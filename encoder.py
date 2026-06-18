from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import queue
import subprocess
import threading
import time
import uuid
from typing import Callable

import numpy as np

from config import AppConfig, CODECS, ProcessOptions
from rife_engine import load_rife
from scene_detector import SceneDetector
from video_loader import DecodeWorker, VideoInfo, probe_video


@dataclass
class PipelineStats:
    state: str = "Idle"
    input_fps: float = 0.0
    output_fps: float = 0.0
    processing_fps: float = 0.0
    progress: float = 0.0
    eta: float = 0.0
    decoded: int = 0
    encoded: int = 0
    interpolated: int = 0
    scene_cuts: int = 0
    inference_ms: float = 0.0
    gpu_name: str = "Unavailable"
    gpu_usage: float = 0.0
    vram_used: float = 0.0
    vram_total: float = 0.0
    started: float = 0.0

    def as_dict(self) -> dict:
        return vars(self).copy()


class EncodeWorker(threading.Thread):
    def __init__(self, info: VideoInfo, options: ProcessOptions, frames: queue.Queue, stop: threading.Event,
                 temp_path: Path, config: AppConfig, stats: PipelineStats, log: Callable[[str], None]) -> None:
        super().__init__(name="encode-worker", daemon=True)
        self.info, self.options, self.frames, self.stop = info, options, frames, stop
        self.temp_path, self.config, self.stats, self.log = temp_path, config, stats, log
        self.error: Exception | None = None
        self.process: subprocess.Popen | None = None

    def run(self) -> None:
        encoder = CODECS[self.options.codec][0]
        # Legacy and current FFmpeg builds both accept these named presets.
        preset = {"p1": "fast", "p2": "fast", "p3": "medium", "p4": "medium",
                  "p5": "hq", "p6": "slow", "p7": "slow"}[self.options.preset]
        command = [
            self.config.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v", f"{self.info.width}x{self.info.height}",
            "-r", f"{self.options.target_fps:.8f}", "-i", "pipe:0", "-an",
            "-c:v", encoder, "-preset", preset,
            "-rc", "vbr", "-cq", str(self.options.cq), "-b:v", "0",
            "-pix_fmt", "yuv420p", str(self.temp_path),
        ]
        self.log(f"Encode worker: {encoder} ({self.options.preset}, CQ {self.options.cq})")
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            assert self.process.stdin is not None
            while not self.stop.is_set():
                try:
                    frame = self.frames.get(timeout=0.1)
                except queue.Empty:
                    continue
                if frame is None:
                    break
                self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
                self.stats.encoded += 1
            self.process.stdin.close()
            code = self.process.wait()
            if code and not self.stop.is_set():
                stderr = self.process.stderr.read().decode(errors="replace") if self.process.stderr else ""
                raise RuntimeError(f"FFmpeg NVENC failed: {stderr[-1200:]}")
        except Exception as exc:
            self.error = exc
            self.stop.set()
        finally:
            if self.process and self.process.poll() is None:
                self.process.terminate()


class InferenceWorker(threading.Thread):
    def __init__(self, info: VideoInfo, options: ProcessOptions, decoded: queue.Queue, frames: queue.Queue,
                 stop: threading.Event, config: AppConfig, stats: PipelineStats, log: Callable[[str], None]) -> None:
        super().__init__(name="inference-worker", daemon=True)
        self.info, self.options, self.decoded, self.frames, self.stop = info, options, decoded, frames, stop
        self.config, self.stats, self.log = config, stats, log
        self.error: Exception | None = None

    def _put(self, frame: np.ndarray) -> bool:
        while not self.stop.is_set():
            try:
                self.frames.put(frame, timeout=0.1)
                return True
            except queue.Full:
                pass
        return False

    def run(self) -> None:
        try:
            model = load_rife(self.options, self.config, self.log)
            warmup = model.warmup(self.info.width, self.info.height)
            self.log(f"RIFE warmup complete: {warmup:.1f} ms/frame")
            detector = SceneDetector(self.options.scene_threshold)
            first_item = None
            received = False
            while not received and not self.stop.is_set():
                try:
                    first_item = self.decoded.get(timeout=0.1)
                    received = True
                except queue.Empty:
                    pass
            if first_item is None:
                raise RuntimeError("Decoder produced no frames.")
            source_index, first = first_item
            output_index = 0
            while not self.stop.is_set():
                try:
                    item = self.decoded.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is None:
                    # Hold the last frame for its display interval to preserve duration.
                    while output_index / self.options.target_fps < self.info.duration - 1e-9:
                        if not self._put(first):
                            break
                        output_index += 1
                    break
                next_index, second = item
                self.stats.decoded = next_index + 1
                interval_start = source_index / self.info.fps
                interval_end = next_index / self.info.fps
                cut = detector.is_cut(first, second)
                if cut:
                    self.stats.scene_cuts += 1
                while output_index / self.options.target_fps < interval_end - 1e-9:
                    timestamp = output_index / self.options.target_fps
                    alpha = min(1.0, max(0.0, (timestamp - interval_start) / (interval_end - interval_start)))
                    if alpha <= 1e-6:
                        output = first
                    elif cut:
                        output = first if alpha < 0.5 else second
                    else:
                        started = time.perf_counter()
                        output = model.infer(first, second, alpha)
                        latency = (time.perf_counter() - started) * 1000
                        self.stats.inference_ms = latency if not self.stats.inference_ms else self.stats.inference_ms * 0.9 + latency * 0.1
                        self.stats.interpolated += 1
                    if not self._put(output):
                        break
                    output_index += 1
                source_index, first = next_index, second
        except Exception as exc:
            self.error = exc
            self.stop.set()
        finally:
            while not self.stop.is_set():
                try:
                    self.frames.put(None, timeout=0.1)
                    break
                except queue.Full:
                    pass


class VideoPipeline:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        self.config.prepare()
        self.stop_event = threading.Event()
        self.stats = PipelineStats()
        self.logs: list[str] = []
        self._log_lock = threading.Lock()

    def log(self, message: str) -> None:
        with self._log_lock:
            self.logs.append(f"[{time.strftime('%H:%M:%S')}] {message}")
            self.logs = self.logs[-200:]

    def cancel(self) -> None:
        self.stop_event.set()
        self.stats.state = "Cancelling"
        self.log("Cancellation requested")

    def log_text(self) -> str:
        with self._log_lock:
            return "\n".join(self.logs)

    def _gpu_stats(self) -> None:
        command = ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=2, check=True)
            name, usage, used, total = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
            self.stats.gpu_name, self.stats.gpu_usage = name, float(usage)
            self.stats.vram_used, self.stats.vram_total = float(used), float(total)
        except Exception:
            pass

    def _validate_encoder(self, codec: str) -> None:
        encoder = CODECS[codec][0]
        result = subprocess.run([self.config.ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True)
        if result.returncode or encoder not in result.stdout:
            raise RuntimeError(
                f"This FFmpeg build does not provide {encoder}. Install a current NVIDIA-enabled FFmpeg build."
            )

    def _remux(self, video_path: Path, source: str, output: Path) -> None:
        base = [
            self.config.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(video_path), "-i", source, "-map", "0:v:0", "-map", "1:a?",
            "-map_metadata", "1", "-map_chapters", "1",
        ]
        command = base + ["-c", "copy", "-shortest", str(output)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            self.log("Audio stream copy was incompatible with the output container; transcoding audio to AAC")
            command = base + ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(output)]
            result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"Final audio/metadata mux failed: {result.stderr[-1200:]}")

    def process(self, source: str, options: ProcessOptions, update: Callable[[dict, str], None] | None = None) -> str:
        self.stop_event = threading.Event()
        self.logs = []
        info = probe_video(source, self.config)
        if info.fps <= 0:
            raise ValueError("FFprobe could not determine the input frame rate.")
        if options.target_fps <= info.fps:
            raise ValueError(f"Target FPS ({options.target_fps:g}) must exceed input FPS ({info.fps:.3f}).")
        self._validate_encoder(options.codec)
        self.stats = PipelineStats(state="Starting", input_fps=info.fps, output_fps=options.target_fps, started=time.perf_counter())
        self.log(f"Input: {info.width}x{info.height}, {info.fps:.3f} FPS, {info.codec}")
        self._gpu_stats()

        decoded: queue.Queue = queue.Queue(self.config.decode_queue_size)
        frames: queue.Queue = queue.Queue(self.config.encode_queue_size)
        stem = "".join(c for c in Path(source).stem if c.isalnum() or c in "-_")[:80] or "video"
        extension = CODECS[options.codec][1]
        token = uuid.uuid4().hex[:8]
        temp = self.config.output_dir / f".{stem}_{token}_video{extension}"
        output = self.config.output_dir / f"{stem}_{options.target_fps:g}fps_{token}{extension}"
        decoder = DecodeWorker(info, decoded, self.stop_event, self.config, options.use_nvdec, self.log)
        inference = InferenceWorker(info, options, decoded, frames, self.stop_event, self.config, self.stats, self.log)
        encoder = EncodeWorker(info, options, frames, self.stop_event, temp, self.config, self.stats, self.log)
        workers = [decoder, inference, encoder]
        for worker in workers:
            worker.start()
        self.stats.state = "Processing"
        expected = max(1, round(info.duration * options.target_fps))
        last_gpu = 0.0
        while any(worker.is_alive() for worker in workers):
            elapsed = time.perf_counter() - self.stats.started
            self.stats.processing_fps = self.stats.encoded / elapsed if elapsed else 0.0
            self.stats.progress = min(0.999, self.stats.encoded / expected)
            remaining = max(0, expected - self.stats.encoded)
            self.stats.eta = remaining / self.stats.processing_fps if self.stats.processing_fps else 0.0
            if time.perf_counter() - last_gpu > 1:
                self._gpu_stats()
                last_gpu = time.perf_counter()
            if update:
                update(self.stats.as_dict(), self.log_text())
            time.sleep(0.2)
        for worker in workers:
            worker.join()
        errors = [worker.error for worker in workers if worker.error]
        if errors:
            if temp.exists():
                temp.unlink()
            self.stats.state = "Failed"
            raise errors[0]
        if self.stop_event.is_set():
            if temp.exists():
                temp.unlink()
            self.stats.state = "Cancelled"
            raise RuntimeError("Processing cancelled.")
        self.stats.state = "Muxing audio and metadata"
        if update:
            update(self.stats.as_dict(), self.log_text())
        self._remux(temp, source, output)
        temp.unlink(missing_ok=True)
        elapsed = time.perf_counter() - self.stats.started
        self.stats.processing_fps = self.stats.encoded / elapsed if elapsed else 0.0
        self.stats.state, self.stats.progress, self.stats.eta = "Complete", 1.0, 0.0
        self.log(f"Complete: {output.name}")
        return str(output)
