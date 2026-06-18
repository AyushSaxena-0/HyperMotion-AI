from __future__ import annotations

from dataclasses import dataclass
import json
import queue
import subprocess
import threading
from typing import Callable

import numpy as np

from config import AppConfig


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frames: int
    duration: float
    codec: str
    pixel_format: str
    has_audio: bool


def _rate(value: str | None) -> float:
    if not value or value == "0/0":
        return 0.0
    numerator, denominator = value.split("/", 1)
    return float(numerator) / float(denominator)


def probe_video(path: str, config: AppConfig) -> VideoInfo:
    command = [
        config.ffprobe, "-v", "error", "-show_streams", "-show_format",
        "-of", "json", path,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    videos = [s for s in data["streams"] if s.get("codec_type") == "video"]
    if not videos:
        raise ValueError("The input has no video stream.")
    stream = videos[0]
    fps = _rate(stream.get("avg_frame_rate")) or _rate(stream.get("r_frame_rate"))
    duration = float(stream.get("duration") or data.get("format", {}).get("duration") or 0)
    frames = int(stream.get("nb_frames") or round(duration * fps))
    return VideoInfo(
        path=path,
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=fps,
        frames=frames,
        duration=duration,
        codec=stream.get("codec_name", "unknown"),
        pixel_format=stream.get("pix_fmt", "unknown"),
        has_audio=any(s.get("codec_type") == "audio" for s in data["streams"]),
    )


class DecodeWorker(threading.Thread):
    """NVDEC-backed producer. Raw RGB is the FFmpeg/Python transfer boundary."""

    def __init__(
        self,
        info: VideoInfo,
        output: queue.Queue,
        stop: threading.Event,
        config: AppConfig,
        use_nvdec: bool,
        log: Callable[[str], None],
    ) -> None:
        super().__init__(name="decode-worker", daemon=True)
        self.info, self.output, self.stop = info, output, stop
        self.config, self.use_nvdec, self.log = config, use_nvdec, log
        self.error: Exception | None = None
        self.process: subprocess.Popen | None = None

    def run(self) -> None:
        command = [self.config.ffmpeg, "-hide_banner", "-loglevel", "error"]
        if self.use_nvdec:
            command += ["-hwaccel", "cuda"]
        command += [
            "-i", self.info.path, "-map", "0:v:0", "-an", "-sn", "-dn",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
        ]
        size = self.info.width * self.info.height * 3
        self.log("Decode worker: FFmpeg NVDEC" if self.use_nvdec else "Decode worker: FFmpeg software decode")
        try:
            self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=size * 2)
            index = 0
            assert self.process.stdout is not None
            while not self.stop.is_set():
                data = self.process.stdout.read(size)
                if not data:
                    break
                if len(data) != size:
                    raise RuntimeError(f"Truncated decoded frame ({len(data)} of {size} bytes)")
                frame = np.frombuffer(data, dtype=np.uint8).reshape(self.info.height, self.info.width, 3).copy()
                while not self.stop.is_set():
                    try:
                        self.output.put((index, frame), timeout=0.1)
                        break
                    except queue.Full:
                        pass
                index += 1
            if not self.stop.is_set():
                return_code = self.process.wait()
                if return_code:
                    stderr = self.process.stderr.read().decode(errors="replace") if self.process.stderr else ""
                    raise RuntimeError(f"FFmpeg decode failed: {stderr[-1000:]}")
        except Exception as exc:
            self.error = exc
            self.stop.set()
        finally:
            if self.process and self.process.poll() is None:
                self.process.terminate()
            while not self.stop.is_set():
                try:
                    self.output.put(None, timeout=0.1)
                    break
                except queue.Full:
                    pass
