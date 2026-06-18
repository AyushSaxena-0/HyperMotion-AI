from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os


ROOT = Path(__file__).resolve().parent


def _binary(name: str) -> str:
    local = ROOT / (f"{name}.exe" if os.name == "nt" else name)
    return str(local) if local.exists() else name


@dataclass(frozen=True)
class AppConfig:
    ffmpeg: str = field(default_factory=lambda: _binary("ffmpeg"))
    ffprobe: str = field(default_factory=lambda: _binary("ffprobe"))
    output_dir: Path = field(default_factory=lambda: ROOT / "outputs")
    decode_queue_size: int = 8
    encode_queue_size: int = 16
    scene_threshold: float = 0.32
    preview_width: int = 640
    ort_device_id: int = 0
    trt_cache_dir: Path = field(default_factory=lambda: ROOT / ".trt_cache")

    def prepare(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trt_cache_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ProcessOptions:
    target_fps: float
    codec: str = "H.264"
    backend: str = "TensorRT FP16"
    model_path: str = ""
    scene_threshold: float = 0.32
    cq: int = 19
    preset: str = "p4"
    use_nvdec: bool = True


CODECS = {
    "H.264": ("h264_nvenc", ".mp4"),
    "H.265": ("hevc_nvenc", ".mp4"),
    "AV1": ("av1_nvenc", ".mkv"),
}
