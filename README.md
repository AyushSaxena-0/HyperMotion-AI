# HyperMotion AI - RIFE Video Frame Interpolation

**HyperMotion AI is an NVIDIA-accelerated AI video frame interpolation app that converts 24/30 FPS footage into smooth 60, 120, or 144 FPS video using RIFE, CUDA, TensorRT, NVDEC, NVENC, FFmpeg, and Gradio.**

Created and engineered by [Ayush Saxena](https://github.com/AyushSaxena-0).

## AI Video FPS Converter

HyperMotion AI generates motion-aware intermediate frames between existing video frames. It is designed for creators, animators, gamers, and video enthusiasts who want high-frame-rate video, fluid anime playback, slow-motion footage, or smoother 4K/1080p motion without simple frame duplication.

### Key Features

- Convert 24, 30, or 60 FPS video to **60, 120, 144, or custom FPS**
- Built-in **RIFE v4.25 FP16** inference optimized for NVIDIA RTX GPUs
- Optional **ONNX Runtime CUDA** and **TensorRT FP16** model backends
- FFmpeg **NVDEC hardware decoding** and **NVENC hardware encoding**
- H.264, H.265/HEVC, and AV1 GPU export
- Hard-cut scene detection to prevent interpolation ghosting
- Three-worker asynchronous decode, inference, and encode pipeline
- Live processing percentage, FPS, ETA, GPU usage, VRAM, and logs
- Source audio, chapters, and metadata preservation
- Gradio web interface with local, privacy-friendly processing

## How It Works

```text
Input video
  -> FFmpeg NVDEC hardware decode
  -> bounded frame queue
  -> RIFE optical flow and intermediate-frame generation
  -> bounded encode queue
  -> FFmpeg NVENC hardware encode
  -> audio and metadata mux
  -> smooth high-FPS output video
```

Arbitrary target rates use timestamp-based scheduling, so conversion is not limited to integer FPS multipliers. Hard scene cuts bypass neural blending to avoid cross-shot artifacts.

## Requirements

- Windows 10/11 or Linux
- Python 3.10 or newer
- NVIDIA CUDA-capable GPU; RTX 30/40 series recommended
- Current NVIDIA driver
- Current FFmpeg build with CUDA/NVENC support
- Approximately 4-6 GB VRAM for typical 1080p workloads

## Installation

```bash
git clone https://github.com/AyushSaxena-0/HyperMotion-AI.git
cd HyperMotion-AI
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860), upload a video, choose the target FPS and codec, then select **Interpolate**.

FFmpeg must be available on `PATH` or placed beside `app.py` as `ffmpeg.exe` and `ffprobe.exe` on Windows. Verify hardware support:

```powershell
ffmpeg -hwaccels
ffmpeg -encoders | Select-String nvenc
```

## Inference Backends

### Built-in RIFE FP16

The default backend uses the included RIFE v4.25 model with PyTorch CUDA FP16. No external ONNX model path is required.

### ONNX Runtime and TensorRT

Install `onnxruntime-gpu` for CUDA or TensorRT Execution Provider inference. Direct serialized `.engine`/`.plan` loading additionally requires `tensorrt` and `pycuda`. TensorRT plans are specific to GPU architecture, TensorRT version, and often input resolution.

## RTX Performance Notes

- Use the built-in FP16 or TensorRT FP16 backend.
- Use NVDEC and NVENC to reduce CPU load.
- Preset `p4` provides a practical quality/performance balance.
- Lower CQ values improve visual quality but increase output size.
- Processing speed depends on resolution, target FPS, scene complexity, GPU power limits, and model backend.

Stock FFmpeg subprocess pipes require RGB frames to cross host memory at the decoder and encoder boundaries. True end-to-end CUDA zero-copy requires an in-process C++/CUDA implementation using NVIDIA Video Codec SDK surfaces.

## Project Structure

| File | Purpose |
| --- | --- |
| `app.py` | Application launcher and deployment metadata |
| `ui.py` | Professional Gradio dashboard |
| `encoder.py` | Worker orchestration, NVENC export, and remuxing |
| `rife_engine.py` | PyTorch, ONNX Runtime, and TensorRT backends |
| `video_loader.py` | FFprobe metadata and NVDEC decode worker |
| `scene_detector.py` | Hard-cut detection |
| `benchmark.py` | Model inference benchmark |
| `config.py` | Runtime configuration |

## Search Keywords

AI video frame interpolation, RIFE frame interpolation, video FPS converter, 24 FPS to 60 FPS, 30 FPS to 120 FPS, 144 FPS video converter, NVIDIA CUDA video processing, TensorRT RIFE, NVENC video encoder, NVDEC video decoder, smooth anime video, AI slow motion, high frame rate video, FFmpeg GPU acceleration, Gradio AI video app.

## License

HyperMotion AI is released under the MIT License. RIFE-derived components retain their original copyright and license notices; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
