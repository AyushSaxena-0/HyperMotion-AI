from __future__ import annotations

import json
import html
import os
from pathlib import Path
import threading
import time

import gradio as gr

from benchmark import run_benchmark
from config import AppConfig, ProcessOptions
from encoder import VideoPipeline
from video_loader import probe_video


CSS = """
.gradio-container { max-width: 1480px !important; background: #090d14; }
.hero { padding: 28px 32px; border: 1px solid #263246; border-radius: 14px;
  background: linear-gradient(110deg, #111827, #111c2f); margin-bottom: 18px; box-shadow: 0 12px 40px #0005; }
.eyebrow { color:#60a5fa; font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:.14em; margin-bottom:9px; }
.hero h1 { margin: 0; font-size: 2.35rem; letter-spacing: -.035em; color:#f8fafc; }
.hero p { color: #94a3b8; margin: 9px 0 0; max-width:760px; line-height:1.5; }
.hero-meta { margin-top:15px; color:#64748b; font-size:12px; }
.hero-meta strong { color:#cbd5e1; }
.panel { border: 1px solid #263246 !important; border-radius: 12px !important; box-shadow: 0 6px 24px #0004; }
.metric textarea { font-family: ui-monospace, Consolas, monospace !important; }
.metric-grid { display: grid; grid-template-columns: repeat(4,minmax(0,1fr)); gap: 10px; margin: 4px 0 12px; }
.metric-card { padding: 13px 14px; border-radius: 9px; border: 1px solid #263246; background: #101722; }
.metric-card small { display:block; color:#94a3b8; font-size:11px; text-transform:uppercase; letter-spacing:.08em; }
.metric-card strong { display:block; color:#f1f5f9; font-size:18px; margin-top:3px; }
.progress-shell { padding: 15px; border: 1px solid #263246; border-radius: 10px; background: #101722; }
.progress-head { display:flex; justify-content:space-between; color:#cbd5e1; font-weight:600; margin-bottom:9px; }
.progress-track { height:13px; border-radius:99px; overflow:hidden; background:#1e293b; }
.progress-fill { height:100%; border-radius:99px; background:linear-gradient(90deg,#2563eb,#38bdf8); transition:width .25s ease; }
.preview-note { color:#94a3b8; font-size:12px; text-align:center; margin-top:5px; }
.footer { text-align:center; color:#64748b; font-size:12px; padding:22px 0 8px; }
.footer a { color:#94a3b8; text-decoration:none; } .footer a:hover { color:#60a5fa; }
@media(max-width:800px){.metric-grid{grid-template-columns:repeat(2,minmax(0,1fr));}}
"""

HEAD = """
<meta name="description" content="HyperMotion AI converts low-FPS videos to smooth 60, 120, or 144 FPS using RIFE, NVIDIA CUDA, TensorRT, NVDEC, NVENC, and FFmpeg.">
<meta name="author" content="Ayush Saxena">
<meta name="keywords" content="AI video frame interpolation, RIFE frame interpolation, 60 FPS converter, 120 FPS video, 144 FPS video, NVIDIA CUDA, TensorRT, NVENC, FFmpeg, Gradio">
<meta property="og:title" content="HyperMotion AI - RIFE Video Frame Interpolation">
<meta property="og:description" content="Convert low-FPS footage into fluid 60, 120, or 144 FPS video with GPU-accelerated RIFE interpolation.">
<meta property="og:type" content="website">
<meta property="og:url" content="https://github.com/AyushSaxena-0/HyperMotion-AI">
<meta name="twitter:card" content="summary_large_image">
"""


def _default_model() -> str:
    configured = os.environ.get("RIFE_MODEL", "")
    if configured and Path(configured).is_file():
        return configured
    candidates = list(Path(__file__).parent.glob("models/*.onnx")) + list(Path(__file__).parent.glob("models/*.engine"))
    return str(candidates[0]) if candidates else ""


def _target(choice: str, custom: float) -> float:
    return float(custom if choice == "Custom" else choice)


def _metrics(stats: dict) -> str:
    eta = stats.get("eta", 0)
    cards = [
        ("Processing", f"{stats.get('processing_fps', 0):.1f} FPS"),
        ("Output", f"{stats.get('output_fps', 0):g} FPS"),
        ("ETA", f"{eta:.0f}s" if eta else "--"),
        ("GPU Load", f"{stats.get('gpu_usage', 0):.0f}%"),
        ("AI Frames", f"{stats.get('interpolated', 0):,}"),
        ("Encoded", f"{stats.get('encoded', 0):,}"),
        ("Inference", f"{stats.get('inference_ms', 0):.1f} ms"),
        ("VRAM", f"{stats.get('vram_used', 0):.0f}/{stats.get('vram_total', 0):.0f} MB"),
    ]
    body = "".join(f"<div class='metric-card'><small>{label}</small><strong>{value}</strong></div>" for label, value in cards)
    state = html.escape(str(stats.get("state", "Idle")))
    gpu = html.escape(str(stats.get("gpu_name", "Unavailable")))
    return f"<div class='metric-grid'>{body}</div><div class='preview-note'>{state} · {gpu} · Scene cuts: {stats.get('scene_cuts', 0)}</div>"


def _progress(stats: dict) -> str:
    percent = min(100.0, max(0.0, float(stats.get("progress", 0))) * 100)
    state = html.escape(str(stats.get("state", "Idle")))
    return (f"<div class='progress-shell'><div class='progress-head'><span>{state}</span>"
            f"<span>{percent:.1f}%</span></div><div class='progress-track'>"
            f"<div class='progress-fill' style='width:{percent:.2f}%'></div></div></div>")


def build_app() -> gr.Blocks:
    config = AppConfig()
    config.prepare()
    pipeline = VideoPipeline(config)

    def inspect_video(path: str | None):
        if not path:
            return "Upload a video to inspect it.", 30.0
        try:
            info = probe_video(path, config)
            text = (
                f"**{Path(path).name}**  \n"
                f"{info.width}x{info.height} | **{info.fps:.3f} FPS** | {info.duration:.2f}s  \n"
                f"{info.frames} frames | {info.codec} / {info.pixel_format} | Audio: {'yes' if info.has_audio else 'no'}"
            )
            return text, round(info.fps, 3)
        except Exception as exc:
            return f"**Probe failed:** {exc}", 30.0

    def process_video(path, target_choice, custom_fps, codec, backend, model_path, threshold, cq, preset, nvdec):
        if not path:
            raise gr.Error("Upload an input video first.")
        options = ProcessOptions(
            target_fps=_target(target_choice, custom_fps), codec=codec, backend=backend,
            model_path=model_path or "", scene_threshold=float(threshold), cq=int(cq),
            preset=preset, use_nvdec=bool(nvdec),
        )
        result: dict[str, str | Exception | None] = {"path": None, "error": None}

        def work():
            try:
                result["path"] = pipeline.process(path, options)
            except Exception as exc:
                result["error"] = exc

        thread = threading.Thread(target=work, name="pipeline-controller", daemon=True)
        thread.start()
        while thread.is_alive():
            stats = pipeline.stats.as_dict()
            yield None, _metrics(stats), pipeline.log_text(), _progress(stats)
            time.sleep(0.2)
        thread.join()
        stats = pipeline.stats.as_dict()
        if result["error"]:
            yield None, _metrics(stats), pipeline.log_text() + f"\nERROR: {result['error']}", _progress(stats)
            raise gr.Error(str(result["error"]))
        yield result["path"], _metrics(stats), pipeline.log_text(), _progress(stats)

    def benchmark_model(model_path, backend, resolution, rounds):
        if backend != "Built-in RIFE FP16" and not model_path:
            raise gr.Error("Select a RIFE model first.")
        width, height = (int(value) for value in resolution.split("x"))
        try:
            return json.dumps(run_benchmark(model_path, backend, width, height, int(rounds)), indent=2)
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    with gr.Blocks(title="HyperMotion AI | AI Video Frame Interpolation") as demo:
        gr.HTML("""<header class='hero'><div class='eyebrow'>AI Video Frame Interpolation</div>
        <h1>HyperMotion AI</h1><p>Convert low-frame-rate footage into fluid 60, 120, or 144 FPS video with
        RIFE neural interpolation and NVIDIA hardware acceleration.</p>
        <div class='hero-meta'>Created and engineered by <strong>Ayush Saxena</strong> · RTX-optimized · Local processing</div></header>""")
        input_fps_state = gr.State(30.0)
        with gr.Row():
            with gr.Column(scale=6, elem_classes=["panel"]):
                source = gr.Video(label="Input Video", sources=["upload"], format=None)
                source_info = gr.Markdown("Upload a video to inspect it.")
            with gr.Column(scale=5, elem_classes=["panel"]):
                model_path = gr.Textbox(label="Optional ONNX / TensorRT Engine", value=_default_model(), placeholder="Not needed for Built-in RIFE")
                backend = gr.Dropdown(["Built-in RIFE FP16", "TensorRT FP16", "CUDA", "CPU"], value="Built-in RIFE FP16", label="Inference Backend")
                with gr.Row():
                    target_choice = gr.Radio(["60", "120", "144", "Custom"], value="60", label="Target FPS")
                    custom_fps = gr.Number(value=90, minimum=1, maximum=480, label="Custom FPS")
                with gr.Row():
                    codec = gr.Dropdown(["H.264", "H.265", "AV1"], value="H.264", label="NVENC Codec")
                    preset = gr.Dropdown(["p1", "p2", "p3", "p4", "p5", "p6", "p7"], value="p4", label="NVENC Preset")
                    cq = gr.Slider(0, 51, value=19, step=1, label="Quality (CQ)")
                scene_threshold = gr.Slider(0.05, 0.8, value=config.scene_threshold, step=0.01, label="Scene Cut Threshold")
                nvdec = gr.Checkbox(value=True, label="Use FFmpeg NVDEC")
                with gr.Row():
                    start = gr.Button("Interpolate", variant="primary")
                    cancel = gr.Button("Cancel", variant="stop")

        with gr.Row():
            with gr.Column(scale=6, elem_classes=["panel"]):
                output = gr.Video(label="Completed Video")
            with gr.Column(scale=5, elem_classes=["panel"]):
                progress = gr.HTML(_progress({}))
                metrics = gr.HTML(_metrics({}))
                logs = gr.Textbox(label="Live Logs", lines=10, elem_classes=["metric"], interactive=False)

        with gr.Accordion("Benchmark Panel", open=False):
            with gr.Row():
                resolution = gr.Dropdown(["640x360", "1280x720", "1920x1080"], value="1920x1080", label="Resolution")
                rounds = gr.Slider(3, 100, value=20, step=1, label="Rounds")
                benchmark_button = gr.Button("Run Benchmark")
            benchmark_output = gr.Code(label="Benchmark Result", language="json")

        gr.HTML("""<footer class='footer'>HyperMotion AI · Built by <strong>Ayush Saxena</strong> ·
        <a href='https://github.com/AyushSaxena-0/HyperMotion-AI' target='_blank'>View source on GitHub</a></footer>""")

        source.change(inspect_video, source, [source_info, input_fps_state])
        run_event = start.click(
            process_video,
            [source, target_choice, custom_fps, codec, backend, model_path, scene_threshold, cq, preset, nvdec],
            [output, metrics, logs, progress],
        )
        cancel.click(lambda: pipeline.cancel(), outputs=None, cancels=[run_event], queue=False)
        benchmark_button.click(benchmark_model, [model_path, backend, resolution, rounds], benchmark_output)
    return demo
