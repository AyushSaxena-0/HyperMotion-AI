from __future__ import annotations

import argparse

import gradio as gr

from ui import CSS, HEAD, build_app


def main() -> None:
    parser = argparse.ArgumentParser(description="HyperMotion AI video frame interpolation by Ayush Saxena")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=7860, type=int)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    demo = build_app()
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        css=CSS,
        theme=gr.themes.Base(),
        head=HEAD,
    )


if __name__ == "__main__":
    main()
