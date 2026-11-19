"""python -m support_bot.demo [--port 8000] [--no-browser]"""
from __future__ import annotations

import argparse
import threading
import webbrowser

import uvicorn

from .server import create_demo_app


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Support Bot local demo")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    url = f"http://127.0.0.1:{args.port}/demo"
    print(f"\n  AI Support Bot demo: {url}\n")
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_demo_app(), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
