"""Entrypoint for Universal RAG System.

This module loads configuration, initializes logging, and starts the Gradio UI.
"""

import os
import sys


def _ensure_src_on_path() -> None:
    """Ensure the local `src` directory is importable when running `python main.py`."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(project_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


_ensure_src_on_path()

from utils.logger import init_logger  # noqa: E402
from config import AppConfig, build_app_config, load_settings  # noqa: E402
from api.gradio_ui import create_interface  # noqa: E402


def main() -> None:
    """Application bootstrap: config, logger, UI."""
    settings = load_settings()
    config: AppConfig = build_app_config(settings)
    logger = init_logger(config)
    logger.info("Starting Universal RAG System UI")

    iface = create_interface(config)
    # Launch in local mode; adjust share parameter if needed
    iface.launch(share=False, inbrowser=False)


if __name__ == "__main__":
    main()


