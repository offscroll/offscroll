"""OffScroll logging configuration.

All components use standard library logging:

    import logging
    logger = logging.getLogger(__name__)
    logger.info("Ingested %d items", count)

Do not use print() for operational output. Use click.echo() in CLI
commands only. Use logging everywhere else.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(level: int = logging.INFO, log_file: str | None = None) -> None:
    """Configure root logger with console and optional file handler."""
    root = logging.getLogger("offscroll")
    root.setLevel(level)

    # Console handler -- always
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.addHandler(console)

    # File handler -- if configured
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
            )
        )
        root.addHandler(file_handler)
