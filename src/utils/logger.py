import logging
import sys
from pathlib import Path
from typing import Optional

# Default log format (human-readable, production-safe)
LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | "
    "%(message)s"
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
) -> None:
    """
    Configure global logging for the application.
    This should be called ONCE at application startup.
    """

    level = getattr(logging, log_level.upper(), logging.INFO)

    handlers = []

    # Console handler (stdout) → REQUIRED for Docker/K8s
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    handlers.append(console_handler)

    # Optional file handler (local / debugging)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        handlers=handlers,
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a module-specific logger.
    Example: logger = get_logger(__name__)
    """
    return logging.getLogger(name)
