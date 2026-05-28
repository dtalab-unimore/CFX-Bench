import logging


def setup_logging(log_file="bt.log", level=logging.INFO):
    """Configure logging to both stderr and a file."""
    log_format = "%(asctime)s | %(levelname)-7s | %(message)s"
    date_format = "%H:%M:%S"

    # Clear any existing handlers (helps when re-running in notebooks)
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    formatter = logging.Formatter(log_format, datefmt=date_format)

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    # File handler — captures everything at DEBUG level for post-hoc inspection
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    root.setLevel(logging.DEBUG)
    return logging.getLogger(__name__)

