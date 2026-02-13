"""
Classifier package initialization and logging configuration.

This module sets up a robust logging system for the classifier package, including:
- Rotating file logging to keep log files manageable.
- Console logging for real-time feedback.
- Automatic creation of a logs directory if it does not exist.
- Logging of Python version and platform information at startup.

Attributes
----------
logger : logging.Logger
    The configured logger for the classifier package.

Functions
---------
get_logger() -> logging.Logger
    Returns the configured logger instance for use throughout the package.
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
import platform

logging_st = "[%(asctime)s] %(levelname)s: %(module)s: %(message)s]"
log_dir = "logs"
log_filepath = os.path.join(log_dir, "current_log.log")

os.makedirs(log_dir, exist_ok=True)


file_handler = RotatingFileHandler(
    log_filepath, maxBytes=5*1024*1024, backupCount=5
)
file_handler.setFormatter(logging.Formatter(logging_st))

# Stream handler for console output
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter(logging_st))


logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, stream_handler]
)

logger = logging.getLogger("Classifier")

logger.info(f"Logging initialized. Log file: {log_filepath}")
logger.info(f"Python version: {platform.python_version()} | Platform: {platform.platform()}")

def get_logger():
    """
    Returns the configured logger for the classifier package.

    Returns
    -------
    logging.Logger
        The logger instance configured for the classifier package.
    """
    return logger

__all__ = ["logger"]