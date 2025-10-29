"""
Simple logging system for the Mangetamain application.

Provides a centralized logger with appropriate levels and file output.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


class MangetamainLogger:
    """Simple logger for the application with console and file output."""

    def __init__(
        self,
        name: str = "mangetamain",
        level: str = "INFO",
        debug_log_file: Optional[Path] = None,
        error_log_file: Optional[Path] = None,
    ):
        """
        Initialize the logger.

        Args:
            name: Logger name
            level: Logging level (DEBUG, INFO, WARNING, ERROR)
            debug_log_file: Optional file for debug logs (DEBUG+)
            error_log_file: Optional file for error logs (ERROR+)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))

        # Avoid adding handlers multiple times
        if not self.logger.handlers:
            self._setup_handlers(debug_log_file, error_log_file)

    def _setup_handlers(
        self,
        debug_log_file: Optional[Path] = None,
        error_log_file: Optional[Path] = None,
    ):
        """Setup console and optional file handlers."""
        # Console handler with nice formatting - only INFO+ for performance
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # Simple format for console
        console_format = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)

        # Debug file handler if specified (DEBUG+)
        if debug_log_file:
            debug_log_file.parent.mkdir(exist_ok=True)
            debug_handler = logging.FileHandler(debug_log_file)
            debug_handler.setLevel(logging.DEBUG)

            # Detailed format for debug file
            debug_format = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            debug_handler.setFormatter(debug_format)
            self.logger.addHandler(debug_handler)

        # Error file handler if specified (ERROR+)
        if error_log_file:
            error_log_file.parent.mkdir(exist_ok=True)
            error_handler = logging.FileHandler(error_log_file)
            error_handler.setLevel(logging.ERROR)

            # Detailed format for error file
            error_format = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            error_handler.setFormatter(error_format)
            self.logger.addHandler(error_handler)

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(message, **kwargs)

    def exception(self, message: str, **kwargs):
        """Log exception with traceback."""
        self.logger.exception(message, **kwargs)


# Global logger instance
_logger_instance: Optional[MangetamainLogger] = None


def get_logger(name: str = "mangetamain") -> MangetamainLogger:
    """Get the global logger instance."""
    global _logger_instance

    if _logger_instance is None:
        # Default setup with separate debug and error files
        debug_file = Path("debug/debug.log")
        error_file = Path("debug/errors.log")
        _logger_instance = MangetamainLogger(
            name=name,
            level="INFO",  # Production level - less verbose
            debug_log_file=debug_file,
            error_log_file=error_file,
        )

    return _logger_instance


def setup_logging(
    level: str = "INFO",
    debug_log_file: Optional[Path] = None,
    error_log_file: Optional[Path] = None,
) -> MangetamainLogger:
    """
    Setup application logging with specified level and optional files.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        debug_log_file: Optional file for debug logs (DEBUG+)
        error_log_file: Optional file for error logs (ERROR+)

    Returns:
        Configured logger instance
    """
    global _logger_instance

    if debug_log_file is None:
        debug_log_file = Path("debug/debug.log")

    if error_log_file is None:
        error_log_file = Path("debug/errors.log")

    _logger_instance = MangetamainLogger(
        name="mangetamain",
        level=level,
        debug_log_file=debug_log_file,
        error_log_file=error_log_file,
    )
    return _logger_instance


# Convenience functions for quick logging
def log_info(message: str):
    """Quick info log."""
    get_logger().info(message)


def log_warning(message: str):
    """Quick warning log."""
    get_logger().warning(message)


def log_error(message: str):
    """Quick error log."""
    get_logger().error(message)


def log_debug(message: str):
    """Quick debug log."""
    get_logger().debug(message)
