"""Package initialization for iadata700-mangetamain app modules."""

from .app import (
    App,
    AppConfig,
)  # re-export for convenience  # noqa: F401

__all__ = [
    "App",
    "AppConfig",
]
