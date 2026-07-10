"""Exception types."""

from __future__ import annotations


class TwelveAngryError(Exception):
    """Base class for package errors."""


class ParseError(TwelveAngryError):
    """A judge's raw output could not be parsed into a valid annotation."""


class JudgeError(TwelveAngryError):
    """A judge failed to produce a usable annotation after all retries."""


class MissingDependencyError(TwelveAngryError):
    """An optional dependency is required for this feature."""

    def __init__(self, package: str, extra: str) -> None:
        super().__init__(
            f"This feature requires the '{package}' package. "
            f'Install it with: pip install "twelve-angry-llms[{extra}]"'
        )
