from __future__ import annotations

import abc


class BaseVideoWriter(abc.ABC):
    """Base video writer class."""

    frame_sequence: list

    @abc.abstractmethod
    def write(self) -> None:
        """Write frame image to video."""
        ...
