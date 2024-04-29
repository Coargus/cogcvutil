"""Color Converter Module."""

from __future__ import annotations


def hex_to_bgr(hex_color: str) -> tuple:
    """Convert a hex color to BGR (Blue, Green, Red).

    Args:
        hex_color (str): The color in hex format.

    Returns:
        tuple: The color in BGR format.
    """
    hex_color = hex_color.lstrip("#")
    lv = len(hex_color)
    return tuple(
        int(hex_color[i : i + lv // 3], 16) for i in range(0, lv, lv // 3)
    )[::-1]
