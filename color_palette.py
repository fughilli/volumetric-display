"""
Color palette management for conference scenes.
Provides predefined color palettes and utilities to pick colors.
"""

import random
from typing import List, Tuple

from artnet import RGB


class ColorPalette:
    """A color palette containing multiple colors"""

    def __init__(self, name: str, colors: List[Tuple[int, int, int]]):
        self.name = name
        self.colors = [RGB(r, g, b) for r, g, b in colors]

    def get_color(self, index: int) -> RGB:
        """Get a color by index (wraps around if index exceeds palette size)"""
        return self.colors[index % len(self.colors)]

    def get_random_color(self) -> RGB:
        """Get a random color from the palette"""
        return random.choice(self.colors)

    def get_colors(self, count: int) -> List[RGB]:
        """Get a list of colors, cycling through the palette if needed"""
        return [self.get_color(i) for i in range(count)]


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color string to RGB tuple"""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


# Define the three default palettes
PALETTE_1 = ColorPalette(
    "Microsoft Brand",
    [
        hex_to_rgb("#F14F21"),  # Orange
        hex_to_rgb("#7EB900"),  # Green
        hex_to_rgb("#00A3EE"),  # Blue
        hex_to_rgb("#FEB800"),  # Yellow
        hex_to_rgb("#FFFFFF"),  # White
    ],
)

PALETTE_2 = ColorPalette(
    "Azure Blues",
    [
        hex_to_rgb("#104581"),  # Dark Blue
        hex_to_rgb("#3CCBF4"),  # Light Cyan
        hex_to_rgb("#0078D4"),  # Azure Blue
        hex_to_rgb("#FFFFFF"),  # White
    ],
)

PALETTE_3 = ColorPalette(
    "Primary & Secondary",
    [
        hex_to_rgb("#FF0000"),  # Red
        hex_to_rgb("#FF7F00"),  # Orange
        hex_to_rgb("#FFFF00"),  # Yellow
        hex_to_rgb("#00FF00"),  # Green
        hex_to_rgb("#0000FF"),  # Blue
        hex_to_rgb("#8000FF"),  # Purple
    ],
)

# Default palette selection
DEFAULT_PALETTE = PALETTE_1

# All available palettes
PALETTES = {
    "brand": PALETTE_1,
    "azure": PALETTE_2,
    "rainbow": PALETTE_3,
}


def get_palette(name: str = None) -> ColorPalette:
    """Get a palette by name, or return the default palette"""
    if name is None:
        return DEFAULT_PALETTE
    return PALETTES.get(name, DEFAULT_PALETTE)


def get_palette_color(index: int, palette_name: str = None) -> RGB:
    """Get a color from a palette by index"""
    palette = get_palette(palette_name)
    return palette.get_color(index)


def get_random_palette_color(palette_name: str = None) -> RGB:
    """Get a random color from a palette"""
    palette = get_palette(palette_name)
    return palette.get_random_color()
