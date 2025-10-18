import numpy as np

from artnet import Raster
from games.util.menu_animations.base import MenuAnimation


def vectorized_hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Fast, vectorized HSV to RGB conversion.

    Args:
        h: Hue array (0-255)
        s: Saturation array (0-255)
        v: Value array (0-255)

    Returns:
        RGB array with shape h.shape + (3,) and dtype uint8
    """
    h_norm = h / 255.0
    s_norm = s / 255.0
    v_norm = v / 255.0

    i = np.floor(h_norm * 6)
    f = h_norm * 6 - i
    p = v_norm * (1 - s_norm)
    q = v_norm * (1 - f * s_norm)
    t = v_norm * (1 - (1 - f) * s_norm)

    i = i.astype(np.int32) % 6

    # Create RGB output array
    rgb = np.zeros(h.shape + (3,), dtype=np.float32)

    # Handle each HSV case
    mask = i == 0
    rgb[mask] = np.stack([v_norm[mask], t[mask], p[mask]], axis=-1)
    mask = i == 1
    rgb[mask] = np.stack([q[mask], v_norm[mask], p[mask]], axis=-1)
    mask = i == 2
    rgb[mask] = np.stack([p[mask], v_norm[mask], t[mask]], axis=-1)
    mask = i == 3
    rgb[mask] = np.stack([p[mask], q[mask], v_norm[mask]], axis=-1)
    mask = i == 4
    rgb[mask] = np.stack([t[mask], p[mask], v_norm[mask]], axis=-1)
    mask = i == 5
    rgb[mask] = np.stack([v_norm[mask], p[mask], q[mask]], axis=-1)

    return (rgb * 255).astype(np.uint8)


class RainbowAnimation(MenuAnimation):
    """A rainbow pattern animation for the menu screen."""

    def __init__(self, width: int, height: int, length: int):
        super().__init__(width, height, length)
        self.coords = None
        self.base_speed = 50  # Base animation speed

    def render(self, raster: Raster):
        """Render the rainbow animation."""
        # Create coordinate grids on first frame
        if self.coords is None or self.coords[0].shape != (
            raster.length,
            raster.height,
            raster.width,
        ):
            self.coords = np.indices((raster.length, raster.height, raster.width), sparse=True)

        # Get current time and adjust speed based on input
        current_time = self.last_update_time
        speed = self.base_speed * (1 + self.state.input_intensity * 2)

        # Calculate base hue pattern
        z_coords, y_coords, x_coords = self.coords
        hue = (x_coords + y_coords + z_coords) * 4 + current_time * speed
        hue = hue.astype(np.int32) % 256

        # Create saturation and value arrays
        saturation = np.full_like(hue, 255, dtype=np.uint8)
        value = np.full_like(hue, 255, dtype=np.uint8)

        # Adjust value based on voting state
        if self.state.active_players:
            vote_percentage = len(self.state.voted_players) / len(self.state.active_players)
            value = (value * (0.5 + vote_percentage * 0.5)).astype(np.uint8)

        # Convert to RGB and assign to raster
        raster.data[:] = vectorized_hsv_to_rgb(hue, saturation, value)
