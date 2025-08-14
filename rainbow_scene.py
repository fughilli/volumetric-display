import numpy as np

from artnet import Scene  # RGB and HSV are no longer needed for this implementation


def vectorized_hsv_to_rgb(h, s, v):
    """
    A fast, NumPy-based conversion from HSV to RGB.
    Inputs H, S, V are NumPy arrays of shape (L, H, W).
    Output is a NumPy array of shape (L, H, W, 3) with dtype=uint8.
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

    # Create an empty array for the RGB output
    rgb = np.zeros(h.shape + (3,), dtype=np.float32)

    # Use boolean array indexing for each of the 6 HSV cases
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


class RainbowScene(Scene):
    def __init__(self, **kwargs):
        self.coords = None

    def render(self, raster, time):
        """Renders a pattern on a single, large raster using NumPy."""
        # Create coordinate grids once on the first frame
        if self.coords is None or self.coords[0].shape != (
            raster.length,
            raster.height,
            raster.width,
        ):
            self.coords = np.indices((raster.length, raster.height, raster.width), sparse=True)

        z_coords, y_coords, x_coords = self.coords

        # Calculate hue for all voxels at once
        hue = (x_coords + y_coords + z_coords) * 4 + time * 50
        hue = hue.astype(np.int32) % 256

        # Create full arrays for saturation and value
        saturation = np.full_like(hue, 255, dtype=np.uint8)
        value = np.full_like(hue, 255, dtype=np.uint8)

        # Convert the entire HSV buffer to RGB in one go and assign it
        raster.data[:] = vectorized_hsv_to_rgb(hue, saturation, value)
