import numpy as np

from artnet import (  # RGB and HSV are no longer needed for this implementation
    Raster,
    Scene,
)


class WaveScene(Scene):
    def __init__(self, **kwargs):
        pass

    def render(self, raster: Raster, time: float):
        raster.data.fill(0)

        ht = raster.height
        frequency = 2 * np.pi / 80
        ts = time / 2

        # Create coordinate grids
        x_coords = np.arange(raster.width)
        z_coords = np.arange(raster.length)

        # Create meshgrids for vectorized operations
        X, Z = np.meshgrid(x_coords, z_coords, indexing="ij")

        # Normalize x and z to a range [0, 2π] for smooth sine behavior
        xf = X * frequency
        zf = Z * frequency

        # Calculate wave height for each (x, z) position
        # change the wave shape over time
        Y = (np.sin(xf * np.cos(ts) * 4 + zf * np.sin(ts) + ts) * 20) + (raster.height // 2)
        Y = np.round(Y).astype(int)

        # Clamp Y to valid grid bounds
        Y = np.clip(Y, 0, raster.height - 1)

        # Create 3D coordinate arrays for vectorized assignment
        # We need to find which (x, z) positions have valid y values
        valid_mask = (Y >= 0) & (Y < raster.height)

        # Get valid coordinates
        valid_x = X[valid_mask]
        valid_z = Z[valid_mask]
        valid_y = Y[valid_mask]

        # Calculate colors for all valid positions at once
        # Array is indexed as [z, y, x, channels] based on actual shape (20, 40, 40, 3)
        color_r = 128 + 110 * np.sin(valid_x / ht * time + valid_y / ht * 1.5 + time)
        color_g = 128 + 110 * np.sin(valid_z / ht * 2 + valid_y / ht + time * 2 + np.pi / 2)
        color_b = 128 + 110 * np.sin(valid_x / ht + valid_z / ht + time * 3 + np.pi)

        # Assign colors to raster data
        raster.data[valid_z, valid_y, valid_x, 0] = color_r
        raster.data[valid_z, valid_y, valid_x, 1] = color_g
        raster.data[valid_z, valid_y, valid_x, 2] = color_b
