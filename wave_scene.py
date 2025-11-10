import math
from typing import Tuple

import numpy as np

from artnet import HSV, RGB, Raster, Scene


class SinusoidalWaveScene(Scene):
    """
    A sinusoidal plane wave visualization oriented in the YZ plane.
    The wave propagates along the X axis with animated amplitude, offset, and phase.
    """

    def __init__(self, **kwargs):
        properties = kwargs.get("properties")
        if not properties:
            raise ValueError("SinusoidalWaveScene requires a 'properties' object.")

        self.width = properties.width
        self.height = properties.height
        self.length = properties.length

        # Determine minor axis for period calculation
        self.minor_axis = min(self.height, self.length)

        # Base wave parameters - period matches minor axis
        # For one complete wave cycle over minor axis: k = 2π / minor_axis
        self.base_wave_number = 2 * math.pi / self.minor_axis

        # Wave coefficient modulation
        self.ky_oscillation_speed = 0.15  # How fast Y coefficient changes
        self.kz_oscillation_speed = 0.23  # How fast Z coefficient changes (different for variety)
        self.ky_range = 0.5  # Variation range for Y coefficient (0.5 to 1.5 of base)
        self.kz_range = 0.5  # Variation range for Z coefficient

        self.time_speed = 0.5  # How fast phase changes over time (reduced from 1.5)
        self.amplitude_speed = 0.5  # How fast amplitude oscillates

        # Amplitude modulation - increased range to reach ceiling
        self.min_amplitude = 3.0
        self.max_amplitude = 12.0

        # Offset will be inversely proportional to amplitude
        # When amplitude is small, offset pushes towards ceiling (negative X)
        # When amplitude is large, offset is more neutral
        self.max_offset = -8.0  # Negative = towards ceiling (X=0)
        self.min_offset = 0.0  # Neutral when amplitude is large

        # Color parameters
        self.hue_shift_speed = 20.0  # Degrees per second
        self.base_hue = 180.0  # Start with cyan

        print(f"SinusoidalWaveScene initialized: {self.width}x{self.height}x{self.length}")
        print(f"Wave period set to minor axis: {self.minor_axis} units")

    def _get_wave_parameters(self, time: float) -> Tuple[float, float, float, float, float, float]:
        """Calculate animated wave parameters based on time"""
        # Amplitude oscillates smoothly
        amplitude_factor = 0.5 + 0.5 * math.sin(time * self.amplitude_speed)
        amplitude = (
            self.min_amplitude + (self.max_amplitude - self.min_amplitude) * amplitude_factor
        )

        # Offset inversely proportional to amplitude
        # When amplitude is at minimum (amplitude_factor = 0), offset is at max_offset (towards ceiling)
        # When amplitude is at maximum (amplitude_factor = 1), offset is at min_offset (neutral)
        offset = self.max_offset + (self.min_offset - self.max_offset) * amplitude_factor

        # Phase advances over time (creates wave motion)
        phase = time * self.time_speed * 2 * math.pi

        # Wave numbers vary independently over time
        # Y coefficient oscillates between 0.5x and 1.5x the base value
        ky_factor = 1.0 + self.ky_range * math.sin(time * self.ky_oscillation_speed)
        wave_number_y = self.base_wave_number * ky_factor

        # Z coefficient oscillates at a different rate for more interesting patterns
        kz_factor = 1.0 + self.kz_range * math.sin(time * self.kz_oscillation_speed)
        wave_number_z = self.base_wave_number * kz_factor

        # Hue shifts over time
        hue = (self.base_hue + time * self.hue_shift_speed) % 360.0

        return amplitude, offset, phase, hue, wave_number_y, wave_number_z

    def render(self, raster: Raster, time: float):
        """Render the sinusoidal wave"""
        # Clear raster
        raster.data.fill(0)

        # Get current wave parameters
        amplitude, offset, phase, hue, wave_number_y, wave_number_z = self._get_wave_parameters(
            time
        )

        # Create coordinate grids for Y and Z
        y_coords = np.arange(self.height)
        z_coords = np.arange(self.length)
        y_grid, z_grid = np.meshgrid(y_coords, z_coords, indexing="ij")

        # Center the coordinates
        y_centered = y_grid - self.height / 2
        z_centered = z_grid - self.length / 2

        # Calculate the 2D sinusoidal wave varying in both Y and Z
        # Wave equation: x = A * sin(k_y*y + k_z*z + phase) + offset
        # Wave numbers vary over time, creating evolving patterns
        x_wave = (
            amplitude * np.sin(wave_number_y * y_centered + wave_number_z * z_centered + phase)
            + offset
        )

        # Shift to raster coordinates (centered)
        x_wave_coords = x_wave + self.width / 2

        # Render the wave with thickness for visibility
        thickness = 0.75  # Voxels thick

        for y_idx in range(self.height):
            for z_idx in range(self.length):
                x_center = x_wave_coords[y_idx, z_idx]

                # Full brightness and saturation for vibrant colors
                brightness = 255
                saturation = 255

                # Convert hue from 0-360 to 0-255 range for HSV
                hue_255 = int((hue / 360.0) * 255)

                color = RGB.from_hsv(HSV(hue_255, saturation, brightness))

                # Rasterize with thickness
                x_min = int(x_center - thickness)
                x_max = int(x_center + thickness) + 1

                for x in range(x_min, x_max):
                    if 0 <= x < self.width:
                        # Distance-based falloff for smooth appearance
                        dist = abs(x - x_center)
                        if dist <= thickness:
                            intensity = 1.0 - (dist / thickness) * 0.5
                            intensity = max(0.0, min(1.0, intensity))

                            # Apply color with intensity
                            raster.data[z_idx, y_idx, x, 0] = int(color.red * intensity)
                            raster.data[z_idx, y_idx, x, 1] = int(color.green * intensity)
                            raster.data[z_idx, y_idx, x, 2] = int(color.blue * intensity)
