import math
import random
from dataclasses import dataclass
from typing import List

from artnet import HSV, RGB, Raster, Scene


@dataclass
class SphericalWave:
    """A spherical wave expanding from a source"""

    x: float
    y: float
    z: float
    radius: float
    max_radius: float
    speed: float
    birth_time: float
    response_type: str  # 'success', 'error', 'timeout'


class SphericalWavesScene(Scene):
    """
    Concentric spherical waves emanating from various points.
    Represents API request/response patterns.
    """

    def __init__(self, **kwargs):
        properties = kwargs.get("properties")
        if not properties:
            raise ValueError("SphericalWavesScene requires a 'properties' object.")

        self.width = properties.width
        self.height = properties.height
        self.length = properties.length

        # Wave configuration
        self.waves: List[SphericalWave] = []
        self.max_concurrent_waves = 3  # Limit for performance
        self.wave_spawn_rate = 1.5  # per second
        self.next_wave_spawn = 0.0

        # Response types with colors
        self.response_types = {
            "success": {"hue": 120, "probability": 0.7},  # Green
            "error": {"hue": 0, "probability": 0.2},  # Red
            "timeout": {"hue": 45, "probability": 0.1},  # Yellow/orange
        }

        # Wave properties
        self.wave_speed = 10.0  # units per second
        self.wave_thickness = 2.0
        self.max_wave_radius = 25.0

        print("SphericalWavesScene initialized")

    def _spawn_wave(self, time: float):
        """Spawn a new spherical wave"""
        # Random source position
        x = random.uniform(self.width * 0.2, self.width * 0.8)
        y = random.uniform(self.height * 0.2, self.height * 0.8)
        z = random.uniform(self.length * 0.2, self.length * 0.8)

        # Choose response type based on probabilities
        rand = random.random()
        cumulative = 0.0
        response_type = "success"
        for rtype, props in self.response_types.items():
            cumulative += props["probability"]
            if rand <= cumulative:
                response_type = rtype
                break

        wave = SphericalWave(
            x=x,
            y=y,
            z=z,
            radius=0.0,
            max_radius=self.max_wave_radius,
            speed=self.wave_speed,
            birth_time=time,
            response_type=response_type,
        )
        self.waves.append(wave)

    def _update_waves(self, dt: float):
        """Update wave radii"""
        updated_waves = []
        for wave in self.waves:
            wave.radius += wave.speed * dt
            if wave.radius < wave.max_radius:
                updated_waves.append(wave)
        self.waves = updated_waves

    def render(self, raster: Raster, time: float):
        """Render the spherical waves"""
        dt = 1.0 / 60.0
        raster.data.fill(0)

        # Spawn new waves (only if under limit)
        if time >= self.next_wave_spawn and len(self.waves) < self.max_concurrent_waves:
            self._spawn_wave(time)
            self.next_wave_spawn = time + (1.0 / self.wave_spawn_rate)

        self._update_waves(dt)

        # Render waves as hollow spheres
        for wave in self.waves:
            response_props = self.response_types[wave.response_type]
            hue = response_props["hue"]

            # Fade as wave expands
            age_factor = 1.0 - (wave.radius / wave.max_radius)
            brightness = int(255 * age_factor)

            if brightness < 10:
                continue

            color = RGB.from_hsv(HSV(hue, 255, brightness))

            # Render spherical shell
            self._draw_spherical_shell(
                raster, wave.x, wave.y, wave.z, wave.radius, self.wave_thickness, color
            )

    def _draw_spherical_shell(self, raster, cx, cy, cz, radius, thickness, color):
        """Draw a hollow spherical shell"""
        # Bounding box
        x_min = max(0, int(cx - radius - thickness))
        x_max = min(self.width - 1, int(cx + radius + thickness))
        y_min = max(0, int(cy - radius - thickness))
        y_max = min(self.height - 1, int(cy + radius + thickness))
        z_min = max(0, int(cz - radius - thickness))
        z_max = min(self.length - 1, int(cz + radius + thickness))

        for z in range(z_min, z_max + 1):
            for y in range(y_min, y_max + 1):
                for x in range(x_min, x_max + 1):
                    dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)

                    # Check if on shell surface
                    if abs(dist - radius) <= thickness:
                        # Intensity based on distance from shell center
                        shell_dist = abs(dist - radius)
                        intensity = 1.0 - (shell_dist / thickness)
                        intensity = max(0.0, min(1.0, intensity))

                        # Additive blending for interference patterns
                        raster.data[z, y, x, 0] = min(
                            255, raster.data[z, y, x, 0] + int(color.red * intensity)
                        )
                        raster.data[z, y, x, 1] = min(
                            255, raster.data[z, y, x, 1] + int(color.green * intensity)
                        )
                        raster.data[z, y, x, 2] = min(
                            255, raster.data[z, y, x, 2] + int(color.blue * intensity)
                        )
