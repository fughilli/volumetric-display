import math
from dataclasses import dataclass
from typing import List

from artnet import RGB, Raster, Scene
from color_palette import get_palette


@dataclass
class HelixPulse:
    """A pulse traveling along a helix"""

    helix_idx: int
    progress: float  # 0.0 to 1.0 along helix height
    speed: float


class HelixRedundancyScene(Scene):
    """
    Three intertwined helical strands rotating around a central axis.
    Represents data replication and availability zones.
    """

    def __init__(self, **kwargs):
        properties = kwargs.get("properties")
        if not properties:
            raise ValueError("HelixRedundancyScene requires a 'properties' object.")

        self.width = properties.width
        self.height = properties.height
        self.length = properties.length

        # Helix configuration
        self.num_helices = 3
        # Diameter equals minor axis, so radius is half of minor axis
        self.helix_radius = min(self.width, self.height) / 2.0
        self.rotation_speed = 0.3  # radians per second
        self.pitch = self.length * 0.8  # Length per full rotation

        # Center of helices (rotating in XY plane, progressing along Z)
        self.center_x = self.width / 2
        self.center_y = self.height / 2

        # Get color palette
        palette = get_palette()

        # Helix colors (different availability zones) from palette
        self.helix_colors = [
            {"color": palette.get_color(2), "name": "Zone A"},  # Blue
            {"color": palette.get_color(1), "name": "Zone B"},  # Green
            {"color": palette.get_color(0), "name": "Zone C"},  # Orange
        ]

        # Pulses traveling up helices
        self.pulses: List[HelixPulse] = []
        self.pulse_spawn_rate = 2.0  # per second
        self.next_pulse_spawn = 0.0

        # Sync points (when all helices pulse together)
        self.sync_interval = 4.0
        self.next_sync = 2.0
        self.last_sync_time = 0.0

        print(f"HelixRedundancyScene initialized with {self.num_helices} helices")

    def _get_helix_point(self, helix_idx: int, progress: float, time: float) -> tuple:
        """Calculate 3D position on helix"""
        # Base angle for this helix (evenly spaced around circle)
        base_angle = (helix_idx / self.num_helices) * 2 * math.pi

        # Add rotation over time
        angle = base_angle + time * self.rotation_speed

        # Progress along Z axis (helix axis)
        z = progress * self.length

        # Add helical twist based on position along Z
        twist = (progress * 2 * math.pi * self.length) / self.pitch
        angle += twist

        # Calculate X, Y on circle (perpendicular to helix axis)
        x = self.center_x + self.helix_radius * math.cos(angle)
        y = self.center_y + self.helix_radius * math.sin(angle)

        return (x, y, z)

    def _spawn_pulse(self, time: float, helix_idx: int = None):
        """Spawn a pulse on a helix"""
        if helix_idx is None:
            helix_idx = int(len(self.pulses) % self.num_helices)

        pulse = HelixPulse(
            helix_idx=helix_idx, progress=0.0, speed=0.25
        )  # 0.25 = 4 seconds to traverse
        self.pulses.append(pulse)

    def _update_pulses(self, dt: float):
        """Update pulse positions"""
        updated_pulses = []
        for pulse in self.pulses:
            pulse.progress += pulse.speed * dt
            if pulse.progress < 1.0:
                updated_pulses.append(pulse)
        self.pulses = updated_pulses

    def render(self, raster: Raster, time: float):
        """Render the helical strands"""
        dt = 1.0 / 60.0
        raster.data.fill(0)

        # Check for sync event
        sync_active = False
        if time >= self.next_sync:
            self.last_sync_time = time
            self.next_sync = time + self.sync_interval
            # Spawn pulse on all helices simultaneously
            for i in range(self.num_helices):
                self._spawn_pulse(time, i)
            sync_active = True

        # Regular pulse spawning
        if time >= self.next_pulse_spawn:
            self._spawn_pulse(time)
            self.next_pulse_spawn = time + (1.0 / self.pulse_spawn_rate)

        self._update_pulses(dt)

        # Render helix strands
        for helix_idx in range(self.num_helices):
            color_info = self.helix_colors[helix_idx]
            base_color = color_info["color"]
            # Dim the base color for the strand
            dimmed_color = RGB(
                int(base_color.red * 0.3), int(base_color.green * 0.3), int(base_color.blue * 0.3)
            )

            # Draw helix strand
            num_segments = 200
            for i in range(num_segments):
                progress = i / num_segments
                x, y, z = self._get_helix_point(helix_idx, progress, time)

                # Draw small sphere at this point
                self._draw_point(raster, x, y, z, dimmed_color, 0.4)

        # Render pulses (bright moving points)
        for pulse in self.pulses:
            x, y, z = self._get_helix_point(pulse.helix_idx, pulse.progress, time)
            color_info = self.helix_colors[pulse.helix_idx]
            bright_color = color_info["color"]

            self._draw_sphere(raster, x, y, z, 1.5, bright_color)

        # Sync visualization (bright flash at all helix points)
        if sync_active or (time - self.last_sync_time < 0.5):
            age = time - self.last_sync_time
            if age < 0.5:
                intensity_factor = 1.0 - age / 0.5
                # Flash all helices with their colors at full brightness
                for helix_idx in range(self.num_helices):
                    color_info = self.helix_colors[helix_idx]
                    base_color = color_info["color"]
                    sync_color = RGB(
                        int(base_color.red * intensity_factor),
                        int(base_color.green * intensity_factor),
                        int(base_color.blue * intensity_factor),
                    )
                    for i in range(0, 200, 10):  # Sample points
                        progress = i / 200
                        x, y, z = self._get_helix_point(helix_idx, progress, time)
                        self._draw_point(raster, x, y, z, sync_color, 0.6)

    def _draw_point(self, raster, x, y, z, color, size=0.5):
        """Draw a small point"""
        ix, iy, iz = int(round(x)), int(round(y)), int(round(z))
        if 0 <= ix < self.width and 0 <= iy < self.height and 0 <= iz < self.length:
            raster.data[iz, iy, ix, 0] = max(raster.data[iz, iy, ix, 0], color.red)
            raster.data[iz, iy, ix, 1] = max(raster.data[iz, iy, ix, 1], color.green)
            raster.data[iz, iy, ix, 2] = max(raster.data[iz, iy, ix, 2], color.blue)

    def _draw_sphere(self, raster, cx, cy, cz, radius, color):
        """Draw a sphere"""
        x_min = max(0, int(cx - radius))
        x_max = min(self.width - 1, int(cx + radius))
        y_min = max(0, int(cy - radius))
        y_max = min(self.height - 1, int(cy + radius))
        z_min = max(0, int(cz - radius))
        z_max = min(self.length - 1, int(cz + radius))

        for z in range(z_min, z_max + 1):
            for y in range(y_min, y_max + 1):
                for x in range(x_min, x_max + 1):
                    dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
                    if dist <= radius:
                        intensity = 1.0 - (dist / radius) * 0.3
                        raster.data[z, y, x, 0] = max(
                            raster.data[z, y, x, 0], int(color.red * intensity)
                        )
                        raster.data[z, y, x, 1] = max(
                            raster.data[z, y, x, 1], int(color.green * intensity)
                        )
                        raster.data[z, y, x, 2] = max(
                            raster.data[z, y, x, 2], int(color.blue * intensity)
                        )
