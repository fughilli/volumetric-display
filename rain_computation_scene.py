import math
import random
from dataclasses import dataclass
from typing import List

from artnet import RGB, Raster, Scene
from color_palette import get_palette


@dataclass
class ComputeStream:
    """A vertical stream of computation"""

    y: float
    z: float
    speed: float
    intensity: int  # Base brightness
    phase: float  # For oscillation
    last_fork_x: float
    hue: int


@dataclass
class StreamConnection:
    """A horizontal connection between streams"""

    y1: float
    z1: float
    y2: float
    z2: float
    x: float
    birth_time: float
    lifetime: float


class RainComputationScene(Scene):
    """
    Vertical streams of computation flowing downward with occasional connections.
    Represents parallel processing and thread synchronization.
    """

    def __init__(self, **kwargs):
        properties = kwargs.get("properties")
        if not properties:
            raise ValueError("RainComputationScene requires a 'properties' object.")

        self.width = properties.width
        self.height = properties.height
        self.length = properties.length

        # Stream configuration
        self.num_streams = 50  # Increased for higher line density
        self.streams: List[ComputeStream] = []

        # Get color palette
        palette = get_palette()

        # Available stream colors from palette (we'll use indices 2, 1, 0)
        stream_colors = [palette.get_color(2), palette.get_color(1), palette.get_color(0)]

        # Initialize streams
        for _ in range(self.num_streams):
            stream = ComputeStream(
                y=random.uniform(0, self.height),
                z=random.uniform(0, self.length),
                speed=random.uniform(3.0, 12.0),
                intensity=random.randint(150, 255),
                phase=random.uniform(0, 2 * math.pi),
                last_fork_x=0.0,
                hue=0,  # Will be replaced by color below
            )
            # Store color instead of hue (reusing hue field for compatibility)
            stream.color = random.choice(stream_colors)
            self.streams.append(stream)

        # Connections between streams
        self.connections: List[StreamConnection] = []
        self.next_connection_spawn = 0.0
        self.connection_spawn_rate = 2.0  # per second

        print(f"RainComputationScene initialized with {self.num_streams} streams")

    def _spawn_connection(self, time: float):
        """Spawn a horizontal connection between two nearby streams"""
        if len(self.streams) < 2:
            return

        # Pick random X position
        x = random.uniform(self.width * 0.2, self.width * 0.8)

        # Pick two random streams
        s1, s2 = random.sample(self.streams, 2)

        connection = StreamConnection(
            y1=s1.y,
            z1=s1.z,
            y2=s2.y,
            z2=s2.z,
            x=x,
            birth_time=time,
            lifetime=1.0,
        )
        self.connections.append(connection)

    def _update_connections(self, time: float):
        """Remove expired connections"""
        self.connections = [c for c in self.connections if time - c.birth_time < c.lifetime]

    def render(self, raster: Raster, time: float):
        """Render the computation rain"""
        raster.data.fill(0)

        # Spawn connections
        if time >= self.next_connection_spawn:
            self._spawn_connection(time)
            self.next_connection_spawn = time + (1.0 / self.connection_spawn_rate)

        self._update_connections(time)

        # Render streams as continuous vertical lines with smooth modulation
        for stream in self.streams:
            # Fixed Y, Z position for this stream (no drift to avoid jitter)
            y = stream.y
            z = stream.z

            # Global intensity oscillation for this stream
            intensity_mod = 0.7 + 0.3 * math.sin(time * 0.5 + stream.phase)
            base_intensity = stream.intensity * intensity_mod / 255.0

            # Calculate wavelength for modulation (based on speed for visual variety)
            wavelength = 8.0 + stream.speed * 0.5  # Longer wavelengths for faster streams

            # Phase that moves downward over time (along X axis)
            # Use negative to make it move down (increasing X direction)
            phase_offset = -time * stream.speed * 0.5

            # Draw continuous vertical line along X axis
            for x in range(self.width):
                # Smooth sinusoidal modulation along the line
                # Position in the wave (moves downward with phase_offset)
                wave_position = (x + phase_offset) / wavelength
                modulation = 0.3 + 0.7 * (0.5 + 0.5 * math.sin(2 * math.pi * wave_position))

                # Combined brightness factor
                brightness_factor = base_intensity * modulation

                if brightness_factor > 0.05:  # Skip very dim points
                    # Scale stream color by brightness factor
                    base_color = stream.color
                    color = RGB(
                        int(base_color.red * brightness_factor),
                        int(base_color.green * brightness_factor),
                        int(base_color.blue * brightness_factor),
                    )
                    self._draw_point(raster, x, y, z, color, 0.5)

        # Render connections (horizontal lines between streams)
        for conn in self.connections:
            age = time - conn.birth_time
            fade = 1.0 - (age / conn.lifetime)
            brightness = int(255 * fade)

            if brightness > 10:
                # Yellow/white for sync connections
                color = RGB(brightness, brightness, brightness)
                self._draw_line(
                    raster, conn.x, conn.y1, conn.z1, conn.x, conn.y2, conn.z2, color, 0.5
                )

    def _draw_point(self, raster, x, y, z, color, size=0.5):
        """Draw an antialiased point with optional thickness"""
        # Integer coordinates of nearest voxel
        ix = int(x)
        iy = int(y)
        iz = int(z)

        # Fractional parts for antialiasing
        fx = x - ix
        fy = y - iy
        fz = z - iz

        # Draw 8 neighboring voxels with interpolated weights (trilinear)
        for dx_offset in [0, 1]:
            for dy_offset in [0, 1]:
                for dz_offset in [0, 1]:
                    px = ix + dx_offset
                    py = iy + dy_offset
                    pz = iz + dz_offset

                    if 0 <= px < self.width and 0 <= py < self.height and 0 <= pz < self.length:
                        # Trilinear interpolation weight
                        weight_x = fx if dx_offset == 1 else (1 - fx)
                        weight_y = fy if dy_offset == 1 else (1 - fy)
                        weight_z = fz if dz_offset == 1 else (1 - fz)
                        weight = weight_x * weight_y * weight_z

                        # Apply antialiased color with max blending
                        raster.data[pz, py, px, 0] = max(
                            raster.data[pz, py, px, 0], int(color.red * weight)
                        )
                        raster.data[pz, py, px, 1] = max(
                            raster.data[pz, py, px, 1], int(color.green * weight)
                        )
                        raster.data[pz, py, px, 2] = max(
                            raster.data[pz, py, px, 2], int(color.blue * weight)
                        )

    def _draw_line(self, raster, x0, y0, z0, x1, y1, z1, color, thickness=0.5):
        """Draw an antialiased line"""
        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
        steps = int(max(abs(dx), abs(dy), abs(dz)) * 2)
        if steps == 0:
            return
        x_inc, y_inc, z_inc = dx / steps, dy / steps, dz / steps
        x, y, z = x0, y0, z0
        for _ in range(steps + 1):
            # Use the antialiased draw_point method
            self._draw_point(raster, x, y, z, color, thickness)
            x += x_inc
            y += y_inc
            z += z_inc
