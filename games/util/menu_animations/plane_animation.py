import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from artnet import HSV, RGB, Raster
from games.util.menu_animations import MenuAnimation


@dataclass
class Plane:
    """A colored plane that moves through space."""

    position: float  # Position along normal vector
    normal: List[float]  # Normal vector [x, y, z]
    velocity: float  # Movement speed
    color: RGB  # Plane color
    end_position: float  # Position at which plane dies

    @classmethod
    def create_random(cls, dimensions: Tuple[int, int, int]) -> "Plane":
        """Create a random plane.

        Args:
            dimensions: (width, height, length) of the display
        """
        # Compute the diagonal of the raster
        raster_size = math.sqrt(sum(d**2 for d in dimensions))
        position = -(raster_size / 2 + 1)
        end_position = -position

        # Create random normal vector
        normal = [random.uniform(-1, 1) for _ in range(3)]
        norm = math.sqrt(sum(n**2 for n in normal))
        normal = [n / norm for n in normal]  # Normalize

        # Random velocity and color
        velocity = random.uniform(0.05, 0.2)
        color = RGB.from_hsv(HSV(random.randint(0, 255), 255, 255))

        return cls(position, normal, velocity, color, end_position)

    def update(self, dt: float) -> None:
        """Update plane position.

        Args:
            dt: Time delta in seconds
        """
        self.position += self.velocity * dt

    def is_alive(self) -> bool:
        """Check if plane is still active."""
        return self.position < self.end_position

    def distance_to_point(self, point: Tuple[float, float, float]) -> float:
        """Calculate distance from a point to the plane.

        Args:
            point: (x, y, z) coordinates
        """
        plane_point = [self.position * n for n in self.normal]
        return abs(np.dot(self.normal, [point[i] - plane_point[i] for i in range(3)]))


class PlaneAnimation(MenuAnimation):
    """A moving planes animation for the menu screen."""

    def __init__(self, width: int, height: int, length: int):
        super().__init__(width, height, length)
        self.planes: List[Plane] = []
        self.dimensions = (width, height, length)

        # Animation parameters
        self.max_planes = 3
        self.spawn_chance = 0.1  # Base spawn chance per frame

    def render(self, raster: Raster):
        """Render the planes animation."""
        # Clear the raster
        raster.data.fill(0)

        # Possibly spawn new plane
        spawn_chance = self.spawn_chance * (1 + self.state.input_intensity)
        if random.random() < spawn_chance and len(self.planes) < self.max_planes:
            self.planes.append(Plane.create_random(self.dimensions))

        # Update planes
        for plane in self.planes:
            plane.update(1.0)  # Using fixed time step for simplicity
        self.planes = [plane for plane in self.planes if plane.is_alive()]

        # Calculate center offsets
        center_x = self.width / 2
        center_y = self.height / 2
        center_z = self.length / 2

        # Render each point
        for x in range(self.width):
            for y in range(self.height):
                for z in range(self.length):
                    point = [
                        x - center_x,
                        y - center_y,
                        z - center_z,
                    ]

                    # Find nearby planes
                    colors = []
                    distances = []
                    for plane in self.planes:
                        distance = plane.distance_to_point(point)
                        if distance < 0.5:  # Consider planes that are close enough
                            colors.append(plane.color)
                            distances.append(distance)

                    # If point is near any planes, calculate color
                    if colors:
                        # Calculate weighted average color
                        total_distance = sum(1 / d for d in distances)
                        interpolated_color = RGB(0, 0, 0)
                        for color, distance in zip(colors, distances):
                            weight = (1 / distance) / total_distance
                            interpolated_color.red += int(color.red * weight)
                            interpolated_color.green += int(color.green * weight)
                            interpolated_color.blue += int(color.blue * weight)

                        # Adjust color intensity based on voting state
                        if self.state.active_players:
                            intensity = (
                                0.5
                                + len(self.state.voted_players)
                                / len(self.state.active_players)
                                * 0.5
                            )
                            interpolated_color.red = int(interpolated_color.red * intensity)
                            interpolated_color.green = int(interpolated_color.green * intensity)
                            interpolated_color.blue = int(interpolated_color.blue * intensity)

                        raster.set_pix(x, y, z, interpolated_color)
                    else:
                        raster.set_pix(x, y, z, RGB(0, 0, 0))
