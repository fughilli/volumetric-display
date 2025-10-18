import math
import random
from dataclasses import dataclass
from typing import List, Tuple

from artnet import HSV, RGB, Raster
from games.util.menu_animations.base import MenuAnimation


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
        # Calculate the diagonal length of the cube for spawn distance
        width, height, length = dimensions
        diagonal = math.sqrt(width**2 + height**2 + length**2)

        # Spawn planes at 1.5x diagonal distance to ensure smooth entry/exit
        spawn_distance = diagonal * 1.5

        # Randomly choose whether plane enters from positive or negative direction
        if random.random() < 0.5:
            position = -spawn_distance
            end_position = spawn_distance
            direction = 1
        else:
            position = spawn_distance
            end_position = -spawn_distance
            direction = -1

        # Create random normal vector
        normal = [random.uniform(-1, 1) for _ in range(3)]
        norm = math.sqrt(sum(n**2 for n in normal))
        normal = [n / norm for n in normal]  # Normalize

        # Random velocity and color
        velocity = (
            random.uniform(0.3, 0.6) * direction
        )  # Velocity direction based on spawn position

        color = RGB.from_hsv(HSV(random.randint(0, 255), 255, 255))

        return cls(position, normal, velocity, color, end_position)

    def update(self, dt: float) -> None:
        """Update plane position."""
        self.position += self.velocity * dt

    def is_alive(self) -> bool:
        """Check if plane is still active."""
        return self.position < self.end_position

    def get_plane_points(
        self, dimensions: Tuple[int, int, int], thickness: float = 0.5
    ) -> List[Tuple[int, int, int]]:
        """Get the points that make up the plane using a 3D line drawing approach.

        Args:
            dimensions: (width, height, length) of the display
            thickness: Thickness of the plane in voxels

        Returns:
            List of (x, y, z) points that make up the plane
        """
        width, height, length = dimensions
        nx, ny, nz = self.normal

        # Calculate center offsets
        center_x = width / 2
        center_y = height / 2
        center_z = length / 2

        # Adjust plane equation for centered coordinate system
        # Original equation: nx*x + ny*y + nz*z = d
        # After centering: nx*(x-cx) + ny*(y-cy) + nz*(z-cz) = d
        # Expanded: nx*x + ny*y + nz*z = d + nx*cx + ny*cy + nz*cz
        d = self.position + nx * center_x + ny * center_y + nz * center_z

        points = []

        # Find the dominant axis of the normal vector
        dominant_axis = max(range(3), key=lambda i: abs([nx, ny, nz][i]))

        # Based on the dominant axis, we'll iterate over the other two axes
        if dominant_axis == 0:  # X is dominant
            for y in range(height):
                for z in range(length):
                    # Solve for x: nx*x + ny*y + nz*z = d
                    if abs(nx) > 1e-6:  # Avoid division by zero
                        x = (d - ny * y - nz * z) / nx
                        x_int = int(round(x))
                        if 0 <= x_int < width:
                            # Add points within thickness
                            for dx in range(-1, 2):
                                px = x_int + dx
                                if (
                                    0 <= px < width
                                    and abs(nx * px + ny * y + nz * z - d) <= thickness
                                ):
                                    points.append((px, y, z))
        elif dominant_axis == 1:  # Y is dominant
            for x in range(width):
                for z in range(length):
                    # Solve for y: nx*x + ny*y + nz*z = d
                    if abs(ny) > 1e-6:
                        y = (d - nx * x - nz * z) / ny
                        y_int = int(round(y))
                        if 0 <= y_int < height:
                            for dy in range(-1, 2):
                                py = y_int + dy
                                if (
                                    0 <= py < height
                                    and abs(nx * x + ny * py + nz * z - d) <= thickness
                                ):
                                    points.append((x, py, z))
        else:  # Z is dominant
            for x in range(width):
                for y in range(height):
                    # Solve for z: nx*x + ny*y + nz*z = d
                    if abs(nz) > 1e-6:
                        z = (d - nx * x - ny * y) / nz
                        z_int = int(round(z))
                        if 0 <= z_int < length:
                            for dz in range(-1, 2):
                                pz = z_int + dz
                                if (
                                    0 <= pz < length
                                    and abs(nx * x + ny * y + nz * pz - d) <= thickness
                                ):
                                    points.append((x, y, pz))

        return points


class PlaneAnimation(MenuAnimation):
    """A moving planes animation for the menu screen."""

    def __init__(self, width: int, height: int, length: int):
        super().__init__(width, height, length)
        self.planes: List[Plane] = []
        self.dimensions = (width, height, length)

        # Animation parameters
        self.max_planes = 4
        self.spawn_chance = 0.15  # Base spawn chance per frame

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

        # Render each plane using the optimized point generation
        for plane in self.planes:
            # Get all points for this plane
            points = plane.get_plane_points(self.dimensions)

            # Calculate color intensity based on voting state
            intensity = 1.0
            if self.state.active_players:
                intensity = (
                    0.5 + len(self.state.voted_players) / len(self.state.active_players) * 0.5
                )

            # Apply color to all points
            color = RGB(
                int(plane.color.red * intensity),
                int(plane.color.green * intensity),
                int(plane.color.blue * intensity),
            )

            # Set all points in one go
            for x, y, z in points:
                raster.set_pix(x, y, z, color)
