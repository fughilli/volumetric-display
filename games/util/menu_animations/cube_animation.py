import math
import random
import time
from dataclasses import dataclass
from typing import List, Tuple

from artnet import RGB, Raster
from games.util.menu_animations.base import MenuAnimation


@dataclass
class CubeRotationState:
    """State for the rotating cube animation."""

    angles: List[float]  # x, y, z angles
    velocities: List[float]  # Angular velocities
    size: float  # Current size
    target_size: float  # Target size
    size_velocity: float  # Size change velocity
    last_time: float  # Last update time


class CubeAnimation(MenuAnimation):
    """A rotating cube animation for the menu screen."""

    def __init__(self, width: int, height: int, length: int):
        super().__init__(width, height, length)

        # Initialize cube-specific state
        self.rotation_state = CubeRotationState(
            angles=[0, 0, 0],  # x, y, z angles
            velocities=[0.2, 0.3, 0.1],  # Angular velocities
            size=5.0,  # Current size
            target_size=5.0,  # Target size
            size_velocity=0.0,  # Size change velocity
            last_time=time.monotonic(),
        )

        # Animation parameters
        self.spring_k = 30.0  # Spring constant
        self.damping = 4.0  # Damping factor
        self.base_colors = [
            RGB(255, 0, 0),  # Red
            RGB(0, 255, 0),  # Green
            RGB(0, 0, 255),  # Blue
            RGB(255, 255, 0),  # Yellow
            RGB(255, 0, 255),  # Purple
            RGB(0, 255, 255),  # Cyan
        ]

    def _update_physics(self, current_time: float):
        """Update cube physics state."""
        state = self.rotation_state
        dt = current_time - state.last_time
        state.last_time = current_time

        # Update rotation angles with varying velocities
        for i in range(3):
            state.angles[i] = (state.angles[i] + state.velocities[i] * dt) % (2 * math.pi)
            # Slowly vary velocities with noise
            state.velocities[i] += (random.random() - 0.5) * 0.1 * dt
            state.velocities[i] = max(min(state.velocities[i], 0.5), -0.5)

        # Spring physics for size
        spring_force = (state.target_size - state.size) * self.spring_k
        damping_force = -state.size_velocity * self.damping

        # Update size
        state.size_velocity += (spring_force + damping_force) * dt
        state.size += state.size_velocity * dt

        # Handle input events
        if self.state.input_intensity > 0:
            # Add rotational kick
            kick_strength = 6.0 * self.state.input_intensity
            for i in range(3):
                state.velocities[i] += (random.random() - 0.5) * kick_strength

            # Compress the cube
            state.target_size = 4.0
            state.size_velocity += 10.0 * self.state.input_intensity
        else:
            state.target_size = 5.0

    def _rotate_point(self, point: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Apply current rotation to a point."""
        x, y, z = point

        # Apply all rotations using rotation matrices
        for axis, angle in enumerate(self.rotation_state.angles):
            if axis == 0:  # X rotation
                y, z = (
                    y * math.cos(angle) - z * math.sin(angle),
                    y * math.sin(angle) + z * math.cos(angle),
                )
            elif axis == 1:  # Y rotation
                x, z = (
                    x * math.cos(angle) - z * math.sin(angle),
                    x * math.sin(angle) + z * math.cos(angle),
                )
            else:  # Z rotation
                x, y = (
                    x * math.cos(angle) - y * math.sin(angle),
                    x * math.sin(angle) + y * math.cos(angle),
                )

        return (x, y, z)

    def render(self, raster: Raster):
        """Render the rotating cube animation."""
        # Update physics
        self._update_physics(time.monotonic())

        # Clear the raster
        for x in range(self.width):
            for y in range(self.height):
                for z in range(self.length):
                    raster.set_pix(x, y, z, RGB(0, 0, 0))

        # Calculate cube points
        center_x = self.width // 2
        center_y = self.height // 2
        center_z = self.length // 2

        points = []
        size = self.rotation_state.size

        # Generate cube vertices
        for x in range(-1, 2, 1):
            for y in range(-1, 2, 1):
                for z in range(-1, 2, 1):
                    # Scale points by size
                    px, py, pz = x * size, y * size, z * size

                    # Apply rotation
                    px, py, pz = self._rotate_point((px, py, pz))

                    # Add to center and convert to integer coordinates
                    screen_x = int(center_x + px)
                    screen_y = int(center_y + py)
                    screen_z = int(center_z + pz)

                    if (
                        0 <= screen_x < self.width
                        and 0 <= screen_y < self.height
                        and 0 <= screen_z < self.length
                    ):
                        points.append((screen_x, screen_y, screen_z))

        # Choose color based on vote state
        vote_percentage = (
            len(self.state.voted_players) * 100 // len(self.state.active_players)
            if self.state.active_players
            else 0
        )
        color_idx = vote_percentage % len(self.base_colors)

        # Render points with chosen color
        for point in points:
            raster.set_pix(point[0], point[1], point[2], self.base_colors[color_idx])
