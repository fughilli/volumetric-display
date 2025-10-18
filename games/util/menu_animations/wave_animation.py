import numpy as np

from artnet import Raster
from games.util.menu_animations import MenuAnimation


class WaveAnimation(MenuAnimation):
    """A wave animation for the menu screen."""

    def __init__(self, width: int, height: int, length: int):
        super().__init__(width, height, length)
        self.frequency = 2 * np.pi / 80
        self.amplitude = (height // 2) - 1
        self.mid = height // 2

        # Wave parameters that can be affected by input
        self.wave_speed = 2.0  # Base wave speed
        self.color_speed = 1.0  # Base color speed

    def render(self, raster: Raster):
        """Render the wave animation."""
        # Clear the raster
        raster.data.fill(0)

        # Get current time for animation
        current_time = self.last_update_time

        # Adjust speeds based on input intensity
        wave_speed = self.wave_speed * (1 + self.state.input_intensity * 2)
        color_speed = self.color_speed * (1 + self.state.input_intensity * 2)

        # Calculate time values
        ts = current_time / wave_speed

        # Render wave for each point
        for x in range(self.width):
            for z in range(self.length):
                xf = x * self.frequency
                zf = z * self.frequency

                # Calculate wave height
                y = int(
                    np.round(
                        np.sin(xf * np.cos(ts) * 4 + zf * np.sin(ts) + ts) * self.amplitude
                        + self.mid
                    )
                )

                if 0 <= y < self.height:
                    # Calculate color based on position and time
                    r = int(
                        128
                        + 110
                        * np.sin(
                            x / self.height * current_time * color_speed + y / self.height * 1.5
                        )
                    )
                    g = int(
                        128
                        + 110
                        * np.sin(
                            z / self.height * 2
                            + y / self.height
                            + current_time * color_speed * 2
                            + np.pi / 2
                        )
                    )
                    b = int(
                        128
                        + 110
                        * np.sin(
                            x / self.height
                            + z / self.height
                            + current_time * color_speed * 3
                            + np.pi
                        )
                    )

                    # Adjust color intensity based on voting state
                    intensity = 1.0
                    if self.state.active_players:
                        vote_percentage = len(self.state.voted_players) / len(
                            self.state.active_players
                        )
                        intensity = 0.5 + vote_percentage * 0.5

                    raster.data[z, y, x] = [
                        int(r * intensity),
                        int(g * intensity),
                        int(b * intensity),
                    ]
