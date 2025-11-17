import math
import random

from artnet import RGB, Raster, Scene
from color_palette import get_palette


class DataCenterLayersScene(Scene):
    """
    Horizontal layers representing server racks in a data center.
    Each layer pulses independently with different activity patterns.
    """

    def __init__(self, **kwargs):
        properties = kwargs.get("properties")
        if not properties:
            raise ValueError("DataCenterLayersScene requires a 'properties' object.")

        self.width = properties.width
        self.height = properties.height
        self.length = properties.length

        # Layer configuration
        self.num_layers = 8
        self.layer_thickness = 1.5
        self.layer_spacing = self.width / (self.num_layers + 1)

        # Get color palette
        palette = get_palette()

        # Activity types with colors and frequencies
        activity_configs = [
            {"name": "compute", "base_freq": 2.0},
            {"name": "storage", "base_freq": 1.5},
            {"name": "network", "base_freq": 3.0},
        ]

        # Initialize each layer with sequential colors from palette
        self.layers = []
        for i in range(self.num_layers):
            # Use sequential colors from palette (cycling if needed)
            color = palette.get_color(i)
            # Pick a random activity config (frequency only)
            activity_config = random.choice(activity_configs)

            layer = {
                "x_position": (i + 1) * self.layer_spacing,
                "color": color,
                "name": activity_config["name"],
                "base_freq": activity_config["base_freq"],
                "phase_offset": random.uniform(0, 2 * math.pi),
                "hot_spots": [],  # List of (y, z, birth_time, intensity)
            }
            self.layers.append(layer)

        # Hot spot spawn rate
        self.next_hot_spot_spawn = 0.0
        self.hot_spot_interval = 0.5

        print(f"DataCenterLayersScene initialized with {self.num_layers} layers")

    def _update_hot_spots(self, time: float, dt: float):
        """Update hot spot positions and spawn new ones"""
        # Spawn new hot spots periodically
        if time >= self.next_hot_spot_spawn:
            layer = random.choice(self.layers)
            hot_spot = {
                "y": random.uniform(0, self.height),
                "z": random.uniform(0, self.length),
                "birth_time": time,
                "lifetime": random.uniform(1.0, 2.0),
            }
            layer["hot_spots"].append(hot_spot)
            self.next_hot_spot_spawn = time + self.hot_spot_interval

        # Remove expired hot spots
        for layer in self.layers:
            layer["hot_spots"] = [
                hs for hs in layer["hot_spots"] if time - hs["birth_time"] < hs["lifetime"]
            ]

    def render(self, raster: Raster, time: float):
        """Render the data center layers"""
        dt = 1.0 / 60.0
        raster.data.fill(0)

        self._update_hot_spots(time, dt)

        # Render each layer
        for layer in self.layers:
            x_center = layer["x_position"]

            # Calculate base intensity using sinusoidal pulse
            pulse = 0.5 + 0.5 * math.sin(layer["base_freq"] * time + layer["phase_offset"])
            base_intensity = int(100 + 155 * pulse)

            # Render layer plane
            x_min = int(x_center - self.layer_thickness)
            x_max = int(x_center + self.layer_thickness) + 1

            for x in range(x_min, x_max):
                if not (0 <= x < self.width):
                    continue

                # Distance falloff from center
                dist = abs(x - x_center)
                falloff = (
                    1.0 - (dist / self.layer_thickness) if dist < self.layer_thickness else 0.0
                )

                for y in range(self.height):
                    for z in range(self.length):
                        # Base color
                        intensity = int(base_intensity * falloff)

                        # Check for nearby hot spots
                        hot_spot_boost = 0
                        for hs in layer["hot_spots"]:
                            dy = y - hs["y"]
                            dz = z - hs["z"]
                            dist_hs = math.sqrt(dy * dy + dz * dz)
                            hot_spot_radius = 8.0

                            if dist_hs < hot_spot_radius:
                                # Calculate hot spot contribution
                                age = time - hs["birth_time"]
                                age_factor = math.sin(
                                    math.pi * age / hs["lifetime"]
                                )  # Rise and fall
                                spatial_factor = 1.0 - (dist_hs / hot_spot_radius)
                                hot_spot_boost = max(
                                    hot_spot_boost, int(155 * age_factor * spatial_factor)
                                )

                        final_intensity = min(255, intensity + hot_spot_boost)

                        if final_intensity > 10:
                            # Scale palette color by intensity
                            base_color = layer["color"]
                            intensity_factor = final_intensity / 255.0
                            color = RGB(
                                int(base_color.red * intensity_factor),
                                int(base_color.green * intensity_factor),
                                int(base_color.blue * intensity_factor),
                            )
                            raster.data[z, y, x, 0] = max(raster.data[z, y, x, 0], color.red)
                            raster.data[z, y, x, 1] = max(raster.data[z, y, x, 1], color.green)
                            raster.data[z, y, x, 2] = max(raster.data[z, y, x, 2], color.blue)
