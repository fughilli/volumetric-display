import argparse
import time
import math
import json
from typing import List, Dict, Tuple
from artnet import ArtNetController, Raster, RGB, Scene, load_scene

# Configuration
ARTNET_IP = "192.168.1.11"  # Replace with your controller's IP
ARTNET_PORT = 6454  # Default ArtNet UDP port

# Universe and DMX settings
UNIVERSE = 0  # Universe ID
CHANNELS = 512  # Max DMX channels

class DisplayConfig:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Parse geometry
        width, height, length = map(int, config['geometry'].split('x'))
        self.width = width
        self.height = height
        self.length = length
        
        # Parse z mapping
        self.z_mapping = []
        for mapping in config['z_mapping']:
            self.z_mapping.append({
                'ip': mapping['ip'],
                'z_indices': mapping['z_idx']
            })

        # Parse orientation
        self.orientation = config.get('orientation', ['X', 'Y', 'Z'])
        self._validate_orientation()
        self._compute_transform()

    def _validate_orientation(self):
        """Validate that orientation contains valid coordinate mappings."""
        valid_coords = {'X', 'Y', 'Z', '-X', '-Y', '-Z'}
        if len(self.orientation) != 3:
            raise ValueError("Orientation must specify exactly 3 coordinates")
        if not all(coord in valid_coords for coord in self.orientation):
            raise ValueError(f"Invalid coordinate in orientation. Must be one of: {valid_coords}")
        # Extract just the axis letters and check for uniqueness
        axes = [coord[-1] for coord in self.orientation]
        if len(set(axes)) != 3:
            raise ValueError("Each coordinate axis must appear exactly once in orientation")

    def _compute_transform(self):
        """Compute the transformation matrix for coordinate mapping."""
        self.transform = []
        for coord in self.orientation:
            axis = coord[-1]  # Get the axis (X, Y, or Z)
            sign = -1 if coord.startswith('-') else 1
            if axis == 'X':
                self.transform.append((0, sign))
            elif axis == 'Y':
                self.transform.append((1, sign))
            else:  # Z
                self.transform.append((2, sign))

    def transform_coordinates(self, x: int, y: int, z: int) -> Tuple[int, int, int]:
        """Transform coordinates according to the orientation configuration."""
        coords = [x, y, z]
        result = [0, 0, 0]
        for i, (axis, sign) in enumerate(self.transform):
            result[i] = coords[axis] * sign
        return tuple(result)

def create_default_scene():
    """Creates a built-in default scene with the original wave pattern"""

    class WaveScene(Scene):

        def render(self, raster, time):
            for y in range(raster.height):
                for x in range(raster.width):
                    for z in range(raster.length):
                        # Calculate color
                        red = int(127 * math.sin(0.5 * math.sin(time * 5) * x +
                                                 z * 0.2 + time * 10) + 128)
                        green = int(127 * math.sin(0.5 * math.cos(time * 4) *
                                                   y + z * 0.2 + time * 10) +
                                    128)
                        blue = int(127 * math.sin(0.5 * math.sin(time * 3) *
                                                  (x + y + z) + time * 10) +
                                   128)

                        # Set pixel color using set_pix
                        raster.set_pix(x, y, z, RGB(red, green, blue))

    return WaveScene()


def main():
    parser = argparse.ArgumentParser(
        description="ArtNet DMX Transmission with Sync")
    parser.add_argument("--config",
                        type=str,
                        required=True,
                        help="Path to display configuration JSON file")
    parser.add_argument("--layer-span",
                        type=int,
                        default=1,
                        help="Layer span (1 for 1:1 mapping)")
    parser.add_argument("--brightness",
                        type=float,
                        default=0.05,
                        help="Brightness factor (0.0 to 1.0)")
    parser.add_argument("--scene",
                        type=str,
                        help="Path to a scene plugin file")
    args = parser.parse_args()

    # Load display configuration
    display_config = DisplayConfig(args.config)
    
    # Create raster with full geometry
    raster = Raster(width=display_config.width, 
                   height=display_config.height, 
                   length=display_config.length,
                   orientation=display_config.orientation)
    raster.brightness = args.brightness

    # Create controllers for each IP
    controllers = {}
    controller_mappings = []
    for mapping in display_config.z_mapping:
        ip = mapping['ip']
        if ip not in controllers:
            controllers[ip] = ArtNetController(ip, ARTNET_PORT)
        controller_mappings.append((controllers[ip], mapping))

    # Load scene
    try:
        # Load raw config for scene
        with open(args.config, 'r') as f:
            config = json.load(f)
        scene = load_scene(args.scene, config=config) if args.scene else create_default_scene()
    except (ImportError, ValueError) as e:
        print(f"Error loading scene: {e}")
        raise e

    start_time = time.monotonic()
    print("🚀 Starting ArtNet DMX Transmission with Sync...")
    print(f"Using {len(controllers)} controllers for {display_config.length} z-slices")
    print(f"Orientation: {display_config.orientation}")

    try:
        while True:
            current_time = time.monotonic() - start_time

            # Update the raster using the scene
            scene.render(raster, current_time)

            # Send the updated raster
            for controller, mapping in controller_mappings:
                controller.send_dmx(UNIVERSE, raster, z_indices=mapping['z_indices'])
            time.sleep(0.01)  # Send updates at 100Hz

    except KeyboardInterrupt:
        print("\n🛑 Transmission stopped by user.")


if __name__ == "__main__":
    main()
