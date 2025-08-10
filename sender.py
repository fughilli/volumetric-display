import argparse
import json

from artnet import ArtNetController, Raster, load_scene

# Try to use Rust-based control port for web monitoring
try:
    from control_port_rust import create_control_port_from_config

    CONTROL_PORT_AVAILABLE = True
    print("Using Rust-based control port with web monitoring")
except ImportError:
    CONTROL_PORT_AVAILABLE = False
    print("Control port not available - web monitoring disabled")

# Configuration
ARTNET_IP = "192.168.1.11"  # Replace with your controller's IP
ARTNET_PORT = 6454  # Default ArtNet UDP port
WEB_MONITOR_PORT = 8080  # Port for web monitoring interface

# Universe and DMX settings
UNIVERSE = 0  # Universe ID
CHANNELS = 512  # Max DMX channels


class DisplayConfig:
    """Configuration for the volumetric display."""

    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            config = json.load(f)

        self.width = config.get("width", 8)
        self.height = config.get("height", 8)
        self.length = config.get("length", 8)
        self.orientation = config.get("orientation", "xyz")

        # Validate dimensions
        if not all(d > 0 for d in [self.width, self.height, self.length]):
            raise ValueError("Display dimensions must be positive integers")


def create_controllers_from_config(config_path: str) -> dict:
    """Create ArtNet controllers based on the configuration file."""
    with open(config_path, "r") as f:
        config = json.load(f)

    controllers = {}
    controller_mappings = []

    # Extract controller mappings from config
    mappings = config.get("controller_mappings", [])
    for mapping in mappings:
        ip = mapping["ip"]
        if ip not in controllers:
            controllers[ip] = ArtNetController(ip, ARTNET_PORT)
        controller_mappings.append((controllers[ip], mapping))

    return controllers, controller_mappings


def main():
    parser = argparse.ArgumentParser(description="Send ArtNet DMX data to volumetric display")
    parser.add_argument("--config", required=True, help="Path to display configuration JSON")
    parser.add_argument("--scene", required=True, help="Path to scene Python file")
    parser.add_argument(
        "--brightness", type=float, default=1.0, help="Brightness multiplier (0.0-1.0)"
    )
    parser.add_argument(
        "--layer-span", type=int, default=1, help="Number of layers to skip between universes"
    )
    parser.add_argument(
        "--web-monitor-port", type=int, default=WEB_MONITOR_PORT, help="Web monitor port"
    )

    args = parser.parse_args()

    # Load display configuration
    display_config = DisplayConfig(args.config)

    # Start control port manager for web monitoring if available
    control_port_manager = None
    if CONTROL_PORT_AVAILABLE:
        try:
            control_port_manager = create_control_port_from_config(
                args.config, args.web_monitor_port
            )
            print(
                f"🌐 Control port manager started with web monitoring on port {args.web_monitor_port}"
            )
        except Exception as e:
            print(f"Warning: Failed to start control port manager: {e}")
            print("Continuing without web monitoring...")

    # Create raster with full geometry
    raster = Raster(
        width=display_config.width,
        height=display_config.height,
        length=display_config.length,
        orientation=display_config.orientation,
    )
    raster.brightness = args.brightness

    # Load and run the scene
    try:
        scene = load_scene(args.scene, raster)
        print(f"🎬 Playing scene: {args.scene}")
        print(f"📐 Display: {display_config.width}x{display_config.height}x{display_config.length}")
        print(f"💡 Brightness: {args.brightness}")
        print(f"🔗 Layer span: {args.layer_span}")

        # Create controllers from config
        controllers, controller_mappings = create_controllers_from_config(args.config)
        print(f"🎛️  Found {len(controllers)} controllers")

        # Run the scene
        scene.run(controllers, controller_mappings, args.layer_span)

    except KeyboardInterrupt:
        print("\n🛑 Transmission stopped by user.")
    finally:
        # Clean up control port manager
        if control_port_manager:
            try:
                control_port_manager.shutdown()
                print("🌐 Control port manager stopped")
            except Exception as e:
                print(f"Error stopping control port manager: {e}")


if __name__ == "__main__":
    main()
