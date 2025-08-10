import argparse
import json
import time

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

        # Parse geometry field if present (format: "20x20x20")
        if "geometry" in config:
            geometry = config["geometry"]
            if isinstance(geometry, str) and "x" in geometry:
                parts = geometry.split("x")
                if len(parts) == 3:
                    self.width = int(parts[0])
                    self.height = int(parts[1])
                    self.length = int(parts[2])
                else:
                    raise ValueError(
                        f"Invalid geometry format: {geometry}. Expected format like '20x20x20'"
                    )
            else:
                raise ValueError(
                    f"Invalid geometry format: {geometry}. Expected string like '20x20x20'"
                )
        else:
            # Fallback to individual fields
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

    # Extract controller mappings from config (using z_mapping field)
    mappings = config.get("z_mapping", [])
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
        # Parse the config file to pass to the scene
        with open(args.config, "r") as f:
            scene_config = json.load(f)
        scene = load_scene(args.scene, scene_config, control_port_manager)
        print(f"🎬 Playing scene: {args.scene}")
        print(f"📐 Display: {display_config.width}x{display_config.height}x{display_config.length}")
        print(f"💡 Brightness: {args.brightness}")
        print(f"🔗 Layer span: {args.layer_span}")

        # Create controllers from config
        controllers, controller_mappings = create_controllers_from_config(args.config)
        print(f"🎛️  Found {len(controllers)} controllers")

        # Main rendering and transmission loop
        print("🎬 Starting main loop...")
        start_time = time.time()

        while True:
            current_time = time.time() - start_time

            # Update the scene (this updates the raster contents)
            scene.render(raster, current_time)

            # Send the raster data to controllers via ArtNet
            for controller, mapping in controller_mappings:
                # Extract z indices for this controller
                z_indices = mapping.get("z_idx", [])
                if z_indices:
                    # Send DMX data for this controller's z layers
                    controller.send_dmx(
                        base_universe=mapping.get("universe", 0),
                        raster=raster,
                        channels_per_universe=510,
                        universes_per_layer=3,
                        channel_span=args.layer_span,
                        z_indices=z_indices,
                    )

            # Small delay to control frame rate
            time.sleep(1.0 / 30.0)  # 30 FPS

    except KeyboardInterrupt:
        print("\n🛑 Transmission stopped by user.")
    except Exception as e:
        print(f"\n❌ Error in main loop: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Clean up scene and controller input handler first
        if "scene" in locals() and hasattr(scene, "input_handler") and scene.input_handler:
            try:
                print("🛑 Stopping controller input handler...")
                scene.input_handler.stop()
                print("✅ Controller input handler stopped")
            except Exception as e:
                print(f"Warning: Error stopping controller input handler: {e}")

        # Clean up control port manager only when the entire program is exiting
        if control_port_manager:
            try:
                control_port_manager.shutdown()
                print("🌐 Control port manager stopped")
            except Exception as e:
                print(f"Error stopping control port manager: {e}")


if __name__ == "__main__":
    main()
