import argparse
import dataclasses
import json
import logging
import time
from collections import defaultdict

import numpy as np

from artnet import RGB, ArtNetController, DisplayProperties, Raster, Scene, load_scene

logger = logging.getLogger(__name__)

# Try to use Rust-based control port for web monitoring
try:
    from control_port_rust import create_control_port_from_config

    CONTROL_PORT_AVAILABLE = True
    logger.debug("Using Rust-based control port with web monitoring")
except ImportError:
    CONTROL_PORT_AVAILABLE = False
    logger.debug("Control port not available - web monitoring disabled")

# Try to use Rust-based sender monitor
try:
    from sender_monitor_rust import create_sender_monitor_with_web_interface

    SENDER_MONITOR_AVAILABLE = True
    logger.debug("Using Rust-based sender monitor with web interface")
except ImportError:
    SENDER_MONITOR_AVAILABLE = False
    logger.debug("Sender monitor not available - monitoring disabled")

# Config (ARTNET IP & PORT are handled via sim_config updates and specified there)
WEB_MONITOR_PORT = 8080  # Port for web monitoring interface
SENDER_MONITOR_PORT = 8082  # Port for sender monitoring interface (changed to avoid conflict)

# Universe and DMX settings
UNIVERSE = 0  # Universe ID
CHANNELS = 512  # Max DMX channels


class ArtNetManager:
    """Manages ArtNet controllers and data mappings based on a config file."""

    def __init__(self, config: dict):
        if "cubes" not in config or not config["cubes"]:
            raise ValueError("Configuration must contain at least one cube.")

        self.config = config
        self.cubes = config["cubes"]

        # Parse cube geometry
        self.width, self.height, self.length = map(int, config["cube_geometry"].split("x"))

        # These will be populated by _initialize_mappings
        self.controllers_cache = {}
        self.send_jobs = []

        self._initialize_mappings()

    def _initialize_mappings(self):
        """Parses the config to create ArtNet controllers and send jobs."""
        print("🎛️  Initializing ArtNet mappings...")

        # Create a unique raster buffer for each physical cube
        cube_rasters = {
            tuple(cube_config["position"]): Raster(self.width, self.height, self.length)
            for cube_config in self.cubes
        }

        for cube_config in self.cubes:
            position_tuple = tuple(cube_config["position"])

            for mapping in cube_config.get("artnet_mappings", []):
                ip = mapping["ip"]
                port = int(mapping["port"])
                controller_key = (ip, port)

                # Create a controller for the IP/Port if it doesn't exist
                if controller_key not in self.controllers_cache:
                    self.controllers_cache[controller_key] = ArtNetController(ip, port)

                # A "send job" is a dictionary with everything needed to send one packet
                self.send_jobs.append(
                    {
                        "controller": self.controllers_cache[controller_key],
                        "cube_raster": cube_rasters[position_tuple],
                        "cube_position": cube_config["position"],
                        "z_indices": mapping["z_idx"],
                        "universe": mapping.get("universe", 0),
                    }
                )

        print(
            f"✅ Found {len(self.cubes)} cubes and created {len(self.send_jobs)} send jobs "
            f"across {len(self.controllers_cache)} unique controllers."
        )


def create_default_scene():
    """Creates a built-in default scene with the original wave pattern"""

    class WaveScene(Scene):
        def __init__(self, **kwargs):
            # We will create these grids once, then reuse them
            self.x_coords, self.y_coords, self.z_coords = (None, None, None)

        def render(self, raster: Raster, time: float):
            # One-time setup to create coordinate grids that match the raster size
            if self.x_coords is None or self.x_coords.shape != (
                raster.length,
                raster.height,
                raster.width,
            ):
                # np.indices creates 3D arrays representing the x, y, and z coordinate of each voxel
                self.z_coords, self.y_coords, self.x_coords = np.indices(
                    (raster.length, raster.height, raster.width), sparse=True
                )

            # Perform all math operations on the entire arrays at once.
            red = (
                127
                * np.sin(0.5 * np.sin(time * 5) * self.x_coords + self.z_coords * 0.2 + time * 10)
                + 128
            )
            green = 127 * np.cos((time * 4) * self.y_coords + self.z_coords * 0.2 + time * 10) + 128
            blue = (
                127
                * np.sin(
                    0.5 * np.sin(time * 3) * (self.x_coords + self.y_coords + self.z_coords)
                    + time * 10
                )
                + 128
            )

            # Assign the calculated color channels directly to the raster's NumPy data buffer.
            # np.stack combines the three separate color arrays into one (L, H, W, 3) array.
            raster.data[:] = np.stack([red, green, blue], axis=-1).astype(np.uint8)

    return WaveScene()


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
    parser.add_argument(
        "--sender-monitor-port", type=int, default=SENDER_MONITOR_PORT, help="Sender monitor port"
    )

    args = parser.parse_args()

    # --- Configuration Loading and Setup ---
    with open(args.config, "r") as f:
        config = json.load(f)

    artnet_manager = ArtNetManager(config)

    # --- Start Control Port ---
    control_port_manager = None
    if CONTROL_PORT_AVAILABLE:
        try:
            control_port_manager = create_control_port_from_config(
                args.config, args.web_monitor_port
            )
            logger.debug(
                f"🌐 Control port manager started with web monitoring on port {args.web_monitor_port}"
            )
        except Exception as e:
            logger.debug(f"Warning: Failed to start control port manager: {e}")
            logger.debug("Continuing without web monitoring...")

    # Start sender monitor for web interface if available
    sender_monitor = None
    if SENDER_MONITOR_AVAILABLE:
        try:
            sender_monitor = create_sender_monitor_with_web_interface(
                args.sender_monitor_port, cooldown_seconds=30
            )
            logger.debug(
                f"🌐 Sender monitor started with web interface on port {args.sender_monitor_port}"
            )
        except Exception as e:
            logger.debug(f"Warning: Failed to start sender monitor: {e}")
            logger.debug("Continuing without sender monitoring...")

    # --- World Raster Setup (Single Canvas for Scene) ---
    all_x = [c["position"][0] for c in artnet_manager.cubes]
    all_y = [c["position"][1] for c in artnet_manager.cubes]
    all_z = [c["position"][2] for c in artnet_manager.cubes]
    min_coord = (min(all_x), min(all_y), min(all_z))

    max_coord_x = max(c["position"][0] + artnet_manager.width for c in artnet_manager.cubes)
    max_coord_y = max(c["position"][1] + artnet_manager.height for c in artnet_manager.cubes)
    max_coord_z = max(c["position"][2] + artnet_manager.length for c in artnet_manager.cubes)

    world_width = max_coord_x - min_coord[0]
    world_height = max_coord_y - min_coord[1]
    world_length = max_coord_z - min_coord[2]

    world_raster = Raster(world_width, world_height, world_length)
    world_raster.brightness = args.brightness
    display_props = DisplayProperties(width=world_width, height=world_height, length=world_length)

    # --- Scene Loading ---
    try:
        with open(args.config, "r") as f:
            scene_config = json.load(f)
        scene = (
            load_scene(
                args.scene,
                properties=display_props,
                scene_config=scene_config,
                control_port_manager=control_port_manager,
            )
            if args.scene
            else create_default_scene()
        )

        print("\n🚀 Starting ArtNet Transmission...")
        print(f"🎬 Playing scene: {args.scene}")
        print(f"📐 World raster dimensions: {world_width}x{world_height}x{world_length}")
        print(f"💡 Brightness: {args.brightness}")

        if (
            hasattr(scene, "input_handler")
            and scene.input_handler
            and scene.input_handler.initialized
        ):
            print(f"🎮 Connected {len(scene.input_handler.controllers)} game controllers")

        # --- Main Rendering and Transmission Loop ---
        TARGET_FPS = 60.0
        FRAME_DURATION = 1.0 / TARGET_FPS

        # ⏱️ PROFILING: Setup for logging performance stats
        frame_count = 0
        last_log_time = time.monotonic()

        # Create controllers from config
        controllers, controller_mappings = create_controllers_from_config(args.config)
        logger.info(f"🎛️  Found {len(controllers)} ArtNet controllers for LED output")

        # Register controllers with the monitor if available
        if sender_monitor:
            for controller, mapping in controller_mappings:
                sender_monitor.register_controller(mapping["ip"], ARTNET_PORT)

        # Track controller failures for rate-limited warning messages
        controller_failures = defaultdict(int)  # controller_ip -> failure_count
        last_warning_time = defaultdict(float)  # controller_ip -> last_warning_time
        WARNING_INTERVAL = 10.0  # Only show warnings every 10 seconds per controller

        # Main rendering and transmission loop
        logger.info("🎬 Starting main loop...")
        start_time = time.time()

        print("🔁 Starting main loop...")
        start_time = time.monotonic()
        while True:
            t_loop_start = time.monotonic()

            frame_start_time = time.monotonic()
            current_time = frame_start_time - start_time

            # A. SCENE RENDER: The active scene draws on the single large world_raster.
            scene.render(world_raster, current_time)
            t_render_done = time.monotonic()

            # Report frame to monitor if available
            if sender_monitor:
                sender_monitor.report_frame()

            # Send the raster data to controllers via ArtNet
            for controller, mapping in controller_mappings:
                # Extract z indices for this controller
                z_indices = mapping.get("z_idx", [])
                if z_indices:
                    try:
                        # Send DMX data for this controller's z layers
                        controller.send_dmx(
                            base_universe=mapping.get("universe", 0),
                            raster=raster,
                            channels_per_universe=510,
                            universes_per_layer=3,
                            channel_span=args.layer_span,
                            z_indices=z_indices,
                        )
                        # Reset failure count on successful transmission
                        controller_failures[mapping["ip"]] = 0

                        # Report success to monitor if available
                        if sender_monitor:
                            sender_monitor.report_controller_success(mapping["ip"])

                    except (OSError, ConnectionError, TimeoutError) as e:
                        # Track failures and log warnings periodically
                        controller_failures[mapping["ip"]] += 1
                        current_time_real = time.time()

                        # Report failure to monitor if available
                        if sender_monitor:
                            sender_monitor.report_controller_failure(mapping["ip"], str(e))

                        # Only show warning if enough time has passed since last warning
                        if (
                            current_time_real - last_warning_time[mapping["ip"]]
                        ) >= WARNING_INTERVAL:
                            logger.warning(
                                f"⚠️  Network error sending to controller {mapping['ip']}: {e}"
                            )
                            last_warning_time[mapping["ip"]] = current_time_real
                    except Exception as e:
                        # Log unexpected errors but continue
                        logger.error(f"❌ Unexpected error with controller {mapping['ip']}: {e}")

                        # Report failure to monitor if available
                        if sender_monitor:
                            sender_monitor.report_controller_failure(mapping["ip"], str(e))

            # B. SLICE: Copy data from the world raster to each cube's individual raster.
            processed_cubes = set()
            for job in artnet_manager.send_jobs:
                cube_pos_tuple = tuple(job["cube_position"])

                # This check ensures we only slice a cube's data once per frame,
                # even if it has multiple ArtNet mappings.
                if cube_pos_tuple not in processed_cubes:
                    start_x = job["cube_position"][0] - min_coord[0]
                    start_y = job["cube_position"][1] - min_coord[1]
                    start_z = job["cube_position"][2] - min_coord[2]

                    # Get a reference to the destination raster for clarity
                    cube_raster = job["cube_raster"]
                    world_slice = world_raster.data[
                        start_z : start_z + cube_raster.length,
                        start_y : start_y + cube_raster.height,
                        start_x : start_x + cube_raster.width,
                    ]
                    cube_raster.data[:] = world_slice
                    processed_cubes.add(cube_pos_tuple)
            t_slice_done = time.monotonic()

            # C. SEND: Iterate through all jobs and send the specified Z-layers.
            conversion_cache = {}
            for job in artnet_manager.send_jobs:
                # Get the original raster with its NumPy data
                cube_raster = job["cube_raster"]
                raster_id = id(cube_raster)

                # Convert the NumPy array into the Python list of RGB objects
                # that the Rust library expects.
                if raster_id not in conversion_cache:
                    # If not in cache, do the expensive conversion and store it
                    numpy_data = cube_raster.data.reshape(-1, 3)
                    conversion_cache[raster_id] = [
                        RGB(int(r), int(g), int(b)) for r, g, b in numpy_data
                    ]

                # Create a temporary raster with the (now cached) Python list
                temp_raster = dataclasses.replace(cube_raster)
                temp_raster.data = conversion_cache[raster_id]

                universes_per_layer = 3
                base_universe_offset = min(job["z_indices"]) * universes_per_layer

                job["controller"].send_dmx(
                    base_universe=base_universe_offset,
                    raster=temp_raster,
                    z_indices=job["z_indices"],
                    # --- These params can be customized if needed ---
                    channels_per_universe=510,
                    universes_per_layer=universes_per_layer,
                    channel_span=1,
                )
            t_send_done = time.monotonic()

            # ⏱️ PROFILING: Log stats every second
            frame_count += 1
            """
            if t_send_done - last_log_time > 1.0:
                fps = frame_count / (t_send_done - last_log_time)
                render_ms = (t_render_done - t_loop_start) * 1000
                slice_ms = (t_slice_done - t_render_done) * 1000
                send_ms = (t_send_done - t_slice_done) * 1000
                total_ms = (t_send_done - t_loop_start) * 1000

                print(
                    f"FPS: {fps:<5.1f} | "
                    f"Total: {total_ms:<5.1f}ms | "
                    f"Render: {render_ms:<5.1f}ms | "
                    f"Slice: {slice_ms:<5.1f}ms | "
                    f"Send: {send_ms:<5.1f}ms"
                )

                frame_count = 0
                last_log_time = t_send_done
            """

            elapsed_time = time.monotonic() - frame_start_time
            sleep_time = FRAME_DURATION - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

    except (ImportError, ValueError) as e:
        print(f"Error loading scene: {e}")
        raise
    except KeyboardInterrupt:
        logger.info("\n🛑 Transmission stopped by user.")
    except Exception as e:
        logger.error(f"\n❌ Error in main loop: {e}")
        import traceback

        print(f"\n❌ Error in main loop: {e}")
        traceback.print_exc()
    finally:
        # Cleanup
        if "scene" in locals() and hasattr(scene, "input_handler") and scene.input_handler:
            try:
                logger.info("🛑 Stopping controller input handler...")
                scene.input_handler.stop()
                logger.info("✅ Controller input handler stopped")
            except Exception as e:
                logger.error(f"Warning: Error stopping controller input handler: {e}")

        # Clean up control port manager only when the entire program is exiting
        if control_port_manager:
            try:
                control_port_manager.shutdown()
                logger.info("🌐 Control port manager stopped")
            except Exception as e:
                logger.error(f"Error stopping control port manager: {e}")

        # Clean up sender monitor only when the entire program is exiting
        if sender_monitor:
            try:
                sender_monitor.shutdown()
                logger.info("🌐 Sender monitor stopped")
            except Exception as e:
                logger.error(f"Error stopping sender monitor: {e}")


if __name__ == "__main__":
    logging.basicConfig()
    main()
