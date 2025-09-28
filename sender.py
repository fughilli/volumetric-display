import argparse
import dataclasses
import json
import logging
import math
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
    from sender_monitor_rust import create_sender_monitor_with_web_interface_wrapped

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
        if "world_geometry" not in config:
            raise ValueError("Configuration must contain world_geometry.")

        self.config = config
        self.cubes = config["cubes"]

        # Parse world geometry
        self.world_width, self.world_height, self.world_length = map(
            int, config["world_geometry"].split("x")
        )

        # These will be populated by _initialize_mappings
        self.controllers_cache = {}
        self.send_jobs = []
        self.cube_orientations = {}

        self._initialize_mappings()

    def _initialize_mappings(self):
        """Parses the config to create ArtNet controllers and send jobs."""
        print("🎛️  Initializing ArtNet mappings...")

        # Create a unique raster buffer for each physical cube with individual dimensions
        cube_rasters = {}
        cube_orientations = {}
        for cube_config in self.cubes:
            position = tuple(cube_config["position"])
            # Parse individual cube dimensions
            cube_width, cube_height, cube_length = map(int, cube_config["dimensions"].split("x"))

            # Parse per-cube orientation (optional, falls back to global)
            cube_orientation = cube_config.get(
                "orientation", self.config.get("orientation", ["-Z", "Y", "X"])
            )
            cube_orientations[position] = cube_orientation
            self.cube_orientations[position] = cube_orientation

            # Create cube raster with the cube's specific orientation
            cube_rasters[position] = Raster(cube_width, cube_height, cube_length)
            cube_rasters[position].orientation = cube_orientation
            cube_rasters[position]._compute_transform()  # Recompute transform for cube orientation

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


def hex_to_rgb(hex_color):
    """Convert hex color string to RGB values."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def apply_orientation_transform(world_data, cube_position, cube_dimensions, orientation):
    """
    Apply orientation transformation to slice world data for a cube.

    Args:
        world_data: 3D numpy array of world raster data
        cube_position: (x, y, z) position of cube in world coordinates
        cube_dimensions: (width, height, length) of cube
        orientation: List of 3 strings like ["-Z", "Y", "X"] defining axis mapping

    Returns:
        3D numpy array of transformed cube data
    """
    # Extract cube slice from world data
    start_x, start_y, start_z = cube_position
    cube_width, cube_height, cube_length = cube_dimensions

    # Get the raw slice from world data
    world_slice = world_data[
        start_z : start_z + cube_length,
        start_y : start_y + cube_height,
        start_x : start_x + cube_width,
    ]

    # Apply orientation transformation
    transformed_slice = world_slice.copy()

    # Map orientation to numpy array axes (numpy uses [Z, Y, X] indexing)
    axis_mapping = {"X": 2, "Y": 1, "Z": 0}  # numpy array indexing: [Z, Y, X]

    for i, axis in enumerate(orientation):
        if axis.startswith("-"):
            # Flip the axis
            axis_name = axis[1:]
            if axis_name in axis_mapping:
                numpy_axis = axis_mapping[axis_name]
                transformed_slice = np.flip(transformed_slice, axis=numpy_axis)

    # Apply axis reordering/rotation
    # This is a simplified implementation - for full 3D rotations, we'd need more complex transformations
    # For now, we'll handle common cases like swapping axes

    # Check if we need to swap axes (orientation is in [X, Y, Z] order)
    if orientation == ["Y", "X", "Z"]:  # Swap X and Y
        transformed_slice = np.swapaxes(
            transformed_slice, axis_mapping["Y"], axis_mapping["X"]
        )  # Swap axes 1 and 2
    elif orientation == ["Z", "X", "Y"]:  # Swap Y and Z
        transformed_slice = np.swapaxes(
            transformed_slice, axis_mapping["Y"], axis_mapping["Z"]
        )  # Swap axes 1 and 0
    elif orientation == ["X", "Z", "Y"]:  # Swap Y and Z
        transformed_slice = np.swapaxes(
            transformed_slice, axis_mapping["Y"], axis_mapping["Z"]
        )  # Swap axes 1 and 0
    # Add more orientation mappings as needed

    return transformed_slice


def apply_debug_commands(raster, debug_command, current_time, artnet_manager):
    """Apply debug commands to the raster."""
    if not debug_command:
        return set()

    command_type = debug_command.get("command_type")
    cubes_with_debug_commands = set()

    if command_type == "mapping_tester":
        cubes_with_debug_commands = apply_mapping_tester(raster, debug_command, artnet_manager)
    elif command_type == "power_draw_tester":
        apply_power_draw_tester(raster, debug_command, current_time)
    elif command_type == "clear":
        # Clear the raster - turn off all pixels
        raster.clear()

    return cubes_with_debug_commands


def apply_mapping_tester(raster, debug_command, artnet_manager):
    """Apply mapping tester command to light up a specific plane."""
    mapping_data = debug_command.get("mapping_tester")
    if not mapping_data:
        return set()

    orientation = mapping_data.get("orientation", "xy")
    layer = mapping_data.get("layer", 0)
    color_hex = mapping_data.get("color", "#FF0000")
    target = mapping_data.get("target", "world")

    # Convert hex to RGB
    r, g, b = hex_to_rgb(color_hex)
    color = RGB(r, g, b)

    cubes_with_debug_commands = set()

    if target == "world":
        # Apply to world raster
        apply_mapping_tester_to_raster(raster, orientation, layer, color)
    elif target.startswith("cube_"):
        # Apply to specific cube raster
        cube_index = int(target.split("_")[1])
        if 0 <= cube_index < len(artnet_manager.cubes):
            cube_config = artnet_manager.cubes[cube_index]
            cube_position = tuple(cube_config["position"])

            # Find the cube raster for this position
            cube_raster = None
            for job in artnet_manager.send_jobs:
                if tuple(job["cube_position"]) == cube_position:
                    cube_raster = job["cube_raster"]
                    break

            if cube_raster:
                # For per-cube debug mode, apply debug commands directly to the cube raster
                # without any orientation transformation - this shows the cube's raw coordinate system
                apply_mapping_tester_to_raster(cube_raster, orientation, layer, color)

                cubes_with_debug_commands.add(cube_position)
                logger.debug(
                    f"Applied mapping tester to cube {cube_index} at position {cube_position}"
                )
            else:
                logger.warning(f"Could not find cube raster for cube {cube_index}")
        else:
            logger.warning(f"Invalid cube index: {cube_index}")
    else:
        logger.warning(f"Unknown target: {target}")

    return cubes_with_debug_commands


def apply_mapping_tester_to_raster(raster, orientation, layer, color):
    """Apply mapping tester to a specific raster (world or cube)."""
    # Clear the raster first
    raster.clear()

    # Light up the specified plane
    if orientation == "xy":
        # XY plane at specific Z layer
        for x in range(raster.width):
            for y in range(raster.height):
                raster.set_pix(x, y, layer, color)
    elif orientation == "xz":
        # XZ plane at specific Y layer
        for x in range(raster.width):
            for z in range(raster.length):
                raster.set_pix(x, layer, z, color)
    elif orientation == "yz":
        # YZ plane at specific X layer
        for y in range(raster.height):
            for z in range(raster.length):
                raster.set_pix(layer, y, z, color)


def apply_power_draw_tester(raster, debug_command, current_time):
    """Apply power draw tester command with modulation."""
    power_data = debug_command.get("power_draw_tester")
    if not power_data:
        return

    color_hex = power_data.get("color", "#00FF00")
    modulation_type = power_data.get("modulation_type", "sin")
    frequency = power_data.get("frequency", 1.0)
    amplitude = power_data.get("amplitude", 0.5)
    offset = power_data.get("offset", 0.5)
    global_brightness = power_data.get("global_brightness", 1.0)

    # Convert hex to RGB
    r, g, b = hex_to_rgb(color_hex)
    base_color = RGB(r, g, b)

    # Calculate modulation value
    if modulation_type == "sin":
        modulation = offset + amplitude * math.sin(2 * math.pi * frequency * current_time)
    else:  # square wave
        modulation = offset + amplitude * (
            1 if math.sin(2 * math.pi * frequency * current_time) >= 0 else -1
        )

    # Apply modulation and brightness to all pixels
    for i in range(len(raster.data)):
        modulated_r = int(base_color.red * modulation * global_brightness)
        modulated_g = int(base_color.green * modulation * global_brightness)
        modulated_b = int(base_color.blue * modulation * global_brightness)

        # Clamp values to uint8 range (0-255)
        modulated_r = max(0, min(255, modulated_r))
        modulated_g = max(0, min(255, modulated_g))
        modulated_b = max(0, min(255, modulated_b))

        raster.data[i] = (modulated_r, modulated_g, modulated_b)


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
            sender_monitor = create_sender_monitor_with_web_interface_wrapped(
                args.sender_monitor_port, cooldown_seconds=30
            )
            logger.debug(
                f"🌐 Sender monitor started with web interface on port {args.sender_monitor_port}"
            )
        except Exception as e:
            logger.debug(f"Warning: Failed to start sender monitor: {e}")
            logger.debug("Continuing without sender monitoring...")

    # --- World Raster Setup (Single Canvas for Scene) ---
    # Use explicit world geometry from config
    world_width = artnet_manager.world_width
    world_height = artnet_manager.world_height
    world_length = artnet_manager.world_length

    # Calculate min coordinates for slicing (assuming world starts at origin)
    min_coord = (0, 0, 0)

    world_raster = Raster(world_width, world_height, world_length)
    world_raster.brightness = args.brightness
    display_props = DisplayProperties(width=world_width, height=world_height, length=world_length)

    # Set world dimensions and cube list in sender monitor for mapping tester
    if sender_monitor:
        sender_monitor.set_world_dimensions(world_width, world_height, world_length)

        # Create cube list for mapping tester
        cube_list = []
        for i, cube_config in enumerate(artnet_manager.cubes):
            cube_id = f"Cube {i+1}"
            position = tuple(cube_config["position"])
            # Parse individual cube dimensions
            cube_width, cube_height, cube_length = map(int, cube_config["dimensions"].split("x"))
            dimensions = (cube_width, cube_height, cube_length)
            cube_list.append((cube_id, position, dimensions))

        sender_monitor.set_cube_list(cube_list)

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

        # Register controllers with the monitor if available
        if sender_monitor:
            for controller_key in artnet_manager.controllers_cache.keys():
                ip, port = controller_key
                sender_monitor.register_controller(ip, port)

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

            # Track cubes with active cube-specific debug commands
            cubes_with_debug_commands = set()

            # Check if we're in debug mode and paused
            if sender_monitor and sender_monitor.is_debug_mode() and sender_monitor.is_paused():
                # In debug mode and paused - don't update scene, just apply debug commands
                debug_command = sender_monitor.get_debug_command()
                if debug_command:
                    cubes_with_debug_commands = apply_debug_commands(
                        world_raster, debug_command, current_time, artnet_manager
                    )
                    logger.debug("🔧 Applied debug command")
            else:
                # Normal operation - update the scene
                # A. SCENE RENDER: The active scene draws on the single large world_raster.
                scene.render(world_raster, current_time)
            t_render_done = time.monotonic()

            # Report frame to monitor if available
            if sender_monitor:
                sender_monitor.report_frame()

            # Note: ArtNet transmission is now handled in the "C. SEND" section below
            # using the artnet_manager.send_jobs infrastructure

            # B. SLICE: Copy data from the world raster to each cube's individual raster.
            # Skip cubes that have active cube-specific debug commands
            processed_cubes = set()
            for job in artnet_manager.send_jobs:
                cube_pos_tuple = tuple(job["cube_position"])

                # This check ensures we only slice a cube's data once per frame,
                # even if it has multiple ArtNet mappings.
                if cube_pos_tuple not in processed_cubes:
                    # Skip slicing if this cube has an active cube-specific debug command
                    if cube_pos_tuple not in cubes_with_debug_commands:
                        # Get cube position relative to world origin
                        cube_position = (
                            job["cube_position"][0] - min_coord[0],
                            job["cube_position"][1] - min_coord[1],
                            job["cube_position"][2] - min_coord[2],
                        )

                        # Get cube dimensions
                        cube_raster = job["cube_raster"]
                        cube_dimensions = (
                            cube_raster.width,
                            cube_raster.height,
                            cube_raster.length,
                        )

                        # Get cube orientation
                        cube_orientation = artnet_manager.cube_orientations.get(
                            cube_pos_tuple, ["-Z", "Y", "X"]
                        )

                        # Apply orientation transformation
                        transformed_slice = apply_orientation_transform(
                            world_raster.data, cube_position, cube_dimensions, cube_orientation
                        )

                        cube_raster.data[:] = transformed_slice
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

                # Get controller IP and port for monitoring
                controller_ip = job["controller"].get_ip()
                controller_port = job["controller"].get_port()

                try:
                    job["controller"].send_dmx(
                        base_universe=base_universe_offset,
                        raster=temp_raster,
                        z_indices=job["z_indices"],
                        # --- These params can be customized if needed ---
                        channels_per_universe=510,
                        universes_per_layer=universes_per_layer,
                        channel_span=1,
                    )
                    # Reset failure count on successful transmission
                    controller_failures[controller_ip] = 0

                    # Report success to monitor if available
                    if sender_monitor:
                        sender_monitor.report_controller_success(controller_ip, controller_port)

                except (OSError, ConnectionError, TimeoutError) as e:
                    # Track failures and log warnings periodically
                    controller_failures[controller_ip] += 1
                    current_time_real = time.time()

                    # Report failure to monitor if available
                    if sender_monitor:
                        sender_monitor.report_controller_failure(
                            controller_ip, controller_port, str(e)
                        )

                    # Only show warning if enough time has passed since last warning
                    if (current_time_real - last_warning_time[controller_ip]) >= WARNING_INTERVAL:
                        logger.warning(
                            f"⚠️  Network error sending to controller {controller_ip}: {e}"
                        )
                        last_warning_time[controller_ip] = current_time_real
                except Exception as e:
                    # Log unexpected errors but continue
                    logger.error(f"❌ Unexpected error with controller {controller_ip}: {e}")

                    # Report failure to monitor if available
                    if sender_monitor:
                        sender_monitor.report_controller_failure(
                            controller_ip, controller_port, str(e)
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
