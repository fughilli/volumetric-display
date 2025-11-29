"""Midnight Walk scene: A glowing fractal crack rift with planar ghost silhouettes spiraling upward."""

import io
import math
import os
import random
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from artnet import RGB, Scene

# Try to import requests for HTTP client
try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Try to import SVG loading libraries
# Prefer svglib+reportlab (pure Python, no system dependencies)
try:
    from PIL import Image
    from reportlab.graphics import renderPM
    from svglib.svglib import svg2rlg

    SVGLIB_AVAILABLE = True
except ImportError:
    SVGLIB_AVAILABLE = False

# Fallback to cairosvg if available (requires Cairo system library)
try:
    import cairosvg

    CAIROSVG_AVAILABLE = True
except ImportError:
    CAIROSVG_AVAILABLE = False


@dataclass
class GhostSprite:
    """A single ghost sprite streaming from the rift."""

    x: float  # Position
    y: float
    z: float
    vx: float  # Velocity
    vy: float
    vz: float
    birth_time: float
    lifetime: float
    base_size: float  # Base visual size of the ghost
    spiral_angle: float  # Current angle in spiral
    spiral_speed: float  # Angular velocity for spiral
    distance_from_center: float  # Distance from rift center
    silhouette_index: int = 0  # Index of which ghost SVG silhouette to use

    # Animation parameters
    FADE_IN_TIME = 0.3  # Time to fade in
    FADE_OUT_TIME = 0.5  # Time to fade out at end
    RISE_SPEED_MIN = 3.0  # Minimum upward velocity
    RISE_SPEED_MAX = 8.0  # Maximum upward velocity
    SPIRAL_SPEED_MIN = 1.5  # Minimum spiral angular velocity
    SPIRAL_SPEED_MAX = 4.0  # Maximum spiral angular velocity
    BASE_SIZE_MIN = 3.0  # Minimum base size (increased)
    BASE_SIZE_MAX = 5.5  # Maximum base size (increased)
    LIFETIME_MIN = 10.0  # Minimum lifetime
    LIFETIME_MAX = 11.0  # Maximum lifetime
    ELONGATION_FACTOR = 2.5  # How much ghosts elongate at max speed
    RADIUS_GROWTH_RATE = 1.0  # How fast the spiral radius grows (increased)

    def update(
        self,
        dt: float,
        bounds: tuple[float, float, float],
        rift_center_x: float,
        rift_center_z: float,
        current_time: float,
    ):
        """Update ghost position with spiral motion outward and upward."""
        width, height, length = bounds

        # Update spiral angle
        self.spiral_angle += self.spiral_speed * dt

        # Increase distance from center over time (spiral outward)
        age = current_time - self.birth_time
        time_based_expansion = age * self.RADIUS_GROWTH_RATE
        # velocity_based_expansion = math.sqrt(self.vx * self.vx + self.vz * self.vz) * dt * 1.2
        # self.distance_from_center += velocity_based_expansion + time_based_expansion * dt
        self.distance_from_center += time_based_expansion * dt

        # Calculate position relative to rift center
        rel_x = math.cos(self.spiral_angle) * self.distance_from_center
        rel_z = math.sin(self.spiral_angle) * self.distance_from_center

        # Update velocity components for spiral motion
        self.vx = math.cos(self.spiral_angle) * self.distance_from_center * 0.3
        self.vz = math.sin(self.spiral_angle) * self.distance_from_center * 0.3

        # Update position
        self.x = rift_center_x + rel_x
        self.y += self.vy * dt
        self.z = rift_center_z + rel_z

        # Increase upward velocity over time (acceleration)
        self.vy += 2.0 * dt

        # Remove if completely off the top (need to account for ghost size)
        # Get current size scale to determine when ghost is fully off screen
        age = current_time - self.birth_time
        size_scale = 0.5 + (age / self.lifetime) * 0.5  # Grows from 0.5 to 1.0
        current_height = self.base_size * 4.0 * size_scale  # Account for 4x scaling
        # Remove when bottom of ghost (y - current_height/2) is above display top
        if self.y - current_height / 2 > height:
            return False
        # Also remove if out of horizontal bounds
        if self.x < 0 or self.x >= width or self.z < 0 or self.z >= length:
            return False
        return True

    def is_expired(self, current_time: float) -> bool:
        """Check if ghost has expired."""
        return current_time - self.birth_time > self.lifetime

    def get_opacity(self, current_time: float) -> float:
        """Get current opacity based on fade in/out."""
        age = current_time - self.birth_time
        if age < self.FADE_IN_TIME:
            # Fade in
            return age / self.FADE_IN_TIME
        elif age > self.lifetime - self.FADE_OUT_TIME:
            # Fade out
            fade_out_age = age - (self.lifetime - self.FADE_OUT_TIME)
            return 1.0 - (fade_out_age / self.FADE_OUT_TIME)
        else:
            return 1.0

    def get_speed(self) -> float:
        """Get current speed magnitude."""
        return math.sqrt(self.vx * self.vx + self.vy * self.vy + self.vz * self.vz)

    def get_elongation(self) -> float:
        """Get elongation factor based on speed."""
        speed = self.get_speed()
        max_speed = self.RISE_SPEED_MAX * 2.0  # Approximate max speed
        speed_ratio = min(speed / max_speed, 1.0)
        return 1.0 + (speed_ratio * (self.ELONGATION_FACTOR - 1.0))

    def get_size_scale(self, current_time: float) -> float:
        """Get size scale factor that grows from 0.5 to 1.0 over lifetime."""
        age = current_time - self.birth_time
        if age <= 0:
            return 0.5
        if age >= self.lifetime:
            return 1.0
        # Linear growth from 0.5 to 1.0
        return 0.5 + (age / self.lifetime) * 0.5


class MidnightScene(Scene):
    """Idle animation scene with a glowing fractal crack rift and streaming ghost sprites."""

    def __init__(self, **kwargs):
        """Initialize the midnight scene."""
        properties = kwargs.get("properties")
        if not properties:
            raise ValueError("MidnightScene requires a 'properties' object.")

        self.width = properties.width
        self.height = properties.height
        self.length = properties.length
        self.bounds = (self.width, self.height, self.length)

        # Rift parameters
        self.rift_center_x = self.width / 2.0
        self.rift_center_z = self.length / 2.0
        self.rift_max_radius = min(self.width, self.length) * 0.25  # Max extent of crack
        self.rift_y = 0.5  # Rift is on the floor (low y)

        # Ghost sprites
        self.ghosts: List[GhostSprite] = []
        self.next_spawn = 0.0
        self.spawn_interval = 3.0  # Spawn a ghost every 3 seconds (lower spawn rate)

        # Spherical pulse effect parameters
        self.pulse_duration = 2.0  # Duration for pulse to expand fully (seconds)
        self.pulse_max_radius = min(self.width, self.length) * 2.0  # Max radius of pulse
        self.pulse_amplitude = 0.3  # Amplitude (30% max brightness)
        self.active_pulses: List[float] = []  # List of pulse spawn times

        # Generate fractal crack pattern
        self.crack_segments: List[Tuple[float, float, float, float]] = []
        self._generate_fractal_crack()

        # Load ghost SVG silhouettes
        self.ghost_silhouettes: Dict[int, np.ndarray] = {}
        self.ghost_svg_paths: Optional[List[str]] = kwargs.get("ghost_svg_paths")
        self._load_ghost_silhouettes()

        # Tharok encounter state
        self.tharok_active = False
        self.tharok_encounter_start_time: Optional[float] = None
        self.tharok_thaw_progress = 0.0  # 0.0 to 1.0
        self.tharok_thaw_start_time: Optional[float] = None
        self.tharok_flame_color_factor = 0.0  # 0.0 = blue, 1.0 = red
        self.tharok_exploding = False
        self.tharok_explosion_start_time: Optional[float] = None
        self.tharok_particles: List[Dict] = []  # Particle system for explosion
        self.tharok_spirit_id: Optional[str] = None  # Track Tharok's spirit ID

        # Server integration
        self.server_url = kwargs.get("server_url", "https://midnight.fughil.li")
        self.last_encounter_check = 0.0
        self.last_player_check = 0.0
        self.encounter_check_interval = 1.0  # Check every second
        self.player_check_interval = 0.5  # Check every 0.5 seconds
        self._lock = threading.Lock()  # Thread lock for state updates
        self._encounter_check_pending = False  # Track if check is in progress
        self._player_check_pending = False  # Track if check is in progress

        # Control port manager for NFC tag handling
        self.control_port_manager = kwargs.get("control_port_manager")
        if self.control_port_manager:
            self._register_nfc_tag_callbacks()

        # Profiling state
        self.profiling_enabled = True
        self.frame_times: List[float] = []  # Track last 60 frame times
        self.last_profile_report = 0.0
        self.profile_report_interval = 5.0  # Report every 5 seconds
        self.section_times: Dict[str, List[float]] = {}  # Track times per section

    def _load_ghost_silhouettes(self):
        """Load all ghost silhouette SVG files (ghost0.svg through ghost3.svg)."""
        # Get the directory where this scene file is located
        scene_dir = os.path.dirname(os.path.abspath(__file__))

        # Find all ghost SVG files, prioritizing relative to scene file location
        possible_dirs = [
            os.path.join(
                scene_dir, "midnight_walk"
            ),  # Look in midnight_walk relative to scene file
            os.path.join(scene_dir, "resources"),  # Also check resources
            scene_dir,  # Check scene file's directory
            "midnight_walk",  # Fallback: relative to current working directory
            "resources",  # Fallback: relative to current working directory
            "",  # Current working directory
        ]

        found_files = []
        seen_indices = set()  # Avoid duplicates

        for directory in possible_dirs:
            for i in range(4):  # ghost0.svg through ghost3.svg
                if i in seen_indices:
                    continue  # Already found this index
                filename = f"ghost{i}.svg"
                path = os.path.join(directory, filename) if directory else filename
                if os.path.exists(path):
                    found_files.append((i, path))
                    seen_indices.add(i)

        # Also check if custom paths were provided
        if self.ghost_svg_paths:
            for i, path in enumerate(self.ghost_svg_paths):
                if i not in seen_indices and os.path.exists(path):
                    found_files.append((i, path))
                    seen_indices.add(i)

        if not found_files:
            print("Warning: No ghost SVG files found (ghost0.svg through ghost3.svg).")
            print(f"Searched in directories: {possible_dirs}")
            return

        # Load each found SVG
        for index, svg_path in found_files:
            try:
                silhouette = self._load_single_svg(svg_path)
                if silhouette is not None:
                    self.ghost_silhouettes[index] = silhouette
                    print(
                        f"Loaded ghost silhouette {index} from {svg_path} (shape: {silhouette.shape})"
                    )
                else:
                    print(f"Failed to load ghost silhouette {index} from {svg_path}: returned None")
            except Exception as e:
                print(f"Error loading ghost SVG {svg_path}: {e}")
                import traceback

                traceback.print_exc()

    def _register_nfc_tag_callbacks(self):
        """Register NFC tag callbacks on all control ports."""
        if not self.control_port_manager:
            return

        control_ports = self.control_port_manager.get_all_control_ports()
        for dip, control_port in control_ports.items():
            try:
                control_port.register_nfc_tag_callback(self._handle_nfc_tag)
                print(f"Registered NFC tag callback for control port DIP {dip}")
            except Exception as e:
                print(f"Failed to register NFC tag callback for DIP {dip}: {e}")

    def _handle_nfc_tag(self, tag_data):
        """Handle NFC tag scan event (called from Rust broadcast receiver thread)."""
        # Debug log all received data
        print(f"[PYTHON-DEBUG] NFC tag callback received: {tag_data}")

        # Process in a separate thread to avoid blocking the Rust receiver thread
        threading.Thread(target=self._process_nfc_tag, args=(tag_data,), daemon=True).start()

    def _process_nfc_tag(self, tag_data):
        """Process NFC tag data in a background thread."""
        url = tag_data.get("url")
        uid = tag_data.get("uid", "")
        data = tag_data.get("data")
        record_type = tag_data.get("record_type", "")

        print(
            f"[PYTHON-DEBUG] Processing NFC tag - UID: {uid!r}, "
            f"data: {data!r}, url: {url!r}, record_type: {record_type!r}"
        )

        # If no URL in url field, try to extract from data field
        if not url and data and record_type == "uri":
            # The data field might contain hex-encoded URL or raw URL
            try:
                # Try to decode as hex first (common NFC encoding)
                if all(c in "0123456789abcdefABCDEF" for c in data):
                    # Decode hex string
                    decoded = bytes.fromhex(data).decode("utf-8", errors="ignore")
                    print(f"[PYTHON-DEBUG] Decoded hex data: {decoded!r}")
                    # Look for URL pattern
                    if "http" in decoded or "midnight.fughil.li" in decoded:
                        # Extract URL from decoded data
                        import re

                        url_match = re.search(r"https?://[^\s\x00]+", decoded)
                        if url_match:
                            url = url_match.group(0)
                            print(f"[PYTHON-DEBUG] Extracted URL from decoded data: {url}")
                else:
                    # Try as direct string
                    if "http" in data or "midnight.fughil.li" in data:
                        url = data
                        print(f"[PYTHON-DEBUG] Using data as direct URL: {url}")
            except Exception as e:
                print(f"[PYTHON-DEBUG] Error parsing data field for URL: {e}")
                import traceback

                traceback.print_exc()

        if not url:
            print(
                f"[PYTHON-DEBUG] NFC tag scanned but no URL found "
                f"(UID: {uid!r}, data: {data!r}, record_type: {record_type!r})"
            )
            return

        # Extract beacon UUID from URL
        try:
            from urllib.parse import parse_qs, urlparse

            parsed = urlparse(url)
            query_params = parse_qs(parsed.query)
            beacon_id = query_params.get("id", [None])[0]

            if not beacon_id:
                print(f"[PYTHON-DEBUG] Could not extract beacon ID from URL: {url}")
                return

            print(f"[PYTHON-DEBUG] Extracted beacon ID: {beacon_id}")

            # Make HTTP request to server
            if not REQUESTS_AVAILABLE:
                print(
                    "[PYTHON-DEBUG] Warning: requests library not available, cannot handle NFC tag"
                )
                return

            # Make request in a separate thread to avoid blocking
            threading.Thread(
                target=self._lookup_beacon_and_start_encounter, args=(beacon_id,), daemon=True
            ).start()

        except Exception as e:
            print(f"[PYTHON-DEBUG] Error handling NFC tag: {e}")
            import traceback

            traceback.print_exc()

    def _lookup_beacon_and_start_encounter(self, beacon_id: str):
        """Look up beacon and start encounter (runs in background thread)."""
        try:
            response = requests.get(
                f"{self.server_url}/api/nfc-lookup",
                params={"beacon_uid": beacon_id, "start_encounter": True},
                timeout=5.0,
            )

            if response.status_code == 404:
                # Beacon not found - display on all controllers
                self._display_message_on_all_controllers("Beacon not found")
                return

            if not response.ok:
                print(f"[PYTHON-DEBUG] Failed to lookup beacon: HTTP {response.status_code}")
                return

            data = response.json()
            print(f"[PYTHON-DEBUG] Server response: {data}")

            if data.get("error") == "active_encounter":
                # Another encounter is active
                self._display_message_on_all_controllers(
                    ["Encounter active", "Wait for it to", "complete"]
                )
            elif data.get("error"):
                # Other error
                error_msg = data.get("message", "Unknown error")
                self._display_message_on_all_controllers(error_msg[:20])
            elif data.get("spirit_id") and data.get("spirit_name"):
                # Encounter started successfully
                spirit_id = data.get("spirit_id")
                spirit_name = data.get("spirit_name", "")
                display_name = spirit_name[:20] if len(spirit_name) > 20 else spirit_name
                self._display_message_on_all_controllers(["Encounter:", display_name])

                # Check if this is Tharok and trigger encounter visual (case-insensitive)
                if "tharok" in spirit_name.lower():
                    current_time = time.time()
                    with self._lock:
                        self.tharok_spirit_id = spirit_id
                        self.set_tharok_encounter(True, current_time)
                    print(
                        f"[PYTHON-DEBUG] 🔥 Tharok encounter triggered! Spirit ID: {spirit_id}, Name: {spirit_name}"
                    )
                    print(
                        f"[PYTHON-DEBUG] Tharok active: {self.tharok_active}, "
                        f"encounter start time: {self.tharok_encounter_start_time}"
                    )
                else:
                    print(f"[PYTHON-DEBUG] Encounter started with {spirit_name} (ID: {spirit_id})")

        except requests.exceptions.RequestException as e:
            print(f"[PYTHON-DEBUG] Network error looking up beacon: {e}")
            import traceback

            traceback.print_exc()
        except Exception as e:
            print(f"[PYTHON-DEBUG] Error in beacon lookup: {e}")
            import traceback

            traceback.print_exc()

    def _display_message_on_all_controllers(self, message):
        """Display a message on all connected controllers (runs in background thread)."""
        if not self.control_port_manager:
            return

        if isinstance(message, str):
            lines = [message]
        else:
            lines = message

        # Run display operations in a separate thread to avoid blocking
        def update_display():
            try:
                control_ports = self.control_port_manager.get_all_control_ports()
                for dip, control_port in control_ports.items():
                    try:
                        control_port.clear_display()
                        for y, line in enumerate(lines[:4]):  # Max 4 lines
                            control_port.write_display(0, y, line)
                        # Note: commit_display is async, but we're in a thread
                        # The display will be committed on the next render cycle
                    except Exception as e:
                        print(f"[PYTHON-DEBUG] Failed to display message on DIP {dip}: {e}")
            except Exception as e:
                print(f"[PYTHON-DEBUG] Error in display update: {e}")

        # Run in separate thread to avoid any potential blocking
        threading.Thread(target=update_display, daemon=True).start()

    def _load_single_svg(self, svg_path: str) -> Optional[np.ndarray]:
        """Load a single SVG file and return as numpy array."""
        try:
            # Prefer svglib+reportlab (pure Python, no system dependencies)
            if SVGLIB_AVAILABLE:
                # Use svglib + reportlab
                drawing = svg2rlg(svg_path)
                if drawing:
                    # Render to PIL Image at higher resolution for better quality
                    # Use 256x256 for better detail, then we can sample from it
                    img = renderPM.drawToPIL(drawing, dpi=144)  # Higher DPI for better quality
                    # Resize to 256x256 for better detail
                    if img.size != (256, 256):
                        img = img.resize((256, 256), Image.Resampling.LANCZOS)
                    img_array = np.array(img.convert("L"))  # Convert to grayscale
                    # For silhouettes, we want: shape pixels = low value (0.0), background = high value (1.0)
                    # This matches how we render (invert alpha to get shape mask)
                    # Check if image is mostly dark (black shape on white background) or light
                    # (white shape on black background)
                    mean_brightness = np.mean(img_array)
                    if mean_brightness > 128:
                        # Mostly light = black shape on white background, invert so shape is dark
                        img_array = img_array
                    # Normalize to 0-1 range (now shape=low value ~0.0, background=high value ~1.0)
                    return (img_array / 255.0).astype(np.float32)
            elif CAIROSVG_AVAILABLE:
                # Fallback to cairosvg if available
                png_data = cairosvg.svg2png(url=svg_path, output_width=256, output_height=256)
                img = Image.open(io.BytesIO(png_data))
                img_array = np.array(img.convert("L"))  # Convert to grayscale
                # For silhouettes, we want: shape pixels = low value (0.0), background = high value (1.0)
                mean_brightness = np.mean(img_array)
                if mean_brightness > 128:
                    # Mostly light = black shape on white background, invert so shape is dark
                    img_array = 255 - img_array
                # Normalize to 0-1 range (now shape=low value ~0.0, background=high value ~1.0)
                return (img_array / 255.0).astype(np.float32)
            else:
                print(
                    "Warning: No SVG loading library available. Install svglib+reportlab or cairosvg."
                )
                return None
        except Exception as e:
            print(f"Error loading SVG {svg_path}: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _get_sparkle_color(
        self, start_x: float, start_z: float, end_x: float, end_z: float, current_time: float
    ) -> RGB:
        """Get a sparkling color (red, orange, yellow) for a crack segment.

        Uses multiple sine waves at different frequencies to create complex sparkling,
        never goes fully black (minimum brightness ~40%).
        """
        # Create a unique seed for this segment based on its position
        segment_hash = hash(
            (int(start_x * 100), int(start_z * 100), int(end_x * 100), int(end_z * 100))
        )
        seed = abs(segment_hash) % 1000

        # Use multiple sine waves at different frequencies for complex sparkling
        # Different frequencies create non-repetitive patterns
        sparkle1 = math.sin(current_time * 2.3 + seed * 0.1) * 0.5 + 0.5
        sparkle2 = math.sin(current_time * 3.7 + seed * 0.2) * 0.5 + 0.5
        sparkle3 = math.sin(current_time * 5.1 + seed * 0.15) * 0.5 + 0.5

        # Combine the sparkles with different weights
        combined_sparkle = sparkle1 * 0.4 + sparkle2 * 0.35 + sparkle3 * 0.25

        # Ensure minimum brightness (never fully black)
        # Combined sparkle ranges from 0 to 1, we want it from 0.4 to 1.0
        brightness = 0.4 + combined_sparkle * 0.6  # Range: 0.4 to 1.0

        # Choose color hue based on segment position (red, orange, yellow)
        # Use segment hash to pick color family
        color_seed = (seed % 3) / 3.0  # 0.0, 0.33, 0.66

        # Interpolate between red, orange, and yellow
        # Red: (255, 0, 0) -> Orange: (255, 165, 0) -> Yellow: (255, 255, 0)
        if color_seed < 0.33:
            # Red to Orange
            t = color_seed / 0.33
            r = 255
            g = int(165 * t)
            b = 0
        elif color_seed < 0.66:
            # Orange to Yellow
            t = (color_seed - 0.33) / 0.33
            r = 255
            g = int(165 + 90 * t)
            b = 0
        else:
            # Yellow (with slight variation)
            r = 255
            g = 255
            b = int(50 * (color_seed - 0.66) / 0.34)  # Slight variation

        # Apply brightness modulation (never fully black)
        r = int(r * brightness)
        g = int(g * brightness)
        b = int(b * brightness)

        return RGB(r, g, b)

    def _render_spherical_pulse(self, raster, current_time: float):
        """Render spherical pulses emanating from the rift center when ghosts spawn."""
        if not self.active_pulses:
            return

        # Pulse color (red/orange tint)
        pulse_r_base = 255
        pulse_g_base = 0
        pulse_b_base = 0

        # Render each active pulse
        for pulse_time in self.active_pulses:
            # Calculate pulse age and phase (0 to 1)
            pulse_age = current_time - pulse_time
            pulse_phase = pulse_age / self.pulse_duration

            # Skip if pulse is complete
            if pulse_phase >= 1.0:
                continue

            # Current pulse radius (expands from 0 to max_radius)
            current_radius = pulse_phase * self.pulse_max_radius

            # Pulse intensity (fades out as it expands)
            # Use a smooth falloff function
            base_intensity = self.pulse_amplitude * (1.0 - pulse_phase) ** 2  # Quadratic falloff

            # Render pulse as a true 3D sphere centered at the rift
            rift_y = self.rift_y

            # Iterate over voxels in a 3D bounding box around the pulse
            radius_int = int(current_radius) + 2
            x_min = max(0, int(self.rift_center_x - radius_int))
            x_max = min(self.width, int(self.rift_center_x + radius_int) + 1)
            y_min = max(0, int(rift_y - radius_int))
            y_max = min(self.height, int(rift_y + radius_int) + 1)
            z_min = max(0, int(self.rift_center_z - radius_int))
            z_max = min(self.length, int(self.rift_center_z + radius_int) + 1)

            for y_int in range(y_min, y_max):
                for z_int in range(z_min, z_max):
                    for x_int in range(x_min, x_max):
                        # Calculate 3D distance from rift center
                        dx = x_int - self.rift_center_x
                        dy = y_int - rift_y
                        dz = z_int - self.rift_center_z
                        dist_3d = math.sqrt(dx * dx + dy * dy + dz * dz)

                        # Check if this voxel is within the pulse sphere
                        if dist_3d <= current_radius:
                            # Calculate distance from sphere surface (for smooth falloff)
                            dist_from_surface = abs(dist_3d - current_radius)

                            # Smooth falloff based on distance from surface
                            # Use a Gaussian-like falloff
                            falloff = math.exp(
                                -dist_from_surface
                                * dist_from_surface
                                / (current_radius * 0.3 + 1.0)
                            )

                            # Final intensity at this voxel (combine base intensity with falloff)
                            voxel_intensity = base_intensity * falloff

                            # Apply to raster (additive blending)
                            current_r = raster.data[z_int, y_int, x_int, 0]
                            current_g = raster.data[z_int, y_int, x_int, 1]
                            current_b = raster.data[z_int, y_int, x_int, 2]

                            # Add pulse contribution
                            raster.data[z_int, y_int, x_int, 0] = min(
                                255, current_r + int(pulse_r_base * voxel_intensity)
                            )
                            raster.data[z_int, y_int, x_int, 1] = min(
                                255, current_g + int(pulse_g_base * voxel_intensity)
                            )
                            raster.data[z_int, y_int, x_int, 2] = min(
                                255, current_b + int(pulse_b_base * voxel_intensity)
                            )

    def _generate_fractal_crack(self):
        """Generate a fractal crack pattern radiating from the center."""
        self.crack_segments = []
        num_branches = random.randint(8, 15)
        branch_length = self.rift_max_radius * 0.6

        for _ in range(num_branches):
            # Use more discrete angles for sharper appearance (multiples of 30 degrees)
            base_angle = random.randint(0, 11) * (
                math.pi / 6
            )  # 0 to 330 degrees in 30-degree steps
            angle = base_angle + random.uniform(-math.pi / 12, math.pi / 12)  # Small random offset
            self._add_crack_branch(
                self.rift_center_x,
                self.rift_center_z,
                angle,
                branch_length,
                depth=0,
                max_depth=3,
            )

    def _add_crack_branch(
        self,
        start_x: float,
        start_z: float,
        angle: float,
        length: float,
        depth: int,
        max_depth: int,
    ):
        """Recursively add a crack branch."""
        if depth > max_depth or length < 1.0:
            return

        end_x = start_x + length * math.cos(angle)
        end_z = start_z + length * math.sin(angle)

        # Store segment (only if it's within reasonable bounds)
        if (0 <= start_x < self.width and 0 <= start_z < self.length) or (
            0 <= end_x < self.width and 0 <= end_z < self.length
        ):
            self.crack_segments.append((start_x, start_z, end_x, end_z))

        # Randomly branch off with sharper angles
        if random.random() < 0.5 and depth < max_depth:
            # Use discrete angle offsets for sharper appearance
            angle_offsets = [
                -math.pi / 2,
                -math.pi / 3,
                math.pi / 3,
                math.pi / 2,
            ]  # 90, 60, -60, -90 degrees
            branch_angle = (
                angle + random.choice(angle_offsets) + random.uniform(-math.pi / 12, math.pi / 12)
            )
            branch_length = length * random.uniform(0.4, 0.7)
            self._add_crack_branch(end_x, end_z, branch_angle, branch_length, depth + 1, max_depth)

        # Continue main branch with minimal angle deviation for straighter lines
        if random.random() < 0.7:
            continue_angle = angle + random.uniform(
                -math.pi / 12, math.pi / 12
            )  # Smaller deviation
            continue_length = length * random.uniform(0.6, 0.9)
            self._add_crack_branch(
                end_x, end_z, continue_angle, continue_length, depth + 1, max_depth
            )

    def _draw_line(
        self,
        raster,
        x0: float,
        z0: float,
        x1: float,
        z1: float,
        y: int,
        color: RGB,
        width: float = 1.5,
    ):
        """Draw a glowing line segment."""
        # Simple interpolation-based line drawing
        line_length = math.sqrt((x1 - x0) ** 2 + (z1 - z0) ** 2)
        if line_length < 0.1:
            return

        # Number of steps along the line
        num_steps = max(int(line_length), 1)
        width_int = max(1, int(width))

        for i in range(num_steps + 1):
            t = i / num_steps if num_steps > 0 else 0
            px = x0 + t * (x1 - x0)
            pz = z0 + t * (z1 - z0)

            # Draw a circle at this point with the specified width
            for dz_offset in range(-width_int, width_int + 1):
                for dx_offset in range(-width_int, width_int + 1):
                    dist = math.sqrt(dx_offset * dx_offset + dz_offset * dz_offset)
                    if dist <= width:
                        final_px = int(px + dx_offset)
                        final_pz = int(pz + dz_offset)
                        if (
                            0 <= final_px < self.width
                            and 0 <= final_pz < self.length
                            and 0 <= y < self.height
                        ):
                            # Intensity based on distance from line center
                            intensity = 1.0 - (dist / width)
                            intensity = max(0.0, intensity)
                            intensity = intensity**0.7  # Softer falloff

                            raster.data[final_pz, y, final_px, 0] = max(
                                raster.data[final_pz, y, final_px, 0], int(color.red * intensity)
                            )
                            raster.data[final_pz, y, final_px, 1] = max(
                                raster.data[final_pz, y, final_px, 1], int(color.green * intensity)
                            )
                            raster.data[final_pz, y, final_px, 2] = max(
                                raster.data[final_pz, y, final_px, 2], int(color.blue * intensity)
                            )

    def spawn_ghost(self, current_time: float) -> GhostSprite:
        """Spawn a new ghost sprite from the rift."""
        # Random size and lifetime (needed before calculating spawn position)
        base_size = random.uniform(GhostSprite.BASE_SIZE_MIN, GhostSprite.BASE_SIZE_MAX)
        lifetime = random.uniform(GhostSprite.LIFETIME_MIN, GhostSprite.LIFETIME_MAX)

        # Start at rift center, but spawn below the bottom of the display
        # Spawn at half size initially, so position it so bottom edge is at rift_y
        initial_size = base_size * 4.0 * 0.5  # Half size with 4x scaling
        x = self.rift_center_x
        z = self.rift_center_z
        y = self.rift_y - initial_size / 2.0  # Spawn below so bottom edge is at rift_y

        # Random initial spiral angle
        spiral_angle = random.uniform(0, 2 * math.pi)
        spiral_speed = random.uniform(GhostSprite.SPIRAL_SPEED_MIN, GhostSprite.SPIRAL_SPEED_MAX)

        # Random upward velocity
        rise_speed = random.uniform(GhostSprite.RISE_SPEED_MIN, GhostSprite.RISE_SPEED_MAX)

        # Initial velocity components (will be updated by spiral motion)
        vx = 0.0
        vy = rise_speed
        vz = 0.0

        # Randomly pick a silhouette index from available ones
        available_indices = list(self.ghost_silhouettes.keys())
        silhouette_index = random.choice(available_indices) if available_indices else 0

        return GhostSprite(
            x=x,
            y=y,
            z=z,
            vx=vx,
            vy=vy,
            vz=vz,
            birth_time=current_time,
            lifetime=lifetime,
            base_size=base_size,
            spiral_angle=spiral_angle,
            spiral_speed=spiral_speed,
            distance_from_center=0.0,
            silhouette_index=silhouette_index,
        )

    def _check_active_encounter(self, current_time: float):
        """Schedule a non-blocking check for active encounter (Tharok)."""
        if not REQUESTS_AVAILABLE:
            return

        # Don't start a new check if one is already pending
        if self._encounter_check_pending:
            return

        self._encounter_check_pending = True

        def _do_check():
            try:
                response = requests.get(f"{self.server_url}/api/encounter/active", timeout=1.0)
                if response.status_code == 200:
                    data = response.json()
                    spirit_name = data.get("spirit_name", "")
                    # Check if Tharok encounter is active (flexible name matching)
                    if data.get("active") and "Tharok" in spirit_name:
                        # Tharok encounter is active
                        with self._lock:
                            if not self.tharok_active:
                                self.tharok_spirit_id = data.get("spirit_id")
                                check_time = time.time()
                                self.set_tharok_encounter(True, check_time)
                                print(
                                    f"🔥 Tharok encounter started! "
                                    f"Spirit ID: {self.tharok_spirit_id}, Name: {spirit_name}"
                                )
                    else:
                        with self._lock:
                            if self.tharok_active and not data.get("active"):
                                # Encounter ended (but don't deactivate if we're exploding)
                                if not self.tharok_exploding:
                                    check_time = time.time()
                                    self.set_tharok_encounter(False, check_time)
                                    print("Tharok encounter ended")
            except Exception:
                # Silently fail - network issues shouldn't break rendering
                pass
            finally:
                self._encounter_check_pending = False

        # Run in background thread
        thread = threading.Thread(target=_do_check, daemon=True)
        thread.start()

    def _check_players_near_rift(self, current_time: float):
        """Schedule a non-blocking check for player count near rift gate."""
        if not REQUESTS_AVAILABLE:
            return

        # Don't start a new check if one is already pending
        if self._player_check_pending:
            return

        # Check if Tharok is active before starting thread
        with self._lock:
            if not self.tharok_active:
                return

        self._player_check_pending = True

        def _do_check():
            try:
                response = requests.get(
                    f"{self.server_url}/api/rift/players-nearby",
                    params={"radius_meters": 10.0},
                    timeout=1.0,
                )
                if response.status_code == 200:
                    data = response.json()
                    player_count = data.get("player_count", 0)
                    check_time = time.time()
                    with self._lock:
                        print(
                            f"[PYTHON-DEBUG] Players near rift: {player_count}, "
                            f"thaw progress: {self.tharok_thaw_progress:.2f}"
                        )
                        self.update_tharok_thawing(player_count, check_time)
            except Exception as e:
                # Silently fail - network issues shouldn't break rendering
                print(f"[PYTHON-DEBUG] Error checking players near rift: {e}")
            finally:
                self._player_check_pending = False

        # Run in background thread
        thread = threading.Thread(target=_do_check, daemon=True)
        thread.start()

    def _mark_tharok_vanquished(self):
        """Mark Tharok as vanquished on the server."""
        if not REQUESTS_AVAILABLE or not self.tharok_spirit_id:
            return

        try:
            response = requests.post(
                f"{self.server_url}/api/spirits/{self.tharok_spirit_id}/vanquished", timeout=2.0
            )
            if response.status_code == 200:
                print("✅ Tharok marked as vanquished on server")
            else:
                print(f"⚠️ Failed to mark Tharok as vanquished: {response.status_code}")
        except Exception as e:
            print(f"⚠️ Error marking Tharok as vanquished: {e}")

    def render(self, raster, current_time: float):
        """Render the scene with multiple ghosts spiraling outward and upward with radial normals."""
        frame_start = time.time()

        # Clear raster
        raster.data.fill(0)

        # Check for active encounters periodically
        t0 = time.time()
        if current_time - self.last_encounter_check >= self.encounter_check_interval:
            self._check_active_encounter(current_time)
            self.last_encounter_check = current_time
        t1 = time.time()
        self._profile_section("encounter_check", t1 - t0)

        # Check player count near rift periodically (non-blocking, if Tharok is active)
        t0 = time.time()
        with self._lock:
            tharok_active = self.tharok_active
        if tharok_active and current_time - self.last_player_check >= self.player_check_interval:
            self._check_players_near_rift(current_time)
            self.last_player_check = current_time
        t1 = time.time()
        self._profile_section("player_check", t1 - t0)

        # Spawn new ghosts (limit to 2 ghosts max)
        t0 = time.time()
        dt = 1.0 / 60.0  # Assume 60 FPS
        if current_time >= self.next_spawn:
            if len(self.ghost_silhouettes) > 0 and len(self.ghosts) < 2:
                self.ghosts.append(self.spawn_ghost(current_time))
                # Create a pulse when a ghost spawns
                self.active_pulses.append(current_time)
            self.next_spawn = current_time + self.spawn_interval

        # Remove expired pulses (pulses that have fully expanded)
        self.active_pulses = [
            pulse_time
            for pulse_time in self.active_pulses
            if (current_time - pulse_time) < self.pulse_duration
        ]

        # Update and remove expired ghosts
        self.ghosts = [
            ghost
            for ghost in self.ghosts
            if ghost.update(dt, self.bounds, self.rift_center_x, self.rift_center_z, current_time)
            and not ghost.is_expired(current_time)
        ]
        t1 = time.time()
        self._profile_section("ghost_update", t1 - t0)

        # Render rift crack pattern (on the floor) with sparkling effect
        t0 = time.time()
        rift_y_int = int(self.rift_y)
        if 0 <= rift_y_int < self.height:
            for start_x, start_z, end_x, end_z in self.crack_segments:
                # Get sparkling color for this segment
                rift_color = self._get_sparkle_color(start_x, start_z, end_x, end_z, current_time)
                self._draw_line(
                    raster, start_x, start_z, end_x, end_z, rift_y_int, rift_color, width=1.0
                )
        t1 = time.time()
        self._profile_section("rift_render", t1 - t0)

        # Render spherical pulse effect emanating from rift
        t0 = time.time()
        self._render_spherical_pulse(raster, current_time)
        t1 = time.time()
        self._profile_section("pulse_render", t1 - t0)

        # Update and render Tharok if active
        t0 = time.time()
        with self._lock:
            tharok_active = self.tharok_active
            tharok_exploding = self.tharok_exploding
        if tharok_active:
            # Update explosion particles if exploding
            if tharok_exploding:
                self._update_tharok_particles(dt, current_time)

            # Render Tharok
            self._render_tharok(raster, current_time)
        t1 = time.time()
        self._profile_section("tharok_render", t1 - t0)

        # Render all ghosts
        t0 = time.time()
        if len(self.ghost_silhouettes) == 0:
            frame_end = time.time()
            self._profile_frame(frame_end - frame_start)
            return

        for ghost in self.ghosts:
            # Get ghost's silhouette
            silhouette = self.ghost_silhouettes.get(ghost.silhouette_index)
            if silhouette is None:
                continue

            # Calculate radial direction (from center to ghost)
            radial_x = ghost.x - self.rift_center_x
            radial_z = ghost.z - self.rift_center_z
            radial_dist = math.sqrt(radial_x * radial_x + radial_z * radial_z)

            if radial_dist < 0.01:
                # At center, use default direction
                radial_x_norm = 1.0
                radial_z_norm = 0.0
            else:
                # Normalize radial direction
                radial_x_norm = radial_x / radial_dist
                radial_z_norm = radial_z / radial_dist

            # Get ghost properties
            opacity = ghost.get_opacity(current_time)
            elongation = ghost.get_elongation()  # Vertical stretching based on speed
            size_scale = ghost.get_size_scale(
                current_time
            )  # Size grows from 0.5 to 1.0 over lifetime
            ghost_y_int = int(ghost.y)

            # Ghost color with opacity
            ghost_r = int(255 * opacity)
            ghost_g = int(255 * opacity)
            ghost_b = int(255 * opacity)

            # Render ghost as vertical billboard sprite perpendicular to radial direction
            # Make ghosts 4x bigger, with size scaling and vertical stretching
            # Size scale applies uniformly to both axes
            self._render_svg_silhouette(
                raster,
                ghost,
                ghost_r,
                ghost_g,
                ghost_b,
                radial_x_norm,
                radial_z_norm,
                ghost.base_size * 4.0 * size_scale,  # Apply size scale uniformly
                elongation,
                ghost_y_int,
                silhouette,
            )
        t1 = time.time()
        self._profile_section("ghost_render", t1 - t0)

        frame_end = time.time()
        self._profile_frame(frame_end - frame_start)

    def _profile_section(self, section_name: str, duration: float):
        """Track timing for a specific section."""
        if not self.profiling_enabled:
            return
        if section_name not in self.section_times:
            self.section_times[section_name] = []
        self.section_times[section_name].append(duration)
        # Keep only last 60 samples
        if len(self.section_times[section_name]) > 60:
            self.section_times[section_name].pop(0)

    def _profile_frame(self, frame_time: float):
        """Track overall frame time and print periodic reports."""
        if not self.profiling_enabled:
            return

        self.frame_times.append(frame_time)
        # Keep only last 60 frames
        if len(self.frame_times) > 60:
            self.frame_times.pop(0)

        # Print periodic reports
        current_time = time.time()
        if current_time - self.last_profile_report >= self.profile_report_interval:
            self._print_profile_report()
            self.last_profile_report = current_time

    def _print_profile_report(self):
        """Print profiling statistics."""
        if not self.frame_times:
            return

        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        max_frame_time = max(self.frame_times)
        min_frame_time = min(self.frame_times)
        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0

        print(f"\n[PROFILE] Frame Statistics (last {len(self.frame_times)} frames):")
        print(f"  Avg frame time: {avg_frame_time*1000:.2f}ms ({fps:.1f} FPS)")
        print(f"  Min frame time: {min_frame_time*1000:.2f}ms")
        print(f"  Max frame time: {max_frame_time*1000:.2f}ms")

        if self.section_times:
            print("  Section breakdown:")
            for section_name, times in sorted(self.section_times.items()):
                if times:
                    avg_time = sum(times) / len(times)
                    max_time = max(times)
                    pct_of_frame = (avg_time / avg_frame_time * 100) if avg_frame_time > 0 else 0
                    print(
                        f"    {section_name:20s}: "
                        f"avg={avg_time*1000:6.2f}ms "
                        f"max={max_time*1000:6.2f}ms "
                        f"({pct_of_frame:5.1f}% of frame)"
                    )
        print()

    def _render_svg_xy_plane(self, raster, silhouette, center_x, center_y, center_z, size):
        """Render SVG silhouette in XY plane (facing forward, no rotation)."""
        silh_height, silh_width = silhouette.shape

        # Calculate bounding box
        half_size = size / 2.0
        x_min = int(center_x - half_size)
        x_max = int(center_x + half_size) + 1
        y_min = int(center_y - half_size)
        y_max = int(center_y + half_size) + 1

        # Clamp to world bounds
        x_min = max(0, x_min)
        x_max = min(self.width, x_max)
        y_min = max(0, y_min)
        y_max = min(self.height, y_max)
        z_int = int(center_z)
        z_int = max(0, min(z_int, self.length - 1))

        # Ghost color: white
        ghost_r = 255
        ghost_g = 255
        ghost_b = 255

        # Render the silhouette
        for y_int in range(y_min, y_max):
            for x_int in range(x_min, x_max):
                # Calculate offset from center
                x_offset = x_int - center_x
                y_offset = y_int - center_y

                # Map to silhouette UV coordinates (0 to 1)
                # u maps to x (horizontal), v maps to y (vertical)
                silh_u = (x_offset / size) + 0.5
                silh_v = (y_offset / size) + 0.5

                # Check bounds
                if 0.0 <= silh_u <= 1.0 and 0.0 <= silh_v <= 1.0:
                    # Sample silhouette with bilinear interpolation
                    v_float = silh_v * (silh_height - 1)
                    u_float = silh_u * (silh_width - 1)

                    # Bilinear interpolation
                    v0 = int(v_float)
                    v1 = min(v0 + 1, silh_height - 1)
                    u0 = int(u_float)
                    u1 = min(u0 + 1, silh_width - 1)

                    dv = v_float - v0
                    du = u_float - u0

                    # Sample four corners
                    a00 = float(silhouette[v0, u0])
                    a01 = float(silhouette[v0, u1])
                    a10 = float(silhouette[v1, u0])
                    a11 = float(silhouette[v1, u1])

                    # Bilinear interpolation
                    alpha = (
                        a00 * (1 - du) * (1 - dv)
                        + a01 * du * (1 - dv)
                        + a10 * (1 - du) * dv
                        + a11 * du * dv
                    )

                    # The silhouette is inverted: low alpha = shape, high alpha = background
                    # Invert alpha to get the shape mask
                    shape_mask = alpha

                    # Render if this is part of the shape (inverted alpha > threshold)
                    if shape_mask > 0.05:
                        # Always render in white, use shape_mask for intensity/opacity
                        intensity = min(1.0, shape_mask)  # Clamp to 1.0
                        final_r = int(ghost_r * intensity)
                        final_g = int(ghost_g * intensity)
                        final_b = int(ghost_b * intensity)

                        # Render at this Z position
                        if 0 <= z_int < self.length:
                            raster.data[z_int, y_int, x_int, 0] = max(
                                raster.data[z_int, y_int, x_int, 0], final_r
                            )
                            raster.data[z_int, y_int, x_int, 1] = max(
                                raster.data[z_int, y_int, x_int, 1], final_g
                            )
                            raster.data[z_int, y_int, x_int, 2] = max(
                                raster.data[z_int, y_int, x_int, 2], final_b
                            )

    def _render_svg_silhouette(
        self,
        raster,
        ghost,
        ghost_r,
        ghost_g,
        ghost_b,
        radial_x_norm,
        radial_z_norm,
        base_size,
        elongation,
        ghost_y,
        silhouette,
    ):
        """Render ghost as vertical billboard sprite perpendicular to radial direction.

        Args:
            radial_x_norm, radial_z_norm: Normalized radial direction (pointing outward from center)
            elongation: Vertical stretching factor (stretches along Y direction)
        """
        silh_height, silh_width = silhouette.shape

        # Billboard plane is vertical and perpendicular to radial direction
        # Plane normal = radial direction
        # Plane contains: up vector (0, 1, 0) and tangent vector (perpendicular to radial in XZ)

        # Calculate tangent direction (perpendicular to radial in XZ plane)
        # Rotate radial by 90 degrees: tangent = (-radial_z, 0, radial_x)
        tangent_x = -radial_z_norm
        tangent_z = radial_x_norm

        # Apply vertical stretching (elongation along Y direction)
        width_size = base_size  # Width along tangent direction (unchanged)
        height_size = base_size * elongation  # Height along Y direction (stretched vertically)

        # Iterate over the billboard plane directly
        # For each position in the billboard (along tangent and Y), calculate world position
        num_samples_u = int(width_size * 2)
        num_samples_v = int(height_size * 2)

        for v_idx in range(num_samples_v):
            for u_idx in range(num_samples_u):
                # World space coordinates on billboard
                # u maps to tangent direction, v maps to Y direction
                silh_u = u_idx / num_samples_u
                silh_v_world = v_idx / num_samples_v  # World space V coordinate (0 to 1)

                # Convert to billboard plane coordinates
                tangent_offset = (silh_u - 0.5) * width_size
                y_offset = (silh_v_world - 0.5) * height_size

                # Calculate world position on billboard plane
                world_x = ghost.x + tangent_offset * tangent_x
                world_z = ghost.z + tangent_offset * tangent_z
                world_y = ghost_y + y_offset

                # Convert to integer voxel coordinates
                x_int = int(world_x)
                z_int = int(world_z)
                y_int = int(world_y)

                # Check bounds
                if not (
                    0 <= x_int < self.width
                    and 0 <= z_int < self.length
                    and 0 <= y_int < self.height
                ):
                    continue

                # For vertical stretching: compress V coordinate before sampling silhouette
                # This way the image doesn't get distorted, just stretched in world space
                silh_v_compressed = (silh_v_world - 0.5) / elongation + 0.5  # Compress V coordinate

                # Clamp to valid range
                if silh_v_compressed < 0.0 or silh_v_compressed > 1.0:
                    continue

                # Flip V coordinate (image v=0 is top, world y increases upward)
                silh_v = 1.0 - silh_v_compressed

                # Sample silhouette with bilinear interpolation
                v_float = silh_v * (silh_height - 1)
                u_float = silh_u * (silh_width - 1)

                # Bilinear interpolation
                v0 = int(v_float)
                v1 = min(v0 + 1, silh_height - 1)
                u0 = int(u_float)
                u1 = min(u0 + 1, silh_width - 1)

                dv = v_float - v0
                du = u_float - u0

                # Sample four corners
                a00 = float(silhouette[v0, u0])
                a01 = float(silhouette[v0, u1])
                a10 = float(silhouette[v1, u0])
                a11 = float(silhouette[v1, u1])

                # Bilinear interpolation
                alpha = (
                    a00 * (1 - du) * (1 - dv)
                    + a01 * du * (1 - dv)
                    + a10 * (1 - du) * dv
                    + a11 * du * dv
                )

                # The silhouette: low alpha = shape, high alpha = background
                # Invert alpha to get the shape mask
                shape_mask = 1.0 - alpha

                # Render if this is part of the shape
                if shape_mask > 0.05:
                    # Use shape_mask for intensity/opacity
                    intensity = min(1.0, shape_mask)  # Clamp to 1.0
                    final_r = int(ghost_r * intensity)
                    final_g = int(ghost_g * intensity)
                    final_b = int(ghost_b * intensity)

                    # Render at this position (thin billboard - single voxel thick)
                    raster.data[z_int, y_int, x_int, 0] = max(
                        raster.data[z_int, y_int, x_int, 0], final_r
                    )
                    raster.data[z_int, y_int, x_int, 1] = max(
                        raster.data[z_int, y_int, x_int, 1], final_g
                    )
                    raster.data[z_int, y_int, x_int, 2] = max(
                        raster.data[z_int, y_int, x_int, 2], final_b
                    )

    def _render_fallback_ghost(
        self,
        raster,
        ghost,
        ghost_r,
        ghost_g,
        ghost_b,
        vx_norm,
        vz_norm,
        elongation,
        base_size,
        ghost_y,
    ):
        """Fallback rendering if SVG not available."""
        # Simple box fallback
        for z_offset in range(-int(base_size), int(base_size) + 1):
            for x_offset in range(-int(base_size), int(base_size) + 1):
                final_x = ghost.x + x_offset
                final_z = ghost.z + z_offset
                x_int = int(final_x)
                z_int = int(final_z)
                if 0 <= x_int < self.width and 0 <= z_int < self.length:
                    for y_offset in range(-int(base_size // 2), int(base_size // 2) + 1):
                        render_y = ghost_y + y_offset
                        if 0 <= render_y < self.height:
                            raster.data[z_int, render_y, x_int, 0] = max(
                                raster.data[z_int, render_y, x_int, 0], ghost_r
                            )
                            raster.data[z_int, render_y, x_int, 1] = max(
                                raster.data[z_int, render_y, x_int, 1], ghost_g
                            )
                            raster.data[z_int, render_y, x_int, 2] = max(
                                raster.data[z_int, render_y, x_int, 2], ghost_b
                            )

    def set_tharok_encounter(self, active: bool, current_time: float):
        """Set Tharok encounter state. Called when encounter starts/ends."""
        print(
            f"[PYTHON-DEBUG] set_tharok_encounter called: active={active}, current_time={current_time}"
        )
        self.tharok_active = active
        if active:
            self.tharok_encounter_start_time = current_time
            self.tharok_thaw_progress = 0.0
            self.tharok_thaw_start_time = None
            self.tharok_flame_color_factor = 0.0
            self.tharok_exploding = False
            self.tharok_explosion_start_time = None
            self.tharok_particles = []
            print(f"[PYTHON-DEBUG] Tharok encounter activated! tharok_active={self.tharok_active}")
        else:
            self.tharok_encounter_start_time = None
            print("[PYTHON-DEBUG] Tharok encounter deactivated")

    def update_tharok_thawing(self, player_count: int, current_time: float):
        """Update Tharok thawing progress based on player count near rift.

        Args:
            player_count: Number of players within 10m of rift gate
            current_time: Current simulation time
        """
        if not self.tharok_active or self.tharok_exploding:
            return

        THAW_REQUIRED_PLAYERS = 5
        THAW_DURATION_SECONDS = 20.0

        if player_count >= THAW_REQUIRED_PLAYERS:
            if self.tharok_thaw_start_time is None:
                self.tharok_thaw_start_time = current_time

            # Calculate thaw progress
            thaw_elapsed = current_time - self.tharok_thaw_start_time
            self.tharok_thaw_progress = min(1.0, thaw_elapsed / THAW_DURATION_SECONDS)

            # Transition flame color from blue to red as thawing progresses
            self.tharok_flame_color_factor = self.tharok_thaw_progress

            # Check if fully thawed
            if self.tharok_thaw_progress >= 1.0 and not self.tharok_exploding:
                self.tharok_exploding = True
                self.tharok_explosion_start_time = current_time
                # Create explosion particles
                self._create_explosion_particles()
        else:
            # Not enough players, reset thawing
            self.tharok_thaw_start_time = None
            self.tharok_thaw_progress = 0.0
            self.tharok_flame_color_factor = 0.0

    def _create_explosion_particles(self):
        """Create particles for Tharok's explosion."""
        num_particles = 50
        self.tharok_particles = []
        for _ in range(num_particles):
            # Random direction
            angle = random.uniform(0, 2 * math.pi)
            elevation = random.uniform(-math.pi / 3, math.pi / 3)
            speed = random.uniform(2.0, 8.0)

            self.tharok_particles.append(
                {
                    "x": self.rift_center_x,
                    "y": self.height * 0.6,  # Skull position
                    "z": self.rift_center_z,
                    "vx": speed * math.cos(elevation) * math.cos(angle),
                    "vy": speed * math.sin(elevation),
                    "vz": speed * math.cos(elevation) * math.sin(angle),
                    "size": random.uniform(1.0, 3.0),
                    "lifetime": random.uniform(1.0, 2.5),
                }
            )

    def _update_tharok_particles(self, dt: float, current_time: float):
        """Update explosion particles."""
        if not self.tharok_exploding or not self.tharok_explosion_start_time:
            return

        explosion_age = current_time - self.tharok_explosion_start_time

        # Update particles
        for particle in self.tharok_particles[:]:
            particle["x"] += particle["vx"] * dt
            particle["y"] += particle["vy"] * dt
            particle["z"] += particle["vz"] * dt

            # Gravity
            particle["vy"] -= 5.0 * dt

            # Drag
            particle["vx"] *= 0.98
            particle["vy"] *= 0.98
            particle["vz"] *= 0.98

            # Suck toward rift after initial explosion
            if explosion_age > 0.5:
                dx = self.rift_center_x - particle["x"]
                dy = self.rift_y - particle["y"]
                dz = self.rift_center_z - particle["z"]
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                if dist > 0.1:
                    pull_strength = 3.0 * (explosion_age - 0.5)
                    particle["vx"] += dx / dist * pull_strength * dt
                    particle["vy"] += dy / dist * pull_strength * dt
                    particle["vz"] += dz / dist * pull_strength * dt

        # Remove particles that are too old or sucked into rift
        self.tharok_particles = [
            p
            for p in self.tharok_particles
            if explosion_age < p["lifetime"]
            and (
                math.sqrt(
                    (p["x"] - self.rift_center_x) ** 2
                    + (p["y"] - self.rift_y) ** 2
                    + (p["z"] - self.rift_center_z) ** 2
                )
                > 0.5
            )
        ]

    def _render_tharok(self, raster, current_time: float):
        """Render Tharok the Starver: goat skull with horns, yellow eyes, and flames."""
        if not self.tharok_active:
            return

        # Skull position (center of display, mid-height)
        skull_x = self.rift_center_x
        skull_y = self.height * 0.6
        skull_z = self.rift_center_z
        # Make skull fill most of the screen (use width/height for screen dimensions)
        skull_size = min(self.width, self.height) * 0.9

        if self.tharok_exploding:
            # Render explosion particles
            explosion_age = (
                current_time - self.tharok_explosion_start_time
                if self.tharok_explosion_start_time
                else 0.0
            )

            for particle in self.tharok_particles:
                particle_x = int(particle["x"])
                particle_y = int(particle["y"])
                particle_z = int(particle["z"])

                if (
                    0 <= particle_x < self.width
                    and 0 <= particle_y < self.height
                    and 0 <= particle_z < self.length
                ):

                    # Fade out over lifetime
                    age_ratio = explosion_age / particle["lifetime"]
                    intensity = max(0.0, 1.0 - age_ratio)

                    # Orange/yellow particle color
                    r = int(255 * intensity)
                    g = int(200 * intensity)
                    b = int(100 * intensity)

                    size_int = max(1, int(particle["size"] * intensity))
                    for dy in range(-size_int, size_int + 1):
                        for dx in range(-size_int, size_int + 1):
                            for dz in range(-size_int, size_int + 1):
                                if dx * dx + dy * dy + dz * dz <= size_int * size_int:
                                    px = particle_x + dx
                                    py = particle_y + dy
                                    pz = particle_z + dz
                                    if (
                                        0 <= px < self.width
                                        and 0 <= py < self.height
                                        and 0 <= pz < self.length
                                    ):
                                        raster.data[pz, py, px, 0] = max(
                                            raster.data[pz, py, px, 0], r
                                        )
                                        raster.data[pz, py, px, 1] = max(
                                            raster.data[pz, py, px, 1], g
                                        )
                                        raster.data[pz, py, px, 2] = max(
                                            raster.data[pz, py, px, 2], b
                                        )

            # If explosion complete, mark as vanquished and deactivate Tharok
            if explosion_age > 3.0 and len(self.tharok_particles) == 0:
                # Mark Tharok as vanquished on the server
                self._mark_tharok_vanquished()
                self.set_tharok_encounter(False, current_time)
            return

        # Render flames (blue transitioning to red during thawing)
        flame_base_r = int(100 * self.tharok_flame_color_factor)  # Red increases with thawing
        flame_base_g = int(150 * (1.0 - self.tharok_flame_color_factor * 0.5))
        flame_base_b = int(255 * (1.0 - self.tharok_flame_color_factor * 0.7))  # Blue decreases

        # Animated flame flicker
        flame_flicker = 0.8 + 0.2 * math.sin(current_time * 8.0)

        # Render flames around skull
        num_flame_points = 20
        for i in range(num_flame_points):
            angle = (i / num_flame_points) * 2 * math.pi
            radius = skull_size * (0.6 + random.uniform(-0.1, 0.1))
            flame_x = skull_x + math.cos(angle) * radius
            flame_z = skull_z + math.sin(angle) * radius

            # Flames rise up
            flame_height = skull_size * (0.8 + random.uniform(0, 0.4))
            for y_offset in range(int(flame_height)):
                flame_y = skull_y + y_offset
                if 0 <= flame_y < self.height:
                    # Fade with height
                    height_factor = 1.0 - (y_offset / flame_height)
                    intensity = height_factor * flame_flicker

                    r = int(flame_base_r * intensity)
                    g = int(flame_base_g * intensity)
                    b = int(flame_base_b * intensity)

                    fx = int(flame_x)
                    fz = int(flame_z)
                    if 0 <= fx < self.width and 0 <= fz < self.length:
                        raster.data[fz, int(flame_y), fx, 0] = max(
                            raster.data[fz, int(flame_y), fx, 0], r
                        )
                        raster.data[fz, int(flame_y), fx, 1] = max(
                            raster.data[fz, int(flame_y), fx, 1], g
                        )
                        raster.data[fz, int(flame_y), fx, 2] = max(
                            raster.data[fz, int(flame_y), fx, 2], b
                        )

        # Render goat skull (simplified: white bone-colored shape)
        skull_radius = int(skull_size * 0.4)
        for y_offset in range(-skull_radius, skull_radius + 1):
            for x_offset in range(-skull_radius, skull_radius + 1):
                for z_offset in range(-skull_radius // 2, skull_radius // 2 + 1):
                    dist = math.sqrt(
                        x_offset * x_offset + y_offset * y_offset + z_offset * z_offset
                    )
                    if dist <= skull_radius:
                        sx = int(skull_x + x_offset)
                        sy = int(skull_y + y_offset)
                        sz = int(skull_z + z_offset)

                        if 0 <= sx < self.width and 0 <= sy < self.height and 0 <= sz < self.length:

                            # Skull color (bone white)
                            intensity = 1.0 - (dist / skull_radius) * 0.3
                            r = int(240 * intensity)
                            g = int(235 * intensity)
                            b = int(220 * intensity)

                            raster.data[sz, sy, sx, 0] = max(raster.data[sz, sy, sx, 0], r)
                            raster.data[sz, sy, sx, 1] = max(raster.data[sz, sy, sx, 1], g)
                            raster.data[sz, sy, sx, 2] = max(raster.data[sz, sy, sx, 2], b)

        # Render horns (curved upward)
        horn_length = skull_size * 0.6
        for horn_side in [-1, 1]:  # Left and right horn
            horn_base_x = skull_x + horn_side * skull_size * 0.3
            horn_base_z = skull_z
            horn_base_y = skull_y

            # Curved horn
            for t in range(int(horn_length)):
                progress = t / horn_length
                angle = progress * math.pi * 0.3  # Curves upward
                horn_x = horn_base_x + math.cos(angle) * horn_side * progress * 0.3
                horn_y = horn_base_y + math.sin(angle) * horn_length * 0.8
                horn_z = horn_base_z

                horn_radius = int(skull_size * 0.08 * (1.0 - progress * 0.5))
                for dy in range(-horn_radius, horn_radius + 1):
                    for dx in range(-horn_radius, horn_radius + 1):
                        for dz in range(-horn_radius, horn_radius + 1):
                            if dx * dx + dy * dy + dz * dz <= horn_radius * horn_radius:
                                hx = int(horn_x + dx)
                                hy = int(horn_y + dy)
                                hz = int(horn_z + dz)

                                if (
                                    0 <= hx < self.width
                                    and 0 <= hy < self.height
                                    and 0 <= hz < self.length
                                ):
                                    # Horn color (darker bone)
                                    r = int(200)
                                    g = int(195)
                                    b = int(180)

                                    raster.data[hz, hy, hx, 0] = max(raster.data[hz, hy, hx, 0], r)
                                    raster.data[hz, hy, hx, 1] = max(raster.data[hz, hy, hx, 1], g)
                                    raster.data[hz, hy, hx, 2] = max(raster.data[hz, hy, hx, 2], b)

        # Render glowing yellow eyes
        eye_size = max(1, int(skull_size * 0.1))  # Ensure at least 1 to avoid division by zero
        eye_y_offset = int(skull_size * 0.15)
        for eye_side in [-1, 1]:
            eye_x = int(skull_x + eye_side * skull_size * 0.15)
            eye_y = int(skull_y + eye_y_offset)
            eye_z = int(skull_z + skull_size * 0.2)

            # Glowing yellow eyes
            eye_glow = 0.8 + 0.2 * math.sin(current_time * 4.0)
            for dy in range(-eye_size, eye_size + 1):
                for dx in range(-eye_size, eye_size + 1):
                    for dz in range(-eye_size, eye_size + 1):
                        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                        if dist <= eye_size:
                            ex = eye_x + dx
                            ey = eye_y + dy
                            ez = eye_z + dz

                            if (
                                0 <= ex < self.width
                                and 0 <= ey < self.height
                                and 0 <= ez < self.length
                            ):

                                intensity = (1.0 - dist / max(eye_size, 1.0)) * eye_glow
                                r = int(255 * intensity)
                                g = int(255 * intensity)
                                b = int(100 * intensity)

                                raster.data[ez, ey, ex, 0] = max(raster.data[ez, ey, ex, 0], r)
                                raster.data[ez, ey, ex, 1] = max(raster.data[ez, ey, ex, 1], g)
                                raster.data[ez, ey, ex, 2] = max(raster.data[ez, ey, ex, 2], b)
