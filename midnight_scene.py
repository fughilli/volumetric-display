"""Midnight Walk scene: A glowing fractal crack rift with planar ghost silhouettes spiraling upward."""

import io
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from artnet import RGB, Scene

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

        # Rift glow animation
        self.rift_pulse_phase = 0.0
        self.rift_pulse_speed = 1.5  # Pulses per second

        # Generate fractal crack pattern
        self.crack_segments: List[Tuple[float, float, float, float]] = []
        self._generate_fractal_crack()

        # Load ghost SVG silhouettes
        self.ghost_silhouettes: Dict[int, np.ndarray] = {}
        self.ghost_svg_paths: Optional[List[str]] = kwargs.get("ghost_svg_paths")
        self._load_ghost_silhouettes()

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

    def render(self, raster, time: float):
        """Render the scene with multiple ghosts spiraling outward and upward with radial normals."""
        # Clear raster
        raster.data.fill(0)

        # Update rift pulse animation
        dt = 1.0 / 60.0  # Assume 60 FPS
        self.rift_pulse_phase += self.rift_pulse_speed * dt

        # Spawn new ghosts (limit to 2 ghosts max)
        if time >= self.next_spawn:
            if len(self.ghost_silhouettes) > 0 and len(self.ghosts) < 2:
                self.ghosts.append(self.spawn_ghost(time))
            self.next_spawn = time + self.spawn_interval

        # Update and remove expired ghosts
        self.ghosts = [
            ghost
            for ghost in self.ghosts
            if ghost.update(dt, self.bounds, self.rift_center_x, self.rift_center_z, time)
            and not ghost.is_expired(time)
        ]

        # Render rift crack pattern (on the floor)
        rift_y_int = int(self.rift_y)
        if 0 <= rift_y_int < self.height:
            rift_glow = 0.6 + 0.4 * math.sin(self.rift_pulse_phase * 2 * math.pi)
            rift_color = RGB(
                int(255 * rift_glow), int(100 * rift_glow), int(100 * rift_glow)
            )  # Red glow

            for start_x, start_z, end_x, end_z in self.crack_segments:
                self._draw_line(
                    raster, start_x, start_z, end_x, end_z, rift_y_int, rift_color, width=1.0
                )

        # Render all ghosts
        if len(self.ghost_silhouettes) == 0:
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
            opacity = ghost.get_opacity(time)
            elongation = ghost.get_elongation()  # Vertical stretching based on speed
            size_scale = ghost.get_size_scale(time)  # Size grows from 0.5 to 1.0 over lifetime
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
