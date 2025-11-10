import math
import random
from dataclasses import dataclass
from time import perf_counter
from typing import List, Set, Tuple

import numpy as np

from artnet import RGB, Raster, Scene


@dataclass
class Vertex:
    """A vertex in 3D space"""

    x: float
    y: float
    z: float
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    velocity_z: float = 0.0

    def pos(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def distance_to(self, other: "Vertex") -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)


@dataclass
class Pulse:
    """A pulse traveling along an edge"""

    current_vertex_idx: int
    target_vertex_idx: int
    progress: float  # 0.0 to 1.0 along the edge
    speed: float  # units per second

    def update(self, dt: float, vertices: List[Vertex]) -> bool:
        """Returns True if pulse reached target vertex"""
        if self.current_vertex_idx == self.target_vertex_idx:
            return True

        v1 = vertices[self.current_vertex_idx]
        v2 = vertices[self.target_vertex_idx]
        edge_length = v1.distance_to(v2)

        if edge_length < 0.001:
            return True

        # Update progress
        self.progress += (self.speed * dt) / edge_length

        return self.progress >= 1.0

    def get_position(self, vertices: List[Vertex]) -> Tuple[float, float, float]:
        """Get current 3D position of the pulse"""
        v1 = vertices[self.current_vertex_idx]
        v2 = vertices[self.target_vertex_idx]

        t = min(1.0, self.progress)
        return (v1.x + (v2.x - v1.x) * t, v1.y + (v2.y - v1.y) * t, v1.z + (v2.z - v1.z) * t)


@dataclass
class Wave:
    """A wave that sweeps through the volume"""

    position: float  # Current position along the axis
    normal: Tuple[float, float, float]  # Direction vector (normalized)
    velocity: float  # Speed of wave movement
    start_position: float  # Starting position
    end_position: float  # Ending position
    magenta_decay_time: float = 2.0  # How long magenta trail lasts
    cyan_thickness: float = 3.0  # Thickness of cyan leading edge

    def update(self, dt: float):
        """Update wave position"""
        self.position += self.velocity * dt

    def is_alive(self) -> bool:
        """Check if wave is still active"""
        return self.position < self.end_position

    def distance_to_point(self, point: Tuple[float, float, float]) -> float:
        """Calculate signed distance from point to wave plane (negative = behind wave)"""
        # Vector from wave position along normal to the point
        plane_point = tuple(self.position * n for n in self.normal)
        return sum(self.normal[i] * (point[i] - plane_point[i]) for i in range(3))

    def get_cyan_intensity(self, distance: float) -> float:
        """Get cyan intensity for leading edge (distance ahead of wave)"""
        if distance < 0 or distance > self.cyan_thickness:
            return 0.0
        return 1.0 - (distance / self.cyan_thickness)

    def rasterize_plane_to_buffer(
        self, buffer: np.ndarray, width: int, height: int, length: int, intensity: float = 1.0
    ):
        """Vectorized plane rasterization into magenta buffer"""
        nx, ny, nz = self.normal
        plane_offset = self.position

        # Determine which axis to iterate over (choose the one most perpendicular to plane)
        abs_normal = [abs(nx), abs(ny), abs(nz)]
        max_idx = abs_normal.index(max(abs_normal))

        plane_thickness = 1.5

        if max_idx == 0:  # Normal mostly along X axis, iterate over Y-Z plane
            if abs(nx) < 0.001:
                return

            # Create meshgrid for Y-Z plane
            y_grid, z_grid = np.meshgrid(np.arange(height), np.arange(length), indexing="ij")

            # Convert to centered coordinates
            y_centered = y_grid - height / 2
            z_centered = z_grid - length / 2

            # Solve for x: nx*x + ny*y + nz*z = plane_offset
            x_centered = (plane_offset - ny * y_centered - nz * z_centered) / nx
            x_grid = x_centered + width / 2

            # Filter points within bounds
            valid_mask = (x_grid >= -plane_thickness) & (x_grid < width + plane_thickness)

            self._rasterize_grid_with_thickness(
                buffer,
                x_grid[valid_mask],
                y_grid[valid_mask],
                z_grid[valid_mask],
                width,
                height,
                length,
                intensity,
                plane_thickness,
            )

        elif max_idx == 1:  # Normal mostly along Y axis, iterate over X-Z plane
            if abs(ny) < 0.001:
                return

            # Create meshgrid for X-Z plane
            x_grid, z_grid = np.meshgrid(np.arange(width), np.arange(length), indexing="ij")

            # Convert to centered coordinates
            x_centered = x_grid - width / 2
            z_centered = z_grid - length / 2

            # Solve for y
            y_centered = (plane_offset - nx * x_centered - nz * z_centered) / ny
            y_grid = y_centered + height / 2

            valid_mask = (y_grid >= -plane_thickness) & (y_grid < height + plane_thickness)

            self._rasterize_grid_with_thickness(
                buffer,
                x_grid[valid_mask],
                y_grid[valid_mask],
                z_grid[valid_mask],
                width,
                height,
                length,
                intensity,
                plane_thickness,
            )

        else:  # Normal mostly along Z axis, iterate over X-Y plane
            if abs(nz) < 0.001:
                return

            # Create meshgrid for X-Y plane
            x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height), indexing="ij")

            # Convert to centered coordinates
            x_centered = x_grid - width / 2
            y_centered = y_grid - height / 2

            # Solve for z
            z_centered = (plane_offset - nx * x_centered - ny * y_centered) / nz
            z_grid = z_centered + length / 2

            valid_mask = (z_grid >= -plane_thickness) & (z_grid < length + plane_thickness)

            self._rasterize_grid_with_thickness(
                buffer,
                x_grid[valid_mask],
                y_grid[valid_mask],
                z_grid[valid_mask],
                width,
                height,
                length,
                intensity,
                plane_thickness,
            )

    def _rasterize_grid_with_thickness(
        self,
        buffer: np.ndarray,
        x_points: np.ndarray,
        y_points: np.ndarray,
        z_points: np.ndarray,
        width: int,
        height: int,
        length: int,
        intensity: float,
        thickness: float,
    ):
        """Vectorized rasterization of grid points with thickness"""
        # Round to nearest integer voxels
        ix = np.round(x_points).astype(np.int32)
        iy = np.round(y_points).astype(np.int32)
        iz = np.round(z_points).astype(np.int32)

        # Generate offsets for thickness
        t_range = int(thickness) + 1
        offsets = np.arange(-t_range, t_range + 1)

        # For each point, apply thickness by iterating over offset combinations
        for dx in offsets:
            for dy in offsets:
                for dz in offsets:
                    # Calculate offset positions
                    vx = ix + dx
                    vy = iy + dy
                    vz = iz + dz

                    # Bounds check (vectorized)
                    valid = (
                        (vx >= 0)
                        & (vx < width)
                        & (vy >= 0)
                        & (vy < height)
                        & (vz >= 0)
                        & (vz < length)
                    )

                    if not np.any(valid):
                        continue

                    # Get valid indices
                    vx_valid = vx[valid]
                    vy_valid = vy[valid]
                    vz_valid = vz[valid]
                    x_valid = x_points[valid]
                    y_valid = y_points[valid]
                    z_valid = z_points[valid]

                    # Calculate distances (vectorized)
                    dist = np.sqrt(
                        (vx_valid - x_valid) ** 2
                        + (vy_valid - y_valid) ** 2
                        + (vz_valid - z_valid) ** 2
                    )

                    # Apply activation with falloff
                    within_thickness = dist <= thickness
                    if np.any(within_thickness):
                        activation = (1.0 - dist[within_thickness] / thickness) * intensity

                        # Update buffer (use maximum to accumulate)
                        vx_final = vx_valid[within_thickness]
                        vy_final = vy_valid[within_thickness]
                        vz_final = vz_valid[within_thickness]

                        for i in range(len(vx_final)):
                            buffer[vz_final[i], vy_final[i], vx_final[i]] = max(
                                buffer[vz_final[i], vy_final[i], vx_final[i]], activation[i]
                            )


class NeuronsFiringScene(Scene):
    """
    A 3D neural network visualization with:
    - Blue mesh of edges connecting nearby vertices
    - White pulses traveling along edges
    - Morphing mesh that updates connectivity based on distance
    """

    def __init__(self, **kwargs):
        properties = kwargs.get("properties")
        if not properties:
            raise ValueError("NeuronsFiringScene requires a 'properties' object.")

        self.width = properties.width
        self.height = properties.height
        self.length = properties.length

        # Configuration
        self.num_vertices = 200  # Number of vertices in the mesh (increased from 50)
        self.max_edge_length = 10.0  # Max distance for edge connectivity
        self.vertex_speed = 0.2  # Speed of vertex movement
        self.pulse_speed = 16.0  # Speed of pulses
        self.pulse_radius = 0.75  # Render radius for pulses
        self.edge_brightness = 255  # Brightness for blue edges (0-255)
        self.pulse_brightness = 255  # Brightness for white pulses
        self.mesh_update_interval = (
            0.5  # How often to update connectivity (seconds, increased from 0.5)
        )
        self.pulse_spawn_rate = 5.0  # Pulses per second (increased from 2.0)

        # Force-directed layout parameters
        self.target_distance = self.max_edge_length * 0.7  # Target spacing between vertices
        self.repulsion_strength = 50.0  # Strength of repulsion force
        self.attraction_strength = 2.0  # Strength of attraction force
        self.force_damping = 0.95  # Damping factor to stabilize movement

        # Wave parameters
        self.wave_spawn_interval = 8.0  # Seconds between wave spawns
        self.wave_velocity = 20.0  # Speed of wave movement
        self.wave_magenta_decay = 1.0  # How long magenta trail lasts
        self.wave_cyan_thickness = 3.0  # Thickness of cyan leading edge

        # State
        self.vertices: List[Vertex] = []
        self.edges: Set[Tuple[int, int]] = set()  # Set of (v1_idx, v2_idx) tuples
        self.pulses: List[Pulse] = []
        self.waves: List[Wave] = []
        self.magenta_buffer: np.ndarray = (
            None  # Back buffer for magenta wave trail (will be initialized in render)
        )
        self.last_mesh_update = 0.0
        self.next_pulse_spawn = 0.0
        self.next_wave_spawn = 3.0  # First wave after 3 seconds
        self.last_frame_time = 0.0

        # Profiling
        self.profile_enabled = True
        self.last_profile_print = 0.0
        self.profile_interval = 2.0  # Print every 2 seconds
        self.profile_samples = {
            "total": [],
            "decay": [],
            "mesh_update": [],
            "vertices_update": [],
            "forces": [],
            "waves_update": [],
            "waves_rasterize": [],
            "pulses_update": [],
            "composite": [],
            "edges_render": [],
            "pulses_render": [],
        }

        # Initialize vertices with random positions and velocities
        self._initialize_vertices()
        self._update_mesh_connectivity()

        print(f"NeuronsFiringScene initialized with {self.num_vertices} vertices")

    def _initialize_vertices(self):
        """Create initial vertices with random positions and velocities"""
        for _ in range(self.num_vertices):
            vertex = Vertex(
                x=random.uniform(2, self.width - 2),
                y=random.uniform(2, self.height - 2),
                z=random.uniform(2, self.length - 2),
                velocity_x=random.uniform(-1, 1) * self.vertex_speed,
                velocity_y=random.uniform(-1, 1) * self.vertex_speed,
                velocity_z=random.uniform(-1, 1) * self.vertex_speed,
            )
            self.vertices.append(vertex)

    def _spawn_wave(self):
        """Spawn a new wave with random direction"""
        # Compute the diagonal of the raster for wave travel distance
        dimensions = (self.width, self.height, self.length)
        raster_size = math.sqrt(sum(d**2 for d in dimensions))

        # Random direction
        normal = [random.uniform(-1, 1) for _ in range(3)]
        norm = math.sqrt(sum(n**2 for n in normal))
        normal = tuple(n / norm for n in normal)

        wave = Wave(
            position=-(raster_size / 2 + 1),
            normal=normal,
            velocity=self.wave_velocity,
            start_position=-(raster_size / 2 + 1),
            end_position=raster_size / 2 + 1,
            magenta_decay_time=self.wave_magenta_decay,
            cyan_thickness=self.wave_cyan_thickness,
        )
        self.waves.append(wave)

    def _update_waves(self, dt: float, raster: Raster):
        """Update wave positions, rasterize to buffer, and remove dead waves"""
        t_rasterize = 0.0
        for wave in self.waves:
            # Rasterize current wave position to magenta buffer
            t0 = perf_counter()
            wave.rasterize_plane_to_buffer(
                self.magenta_buffer, raster.width, raster.height, raster.length, intensity=1.0
            )
            t_rasterize += perf_counter() - t0
            wave.update(dt)

        if self.profile_enabled:
            self.profile_samples["waves_rasterize"].append(t_rasterize * 1000)  # Convert to ms

        self.waves = [wave for wave in self.waves if wave.is_alive()]

    def _update_mesh_connectivity(self):
        """Update edges based on distance between vertices"""
        self.edges.clear()

        for i in range(len(self.vertices)):
            for j in range(i + 1, len(self.vertices)):
                distance = self.vertices[i].distance_to(self.vertices[j])
                if distance <= self.max_edge_length:
                    self.edges.add((i, j))

    def _apply_forces(self, dt: float):
        """Apply attraction/repulsion forces between vertices using force-directed graph layout"""
        # Calculate forces for each vertex
        forces = [(0.0, 0.0, 0.0) for _ in self.vertices]

        # 1. Apply repulsion between ALL vertex pairs (keeps them spread out)
        for i in range(len(self.vertices)):
            fx, fy, fz = 0.0, 0.0, 0.0
            v1 = self.vertices[i]

            for j in range(len(self.vertices)):
                if i == j:
                    continue

                v2 = self.vertices[j]

                # Calculate distance and direction
                dx = v2.x - v1.x
                dy = v2.y - v1.y
                dz = v2.z - v1.z
                distance = math.sqrt(dx * dx + dy * dy + dz * dz)

                if distance < 0.1:  # Avoid division by zero
                    continue

                # Normalize direction
                nx = dx / distance
                ny = dy / distance
                nz = dz / distance

                # Repulsion force (inverse square law for natural spreading)
                # Pushes vertices away from each other
                repulsion = self.repulsion_strength / (distance * distance)

                # Apply repulsion (negative because we push away from v2)
                fx -= nx * repulsion
                fy -= ny * repulsion
                fz -= nz * repulsion

            forces[i] = (fx, fy, fz)

        # 2. Apply attraction ONLY between connected vertices (keeps edges at reasonable length)
        for v1_idx, v2_idx in self.edges:
            v1 = self.vertices[v1_idx]
            v2 = self.vertices[v2_idx]

            # Calculate distance and direction
            dx = v2.x - v1.x
            dy = v2.y - v1.y
            dz = v2.z - v1.z
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)

            if distance < 0.1:
                continue

            # Normalize direction
            nx = dx / distance
            ny = dy / distance
            nz = dz / distance

            # Attraction force (linear with distance from target)
            # Only pulls together vertices that are connected
            attraction = self.attraction_strength * (distance - self.target_distance)

            # Apply attraction (pull towards each other)
            fx1, fy1, fz1 = forces[v1_idx]
            fx2, fy2, fz2 = forces[v2_idx]

            forces[v1_idx] = (fx1 + nx * attraction, fy1 + ny * attraction, fz1 + nz * attraction)
            forces[v2_idx] = (fx2 - nx * attraction, fy2 - ny * attraction, fz2 - nz * attraction)

        # Apply forces to velocities
        for i, vertex in enumerate(self.vertices):
            fx, fy, fz = forces[i]
            vertex.velocity_x += fx * dt
            vertex.velocity_y += fy * dt
            vertex.velocity_z += fz * dt

            # Apply damping to prevent excessive speeds
            vertex.velocity_x *= self.force_damping
            vertex.velocity_y *= self.force_damping
            vertex.velocity_z *= self.force_damping

    def _update_vertices(self, dt: float):
        """Update vertex positions and bounce off walls"""
        # Apply forces first
        t0 = perf_counter()
        self._apply_forces(dt)
        if self.profile_enabled:
            self.profile_samples["forces"].append((perf_counter() - t0) * 1000)

        for vertex in self.vertices:
            # Update position
            vertex.x += vertex.velocity_x * dt
            vertex.y += vertex.velocity_y * dt
            vertex.z += vertex.velocity_z * dt

            # Bounce off walls
            if vertex.x < 1 or vertex.x > self.width - 1:
                vertex.velocity_x *= -1
                vertex.x = max(1, min(self.width - 1, vertex.x))

            if vertex.y < 1 or vertex.y > self.height - 1:
                vertex.velocity_y *= -1
                vertex.y = max(1, min(self.height - 1, vertex.y))

            if vertex.z < 1 or vertex.z > self.length - 1:
                vertex.velocity_z *= -1
                vertex.z = max(1, min(self.length - 1, vertex.z))

    def _spawn_pulse(self):
        """Spawn a new pulse at a random vertex"""
        if not self.edges:
            return

        # Pick a random vertex that has at least one connection
        vertices_with_edges = set()
        for v1, v2 in self.edges:
            vertices_with_edges.add(v1)
            vertices_with_edges.add(v2)

        if not vertices_with_edges:
            return

        start_vertex = random.choice(list(vertices_with_edges))

        # Find connected vertices
        neighbors = self._get_neighbors(start_vertex)
        if neighbors:
            target_vertex = random.choice(neighbors)
            pulse = Pulse(
                current_vertex_idx=start_vertex,
                target_vertex_idx=target_vertex,
                progress=0.0,
                speed=self.pulse_speed,
            )
            self.pulses.append(pulse)

    def _get_neighbors(self, vertex_idx: int) -> List[int]:
        """Get all vertices connected to the given vertex"""
        neighbors = []
        for v1, v2 in self.edges:
            if v1 == vertex_idx:
                neighbors.append(v2)
            elif v2 == vertex_idx:
                neighbors.append(v1)
        return neighbors

    def _update_pulses(self, dt: float):
        """Update pulse positions and handle vertex transitions"""
        updated_pulses = []

        for pulse in self.pulses:
            reached_target = pulse.update(dt, self.vertices)

            if reached_target:
                # Pulse reached a vertex, choose a new direction
                neighbors = self._get_neighbors(pulse.target_vertex_idx)
                if neighbors:
                    # Choose a random neighbor (excluding where we came from if possible)
                    if len(neighbors) > 1:
                        # Try to avoid going back
                        neighbors = [n for n in neighbors if n != pulse.current_vertex_idx]

                    if neighbors:
                        new_target = random.choice(neighbors)
                        pulse.current_vertex_idx = pulse.target_vertex_idx
                        pulse.target_vertex_idx = new_target
                        pulse.progress = 0.0
                        updated_pulses.append(pulse)
                # If no neighbors, pulse dies
            else:
                updated_pulses.append(pulse)

        self.pulses = updated_pulses

    def _draw_line(
        self,
        raster: Raster,
        x0: float,
        y0: float,
        z0: float,
        x1: float,
        y1: float,
        z1: float,
        color: RGB,
        thickness: float = 0.5,
    ):
        """Draw a line between two 3D points using DDA algorithm"""
        dx = x1 - x0
        dy = y1 - y0
        dz = z1 - z0

        # Find the longest dimension
        steps = int(max(abs(dx), abs(dy), abs(dz)) * 2)  # *2 for smoother lines

        if steps == 0:
            return

        x_inc = dx / steps
        y_inc = dy / steps
        z_inc = dz / steps

        x, y, z = x0, y0, z0

        for _ in range(steps + 1):
            # Draw the point with thickness
            for tx in range(-int(thickness), int(thickness) + 1):
                for ty in range(-int(thickness), int(thickness) + 1):
                    for tz in range(-int(thickness), int(thickness) + 1):
                        px = int(x + tx)
                        py = int(y + ty)
                        pz = int(z + tz)

                        if (
                            0 <= px < raster.width
                            and 0 <= py < raster.height
                            and 0 <= pz < raster.length
                        ):
                            # Use additive blending for edges
                            raster.data[pz, py, px, 0] = min(
                                255, raster.data[pz, py, px, 0] + color.red
                            )
                            raster.data[pz, py, px, 1] = min(
                                255, raster.data[pz, py, px, 1] + color.green
                            )
                            raster.data[pz, py, px, 2] = min(
                                255, raster.data[pz, py, px, 2] + color.blue
                            )

            x += x_inc
            y += y_inc
            z += z_inc

    def _draw_sphere(
        self, raster: Raster, cx: float, cy: float, cz: float, radius: float, color: RGB
    ):
        """Draw a sphere at the given position"""
        x_min = max(0, int(cx - radius))
        x_max = min(raster.width - 1, int(cx + radius))
        y_min = max(0, int(cy - radius))
        y_max = min(raster.height - 1, int(cy + radius))
        z_min = max(0, int(cz - radius))
        z_max = min(raster.length - 1, int(cz + radius))

        for z in range(z_min, z_max + 1):
            for y in range(y_min, y_max + 1):
                for x in range(x_min, x_max + 1):
                    dx = x - cx
                    dy = y - cy
                    dz = z - cz
                    distance = math.sqrt(dx * dx + dy * dy + dz * dz)

                    if distance <= radius:
                        # Soft falloff
                        intensity = 1.0 - (distance / radius) * 0.5
                        intensity = max(0.0, min(1.0, intensity))

                        # Use maximum blending for bright pulses
                        raster.data[z, y, x, 0] = max(
                            raster.data[z, y, x, 0], int(color.red * intensity)
                        )
                        raster.data[z, y, x, 1] = max(
                            raster.data[z, y, x, 1], int(color.green * intensity)
                        )
                        raster.data[z, y, x, 2] = max(
                            raster.data[z, y, x, 2], int(color.blue * intensity)
                        )

    def render(self, raster: Raster, time: float):
        """Render the neural network scene with wave effects"""
        t_frame_start = perf_counter()

        dt = time - self.last_frame_time if self.last_frame_time > 0 else 1.0 / 60.0
        self.last_frame_time = time

        # Initialize magenta buffer on first frame
        if self.magenta_buffer is None:
            self.magenta_buffer = np.zeros(
                (raster.length, raster.height, raster.width), dtype=np.float32
            )

        # Clear raster
        raster.data.fill(0)

        # Decay magenta buffer uniformly (exponential decay)
        t0 = perf_counter()
        decay_rate = 3.0 / self.wave_magenta_decay  # Rate for exponential decay
        decay_factor = math.exp(-decay_rate * dt)
        self.magenta_buffer *= decay_factor
        if self.profile_enabled:
            self.profile_samples["decay"].append((perf_counter() - t0) * 1000)

        # Update mesh connectivity periodically
        t0 = perf_counter()
        if time - self.last_mesh_update >= self.mesh_update_interval:
            self._update_mesh_connectivity()
            self.last_mesh_update = time
            if self.profile_enabled:
                self.profile_samples["mesh_update"].append((perf_counter() - t0) * 1000)

        # Update vertices (includes forces)
        t0 = perf_counter()
        self._update_vertices(dt)
        if self.profile_enabled:
            self.profile_samples["vertices_update"].append((perf_counter() - t0) * 1000)

        # Spawn and update waves (waves rasterize themselves into magenta buffer)
        t0 = perf_counter()
        if time >= self.next_wave_spawn:
            self._spawn_wave()
            self.next_wave_spawn = time + self.wave_spawn_interval
        self._update_waves(dt, raster)
        if self.profile_enabled:
            self.profile_samples["waves_update"].append((perf_counter() - t0) * 1000)

        # Spawn new pulses
        if time >= self.next_pulse_spawn:
            self._spawn_pulse()
            self.next_pulse_spawn = time + (1.0 / self.pulse_spawn_rate)

        # Update pulses
        t0 = perf_counter()
        self._update_pulses(dt)
        if self.profile_enabled:
            self.profile_samples["pulses_update"].append((perf_counter() - t0) * 1000)

        # Composite magenta buffer into output (before edges)
        t0 = perf_counter()
        self._composite_magenta_buffer(raster)
        if self.profile_enabled:
            self.profile_samples["composite"].append((perf_counter() - t0) * 1000)

        # Render edges with wave-influenced colors
        t0 = perf_counter()
        self._render_edges_with_waves(raster, time)
        if self.profile_enabled:
            self.profile_samples["edges_render"].append((perf_counter() - t0) * 1000)

        # Render pulses (white/bright spheres)
        t0 = perf_counter()
        white_color = RGB(self.pulse_brightness, self.pulse_brightness, self.pulse_brightness)
        for pulse in self.pulses:
            px, py, pz = pulse.get_position(self.vertices)
            self._draw_sphere(raster, px, py, pz, self.pulse_radius, white_color)
        if self.profile_enabled:
            self.profile_samples["pulses_render"].append((perf_counter() - t0) * 1000)

        # Total frame time
        if self.profile_enabled:
            self.profile_samples["total"].append((perf_counter() - t_frame_start) * 1000)

            # Print profile stats periodically
            if time - self.last_profile_print >= self.profile_interval:
                self._print_profile_stats()
                self.last_profile_print = time

    def _composite_magenta_buffer(self, raster: Raster):
        """Composite magenta buffer into output raster using vectorized hue-additive blending"""
        # Threshold to avoid processing very dim values
        mask = self.magenta_buffer > 0.01

        if not np.any(mask):
            return

        # Extract intensity values where mask is True (float 0-1)
        intensities = self.magenta_buffer[mask]

        # Get existing RGB values at masked locations (uint8 0-255)
        r_old = raster.data[mask, 0].astype(np.float32)
        g_old = raster.data[mask, 1].astype(np.float32)
        b_old = raster.data[mask, 2].astype(np.float32)

        # Vectorized RGB to HSV conversion
        h_old, s_old, v_old = self._rgb_to_hsv_vectorized(r_old, g_old, b_old)

        # Magenta in HSV: hue=300° (0.833 in 0-1 range), saturation=1.0, value from intensity
        magenta_hue_rad = np.deg2rad(300.0)
        magenta_value = intensities  # Already 0-1 range

        # Convert hues to radians for circular blending
        h_old_rad = h_old * 2 * np.pi

        # Weights based on brightness (value)
        old_weight = v_old
        mag_weight = magenta_value
        total_weight = old_weight + mag_weight

        # Avoid division by zero
        total_weight = np.maximum(total_weight, 1e-6)

        # Circular hue interpolation using unit vectors
        old_x = np.cos(h_old_rad) * old_weight
        old_y = np.sin(h_old_rad) * old_weight
        mag_x = np.cos(magenta_hue_rad) * mag_weight
        mag_y = np.sin(magenta_hue_rad) * mag_weight

        blend_x = (old_x + mag_x) / total_weight
        blend_y = (old_y + mag_y) / total_weight
        h_new_rad = np.arctan2(blend_y, blend_x)
        h_new = (h_new_rad / (2 * np.pi)) % 1.0  # Convert back to 0-1 range

        # Weighted saturation blending (magenta has full saturation = 1.0)
        s_new = (s_old * old_weight + 1.0 * mag_weight) / total_weight

        # Additive value (brightness)
        v_new = np.minimum(1.0, v_old + magenta_value)

        # Vectorized HSV to RGB conversion
        r_new, g_new, b_new = self._hsv_to_rgb_vectorized(h_new, s_new, v_new)

        # Write back to raster (convert to uint8)
        raster.data[mask, 0] = r_new.astype(np.uint8)
        raster.data[mask, 1] = g_new.astype(np.uint8)
        raster.data[mask, 2] = b_new.astype(np.uint8)

    def _rgb_to_hsv_vectorized(self, r, g, b):
        """Vectorized RGB to HSV conversion (inputs 0-255, outputs 0-1)"""
        r, g, b = r / 255.0, g / 255.0, b / 255.0

        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        delta = max_c - min_c

        # Value
        v = max_c

        # Saturation
        s = np.where(max_c > 0, delta / max_c, 0.0)

        # Hue
        h = np.zeros_like(r)

        # Where delta > 0
        mask_delta = delta > 0

        # Red is max
        mask_r = mask_delta & (max_c == r)
        h[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6.0

        # Green is max
        mask_g = mask_delta & (max_c == g)
        h[mask_g] = ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2.0

        # Blue is max
        mask_b = mask_delta & (max_c == b)
        h[mask_b] = ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4.0

        h = h / 6.0  # Normalize to 0-1

        return h, s, v

    def _hsv_to_rgb_vectorized(self, h, s, v):
        """Vectorized HSV to RGB conversion (inputs 0-1, outputs 0-255)"""
        h = h * 6.0  # Scale to 0-6 range

        i = np.floor(h).astype(np.int32)
        f = h - i

        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))

        i = i % 6

        r = np.choose(i, [v, q, p, p, t, v])
        g = np.choose(i, [t, v, v, q, p, p])
        b = np.choose(i, [p, p, t, v, v, q])

        return r * 255.0, g * 255.0, b * 255.0

    def _render_edges_with_waves(self, raster: Raster, time: float):
        """Render edges with color influenced by waves (cyan at wave front, blue otherwise)"""
        for v1_idx, v2_idx in self.edges:
            v1 = self.vertices[v1_idx]
            v2 = self.vertices[v2_idx]

            # Get midpoint of edge for wave distance calculation
            mid_x = (v1.x + v2.x) / 2 - raster.width / 2
            mid_y = (v1.y + v2.y) / 2 - raster.height / 2
            mid_z = (v1.z + v2.z) / 2 - raster.length / 2
            mid_point = (mid_x, mid_y, mid_z)

            # Check if edge is near any wave front (cyan effect)
            max_cyan_intensity = 0.0
            for wave in self.waves:
                distance = wave.distance_to_point(mid_point)
                cyan_intensity = wave.get_cyan_intensity(distance)
                max_cyan_intensity = max(max_cyan_intensity, cyan_intensity)

            # Blend between blue (base) and cyan (wave front)
            if max_cyan_intensity > 0.01:
                # Cyan color (0, 255, 255)
                edge_r = int(0 * (1 - max_cyan_intensity) + 0 * max_cyan_intensity)
                edge_g = int(
                    0 * (1 - max_cyan_intensity) + self.edge_brightness * max_cyan_intensity
                )
                edge_b = int(
                    self.edge_brightness * (1 - max_cyan_intensity)
                    + self.edge_brightness * max_cyan_intensity
                )
            else:
                # Base blue color
                edge_r = 0
                edge_g = 0
                edge_b = self.edge_brightness

            edge_color = RGB(edge_r, edge_g, edge_b)
            self._draw_line(raster, v1.x, v1.y, v1.z, v2.x, v2.y, v2.z, edge_color)

    def _print_profile_stats(self):
        """Print profiling statistics"""
        print("\n=== PROFILING STATS (ms) ===")
        print(f"{'Stage':<20} {'Mean':>8} {'Max':>8} {'Min':>8} {'Samples':>8}")
        print("-" * 60)

        for stage, samples in self.profile_samples.items():
            if samples:
                mean = sum(samples) / len(samples)
                max_val = max(samples)
                min_val = min(samples)
                print(f"{stage:<20} {mean:>8.2f} {max_val:>8.2f} {min_val:>8.2f} {len(samples):>8}")

        # Calculate frame rate
        if self.profile_samples["total"]:
            avg_frame_time = sum(self.profile_samples["total"]) / len(self.profile_samples["total"])
            fps = 1000.0 / avg_frame_time if avg_frame_time > 0 else 0
            print(f"\nAverage FPS: {fps:.1f}")

        print(f"Active waves: {len(self.waves)}")
        print(f"Active pulses: {len(self.pulses)}")
        print(f"Edges: {len(self.edges)}")

        # Clear samples for next interval
        for key in self.profile_samples:
            self.profile_samples[key].clear()
