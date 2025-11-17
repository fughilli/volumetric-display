import math
import random
from dataclasses import dataclass
from typing import List, Set, Tuple

import numpy as np
import wgpu

from artnet import Raster, Scene
from color_palette import get_palette


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


class GPUNeuronsFiringScene(Scene):
    """
    GPU-accelerated neural network visualization using wgpu compute shaders.
    CPU handles physics simulation, GPU handles all rendering.
    """

    def __init__(self, **kwargs):
        properties = kwargs.get("properties")
        if not properties:
            raise ValueError("GPUNeuronsFiringScene requires a 'properties' object.")

        self.width = properties.width
        self.height = properties.height
        self.length = properties.length

        # Configuration
        self.lattice_cell_size = 4.0  # Distance between lattice points
        self.edge_ablation_probability = 0.3  # Probability of removing an edge
        self.pulse_speed = 16.0
        self.pulse_radius = 0.75
        self.edge_brightness = 255
        self.pulse_brightness = 255
        self.pulse_spawn_rate = 3.0

        # State
        self.vertices: List[Vertex] = []
        self.edges: Set[Tuple[int, int]] = set()
        self.pulses: List[Pulse] = []
        self.next_pulse_spawn = 0.0
        self.last_frame_time = 0.0

        # Lattice structure
        self.vertex_grid = {}  # Maps (gx, gy, gz) grid coords to vertex index

        # Fade-in state (no simulation needed for fixed lattice)
        self.freeze_time = 0.0
        self.fade_in_duration = 2.0
        self.start_time = 0.0

        # Color palette
        palette = get_palette()
        self.edge_color_base = palette.get_color(2)

        # Initialize cubic lattice
        self._initialize_lattice()

        # Initialize GPU
        self._init_gpu()

        print(
            f"GPUNeuronsFiringScene initialized with {len(self.vertices)} vertices on cubic lattice"
        )
        print(f"  Lattice cell size: {self.lattice_cell_size}")
        print(f"  Total edges: {len(self.edges)}")

    def _init_gpu(self):
        """Initialize wgpu device and compute shaders"""
        # Get GPU device
        self.adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        self.device = self.adapter.request_device_sync()

        # Create 3D volume texture using r32uint for read-write support
        # We'll pack RGBA into a single uint32 for atomic operations
        self.volume_size = (self.width, self.height, self.length)
        self.volume_texture = self.device.create_texture(
            size=self.volume_size,
            format=wgpu.TextureFormat.r32uint,
            usage=wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.COPY_SRC,
            dimension=wgpu.TextureDimension.d3,
        )

        # Create buffer for reading back data (r32uint = 4 bytes per voxel)
        bytes_per_row = self.width * 4  # r32uint is 4 bytes
        bytes_per_row_aligned = (bytes_per_row + 255) // 256 * 256
        readback_size = bytes_per_row_aligned * self.height * self.length

        self.readback_buffer = self.device.create_buffer(
            size=readback_size,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )

        # Compile shaders
        self._compile_shaders()

        print("GPU initialized successfully")

    def _compile_shaders(self):
        """Compile all compute shaders"""
        # Clear shader - using r32uint format
        clear_shader = """
        @group(0) @binding(0) var volume: texture_storage_3d<r32uint, write>;

        @compute @workgroup_size(8, 8, 8)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            textureStore(volume, vec3<i32>(global_id), vec4<u32>(0u));
        }
        """

        # Edge drawing shader - using r32uint with packed RGBA
        edge_shader = """
        struct Vertex {
            pos: vec3<f32>,
            padding: f32,  // For 16-byte alignment
        }

        struct Edge {
            v0_idx: u32,
            v1_idx: u32,
        }

        struct Uniforms {
            color: vec3<f32>,
            fade_factor: f32,
            num_edges: u32,
        }

        @group(0) @binding(0) var volume: texture_storage_3d<r32uint, read_write>;
        @group(0) @binding(1) var<storage, read> vertices: array<Vertex>;
        @group(0) @binding(2) var<storage, read> edges: array<Edge>;
        @group(0) @binding(3) var<uniform> uniforms: Uniforms;

        // Pack RGB into uint32 (8 bits per channel: 0xRRGGBB00)
        fn pack_rgb(color: vec3<f32>) -> u32 {
            let r = u32(clamp(color.r * 255.0, 0.0, 255.0));
            let g = u32(clamp(color.g * 255.0, 0.0, 255.0));
            let b = u32(clamp(color.b * 255.0, 0.0, 255.0));
            return (r << 24u) | (g << 16u) | (b << 8u);
        }

        // Unpack uint32 to RGB
        fn unpack_rgb(packed: u32) -> vec3<f32> {
            let r = f32((packed >> 24u) & 0xFFu) / 255.0;
            let g = f32((packed >> 16u) & 0xFFu) / 255.0;
            let b = f32((packed >> 8u) & 0xFFu) / 255.0;
            return vec3<f32>(r, g, b);
        }

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let edge_id = global_id.x;
            if (edge_id >= uniforms.num_edges) {
                return;
            }

            let edge = edges[edge_id];
            let v0 = vertices[edge.v0_idx].pos;
            let v1 = vertices[edge.v1_idx].pos;

            draw_line(v0, v1, uniforms.color * uniforms.fade_factor);
        }

        fn draw_line(p0: vec3<f32>, p1: vec3<f32>, color: vec3<f32>) {
            let delta = p1 - p0;
            let steps = max(max(abs(delta.x), abs(delta.y)), abs(delta.z));
            let num_steps = i32(steps);

            if (num_steps == 0) {
                return;
            }

            let inc = delta / steps;
            var pos = p0;

            for (var i = 0; i <= num_steps; i++) {
                let voxel = vec3<i32>(i32(round(pos.x)), i32(round(pos.y)), i32(round(pos.z)));

                // Bounds check
                if (voxel.x >= 0 && voxel.x < i32(textureDimensions(volume).x) &&
                    voxel.y >= 0 && voxel.y < i32(textureDimensions(volume).y) &&
                    voxel.z >= 0 && voxel.z < i32(textureDimensions(volume).z)) {

                    // Read current packed value
                    let current_packed = textureLoad(volume, voxel).r;
                    let current_color = unpack_rgb(current_packed);

                    // Additive blend
                    let new_color = min(vec3<f32>(1.0), current_color + color);
                    let new_packed = pack_rgb(new_color);

                    textureStore(volume, voxel, vec4<u32>(new_packed, 0u, 0u, 0u));
                }

                pos += inc;
            }
        }
        """

        # Pulse drawing shader - using r32uint with packed RGBA
        pulse_shader = """
        struct Pulse {
            pos: vec3<f32>,
            radius: f32,
        }

        struct Uniforms {
            color: vec3<f32>,
            fade_factor: f32,
            num_pulses: u32,
        }

        @group(0) @binding(0) var volume: texture_storage_3d<r32uint, read_write>;
        @group(0) @binding(1) var<storage, read> pulses: array<Pulse>;
        @group(0) @binding(2) var<uniform> uniforms: Uniforms;

        // Pack RGB into uint32 (8 bits per channel: 0xRRGGBB00)
        fn pack_rgb(color: vec3<f32>) -> u32 {
            let r = u32(clamp(color.r * 255.0, 0.0, 255.0));
            let g = u32(clamp(color.g * 255.0, 0.0, 255.0));
            let b = u32(clamp(color.b * 255.0, 0.0, 255.0));
            return (r << 24u) | (g << 16u) | (b << 8u);
        }

        // Unpack uint32 to RGB
        fn unpack_rgb(packed: u32) -> vec3<f32> {
            let r = f32((packed >> 24u) & 0xFFu) / 255.0;
            let g = f32((packed >> 16u) & 0xFFu) / 255.0;
            let b = f32((packed >> 8u) & 0xFFu) / 255.0;
            return vec3<f32>(r, g, b);
        }

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let pulse_id = global_id.x;
            if (pulse_id >= uniforms.num_pulses) {
                return;
            }

            let pulse = pulses[pulse_id];
            draw_sphere(pulse.pos, pulse.radius, uniforms.color * uniforms.fade_factor);
        }

        fn draw_sphere(center: vec3<f32>, radius: f32, color: vec3<f32>) {
            let min_bound = max(vec3<i32>(0), vec3<i32>(center - vec3<f32>(radius)));
            let max_bound = min(
                vec3<i32>(textureDimensions(volume)) - vec3<i32>(1),
                vec3<i32>(center + vec3<f32>(radius))
            );

            for (var z = min_bound.z; z <= max_bound.z; z++) {
                for (var y = min_bound.y; y <= max_bound.y; y++) {
                    for (var x = min_bound.x; x <= max_bound.x; x++) {
                        let pos = vec3<f32>(f32(x), f32(y), f32(z));
                        let dist = distance(pos, center);

                        if (dist <= radius) {
                            let intensity = 1.0 - (dist / radius) * 0.5;
                            let voxel = vec3<i32>(x, y, z);

                            // Read current packed value
                            let current_packed = textureLoad(volume, voxel).r;
                            let current_color = unpack_rgb(current_packed);

                            // Max blend for pulses (take brighter of current or new)
                            let new_color = max(current_color, color * intensity);
                            let new_packed = pack_rgb(new_color);

                            textureStore(volume, voxel, vec4<u32>(new_packed, 0u, 0u, 0u));
                        }
                    }
                }
            }
        }
        """

        # Create shader modules
        self.clear_shader_module = self.device.create_shader_module(code=clear_shader)
        self.edge_shader_module = self.device.create_shader_module(code=edge_shader)
        self.pulse_shader_module = self.device.create_shader_module(code=pulse_shader)

        # Create compute pipelines
        self._create_pipelines()

    def _create_pipelines(self):
        """Create compute pipelines and bind groups"""
        # Clear pipeline
        self.clear_pipeline = self.device.create_compute_pipeline(
            layout="auto",
            compute={"module": self.clear_shader_module, "entry_point": "main"},
        )

        clear_bind_group_layout = self.clear_pipeline.get_bind_group_layout(0)
        self.clear_bind_group = self.device.create_bind_group(
            layout=clear_bind_group_layout,
            entries=[
                {"binding": 0, "resource": self.volume_texture.create_view()},
            ],
        )

        # Edge pipeline
        self.edge_pipeline = self.device.create_compute_pipeline(
            layout="auto",
            compute={"module": self.edge_shader_module, "entry_point": "main"},
        )

        # Pulse pipeline
        self.pulse_pipeline = self.device.create_compute_pipeline(
            layout="auto",
            compute={"module": self.pulse_shader_module, "entry_point": "main"},
        )

    def _initialize_lattice(self):
        """Create vertices on a cubic lattice with ablated edges"""
        # Calculate lattice dimensions
        nx = int(self.width / self.lattice_cell_size)
        ny = int(self.height / self.lattice_cell_size)
        nz = int(self.length / self.lattice_cell_size)

        # Center the lattice
        offset_x = (self.width - (nx - 1) * self.lattice_cell_size) / 2
        offset_y = (self.height - (ny - 1) * self.lattice_cell_size) / 2
        offset_z = (self.length - (nz - 1) * self.lattice_cell_size) / 2

        # Create vertices on lattice points
        vertex_idx = 0
        for gx in range(nx):
            for gy in range(ny):
                for gz in range(nz):
                    x = offset_x + gx * self.lattice_cell_size
                    y = offset_y + gy * self.lattice_cell_size
                    z = offset_z + gz * self.lattice_cell_size

                    # Round to integer coordinates to ensure vertices are exactly on voxel centers
                    x = round(x)
                    y = round(y)
                    z = round(z)

                    vertex = Vertex(x=x, y=y, z=z)
                    self.vertices.append(vertex)
                    self.vertex_grid[(gx, gy, gz)] = vertex_idx
                    vertex_idx += 1

        # Create edges between adjacent lattice points (6-connected)
        all_edges = []
        for gx in range(nx):
            for gy in range(ny):
                for gz in range(nz):
                    v1_idx = self.vertex_grid[(gx, gy, gz)]

                    # Connect to neighbors in positive directions only
                    neighbors = [
                        (gx + 1, gy, gz),  # +X
                        (gx, gy + 1, gz),  # +Y
                        (gx, gy, gz + 1),  # +Z
                    ]

                    for neighbor in neighbors:
                        if neighbor in self.vertex_grid:
                            v2_idx = self.vertex_grid[neighbor]
                            all_edges.append((v1_idx, v2_idx))

        # Ablate edges randomly to create subnetworks
        for edge in all_edges:
            if random.random() > self.edge_ablation_probability:
                self.edges.add(edge)

        print(
            f"  Lattice: {nx}x{ny}x{nz}, Vertices: {len(self.vertices)}, Edges: {len(self.edges)}/{len(all_edges)}"
        )

    def _spawn_pulse(self):
        """Spawn a new pulse"""
        if not self.edges:
            return

        vertices_with_edges = set()
        for v1, v2 in self.edges:
            vertices_with_edges.add(v1)
            vertices_with_edges.add(v2)

        if not vertices_with_edges:
            return

        start_vertex = random.choice(list(vertices_with_edges))
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
        """Get neighbors of a vertex"""
        neighbors = []
        for v1, v2 in self.edges:
            if v1 == vertex_idx:
                neighbors.append(v2)
            elif v2 == vertex_idx:
                neighbors.append(v1)
        return neighbors

    def _update_pulses(self, dt: float):
        """Update pulse positions"""
        updated_pulses = []

        for pulse in self.pulses:
            reached_target = pulse.update(dt, self.vertices)

            if reached_target:
                neighbors = self._get_neighbors(pulse.target_vertex_idx)
                if neighbors:
                    if len(neighbors) > 1:
                        neighbors = [n for n in neighbors if n != pulse.current_vertex_idx]

                    if neighbors:
                        new_target = random.choice(neighbors)
                        pulse.current_vertex_idx = pulse.target_vertex_idx
                        pulse.target_vertex_idx = new_target
                        pulse.progress = 0.0
                        updated_pulses.append(pulse)
            else:
                updated_pulses.append(pulse)

        self.pulses = updated_pulses

    def _get_fade_in_factor(self, time: float) -> float:
        """Calculate fade-in factor for lattice (immediate fade-in, no simulation)"""
        time_since_start = time - self.start_time

        if time_since_start >= self.fade_in_duration:
            return 1.0

        t = time_since_start / self.fade_in_duration
        return t * t * (3.0 - 2.0 * t)

    def render(self, raster: Raster, time: float):
        """Render using GPU"""
        dt = time - self.last_frame_time if self.last_frame_time > 0 else 1.0 / 60.0
        self.last_frame_time = time

        # Initialize start time
        if self.start_time == 0.0:
            self.start_time = time

        # Spawn pulses
        if time >= self.next_pulse_spawn:
            self._spawn_pulse()
            self.next_pulse_spawn = time + (1.0 / self.pulse_spawn_rate)

        self._update_pulses(dt)

        # GPU rendering
        self._render_gpu(time)

        # Read back from GPU
        self._readback_to_cpu(raster)

    def _render_gpu(self, time: float):
        """Execute GPU rendering"""
        command_encoder = self.device.create_command_encoder()

        # Clear volume
        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.clear_pipeline)
        compute_pass.set_bind_group(0, self.clear_bind_group)
        workgroups_x = (self.width + 7) // 8
        workgroups_y = (self.height + 7) // 8
        workgroups_z = (self.length + 7) // 8
        compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z)
        compute_pass.end()

        fade_factor = self._get_fade_in_factor(time)

        if fade_factor > 0.01 and len(self.edges) > 0:
            # Draw edges
            self._draw_edges_gpu(command_encoder, fade_factor)

        if fade_factor > 0.01 and len(self.pulses) > 0:
            # Draw pulses
            self._draw_pulses_gpu(command_encoder, fade_factor)

        self.device.queue.submit([command_encoder.finish()])

    def _draw_edges_gpu(self, command_encoder, fade_factor):
        """Draw edges using GPU"""
        # Prepare vertex data with padding for 16-byte alignment (vec3<f32> + padding)
        vertex_data = np.zeros((len(self.vertices), 4), dtype=np.float32)
        for i, v in enumerate(self.vertices):
            vertex_data[i] = [v.x, v.y, v.z, 0.0]  # 4th component is padding

        # Prepare edge data - convert set to sorted list for deterministic ordering
        edge_list = sorted(list(self.edges))
        edge_data = np.array(edge_list, dtype=np.uint32).reshape(-1, 2)

        # Verify no duplicate edges
        if len(edge_list) != len(set(edge_list)):
            print(f"WARNING: Duplicate edges detected! {len(edge_list)} vs {len(set(edge_list))}")

        # Create buffers
        vertex_buffer = self.device.create_buffer_with_data(
            data=vertex_data, usage=wgpu.BufferUsage.STORAGE
        )
        edge_buffer = self.device.create_buffer_with_data(
            data=edge_data, usage=wgpu.BufferUsage.STORAGE
        )

        # Create uniform buffer
        brightness_factor = self.edge_brightness / 255.0
        color = np.array(
            [
                self.edge_color_base.red * brightness_factor / 255.0,
                self.edge_color_base.green * brightness_factor / 255.0,
                self.edge_color_base.blue * brightness_factor / 255.0,
                fade_factor,
                len(edge_list),
                0,
                0,
                0,  # Padding
            ],
            dtype=np.float32,
        )
        uniform_buffer = self.device.create_buffer_with_data(
            data=color, usage=wgpu.BufferUsage.UNIFORM
        )

        # Create bind group
        bind_group_layout = self.edge_pipeline.get_bind_group_layout(0)
        bind_group = self.device.create_bind_group(
            layout=bind_group_layout,
            entries=[
                {"binding": 0, "resource": self.volume_texture.create_view()},
                {"binding": 1, "resource": {"buffer": vertex_buffer, "size": vertex_data.nbytes}},
                {"binding": 2, "resource": {"buffer": edge_buffer, "size": edge_data.nbytes}},
                {"binding": 3, "resource": {"buffer": uniform_buffer, "size": color.nbytes}},
            ],
        )

        # Dispatch
        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.edge_pipeline)
        compute_pass.set_bind_group(0, bind_group)
        workgroups = (len(edge_list) + 63) // 64
        compute_pass.dispatch_workgroups(workgroups, 1, 1)
        compute_pass.end()

    def _draw_pulses_gpu(self, command_encoder, fade_factor):
        """Draw pulses using GPU"""
        # Prepare pulse data
        pulse_data = np.zeros((len(self.pulses), 4), dtype=np.float32)
        for i, pulse in enumerate(self.pulses):
            pos = pulse.get_position(self.vertices)
            pulse_data[i] = [pos[0], pos[1], pos[2], self.pulse_radius]

        # Create buffer
        pulse_buffer = self.device.create_buffer_with_data(
            data=pulse_data, usage=wgpu.BufferUsage.STORAGE
        )

        # Create uniform buffer
        pulse_brightness = self.pulse_brightness * fade_factor
        color = np.array(
            [
                pulse_brightness / 255.0,
                pulse_brightness / 255.0,
                pulse_brightness / 255.0,
                fade_factor,
                len(self.pulses),
                0,
                0,
                0,  # Padding
            ],
            dtype=np.float32,
        )
        uniform_buffer = self.device.create_buffer_with_data(
            data=color, usage=wgpu.BufferUsage.UNIFORM
        )

        # Create bind group
        bind_group_layout = self.pulse_pipeline.get_bind_group_layout(0)
        bind_group = self.device.create_bind_group(
            layout=bind_group_layout,
            entries=[
                {"binding": 0, "resource": self.volume_texture.create_view()},
                {"binding": 1, "resource": {"buffer": pulse_buffer, "size": pulse_data.nbytes}},
                {"binding": 2, "resource": {"buffer": uniform_buffer, "size": color.nbytes}},
            ],
        )

        # Dispatch
        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pulse_pipeline)
        compute_pass.set_bind_group(0, bind_group)
        workgroups = (len(self.pulses) + 63) // 64
        compute_pass.dispatch_workgroups(workgroups, 1, 1)
        compute_pass.end()

    def _readback_to_cpu(self, raster: Raster):
        """Read GPU texture back to CPU"""
        command_encoder = self.device.create_command_encoder()

        # Copy texture to buffer
        bytes_per_row = self.width * 4
        # Align to 256 bytes as required by WebGPU
        bytes_per_row_aligned = (bytes_per_row + 255) // 256 * 256

        command_encoder.copy_texture_to_buffer(
            {"texture": self.volume_texture},
            {
                "buffer": self.readback_buffer,
                "offset": 0,
                "bytes_per_row": bytes_per_row_aligned,
                "rows_per_image": self.height,
            },
            self.volume_size,
        )

        self.device.queue.submit([command_encoder.finish()])

        # Map and read buffer
        # Map buffer for reading
        self.readback_buffer.map_sync(mode=wgpu.MapMode.READ)

        # Read mapped data
        mapped_data = self.readback_buffer.read_mapped()

        # Unmap buffer
        self.readback_buffer.unmap()

        # Convert to numpy array of uint32
        data_array = np.frombuffer(mapped_data, dtype=np.uint32)

        # Handle row alignment by reading actual data per row
        volume_data = np.zeros((self.length, self.height, self.width, 3), dtype=np.uint8)

        for z in range(self.length):
            for y in range(self.height):
                src_offset = (z * self.height + y) * (bytes_per_row_aligned // 4)
                src_row = data_array[src_offset : src_offset + self.width]

                # Unpack RGB from uint32 (format: 0xRRGGBB00)
                volume_data[z, y, :, 0] = (src_row >> 24) & 0xFF  # R
                volume_data[z, y, :, 1] = (src_row >> 16) & 0xFF  # G
                volume_data[z, y, :, 2] = (src_row >> 8) & 0xFF  # B

        # Copy to raster (RGB only)
        raster.data[:, :, :, :3] = volume_data
