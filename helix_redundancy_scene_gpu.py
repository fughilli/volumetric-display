import struct
from dataclasses import dataclass
from typing import List

import numpy as np
import wgpu

from artnet import Raster, Scene
from color_palette import get_palette


@dataclass
class HelixPulse:
    """A pulse traveling along a helix"""

    helix_idx: int
    progress: float
    speed: float
    color: tuple  # RGB 0-1


class HelixRedundancySceneGPU(Scene):
    """
    GPU-accelerated three intertwined helical strands.
    Represents data replication and availability zones.
    """

    def __init__(self, **kwargs):
        properties = kwargs.get("properties")
        if not properties:
            raise ValueError("HelixRedundancySceneGPU requires a 'properties' object.")

        self.width = properties.width
        self.height = properties.height
        self.length = properties.length

        # Helix configuration
        self.num_helices = 3
        self.helix_radius = min(self.width, self.height) / 2.0
        self.rotation_speed = 0.3
        self.pitch = self.length * 0.8

        self.center_x = self.width / 2
        self.center_y = self.height / 2

        # Get colors from palette
        palette = get_palette()
        self.helix_colors = [
            palette.get_color(2),  # Blue
            palette.get_color(1),  # Green
            palette.get_color(0),  # Orange
        ]

        # Pulses
        self.pulses: List[HelixPulse] = []
        self.pulse_spawn_rate = 2.0
        self.next_pulse_spawn = 0.0
        self.next_helix_idx = 0

        # Sync points
        self.sync_interval = 4.0
        self.next_sync = 2.0
        self.last_sync_time = 0.0

        # Initialize GPU
        self._init_gpu()

        print(f"HelixRedundancySceneGPU initialized with {self.num_helices} helices")

    def _init_gpu(self):
        """Initialize GPU resources"""
        self.adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        self.device = self.adapter.request_device_sync()

        self.volume_size = (self.width, self.height, self.length)
        self.volume_texture = self.device.create_texture(
            size=self.volume_size,
            format=wgpu.TextureFormat.r32uint,
            usage=wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.COPY_SRC,
            dimension=wgpu.TextureDimension.d3,
        )

        # Readback buffer
        bytes_per_row = self.width * 4
        bytes_per_row_aligned = (bytes_per_row + 255) // 256 * 256
        readback_size = bytes_per_row_aligned * self.height * self.length

        self.readback_buffer = self.device.create_buffer(
            size=readback_size,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )

        self._compile_shaders()

    def _compile_shaders(self):
        """Compile compute shaders"""
        # Clear shader
        clear_shader = """
        @group(0) @binding(0) var volume: texture_storage_3d<r32uint, write>;

        @compute @workgroup_size(8, 8, 8)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            textureStore(volume, vec3<i32>(global_id), vec4<u32>(0u));
        }
        """

        # Helix rendering shader
        helix_shader = """
        struct Uniforms {
            center_x: f32,
            center_y: f32,
            radius: f32,
            rotation_speed: f32,
            pitch: f32,
            time: f32,
            length: f32,
            num_helices: u32,
            color0_r: f32,
            color0_g: f32,
            color0_b: f32,
            color1_r: f32,
            color1_g: f32,
            color1_b: f32,
            color2_r: f32,
            color2_g: f32,
            color2_b: f32,
        }

        @group(0) @binding(0) var volume: texture_storage_3d<r32uint, read_write>;
        @group(0) @binding(1) var<uniform> uniforms: Uniforms;

        fn pack_rgb(color: vec3<f32>) -> u32 {
            let r = u32(clamp(color.r * 255.0, 0.0, 255.0));
            let g = u32(clamp(color.g * 255.0, 0.0, 255.0));
            let b = u32(clamp(color.b * 255.0, 0.0, 255.0));
            return (r << 24u) | (g << 16u) | (b << 8u);
        }

        fn get_helix_point(helix_idx: u32, progress: f32) -> vec3<f32> {
            let base_angle = (f32(helix_idx) / f32(uniforms.num_helices)) * 2.0 * 3.14159265;
            let angle = base_angle + uniforms.time * uniforms.rotation_speed;
            let z = progress * uniforms.length;
            let twist = (progress * 2.0 * 3.14159265 * uniforms.length) / uniforms.pitch;
            let total_angle = angle + twist;

            let x = uniforms.center_x + uniforms.radius * cos(total_angle);
            let y = uniforms.center_y + uniforms.radius * sin(total_angle);

            return vec3<f32>(x, y, z);
        }

        fn get_color(helix_idx: u32) -> vec3<f32> {
            if (helix_idx == 0u) {
                return vec3<f32>(uniforms.color0_r, uniforms.color0_g, uniforms.color0_b);
            } else if (helix_idx == 1u) {
                return vec3<f32>(uniforms.color1_r, uniforms.color1_g, uniforms.color1_b);
            } else {
                return vec3<f32>(uniforms.color2_r, uniforms.color2_g, uniforms.color2_b);
            }
        }

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let segment_id = global_id.x;
            let num_segments = 200u;

            if (segment_id >= num_segments) {
                return;
            }

            let progress = f32(segment_id) / f32(num_segments);

            // Draw all helices
            for (var helix_idx = 0u; helix_idx < uniforms.num_helices; helix_idx++) {
                let pos = get_helix_point(helix_idx, progress);
                let color = get_color(helix_idx) * 0.3;  // Dimmed for strand

                // Draw point
                let voxel = vec3<i32>(i32(round(pos.x)), i32(round(pos.y)), i32(round(pos.z)));
                let dims = textureDimensions(volume);

                if (voxel.x >= 0 && voxel.x < i32(dims.x) &&
                    voxel.y >= 0 && voxel.y < i32(dims.y) &&
                    voxel.z >= 0 && voxel.z < i32(dims.z)) {
                    let packed = pack_rgb(color);
                    textureStore(volume, voxel, vec4<u32>(packed, 0u, 0u, 0u));
                }
            }
        }
        """

        # Pulse rendering shader
        pulse_shader = """
        struct Pulse {
            progress: f32,
            helix_idx: f32,
            color_r: f32,
            color_g: f32,
            color_b: f32,
            radius: f32,
            padding1: f32,
            padding2: f32,
        }

        struct Uniforms {
            center_x: f32,
            center_y: f32,
            radius: f32,
            rotation_speed: f32,
            pitch: f32,
            time: f32,
            length: f32,
            num_pulses: u32,
        }

        @group(0) @binding(0) var volume: texture_storage_3d<r32uint, read_write>;
        @group(0) @binding(1) var<storage, read> pulses: array<Pulse>;
        @group(0) @binding(2) var<uniform> uniforms: Uniforms;

        fn pack_rgb(color: vec3<f32>) -> u32 {
            let r = u32(clamp(color.r * 255.0, 0.0, 255.0));
            let g = u32(clamp(color.g * 255.0, 0.0, 255.0));
            let b = u32(clamp(color.b * 255.0, 0.0, 255.0));
            return (r << 24u) | (g << 16u) | (b << 8u);
        }

        fn unpack_rgb(packed: u32) -> vec3<f32> {
            let r = f32((packed >> 24u) & 0xFFu) / 255.0;
            let g = f32((packed >> 16u) & 0xFFu) / 255.0;
            let b = f32((packed >> 8u) & 0xFFu) / 255.0;
            return vec3<f32>(r, g, b);
        }

        fn get_helix_point(helix_idx: u32, progress: f32) -> vec3<f32> {
            let base_angle = (f32(helix_idx) / 3.0) * 2.0 * 3.14159265;
            let angle = base_angle + uniforms.time * uniforms.rotation_speed;
            let z = progress * uniforms.length;
            let twist = (progress * 2.0 * 3.14159265 * uniforms.length) / uniforms.pitch;
            let total_angle = angle + twist;

            let x = uniforms.center_x + uniforms.radius * cos(total_angle);
            let y = uniforms.center_y + uniforms.radius * sin(total_angle);

            return vec3<f32>(x, y, z);
        }

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let pulse_id = global_id.x;
            if (pulse_id >= uniforms.num_pulses) {
                return;
            }

            let pulse = pulses[pulse_id];
            let helix_idx = u32(round(pulse.helix_idx));
            let center = get_helix_point(helix_idx, pulse.progress);
            let color = vec3<f32>(pulse.color_r, pulse.color_g, pulse.color_b);

            // Draw sphere
            let radius = pulse.radius;
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
                            let intensity = 1.0 - (dist / radius) * 0.3;
                            let voxel = vec3<i32>(x, y, z);

                            let current_packed = textureLoad(volume, voxel).r;
                            let current_color = unpack_rgb(current_packed);
                            let new_color = max(current_color, color * intensity);
                            let new_packed = pack_rgb(new_color);

                            textureStore(volume, voxel, vec4<u32>(new_packed, 0u, 0u, 0u));
                        }
                    }
                }
            }
        }
        """

        self.clear_shader_module = self.device.create_shader_module(code=clear_shader)
        self.helix_shader_module = self.device.create_shader_module(code=helix_shader)
        self.pulse_shader_module = self.device.create_shader_module(code=pulse_shader)

        self.clear_pipeline = self.device.create_compute_pipeline(
            layout="auto",
            compute={"module": self.clear_shader_module, "entry_point": "main"},
        )
        self.helix_pipeline = self.device.create_compute_pipeline(
            layout="auto",
            compute={"module": self.helix_shader_module, "entry_point": "main"},
        )
        self.pulse_pipeline = self.device.create_compute_pipeline(
            layout="auto",
            compute={"module": self.pulse_shader_module, "entry_point": "main"},
        )

        # Create clear bind group
        bind_group_layout = self.clear_pipeline.get_bind_group_layout(0)
        self.clear_bind_group = self.device.create_bind_group(
            layout=bind_group_layout,
            entries=[{"binding": 0, "resource": self.volume_texture.create_view()}],
        )

    def _spawn_pulse(self, time: float, helix_idx: int = None):
        """Spawn a pulse on a helix"""
        if helix_idx is None:
            helix_idx = self.next_helix_idx
            self.next_helix_idx = (self.next_helix_idx + 1) % self.num_helices

        color = self.helix_colors[helix_idx]
        pulse = HelixPulse(
            helix_idx=helix_idx,
            progress=0.0,
            speed=0.25,
            color=(color.red / 255.0, color.green / 255.0, color.blue / 255.0),
        )
        self.pulses.append(pulse)

    def _update_pulses(self, dt: float):
        """Update pulse positions"""
        updated_pulses = []
        for pulse in self.pulses:
            pulse.progress += pulse.speed * dt
            if pulse.progress < 1.0:
                updated_pulses.append(pulse)
        self.pulses = updated_pulses

    def render(self, raster: Raster, time: float):
        """Render using GPU"""
        dt = 1.0 / 60.0

        # Check for sync event
        if time >= self.next_sync:
            self.last_sync_time = time
            self.next_sync = time + self.sync_interval
            for i in range(self.num_helices):
                self._spawn_pulse(time, i)

        # Regular pulse spawning
        if time >= self.next_pulse_spawn:
            self._spawn_pulse(time)
            self.next_pulse_spawn = time + (1.0 / self.pulse_spawn_rate)

        self._update_pulses(dt)

        command_encoder = self.device.create_command_encoder()

        # Clear
        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.clear_pipeline)
        compute_pass.set_bind_group(0, self.clear_bind_group)
        workgroups_x = (self.width + 7) // 8
        workgroups_y = (self.height + 7) // 8
        workgroups_z = (self.length + 7) // 8
        compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z)
        compute_pass.end()

        # Render helices
        self._render_helices_gpu(command_encoder, time)

        # Render pulses
        if len(self.pulses) > 0:
            self._render_pulses_gpu(command_encoder, time)

        self.device.queue.submit([command_encoder.finish()])
        self._readback_to_cpu(raster)

    def _render_helices_gpu(self, command_encoder, time):
        """Render helix strands using GPU"""
        uniform_data = struct.pack(
            "<7fI9f",
            self.center_x,
            self.center_y,
            self.helix_radius,
            self.rotation_speed,
            self.pitch,
            time,
            self.length,
            self.num_helices,
            self.helix_colors[0].red / 255.0,
            self.helix_colors[0].green / 255.0,
            self.helix_colors[0].blue / 255.0,
            self.helix_colors[1].red / 255.0,
            self.helix_colors[1].green / 255.0,
            self.helix_colors[1].blue / 255.0,
            self.helix_colors[2].red / 255.0,
            self.helix_colors[2].green / 255.0,
            self.helix_colors[2].blue / 255.0,
        )

        uniform_buffer = self.device.create_buffer_with_data(
            data=uniform_data, usage=wgpu.BufferUsage.UNIFORM
        )

        bind_group_layout = self.helix_pipeline.get_bind_group_layout(0)
        bind_group = self.device.create_bind_group(
            layout=bind_group_layout,
            entries=[
                {"binding": 0, "resource": self.volume_texture.create_view()},
                {"binding": 1, "resource": {"buffer": uniform_buffer, "size": len(uniform_data)}},
            ],
        )

        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.helix_pipeline)
        compute_pass.set_bind_group(0, bind_group)
        workgroups = (200 + 63) // 64
        compute_pass.dispatch_workgroups(workgroups, 1, 1)
        compute_pass.end()

    def _render_pulses_gpu(self, command_encoder, time):
        """Render pulses using GPU"""
        pulse_data = np.zeros((len(self.pulses), 8), dtype=np.float32)
        for i, pulse in enumerate(self.pulses):
            pulse_data[i] = [
                pulse.progress,
                float(pulse.helix_idx),
                pulse.color[0],
                pulse.color[1],
                pulse.color[2],
                1.5,  # radius
                0.0,  # padding
                0.0,  # padding
            ]

        pulse_buffer = self.device.create_buffer_with_data(
            data=pulse_data, usage=wgpu.BufferUsage.STORAGE
        )

        uniform_data = np.array(
            [
                self.center_x,
                self.center_y,
                self.helix_radius,
                self.rotation_speed,
                self.pitch,
                time,
                self.length,
                len(self.pulses),
            ],
            dtype=np.float32,
        )

        uniform_buffer = self.device.create_buffer_with_data(
            data=uniform_data, usage=wgpu.BufferUsage.UNIFORM
        )

        bind_group_layout = self.pulse_pipeline.get_bind_group_layout(0)
        bind_group = self.device.create_bind_group(
            layout=bind_group_layout,
            entries=[
                {"binding": 0, "resource": self.volume_texture.create_view()},
                {"binding": 1, "resource": {"buffer": pulse_buffer, "size": pulse_data.nbytes}},
                {"binding": 2, "resource": {"buffer": uniform_buffer, "size": uniform_data.nbytes}},
            ],
        )

        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pulse_pipeline)
        compute_pass.set_bind_group(0, bind_group)
        workgroups = (len(self.pulses) + 63) // 64
        compute_pass.dispatch_workgroups(workgroups, 1, 1)
        compute_pass.end()

    def _readback_to_cpu(self, raster: Raster):
        """Read GPU texture back to CPU"""
        command_encoder = self.device.create_command_encoder()

        bytes_per_row = self.width * 4
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

        self.readback_buffer.map_sync(mode=wgpu.MapMode.READ)
        mapped_data = self.readback_buffer.read_mapped()
        self.readback_buffer.unmap()

        data_array = np.frombuffer(mapped_data, dtype=np.uint32)
        volume_data = np.zeros((self.length, self.height, self.width, 3), dtype=np.uint8)

        for z in range(self.length):
            for y in range(self.height):
                src_offset = (z * self.height + y) * (bytes_per_row_aligned // 4)
                src_row = data_array[src_offset : src_offset + self.width]
                volume_data[z, y, :, 0] = (src_row >> 24) & 0xFF
                volume_data[z, y, :, 1] = (src_row >> 16) & 0xFF
                volume_data[z, y, :, 2] = (src_row >> 8) & 0xFF

        raster.data[:, :, :, :3] = volume_data
