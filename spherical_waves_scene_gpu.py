import random
from dataclasses import dataclass
from typing import List

import numpy as np
import wgpu

from artnet import Raster, Scene
from color_palette import get_palette


@dataclass
class SphericalWave:
    """A spherical wave expanding from a source"""

    x: float
    y: float
    z: float
    radius: float
    max_radius: float
    speed: float
    birth_time: float
    response_type: str
    color: tuple  # RGB 0-1


class SphericalWavesSceneGPU(Scene):
    """
    GPU-accelerated concentric spherical waves.
    Represents API request/response patterns.
    """

    def __init__(self, **kwargs):
        properties = kwargs.get("properties")
        if not properties:
            raise ValueError("SphericalWavesSceneGPU requires a 'properties' object.")

        self.width = properties.width
        self.height = properties.height
        self.length = properties.length

        # Palette + color cycling
        self.palette = get_palette()
        self.wave_color_index = 0

        # Wave configuration
        self.waves: List[SphericalWave] = []
        self.wave_spawn_interval = 0.9
        self.next_wave_spawn = 0.0
        self.next_source_index = 0

        # Allow explicit source positions to be provided
        source_positions = kwargs.get("source_positions")
        if source_positions is None and hasattr(properties, "spherical_wave_sources"):
            source_positions = getattr(properties, "spherical_wave_sources")

        if source_positions:
            self.source_positions = [
                (
                    float(max(0.0, min(self.width, pos[0]))),
                    float(max(0.0, min(self.height, pos[1]))),
                    float(max(0.0, min(self.length, pos[2]))),
                )
                for pos in source_positions
                if isinstance(pos, (list, tuple)) and len(pos) == 3
            ]
        else:
            self.source_positions = [
                (20.0, 28.0, 10.0),
                (20.0, 28.0, 26.0),
                (20.0, 28.0, 42.0),
                (20.0, 28.0, 58.0),
                (20.0, 28.0, 74.0),
            ]

        # Response probabilities (color chosen per wave)
        self.response_probabilities = {
            "success": 0.7,
            "error": 0.2,
            "timeout": 0.1,
        }

        # Wave properties
        self.wave_speed = 10.0
        self.wave_thickness = 2.0
        self.max_wave_radius = 25.0

        # Initialize GPU
        self._init_gpu()

        print("SphericalWavesSceneGPU initialized")

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

        # Wave rendering shader - renders ONE wave at a time
        wave_shader = """
        struct WaveParams {
            center_x: f32,
            center_y: f32,
            center_z: f32,
            radius: f32,
            thickness: f32,
            color_r: f32,
            color_g: f32,
            color_b: f32,
        }

        @group(0) @binding(0) var volume: texture_storage_3d<r32uint, read_write>;
        @group(0) @binding(1) var<uniform> wave: WaveParams;

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

        @compute @workgroup_size(8, 8, 8)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let dims = textureDimensions(volume);
            if (global_id.x >= dims.x || global_id.y >= dims.y || global_id.z >= dims.z) {
                return;
            }

            let center = vec3<f32>(wave.center_x, wave.center_y, wave.center_z);
            let color = vec3<f32>(wave.color_r, wave.color_g, wave.color_b);
            let pos = vec3<f32>(f32(global_id.x), f32(global_id.y), f32(global_id.z));

            let dist = distance(pos, center);

            // Check if on shell surface
            if (abs(dist - wave.radius) <= wave.thickness) {
                let shell_dist = abs(dist - wave.radius);
                let intensity = 1.0 - (shell_dist / wave.thickness);
                let voxel = vec3<i32>(global_id);

                // Additive blending for interference patterns
                let current_packed = textureLoad(volume, voxel).r;
                let current_color = unpack_rgb(current_packed);
                let new_color = min(vec3<f32>(1.0), current_color + color * intensity);
                let new_packed = pack_rgb(new_color);

                textureStore(volume, voxel, vec4<u32>(new_packed, 0u, 0u, 0u));
            }
        }
        """

        self.clear_shader_module = self.device.create_shader_module(code=clear_shader)
        self.wave_shader_module = self.device.create_shader_module(code=wave_shader)

        self.clear_pipeline = self.device.create_compute_pipeline(
            layout="auto",
            compute={"module": self.clear_shader_module, "entry_point": "main"},
        )
        self.wave_pipeline = self.device.create_compute_pipeline(
            layout="auto",
            compute={"module": self.wave_shader_module, "entry_point": "main"},
        )

        # Create clear bind group
        bind_group_layout = self.clear_pipeline.get_bind_group_layout(0)
        self.clear_bind_group = self.device.create_bind_group(
            layout=bind_group_layout,
            entries=[{"binding": 0, "resource": self.volume_texture.create_view()}],
        )

    def _spawn_wave(self, time: float):
        """Spawn a new spherical wave"""
        # Mostly cycle through sources, with occasional random pick for variety
        if random.random() < 0.7:
            source_x, source_y, source_z = self.source_positions[self.next_source_index]
            self.next_source_index = (self.next_source_index + 1) % len(self.source_positions)
        else:
            source_x, source_y, source_z = random.choice(self.source_positions)

        # Choose response type (probabilities only)
        rand = random.random()
        cumulative = 0.0
        response_type = "success"
        for rtype, probability in self.response_probabilities.items():
            cumulative += probability
            if rand <= cumulative:
                response_type = rtype
                break

        # Cycle through palette colors for each wave
        color = self.palette.get_color(self.wave_color_index)
        self.wave_color_index = (self.wave_color_index + 1) % len(self.palette.colors)
        wave = SphericalWave(
            x=source_x,
            y=source_y,
            z=source_z,
            radius=0.0,
            max_radius=self.max_wave_radius,
            speed=self.wave_speed,
            birth_time=time,
            response_type=response_type,
            color=(color.red / 255.0, color.green / 255.0, color.blue / 255.0),
        )
        self.waves.append(wave)

    def _update_waves(self, dt: float):
        """Update wave radii"""
        updated_waves = []
        for wave in self.waves:
            wave.radius += wave.speed * dt
            if wave.radius < wave.max_radius:
                updated_waves.append(wave)
        self.waves = updated_waves

    def render(self, raster: Raster, time: float):
        """Render using GPU"""
        dt = 1.0 / 60.0

        # Spawn new waves
        if time >= self.next_wave_spawn:
            self._spawn_wave(time)
            # Add small jitter to avoid strict cadence
            jitter = random.uniform(-0.2, 0.2)
            self.next_wave_spawn = time + max(0.4, self.wave_spawn_interval + jitter)

        self._update_waves(dt)

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

        # Render waves
        if len(self.waves) > 0:
            self._render_waves_gpu(command_encoder, time)

        self.device.queue.submit([command_encoder.finish()])
        self._readback_to_cpu(raster)

    def _render_waves_gpu(self, command_encoder, time):
        """Render waves using GPU - one wave per pass to avoid race conditions"""
        bind_group_layout = self.wave_pipeline.get_bind_group_layout(0)

        # Render each wave in a separate pass for proper compositing
        for wave in self.waves:
            # Fade as wave expands
            age_factor = 1.0 - (wave.radius / wave.max_radius)

            # Create uniform buffer for this wave
            wave_data = np.array(
                [
                    wave.x,
                    wave.y,
                    wave.z,
                    wave.radius,
                    self.wave_thickness,
                    wave.color[0] * age_factor,
                    wave.color[1] * age_factor,
                    wave.color[2] * age_factor,
                ],
                dtype=np.float32,
            )

            wave_buffer = self.device.create_buffer_with_data(
                data=wave_data, usage=wgpu.BufferUsage.UNIFORM
            )

            bind_group = self.device.create_bind_group(
                layout=bind_group_layout,
                entries=[
                    {"binding": 0, "resource": self.volume_texture.create_view()},
                    {"binding": 1, "resource": {"buffer": wave_buffer, "size": wave_data.nbytes}},
                ],
            )

            # Dispatch compute pass for this wave
            compute_pass = command_encoder.begin_compute_pass()
            compute_pass.set_pipeline(self.wave_pipeline)
            compute_pass.set_bind_group(0, bind_group)
            workgroups_x = (self.width + 7) // 8
            workgroups_y = (self.height + 7) // 8
            workgroups_z = (self.length + 7) // 8
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z)
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
