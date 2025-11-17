import math
import random
from dataclasses import dataclass

import numpy as np
import wgpu

from artnet import Raster, Scene
from color_palette import get_palette


@dataclass
class HotSpot:
    """A hot spot on a layer"""

    y: float
    z: float
    birth_time: float
    lifetime: float


class DataCenterLayersSceneGPU(Scene):
    """
    GPU-accelerated horizontal layers representing server racks.
    Each layer pulses independently with different activity patterns.
    """

    def __init__(self, **kwargs):
        properties = kwargs.get("properties")
        if not properties:
            raise ValueError("DataCenterLayersSceneGPU requires a 'properties' object.")

        self.width = properties.width
        self.height = properties.height
        self.length = properties.length

        # Layer configuration
        self.num_layers = 8
        self.layer_thickness = 1.5
        self.layer_spacing = self.width / (self.num_layers + 1)

        # Get color palette
        palette = get_palette()

        # Initialize layers with sequential colors
        self.layers = []
        activity_freqs = [2.0, 1.5, 3.0]
        for i in range(self.num_layers):
            color = palette.get_color(i)
            self.layers.append(
                {
                    "x_position": (i + 1) * self.layer_spacing,
                    "color": color,
                    "base_freq": activity_freqs[i % len(activity_freqs)],
                    "phase_offset": random.uniform(0, 2 * math.pi),
                    "hot_spots": [],
                }
            )

        # Hot spot spawning
        self.next_hot_spot_spawn = 0.0
        self.hot_spot_interval = 0.5

        # Initialize GPU
        self._init_gpu()

        print(f"DataCenterLayersSceneGPU initialized with {self.num_layers} layers")

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

        # Layer rendering shader
        layer_shader = """
        struct Layer {
            x_position: f32,
            thickness: f32,
            base_intensity: f32,
            color_r: f32,
            color_g: f32,
            color_b: f32,
            padding1: f32,
            padding2: f32,
        }

        struct HotSpot {
            y: f32,
            z: f32,
            age_factor: f32,
            padding: f32,
        }

        struct Uniforms {
            width: f32,
            height: f32,
            length: f32,
            num_layers: u32,
            num_hot_spots: u32,
        }

        @group(0) @binding(0) var volume: texture_storage_3d<r32uint, read_write>;
        @group(0) @binding(1) var<storage, read> layers: array<Layer>;
        @group(0) @binding(2) var<storage, read> hot_spots: array<HotSpot>;
        @group(0) @binding(3) var<uniform> uniforms: Uniforms;

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

            let x = f32(global_id.x);
            let y = f32(global_id.y);
            let z = f32(global_id.z);

            var final_color = vec3<f32>(0.0, 0.0, 0.0);

            // Process all layers
            for (var layer_idx = 0u; layer_idx < uniforms.num_layers; layer_idx++) {
                let layer = layers[layer_idx];
                let dist = abs(x - layer.x_position);

                if (dist < layer.thickness) {
                    let falloff = 1.0 - (dist / layer.thickness);
                    var intensity = layer.base_intensity * falloff;

                    // Check hot spots for this layer
                    for (var hs_idx = 0u; hs_idx < uniforms.num_hot_spots; hs_idx++) {
                        let hs = hot_spots[hs_idx];
                        let dy = y - hs.y;
                        let dz = z - hs.z;
                        let dist_hs = sqrt(dy * dy + dz * dz);
                        let hot_spot_radius = 8.0;

                        if (dist_hs < hot_spot_radius) {
                            let spatial_factor = 1.0 - (dist_hs / hot_spot_radius);
                            let boost = hs.age_factor * spatial_factor;
                            intensity = min(1.0, intensity + boost);
                        }
                    }

                    if (intensity > 0.01) {
                        let layer_color = vec3<f32>(layer.color_r, layer.color_g, layer.color_b) * intensity;
                        final_color = max(final_color, layer_color);
                    }
                }
            }

            if (final_color.r > 0.0 || final_color.g > 0.0 || final_color.b > 0.0) {
                let packed = pack_rgb(final_color);
                textureStore(volume, vec3<i32>(global_id), vec4<u32>(packed, 0u, 0u, 0u));
            }
        }
        """

        self.clear_shader_module = self.device.create_shader_module(code=clear_shader)
        self.layer_shader_module = self.device.create_shader_module(code=layer_shader)

        self.clear_pipeline = self.device.create_compute_pipeline(
            layout="auto",
            compute={"module": self.clear_shader_module, "entry_point": "main"},
        )
        self.layer_pipeline = self.device.create_compute_pipeline(
            layout="auto",
            compute={"module": self.layer_shader_module, "entry_point": "main"},
        )

        # Create clear bind group
        bind_group_layout = self.clear_pipeline.get_bind_group_layout(0)
        self.clear_bind_group = self.device.create_bind_group(
            layout=bind_group_layout,
            entries=[{"binding": 0, "resource": self.volume_texture.create_view()}],
        )

    def _update_hot_spots(self, time: float):
        """Update hot spots"""
        if time >= self.next_hot_spot_spawn:
            layer = random.choice(self.layers)
            hot_spot = HotSpot(
                y=random.uniform(0, self.height),
                z=random.uniform(0, self.length),
                birth_time=time,
                lifetime=random.uniform(1.0, 2.0),
            )
            layer["hot_spots"].append(hot_spot)
            self.next_hot_spot_spawn = time + self.hot_spot_interval

        # Remove expired hot spots
        for layer in self.layers:
            layer["hot_spots"] = [
                hs for hs in layer["hot_spots"] if time - hs.birth_time < hs.lifetime
            ]

    def render(self, raster: Raster, time: float):
        """Render using GPU"""
        self._update_hot_spots(time)

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

        # Render layers
        self._render_layers_gpu(command_encoder, time)

        self.device.queue.submit([command_encoder.finish()])
        self._readback_to_cpu(raster)

    def _render_layers_gpu(self, command_encoder, time):
        """Render layers using GPU"""
        # Prepare layer data
        layer_data = np.zeros((self.num_layers, 8), dtype=np.float32)
        for i, layer in enumerate(self.layers):
            pulse = 0.5 + 0.5 * math.sin(layer["base_freq"] * time + layer["phase_offset"])
            base_intensity = (100 + 155 * pulse) / 255.0

            color = layer["color"]
            layer_data[i] = [
                layer["x_position"],
                self.layer_thickness,
                base_intensity,
                color.red / 255.0,
                color.green / 255.0,
                color.blue / 255.0,
                0.0,  # padding
                0.0,  # padding
            ]

        # Collect all hot spots with age factors
        all_hot_spots = []
        for layer in self.layers:
            for hs in layer["hot_spots"]:
                age = time - hs.birth_time
                age_factor = math.sin(math.pi * age / hs.lifetime) * 0.6  # normalized to 0-0.6
                all_hot_spots.append([hs.y, hs.z, age_factor, 0.0])

        if len(all_hot_spots) == 0:
            all_hot_spots = [[0.0, 0.0, 0.0, 0.0]]  # Dummy

        hot_spot_data = np.array(all_hot_spots, dtype=np.float32)

        # Create buffers
        layer_buffer = self.device.create_buffer_with_data(
            data=layer_data, usage=wgpu.BufferUsage.STORAGE
        )
        hot_spot_buffer = self.device.create_buffer_with_data(
            data=hot_spot_data, usage=wgpu.BufferUsage.STORAGE
        )

        uniform_data = np.array(
            [self.width, self.height, self.length, self.num_layers, len(all_hot_spots)],
            dtype=np.float32,
        )
        uniform_buffer = self.device.create_buffer_with_data(
            data=uniform_data, usage=wgpu.BufferUsage.UNIFORM
        )

        # Create bind group
        bind_group_layout = self.layer_pipeline.get_bind_group_layout(0)
        bind_group = self.device.create_bind_group(
            layout=bind_group_layout,
            entries=[
                {"binding": 0, "resource": self.volume_texture.create_view()},
                {"binding": 1, "resource": {"buffer": layer_buffer, "size": layer_data.nbytes}},
                {
                    "binding": 2,
                    "resource": {"buffer": hot_spot_buffer, "size": hot_spot_data.nbytes},
                },
                {"binding": 3, "resource": {"buffer": uniform_buffer, "size": uniform_data.nbytes}},
            ],
        )

        # Dispatch
        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.layer_pipeline)
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
