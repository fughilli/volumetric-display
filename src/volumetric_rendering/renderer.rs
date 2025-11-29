use anyhow::{Context as AnyhowContext, Result};
use glam::Mat4;
use glow::*;

use crate::shaders::*;

#[derive(Debug, Clone)]
pub struct VolumeConfig {
    pub resolution: (u32, u32, u32), // (width, height, depth)
    pub slice_thickness: f32,
}

impl Default for VolumeConfig {
    fn default() -> Self {
        Self {
            resolution: (64, 64, 64),
            slice_thickness: 0.01,
        }
    }
}

pub struct SlicedRenderer<'gl> {
    gl: &'gl Context,
    shader_program: NativeProgram,
    vao: NativeVertexArray,
    vbo: NativeBuffer,
    ebo: NativeBuffer,
    fbo: NativeFramebuffer,
    tiled_texture: NativeTexture,
    volume_config: VolumeConfig,
    tile_count_xy: (i32, i32),
    tile_count: i32,
    index_count: i32,
}

impl<'gl> SlicedRenderer<'gl> {
    pub fn new(gl: &'gl Context, volume_config: VolumeConfig) -> Result<Self> {
        unsafe {
            // Calculate tile layout
            let (width, _height, depth) = volume_config.resolution;
            let total_tiles = (depth * 3) as i32; // N slices × 3 orientations
            let tile_count_xy = Self::calculate_tile_layout(total_tiles);
            let tile_count = depth as i32;

            // Compile shaders
            let shader_program = Self::compile_shader_program(&gl)?;

            // Create FBO and texture for tiled output
            let (fbo, tiled_texture) = Self::create_framebuffer(&gl, tile_count_xy, width)?;

            // Create VAO/VBO/EBO (will be set up when geometry is provided)
            let (vao, vbo, ebo) = Self::create_geometry_buffers(&gl)?;

            Ok(Self {
                gl,
                shader_program,
                vao,
                vbo,
                ebo,
                fbo,
                tiled_texture,
                volume_config,
                tile_count_xy,
                tile_count,
                index_count: 0,
            })
        }
    }

    fn calculate_tile_layout(total_tiles: i32) -> (i32, i32) {
        // Calculate minimum square texture size to fit all tiles
        let tiles_per_dim = (total_tiles as f32).sqrt().ceil() as i32;
        (tiles_per_dim, tiles_per_dim)
    }

    unsafe fn compile_shader_program(gl: &Context) -> Result<NativeProgram> {
        // Compile vertex shader
        let vertex_shader = gl
            .create_shader(glow::VERTEX_SHADER)
            .map_err(|e| anyhow::anyhow!("Failed to create vertex shader: {}", e))?;
        gl.shader_source(vertex_shader, SLICED_RENDERING_VERTEX);
        gl.compile_shader(vertex_shader);

        if !gl.get_shader_compile_status(vertex_shader) {
            let info = gl.get_shader_info_log(vertex_shader);
            return Err(anyhow::anyhow!(
                "Vertex shader compilation failed: {}",
                info
            ));
        }

        // Compile fragment shader
        let fragment_shader = gl
            .create_shader(glow::FRAGMENT_SHADER)
            .map_err(|e| anyhow::anyhow!("Failed to create fragment shader: {}", e))?;
        gl.shader_source(fragment_shader, SLICED_RENDERING_FRAGMENT);
        gl.compile_shader(fragment_shader);

        if !gl.get_shader_compile_status(fragment_shader) {
            let info = gl.get_shader_info_log(fragment_shader);
            return Err(anyhow::anyhow!(
                "Fragment shader compilation failed: {}",
                info
            ));
        }

        // Link shader program
        let shader_program = gl
            .create_program()
            .map_err(|e| anyhow::anyhow!("Failed to create shader program: {}", e))?;
        gl.attach_shader(shader_program, vertex_shader);
        gl.attach_shader(shader_program, fragment_shader);
        gl.link_program(shader_program);

        if !gl.get_program_link_status(shader_program) {
            let info = gl.get_program_info_log(shader_program);
            return Err(anyhow::anyhow!("Shader program linking failed: {}", info));
        }

        // Clean up shaders
        gl.delete_shader(vertex_shader);
        gl.delete_shader(fragment_shader);

        Ok(shader_program)
    }

    unsafe fn create_framebuffer(
        gl: &Context,
        tile_count_xy: (i32, i32),
        tile_size: u32,
    ) -> Result<(NativeFramebuffer, NativeTexture)> {
        // Create texture for tiled output
        let texture = gl
            .create_texture()
            .map_err(|e| anyhow::anyhow!("Failed to create texture: {}", e))?;
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));

        let texture_width = (tile_count_xy.0 * tile_size as i32) as i32;
        let texture_height = (tile_count_xy.1 * tile_size as i32) as i32;

        gl.tex_image_2d(
            glow::TEXTURE_2D,
            0,
            glow::RGBA as i32,
            texture_width,
            texture_height,
            0,
            glow::RGBA,
            glow::UNSIGNED_BYTE,
            None,
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_MIN_FILTER,
            glow::LINEAR as i32,
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_MAG_FILTER,
            glow::LINEAR as i32,
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_WRAP_S,
            glow::CLAMP_TO_EDGE as i32,
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_WRAP_T,
            glow::CLAMP_TO_EDGE as i32,
        );

        // Create framebuffer
        let fbo = gl
            .create_framebuffer()
            .map_err(|e| anyhow::anyhow!("Failed to create framebuffer: {}", e))?;
        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
        gl.framebuffer_texture_2d(
            glow::FRAMEBUFFER,
            glow::COLOR_ATTACHMENT0,
            glow::TEXTURE_2D,
            Some(texture),
            0,
        );

        // Check framebuffer completeness
        let status = gl.check_framebuffer_status(glow::FRAMEBUFFER);
        if status != glow::FRAMEBUFFER_COMPLETE {
            return Err(anyhow::anyhow!("Framebuffer is not complete: {:?}", status));
        }

        gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        gl.bind_texture(glow::TEXTURE_2D, None);

        Ok((fbo, texture))
    }

    unsafe fn create_geometry_buffers(
        gl: &Context,
    ) -> Result<(NativeVertexArray, NativeBuffer, NativeBuffer)> {
        let vao = gl
            .create_vertex_array()
            .map_err(|e| anyhow::anyhow!("Failed to create vertex array: {}", e))?;
        let vbo = gl
            .create_buffer()
            .map_err(|e| anyhow::anyhow!("Failed to create buffer: {}", e))?;
        let ebo = gl
            .create_buffer()
            .map_err(|e| anyhow::anyhow!("Failed to create buffer: {}", e))?;

        // VAO will be configured when geometry is uploaded
        Ok((vao, vbo, ebo))
    }

    pub fn upload_geometry(&mut self, vertices: &[f32], indices: &[u32]) -> Result<()> {
        unsafe {
            self.gl.bind_vertex_array(Some(self.vao));

            // Upload vertex data
            self.gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.vbo));
            let u8_slice = std::slice::from_raw_parts(
                vertices.as_ptr() as *const u8,
                vertices.len() * std::mem::size_of::<f32>(),
            );
            self.gl
                .buffer_data_u8_slice(glow::ARRAY_BUFFER, u8_slice, glow::STATIC_DRAW);

            // Upload index data
            self.gl
                .bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(self.ebo));
            let u8_slice = std::slice::from_raw_parts(
                indices.as_ptr() as *const u8,
                indices.len() * std::mem::size_of::<u32>(),
            );
            self.gl
                .buffer_data_u8_slice(glow::ELEMENT_ARRAY_BUFFER, u8_slice, glow::STATIC_DRAW);

            // Configure vertex attributes
            // Position (location 0): vec3
            self.gl
                .vertex_attrib_pointer_f32(0, 3, glow::FLOAT, false, 11 * 4, 0);
            self.gl.enable_vertex_attrib_array(0);

            // Normal (location 1): vec3
            self.gl
                .vertex_attrib_pointer_f32(1, 3, glow::FLOAT, false, 11 * 4, 3 * 4);
            self.gl.enable_vertex_attrib_array(1);

            // TexCoord (location 2): vec2
            self.gl
                .vertex_attrib_pointer_f32(2, 2, glow::FLOAT, false, 11 * 4, 6 * 4);
            self.gl.enable_vertex_attrib_array(2);

            // Color (location 3): vec3
            self.gl
                .vertex_attrib_pointer_f32(3, 3, glow::FLOAT, false, 11 * 4, 8 * 4);
            self.gl.enable_vertex_attrib_array(3);

            self.gl.bind_vertex_array(None);

            Ok(())
        }
    }

    pub fn render_to_tiled_texture(
        &self,
        model: Mat4,
        view: Mat4,
        projection: Mat4,
        input_texture: NativeTexture,
        min_z: f32,
        max_z: f32,
    ) -> Result<()> {
        unsafe {
            let (width, height, depth) = self.volume_config.resolution;
            let texture_width = (self.tile_count_xy.0 * width as i32) as i32;
            let texture_height = (self.tile_count_xy.1 * height as i32) as i32;

            // Bind framebuffer
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.fbo));
            self.gl.viewport(0, 0, texture_width, texture_height);
            self.gl.clear_color(0.0, 0.0, 0.0, 0.0);
            self.gl.clear(glow::COLOR_BUFFER_BIT);

            // Use shader program
            self.gl.use_program(Some(self.shader_program));

            // Set uniforms
            let tile_count_xy_loc = self
                .gl
                .get_uniform_location(self.shader_program, "tileCountXY")
                .ok_or_else(|| anyhow::anyhow!("Uniform 'tileCountXY' not found"))?;
            self.gl.uniform_2_i32(
                Some(&tile_count_xy_loc),
                self.tile_count_xy.0,
                self.tile_count_xy.1,
            );

            let tile_count_loc = self
                .gl
                .get_uniform_location(self.shader_program, "tileCount")
                .ok_or_else(|| anyhow::anyhow!("Uniform 'tileCount' not found"))?;
            self.gl
                .uniform_1_i32(Some(&tile_count_loc), self.tile_count);

            let min_z_loc = self
                .gl
                .get_uniform_location(self.shader_program, "minZ")
                .ok_or_else(|| anyhow::anyhow!("Uniform 'minZ' not found"))?;
            self.gl.uniform_1_f32(Some(&min_z_loc), min_z);

            let max_z_loc = self
                .gl
                .get_uniform_location(self.shader_program, "maxZ")
                .ok_or_else(|| anyhow::anyhow!("Uniform 'maxZ' not found"))?;
            self.gl.uniform_1_f32(Some(&max_z_loc), max_z);

            let slice_thickness_loc = self
                .gl
                .get_uniform_location(self.shader_program, "uSliceThickness")
                .ok_or_else(|| anyhow::anyhow!("Uniform 'uSliceThickness' not found"))?;
            self.gl.uniform_1_f32(
                Some(&slice_thickness_loc),
                self.volume_config.slice_thickness,
            );

            let model_loc = self
                .gl
                .get_uniform_location(self.shader_program, "model")
                .ok_or_else(|| anyhow::anyhow!("Uniform 'model' not found"))?;
            self.gl
                .uniform_matrix_4_f32_slice(Some(&model_loc), false, &model.to_cols_array());

            let view_loc = self
                .gl
                .get_uniform_location(self.shader_program, "view")
                .ok_or_else(|| anyhow::anyhow!("Uniform 'view' not found"))?;
            self.gl
                .uniform_matrix_4_f32_slice(Some(&view_loc), false, &view.to_cols_array());

            let projection_loc = self
                .gl
                .get_uniform_location(self.shader_program, "projection")
                .ok_or_else(|| anyhow::anyhow!("Uniform 'projection' not found"))?;
            self.gl.uniform_matrix_4_f32_slice(
                Some(&projection_loc),
                false,
                &projection.to_cols_array(),
            );

            // Bind input texture
            self.gl.active_texture(glow::TEXTURE0);
            self.gl.bind_texture(glow::TEXTURE_2D, Some(input_texture));
            let in_tex_loc = self
                .gl
                .get_uniform_location(self.shader_program, "inTex")
                .ok_or_else(|| anyhow::anyhow!("Uniform 'inTex' not found"))?;
            self.gl.uniform_1_i32(Some(&in_tex_loc), 0);

            // Render instanced
            self.gl.bind_vertex_array(Some(self.vao));
            let instance_count = (depth * 3) as i32; // N slices × 3 orientations
            self.gl.draw_elements_instanced(
                glow::TRIANGLES,
                self.index_count,
                glow::UNSIGNED_INT,
                0,
                instance_count,
            );
            self.gl.bind_vertex_array(None);

            self.gl.bind_framebuffer(glow::FRAMEBUFFER, None);

            Ok(())
        }
    }

    pub fn get_tiled_texture(&self) -> NativeTexture {
        self.tiled_texture
    }

    pub fn get_tile_count_xy(&self) -> (i32, i32) {
        self.tile_count_xy
    }

    pub fn get_volume_config(&self) -> &VolumeConfig {
        &self.volume_config
    }

    pub fn set_index_count(&mut self, count: usize) {
        self.index_count = count as i32;
    }
}

impl<'gl> Drop for SlicedRenderer<'gl> {
    fn drop(&mut self) {
        unsafe {
            self.gl.delete_framebuffer(self.fbo);
            self.gl.delete_texture(self.tiled_texture);
            self.gl.delete_buffer(self.ebo);
            self.gl.delete_buffer(self.vbo);
            self.gl.delete_vertex_array(self.vao);
            self.gl.delete_program(self.shader_program);
        }
    }
}
