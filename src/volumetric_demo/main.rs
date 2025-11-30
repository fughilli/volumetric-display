use anyhow::{Context as _, Result};
use glam::{Mat4, Quat, Vec3};
use glfw::{Action, Context as GlfwContext, Key, MouseButton, WindowEvent};
use glow::*;
use std::time::Instant;
use volumetric_rendering::{SlicedRenderer, VolumeConfig};

fn main() -> Result<()> {
    // Initialize GLFW
    let mut glfw = glfw::init(glfw::fail_on_errors).context("Failed to initialize GLFW")?;

    glfw.window_hint(glfw::WindowHint::ContextVersion(3, 3));
    glfw.window_hint(glfw::WindowHint::OpenGlProfile(
        glfw::OpenGlProfileHint::Core,
    ));
    #[cfg(target_os = "macos")]
    glfw.window_hint(glfw::WindowHint::OpenGlForwardCompat(true));

    // Create window
    let (mut window, events) = glfw
        .create_window(
            1024,
            768,
            "Volumetric Rendering Demo",
            glfw::WindowMode::Windowed,
        )
        .ok_or_else(|| anyhow::anyhow!("Failed to create GLFW window"))?;

    window.make_current();
    window.set_key_polling(true);
    window.set_framebuffer_size_polling(true);
    window.set_cursor_pos_polling(true);
    window.set_scroll_polling(true);
    window.set_mouse_button_polling(true);

    // Initialize glow with GLFW
    let gl =
        unsafe { glow::Context::from_loader_function(|s| window.get_proc_address(s) as *const _) };

    // Create volumetric renderer
    // Calculate slice thickness from Z extent and resolution (matching TD formula)
    let depth = 21u32;
    let min_z = -1.0f32;
    let max_z = 1.0f32;
    let z_extent = (max_z - min_z).abs();
    let slice_count = depth as f32; // Number of slices per orientation
    let tile_tex_edge_count = (slice_count.sqrt()).ceil() as f32;
    let slice_thickness = z_extent / slice_count / tile_tex_edge_count * 1.25;

    let volume_config = VolumeConfig {
        resolution: (21, 21, depth),
        slice_thickness,
    };
    let mut renderer = SlicedRenderer::new(&gl, volume_config.clone())
        .context("Failed to create sliced renderer")?;

    // Create cube geometry
    let (cube_vertices, cube_indices) = create_cube_geometry();
    renderer
        .upload_geometry(&cube_vertices, &cube_indices)
        .context("Failed to upload geometry")?;
    renderer.set_index_count(cube_indices.len());

    // Create a simple texture for the cube
    let cube_texture =
        create_checkerboard_texture(&gl, 256, 256).context("Failed to create cube texture")?;

    // Create shader for converting tiled texture to 3D texture
    let tiled_to_3d_shader =
        create_tiled_to_3d_shader(&gl, &renderer).context("Failed to create tiled to 3D shader")?;

    // Create 3D texture
    let volume_3d_texture =
        create_3d_texture(&gl, &renderer).context("Failed to create 3D texture")?;

    // Create shader for displaying slices from 3D texture (right viewport)
    let slice_viewer_shader = create_3d_slice_viewer_shader(&gl, &renderer)
        .context("Failed to create 3D slice viewer shader")?;

    // Create shader for displaying tiled texture (left viewport)
    let tiled_viewer_shader = create_slice_viewer_shader(&gl, &renderer)
        .context("Failed to create tiled viewer shader")?;

    // Create quad VAO for displaying slices
    let (quad_vao, quad_vbo) =
        create_quad_geometry(&gl).context("Failed to create quad geometry")?;

    // Time tracking for rotation
    let start_time = Instant::now();

    // Main loop
    while !window.should_close() {
        glfw.poll_events();
        for (_, event) in glfw::flush_messages(&events) {
            match event {
                glfw::WindowEvent::Key(glfw::Key::Escape, _, glfw::Action::Press, _) => {
                    window.set_should_close(true);
                }
                glfw::WindowEvent::FramebufferSize(_width, _height) => {
                    // Viewport will be updated in render loop
                }
                _ => {}
            }
        }

        // Calculate rotation based on time
        let elapsed = start_time.elapsed().as_secs_f32();
        let rotation_angle = elapsed * 0.5; // Slow rotation
        let model = Mat4::from_axis_angle(Vec3::Y, rotation_angle)
            * Mat4::from_axis_angle(Vec3::X, rotation_angle * 0.7);

        // Render cube to tiled texture with FIXED view (for right viewport slices)
        // Fixed view for tiled texture rendering - camera translated 1 unit along +Z (looking down -Z)
        let fixed_view = Mat4::from_translation(Vec3::new(0.0, 0.0, -1.0));

        // Use orthographic projection (no perspective)
        // The vertex shader:
        // 1. Divides X/Y by 2.0 (line 69-70)
        // 2. Maps to tile: (instanceID % tileCountXY.x + worldSpacePos.x + 0.5) / tileCountXY.x (line 72-75)
        // 3. Transforms to [-1, 1]: worldSpacePos.x * 2.0 - 1.0 (line 77-78)
        // After mapping, each tile occupies [-1, 1] range in clip space
        // For 8 tiles, the range spans from approximately -1 to 1 (covering all tiles)
        // But we need to ensure the orthographic bounds cover the full extent
        let (tile_count_x, tile_count_y) = renderer.get_tile_count_xy();
        // Orthographic bounds should cover [-1, 1] range for all tiles
        // Since each tile maps to [-1, 1], and we have N tiles, we need to cover the full range
        let fixed_projection = Mat4::orthographic_rh(-1.0, 1.0, -1.0, 1.0, -100.0, 100.0);

        renderer
            .render_to_tiled_texture(model, fixed_view, fixed_projection, cube_texture, -1.0, 1.0)
            .context("Failed to render to tiled texture")?;

        // Convert tiled texture to 3D texture
        convert_tiled_to_3d_texture(
            &gl,
            &tiled_to_3d_shader,
            &renderer,
            volume_3d_texture,
            &quad_vao,
        )
        .context("Failed to convert tiled texture to 3D texture")?;

        // Display side-by-side: debug view (left) and 3D slices (right)
        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);

            // Get framebuffer size (accounts for retina/high DPI displays)
            let (fb_width, fb_height) = window.get_framebuffer_size();
            gl.viewport(0, 0, fb_width as i32, fb_height as i32);
            gl.clear_color(0.1, 0.1, 0.1, 1.0);
            gl.clear(glow::COLOR_BUFFER_BIT | glow::DEPTH_BUFFER_BIT);

            // Left side: Show tiled texture (post-vertex-shader geometry)
            let left_width = fb_width / 2;
            gl.viewport(0, 0, left_width as i32, fb_height as i32);
            gl.disable(glow::DEPTH_TEST);

            gl.use_program(Some(tiled_viewer_shader));
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(renderer.get_tiled_texture()));

            let tex_loc = gl
                .get_uniform_location(tiled_viewer_shader, "uTiledTexture")
                .ok_or_else(|| anyhow::anyhow!("Uniform 'uTiledTexture' not found"))?;
            gl.uniform_1_i32(Some(&tex_loc), 0);

            let tile_count_xy_loc = gl
                .get_uniform_location(tiled_viewer_shader, "uTileCountXY")
                .ok_or_else(|| anyhow::anyhow!("Uniform 'uTileCountXY' not found"))?;
            gl.uniform_2_i32(
                Some(&tile_count_xy_loc),
                renderer.get_tile_count_xy().0,
                renderer.get_tile_count_xy().1,
            );

            let slice_index_loc = gl
                .get_uniform_location(tiled_viewer_shader, "uSliceIndex")
                .ok_or_else(|| anyhow::anyhow!("Uniform 'uSliceIndex' not found"))?;
            let orientation_loc = gl
                .get_uniform_location(tiled_viewer_shader, "uOrientation")
                .ok_or_else(|| anyhow::anyhow!("Uniform 'uOrientation' not found"))?;
            let slice_count_loc = gl
                .get_uniform_location(tiled_viewer_shader, "uSliceCount")
                .ok_or_else(|| anyhow::anyhow!("Uniform 'uSliceCount' not found"))?;

            // Show all tiles (post-vertex-shader view)
            gl.uniform_1_i32(Some(&slice_index_loc), -1); // -1 means show all tiles
            gl.uniform_1_i32(Some(&orientation_loc), -1); // -1 means show all orientations
            gl.uniform_1_i32(Some(&slice_count_loc), renderer.get_slice_count());

            gl.bind_vertex_array(Some(quad_vao));
            gl.draw_arrays(glow::TRIANGLES, 0, 6);
            gl.bind_vertex_array(None);

            // Right side: Display slices from 3D texture arranged as tiles along Z
            let right_width = fb_width / 2;
            gl.viewport(left_width as i32, 0, right_width as i32, fb_height as i32);
            gl.disable(glow::DEPTH_TEST);

            gl.use_program(Some(slice_viewer_shader));
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_3D, Some(volume_3d_texture));

            let tex_loc = gl
                .get_uniform_location(slice_viewer_shader, "uVolume3D")
                .ok_or_else(|| anyhow::anyhow!("Uniform 'uVolume3D' not found"))?;
            gl.uniform_1_i32(Some(&tex_loc), 0);

            let volume_res_loc = gl
                .get_uniform_location(slice_viewer_shader, "uVolumeResolution")
                .ok_or_else(|| anyhow::anyhow!("Uniform 'uVolumeResolution' not found"))?;
            let (res_x, res_y, res_z) = renderer.get_volume_config().resolution;
            gl.uniform_3_i32(
                Some(&volume_res_loc),
                res_x as i32,
                res_y as i32,
                res_z as i32,
            );

            gl.bind_vertex_array(Some(quad_vao));
            gl.draw_arrays(glow::TRIANGLES, 0, 6);
            gl.bind_vertex_array(None);
        }

        window.swap_buffers();
    }

    Ok(())
}

fn create_cube_geometry() -> (Vec<f32>, Vec<u32>) {
    // Cube vertices: position(3) + normal(3) + texcoord(2) + color(3) = 11 floats per vertex
    let vertices = vec![
        // Front face
        -0.5, -0.5, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, // 0
        0.5, -0.5, 0.5, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, // 1
        0.5, 0.5, 0.5, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, // 2
        -0.5, 0.5, 0.5, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, // 3
        // Back face
        -0.5, -0.5, -0.5, 0.0, 0.0, -1.0, 1.0, 0.0, 1.0, 0.0, 1.0, // 4
        0.5, -0.5, -0.5, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, // 5
        0.5, 0.5, -0.5, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 1.0, // 6
        -0.5, 0.5, -0.5, 0.0, 0.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // 7
        // Left face
        -0.5, -0.5, -0.5, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, // 8
        -0.5, -0.5, 0.5, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, // 9
        -0.5, 0.5, 0.5, -1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, // 10
        -0.5, 0.5, -0.5, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, // 11
        // Right face
        0.5, -0.5, -0.5, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, // 12
        0.5, -0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, // 13
        0.5, 0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, // 14
        0.5, 0.5, -0.5, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, // 15
        // Bottom face
        -0.5, -0.5, -0.5, 0.0, -1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, // 16
        0.5, -0.5, -0.5, 0.0, -1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, // 17
        0.5, -0.5, 0.5, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, // 18
        -0.5, -0.5, 0.5, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, // 19
        // Top face
        -0.5, 0.5, -0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, // 20
        0.5, 0.5, -0.5, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, // 21
        0.5, 0.5, 0.5, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, // 22
        -0.5, 0.5, 0.5, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, // 23
    ];

    let indices = vec![
        // Front face
        0, 1, 2, 2, 3, 0, // Back face
        4, 5, 6, 6, 7, 4, // Left face
        8, 9, 10, 10, 11, 8, // Right face
        12, 13, 14, 14, 15, 12, // Bottom face
        16, 17, 18, 18, 19, 16, // Top face
        20, 21, 22, 22, 23, 20,
    ];

    (vertices, indices)
}

fn create_checkerboard_texture(gl: &Context, width: u32, height: u32) -> Result<NativeTexture> {
    unsafe {
        let texture = gl
            .create_texture()
            .map_err(|e| anyhow::anyhow!("Failed to create texture: {}", e))?;
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));

        let mut data = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            for x in 0..width {
                let checker = ((x / 32) + (y / 32)) % 2;
                let r = if checker == 0 { 255 } else { 128 };
                let g = if checker == 0 { 128 } else { 255 };
                let b = if checker == 0 { 64 } else { 192 };
                data.push(r);
                data.push(g);
                data.push(b);
                data.push(255);
            }
        }

        gl.tex_image_2d(
            glow::TEXTURE_2D,
            0,
            glow::RGBA as i32,
            width as i32,
            height as i32,
            0,
            glow::RGBA,
            glow::UNSIGNED_BYTE,
            Some(&data),
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
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::REPEAT as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::REPEAT as i32);

        gl.bind_texture(glow::TEXTURE_2D, None);

        Ok(texture)
    }
}

fn create_slice_viewer_shader(gl: &Context, renderer: &SlicedRenderer) -> Result<NativeProgram> {
    unsafe {
        let vertex_shader_source = r#"
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTexCoord;

out vec2 vTexCoord;

void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    vTexCoord = aTexCoord;
}
"#;

        let fragment_shader_source = r#"
#version 330 core
out vec4 fragColor;

in vec2 vTexCoord;
uniform sampler2D uTiledTexture;
uniform ivec2 uTileCountXY;
uniform int uSliceIndex;  // -1 = show all tiles, >= 0 = show specific slice
uniform int uOrientation; // -1 = all, 0 = X, 1 = Y, 2 = Z
uniform int uSliceCount;  // Number of slices per orientation

void main() {
    vec2 uv = vTexCoord;

    if (uSliceIndex >= 0 && uOrientation >= 0) {
        // Calculate tile index: tiles are arranged as [X slices][Y slices][Z slices]
        // Each orientation has uSliceCount tiles
        int tile_index = uSliceIndex + (uOrientation * uSliceCount);
        int tile_x = tile_index % uTileCountXY.x;
        int tile_y = tile_index / uTileCountXY.x;

        vec2 tile_size = vec2(1.0 / float(uTileCountXY.x), 1.0 / float(uTileCountXY.y));
        vec2 tile_offset = vec2(float(tile_x) / float(uTileCountXY.x), float(tile_y) / float(uTileCountXY.y));

        uv = tile_offset + uv * tile_size;
    }

    fragColor = texture(uTiledTexture, uv);
}
"#;

        // Compile vertex shader
        let vertex_shader = gl
            .create_shader(glow::VERTEX_SHADER)
            .map_err(|e| anyhow::anyhow!("Failed to create vertex shader: {}", e))?;
        gl.shader_source(vertex_shader, vertex_shader_source);
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
        gl.shader_source(fragment_shader, fragment_shader_source);
        gl.compile_shader(fragment_shader);

        if !gl.get_shader_compile_status(fragment_shader) {
            let info = gl.get_shader_info_log(fragment_shader);
            return Err(anyhow::anyhow!(
                "Fragment shader compilation failed: {}",
                info
            ));
        }

        // Link program
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

        gl.delete_shader(vertex_shader);
        gl.delete_shader(fragment_shader);

        Ok(shader_program)
    }
}

fn create_display_shader(gl: &Context) -> Result<NativeProgram> {
    unsafe {
        let vertex_shader_source = r#"
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTexCoord;

out vec2 vTexCoord;

void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    vTexCoord = aTexCoord;
}
"#;

        let fragment_shader_source = r#"
#version 330 core
out vec4 fragColor;

in vec2 vTexCoord;
uniform sampler2D uTexture;

void main() {
    fragColor = texture(uTexture, vTexCoord);
}
"#;

        // Compile vertex shader
        let vertex_shader = gl
            .create_shader(glow::VERTEX_SHADER)
            .map_err(|e| anyhow::anyhow!("Failed to create vertex shader: {}", e))?;
        gl.shader_source(vertex_shader, vertex_shader_source);
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
        gl.shader_source(fragment_shader, fragment_shader_source);
        gl.compile_shader(fragment_shader);

        if !gl.get_shader_compile_status(fragment_shader) {
            let info = gl.get_shader_info_log(fragment_shader);
            return Err(anyhow::anyhow!(
                "Fragment shader compilation failed: {}",
                info
            ));
        }

        // Link program
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

        gl.delete_shader(vertex_shader);
        gl.delete_shader(fragment_shader);

        Ok(shader_program)
    }
}

fn create_quad_geometry(gl: &Context) -> Result<(NativeVertexArray, NativeBuffer)> {
    unsafe {
        // Fullscreen quad: position(2) + texcoord(2) = 4 floats per vertex
        let vertices: [f32; 24] = [
            // Position    TexCoord
            -1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 0.0, 1.0,
        ];

        let vao = gl
            .create_vertex_array()
            .map_err(|e| anyhow::anyhow!("Failed to create vertex array: {}", e))?;
        let vbo = gl
            .create_buffer()
            .map_err(|e| anyhow::anyhow!("Failed to create buffer: {}", e))?;

        gl.bind_vertex_array(Some(vao));
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
        let u8_slice = std::slice::from_raw_parts(
            vertices.as_ptr() as *const u8,
            vertices.len() * std::mem::size_of::<f32>(),
        );
        gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, u8_slice, glow::STATIC_DRAW);

        // Position attribute
        gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, 4 * 4, 0);
        gl.enable_vertex_attrib_array(0);

        // TexCoord attribute
        gl.vertex_attrib_pointer_f32(1, 2, glow::FLOAT, false, 4 * 4, 2 * 4);
        gl.enable_vertex_attrib_array(1);

        gl.bind_vertex_array(None);

        Ok((vao, vbo))
    }
}

fn create_tiled_to_3d_shader(gl: &Context, renderer: &SlicedRenderer) -> Result<NativeProgram> {
    use volumetric_rendering::shaders::SAMPLE_TO_3D_TEX_FRAGMENT;

    unsafe {
        let vertex_shader_source = r#"
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTexCoord;

out vec2 vUV;

void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    vUV = aTexCoord;
}
"#;

        // Compile vertex shader
        let vertex_shader = gl
            .create_shader(glow::VERTEX_SHADER)
            .map_err(|e| anyhow::anyhow!("Failed to create vertex shader: {}", e))?;
        gl.shader_source(vertex_shader, vertex_shader_source);
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
        gl.shader_source(fragment_shader, SAMPLE_TO_3D_TEX_FRAGMENT);
        gl.compile_shader(fragment_shader);

        if !gl.get_shader_compile_status(fragment_shader) {
            let info = gl.get_shader_info_log(fragment_shader);
            return Err(anyhow::anyhow!(
                "Fragment shader compilation failed: {}",
                info
            ));
        }

        // Link program
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

        gl.delete_shader(vertex_shader);
        gl.delete_shader(fragment_shader);

        Ok(shader_program)
    }
}

fn create_3d_texture(gl: &Context, renderer: &SlicedRenderer) -> Result<NativeTexture> {
    unsafe {
        let texture = gl
            .create_texture()
            .map_err(|e| anyhow::anyhow!("Failed to create texture: {}", e))?;
        gl.bind_texture(glow::TEXTURE_3D, Some(texture));

        let (width, height, depth) = renderer.get_volume_config().resolution;

        gl.tex_image_3d(
            glow::TEXTURE_3D,
            0,
            glow::RGBA as i32,
            width as i32,
            height as i32,
            depth as i32,
            0,
            glow::RGBA,
            glow::UNSIGNED_BYTE,
            None,
        );

        gl.tex_parameter_i32(
            glow::TEXTURE_3D,
            glow::TEXTURE_MIN_FILTER,
            glow::NEAREST as i32,
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_3D,
            glow::TEXTURE_MAG_FILTER,
            glow::NEAREST as i32,
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_3D,
            glow::TEXTURE_WRAP_S,
            glow::CLAMP_TO_EDGE as i32,
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_3D,
            glow::TEXTURE_WRAP_T,
            glow::CLAMP_TO_EDGE as i32,
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_3D,
            glow::TEXTURE_WRAP_R,
            glow::CLAMP_TO_EDGE as i32,
        );

        gl.bind_texture(glow::TEXTURE_3D, None);

        Ok(texture)
    }
}

fn convert_tiled_to_3d_texture(
    gl: &Context,
    shader: &NativeProgram,
    renderer: &SlicedRenderer,
    volume_3d_texture: NativeTexture,
    quad_vao: &NativeVertexArray,
) -> Result<()> {
    unsafe {
        let (width, height, depth) = renderer.get_volume_config().resolution;

        // Create framebuffer for each Z slice
        let mut fbo = gl
            .create_framebuffer()
            .map_err(|e| anyhow::anyhow!("Failed to create framebuffer: {}", e))?;

        gl.use_program(Some(*shader));

        // Bind tiled texture
        gl.active_texture(glow::TEXTURE0);
        gl.bind_texture(glow::TEXTURE_2D, Some(renderer.get_tiled_texture()));

        let tiled_tex_loc = gl
            .get_uniform_location(*shader, "tiledTexture")
            .ok_or_else(|| anyhow::anyhow!("Uniform 'tiledTexture' not found"))?;
        gl.uniform_1_i32(Some(&tiled_tex_loc), 0);

        let tile_count_xy_loc = gl
            .get_uniform_location(*shader, "tileCountXY")
            .ok_or_else(|| anyhow::anyhow!("Uniform 'tileCountXY' not found"))?;
        let (tile_x, tile_y) = renderer.get_tile_count_xy();
        gl.uniform_2_i32(Some(&tile_count_xy_loc), tile_x, tile_y);

        let volume_res_loc = gl
            .get_uniform_location(*shader, "volumeResolution")
            .ok_or_else(|| anyhow::anyhow!("Uniform 'volumeResolution' not found"))?;
        gl.uniform_3_i32(
            Some(&volume_res_loc),
            width as i32,
            height as i32,
            depth as i32,
        );

        let z_slice_loc = gl
            .get_uniform_location(*shader, "uZSlice")
            .ok_or_else(|| anyhow::anyhow!("Uniform 'uZSlice' not found"))?;

        // Render each Z slice
        for z in 0..depth {
            // Set the Z slice uniform
            gl.uniform_1_i32(Some(&z_slice_loc), z as i32);

            // Bind 3D texture layer as 2D texture for rendering
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
            gl.framebuffer_texture_layer(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                Some(volume_3d_texture),
                0,
                z as i32,
            );

            // Check framebuffer completeness
            let status = gl.check_framebuffer_status(glow::FRAMEBUFFER);
            if status != glow::FRAMEBUFFER_COMPLETE {
                return Err(anyhow::anyhow!(
                    "Framebuffer is not complete for layer {}: {:?}",
                    z,
                    status
                ));
            }

            // Set viewport to match slice resolution
            gl.viewport(0, 0, width as i32, height as i32);
            gl.clear_color(0.0, 0.0, 0.0, 0.0);
            gl.clear(glow::COLOR_BUFFER_BIT);

            // Render fullscreen quad
            gl.bind_vertex_array(Some(*quad_vao));
            gl.draw_arrays(glow::TRIANGLES, 0, 6);
            gl.bind_vertex_array(None);
        }

        gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        gl.delete_framebuffer(fbo);

        Ok(())
    }
}

fn create_3d_slice_viewer_shader(gl: &Context, renderer: &SlicedRenderer) -> Result<NativeProgram> {
    unsafe {
        let vertex_shader_source = r#"
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTexCoord;

out vec2 vTexCoord;

void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    vTexCoord = aTexCoord;
}
"#;

        let fragment_shader_source = r#"
#version 330 core
out vec4 fragColor;

in vec2 vTexCoord;
uniform sampler3D uVolume3D;
uniform ivec3 uVolumeResolution;

void main() {
    // Sample from 3D texture and arrange slices as tiles along Z
    vec2 uv = vTexCoord;

    // Calculate which tile we're in (arrange slices in a grid)
    int depth = uVolumeResolution.z;
    int tiles_per_row = int(ceil(sqrt(float(depth))));
    int tile_x = int(uv.x * float(tiles_per_row));
    int tile_y = int(uv.y * float(tiles_per_row));
    int tile_index = tile_y * tiles_per_row + tile_x;

    if (tile_index >= depth) {
        fragColor = vec4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    // Calculate UV within the tile
    vec2 tile_uv = vec2(
        fract(uv.x * float(tiles_per_row)),
        fract(uv.y * float(tiles_per_row))
    );

    // Sample from 3D texture at the appropriate Z slice
    // Normalize Z coordinate to [0, 1] range
    float z_coord = (float(tile_index) + 0.5) / float(depth);
    vec3 sample_coord = vec3(tile_uv.x, tile_uv.y, z_coord);

    fragColor = texture(uVolume3D, sample_coord);

    // Debug: show red if sampling fails
    if (fragColor.a == 0.0 && tile_index < depth) {
        fragColor = vec4(1.0, 0.0, 0.0, 1.0); // Red for debugging
    }
}
"#;

        // Compile vertex shader
        let vertex_shader = gl
            .create_shader(glow::VERTEX_SHADER)
            .map_err(|e| anyhow::anyhow!("Failed to create vertex shader: {}", e))?;
        gl.shader_source(vertex_shader, vertex_shader_source);
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
        gl.shader_source(fragment_shader, fragment_shader_source);
        gl.compile_shader(fragment_shader);

        if !gl.get_shader_compile_status(fragment_shader) {
            let info = gl.get_shader_info_log(fragment_shader);
            return Err(anyhow::anyhow!(
                "Fragment shader compilation failed: {}",
                info
            ));
        }

        // Link program
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

        gl.delete_shader(vertex_shader);
        gl.delete_shader(fragment_shader);

        Ok(shader_program)
    }
}
