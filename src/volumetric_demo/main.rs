use anyhow::{Context, Result};
use glam::{Mat4, Vec3};
use glfw::{Action, Context, Key, WindowEvent};
use glow::*;
use std::sync::mpsc::Receiver;
use volumetric_rendering::{SlicedRenderer, VolumeConfig};

fn main() -> Result<()> {
    // Initialize GLFW
    let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).context("Failed to initialize GLFW")?;

    glfw.window_hint(glfw::WindowHint::ContextVersion(3, 3));
    glfw.window_hint(glfw::WindowHint::OpenGlProfile(glfw::OpenGlProfile::Core));
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

    // Initialize glow with GLFW
    let gl = unsafe {
        glow::Context::from_loader_function(|s| {
            let s = std::ffi::CString::new(s).unwrap();
            window.get_proc_address(s.as_c_str()) as *const _
        })
    };

    // Create volumetric renderer
    let volume_config = VolumeConfig {
        resolution: (64, 64, 64),
        slice_thickness: 0.02,
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

    // Create a simple shader for displaying the tiled texture
    let display_shader = create_display_shader(&gl).context("Failed to create display shader")?;

    // Create quad VAO for displaying the texture
    let (quad_vao, quad_vbo) =
        create_quad_geometry(&gl).context("Failed to create quad geometry")?;

    // Main loop
    while !window.should_close() {
        glfw.poll_events();
        for (_, event) in glfw::flush_messages(&events) {
            handle_window_event(&mut window, event);
        }

        // Render cube to tiled texture
        let model = Mat4::IDENTITY;
        let view = Mat4::look_at_rh(Vec3::new(2.0, 2.0, 2.0), Vec3::ZERO, Vec3::Y);
        let projection = Mat4::perspective_rh(45.0f32.to_radians(), 1024.0 / 768.0, 0.1, 100.0);

        renderer
            .render_to_tiled_texture(model, view, projection, cube_texture, -1.0, 1.0)
            .context("Failed to render to tiled texture")?;

        // Display the tiled texture
        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.viewport(0, 0, 1024, 768);
            gl.clear_color(0.1, 0.1, 0.1, 1.0);
            gl.clear(glow::COLOR_BUFFER_BIT);

            gl.use_program(Some(display_shader));
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(renderer.get_tiled_texture()));
            let tex_loc = gl
                .get_uniform_location(display_shader, "uTexture")
                .ok_or_else(|| anyhow::anyhow!("Uniform 'uTexture' not found"))?;
            gl.uniform_1_i32(Some(&tex_loc), 0);

            gl.bind_vertex_array(Some(quad_vao));
            gl.draw_arrays(glow::TRIANGLES, 0, 6);
            gl.bind_vertex_array(None);
        }

        window.swap_buffers();
    }

    Ok(())
}

fn handle_window_event(window: &mut glfw::Window, event: WindowEvent) {
    match event {
        glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) => {
            window.set_should_close(true);
        }
        glfw::WindowEvent::FramebufferSize(width, height) => {
            unsafe {
                // Note: We'd need access to gl here, but for simplicity we'll handle this in the main loop
            }
        }
        _ => {}
    }
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
        let mut vao = 0;
        let mut vbo = 0;

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
