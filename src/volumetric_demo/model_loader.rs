use anyhow::{Context as _, Result};
use glam::Vec3;
use std::path::Path;
use tobj::load_obj;

pub struct LoadedModel {
    pub vertices: Vec<f32>,
    pub indices: Vec<u32>,
    pub texture_path: Option<String>,
}

pub fn load_obj_model(obj_path: &Path, mtl_dir: Option<&Path>) -> Result<LoadedModel> {
    let load_options = tobj::LoadOptions {
        triangulate: true,
        single_index: true,
        ..Default::default()
    };

    let (models, materials) =
        load_obj(obj_path, &load_options).context("Failed to load OBJ file")?;

    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut texture_path = None;

    // Process materials to find texture path
    if let Ok(materials) = materials {
        for material in materials.iter() {
            if let Some(diffuse_texture) = &material.diffuse_texture {
                // Get the directory of the OBJ file
                let obj_dir = obj_path.parent().unwrap_or(Path::new("."));
                // Resolve texture path relative to OBJ file or MTL directory
                let base_dir = mtl_dir.unwrap_or(obj_dir);
                let tex_path = base_dir.join(diffuse_texture);
                texture_path = Some(
                    tex_path
                        .to_str()
                        .ok_or_else(|| anyhow::anyhow!("Invalid texture path"))?
                        .to_string(),
                );
                break; // Use first texture found
            }
        }
    }

    // Process all models and combine their geometry
    for model in models.iter() {
        let mesh = &model.mesh;

        // Calculate normals if not present
        let mut normals = mesh.normals.clone();
        if normals.is_empty() {
            normals = calculate_normals(&mesh.positions, &mesh.indices);
        }

        // Ensure we have texture coordinates (use zeros if missing)
        let mut tex_coords = mesh.texcoords.clone();
        if tex_coords.is_empty() {
            tex_coords = vec![0.0; (mesh.positions.len() / 3) * 2];
        }

        // Build vertex data: position(3) + normal(3) + texcoord(2) + color(3) = 11 floats per vertex
        let vertex_offset = vertices.len() / 11;
        for i in 0..(mesh.positions.len() / 3) {
            let pos_idx = i * 3;
            let norm_idx = if i < normals.len() / 3 { i * 3 } else { 0 };
            let tex_idx = if i < tex_coords.len() / 2 { i * 2 } else { 0 };

            // Position
            vertices.push(mesh.positions[pos_idx]);
            vertices.push(mesh.positions[pos_idx + 1]);
            vertices.push(mesh.positions[pos_idx + 2]);

            // Normal
            vertices.push(normals[norm_idx]);
            vertices.push(normals[norm_idx + 1]);
            vertices.push(normals[norm_idx + 2]);

            // Texture coordinates - flip V coordinate (Y) because PNG has origin at top-left
            // while OpenGL expects bottom-left
            vertices.push(tex_coords[tex_idx]);
            vertices.push(1.0 - tex_coords[tex_idx + 1]);

            // Color (default to white, can be overridden by material)
            vertices.push(1.0);
            vertices.push(1.0);
            vertices.push(1.0);
        }

        // Add indices (offset by current vertex count)
        for &idx in mesh.indices.iter() {
            indices.push(idx + vertex_offset as u32);
        }
    }

    Ok(LoadedModel {
        vertices,
        indices,
        texture_path,
    })
}

fn calculate_normals(positions: &[f32], indices: &[u32]) -> Vec<f32> {
    let mut normals = vec![0.0; positions.len()];

    // Calculate face normals and accumulate
    for i in (0..indices.len()).step_by(3) {
        let i0 = indices[i] as usize * 3;
        let i1 = indices[i + 1] as usize * 3;
        let i2 = indices[i + 2] as usize * 3;

        let v0 = Vec3::new(positions[i0], positions[i0 + 1], positions[i0 + 2]);
        let v1 = Vec3::new(positions[i1], positions[i1 + 1], positions[i1 + 2]);
        let v2 = Vec3::new(positions[i2], positions[i2 + 1], positions[i2 + 2]);

        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let normal = edge1.cross(edge2).normalize();

        // Accumulate normals
        for j in 0..3 {
            let idx = indices[i + j] as usize * 3;
            normals[idx] += normal.x;
            normals[idx + 1] += normal.y;
            normals[idx + 2] += normal.z;
        }
    }

    // Normalize accumulated normals
    for i in (0..normals.len()).step_by(3) {
        let n = Vec3::new(normals[i], normals[i + 1], normals[i + 2]);
        let normalized = n.normalize();
        normals[i] = normalized.x;
        normals[i + 1] = normalized.y;
        normals[i + 2] = normalized.z;
    }

    normals
}

pub fn load_png_texture(gl: &glow::Context, texture_path: &Path) -> Result<glow::NativeTexture> {
    use glow::HasContext;
    use image::GenericImageView;

    let img = image::open(texture_path)
        .context("Failed to open PNG file")?
        .to_rgba8();

    let (width, height) = img.dimensions();
    // Use raw data directly - we're flipping UV coordinates instead (V coordinate flipped in model loader)
    let data = img.as_raw();

    unsafe {
        let texture = gl
            .create_texture()
            .map_err(|e| anyhow::anyhow!("Failed to create texture: {}", e))?;
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));

        gl.tex_image_2d(
            glow::TEXTURE_2D,
            0,
            glow::RGBA as i32,
            width as i32,
            height as i32,
            0,
            glow::RGBA,
            glow::UNSIGNED_BYTE,
            Some(data),
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
