# Volumetric Rendering Library

This library implements a sliced rendering pipeline for volumetric displays, based on
the shaders from TouchDesigner.

## Overview

The sliced rendering pipeline renders 3D geometry into a volumetric texture by
performing N×3 rendering passes (N slices × 3 orientations: X, Y, Z). Each pass
transforms the geometry so that slices intersect the camera view plane, and the
results are tiled into a 2D texture.

## Architecture

- **`shaders.rs`**: Contains GLSL shader source code:

  - `SLICED_RENDERING_VERTEX`: Vertex shader for instanced rendering with
    slice/orientation transforms
  - `SLICED_RENDERING_FRAGMENT`: Fragment shader that discards fragments outside
    slice thickness and tile bounds
  - `SAMPLE_TO_3D_TEX_FRAGMENT`: Fragment shader for converting tiled texture back
    to 3D volume (for future use)

- **`renderer.rs`**: Main rendering implementation:
  - `SlicedRenderer`: Manages shader compilation, FBO setup, and instanced rendering
  - `VolumeConfig`: Configuration for volume resolution and slice thickness

## Usage

```rust
use volumetric_rendering::{SlicedRenderer, VolumeConfig};
use glam::{Mat4, Vec3};

// Create renderer
let config = VolumeConfig {
    resolution: (64, 64, 64),
    slice_thickness: 0.02,
};
let mut renderer = SlicedRenderer::new(config)?;

// Upload geometry (vertices and indices)
renderer.upload_geometry(&vertices, &indices)?;
renderer.set_index_count(indices.len());

// Render to tiled texture
let model = Mat4::IDENTITY;
let view = Mat4::look_at_rh(Vec3::new(2.0, 2.0, 2.0), Vec3::ZERO, Vec3::Y);
let projection = Mat4::perspective_rh(45.0f32.to_radians(), 1.0, 0.1, 100.0);
renderer.render_to_tiled_texture(model, view, projection, input_texture, -1.0, 1.0)?;

// Get the tiled texture
let tiled_texture = renderer.get_tiled_texture();
```

## Demo Application

See `src/volumetric_demo/main.rs` for a complete example that:

- Sets up a GLFW window
- Creates a textured cube
- Renders it using the sliced rendering pipeline
- Displays the tiled texture in the viewport

## Building

### With Cargo

```bash
cd src/volumetric_demo
cargo run
```

### With Bazel

```bash
bazel build //src/volumetric_demo:volumetric_demo
bazel run //src/volumetric_demo:volumetric_demo
```

## Dependencies

- `gl`: OpenGL bindings
- `glam`: Math library (similar to GLM)
- `anyhow`: Error handling
- `thiserror`: Error types
