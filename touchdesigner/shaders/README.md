The shaders in this directory implement a volumetric rendering solution for
textured 3D geometry. For a 3D raster of size NxNxN, this algorithm performs
Nx3 rendering passes each to a texture of size NxN tiled into a larger texture
whose size is the minimum square texture to fit all NxN tiles.

Each pass transforms the input geometry so that a slice of the geometry
intersects the camera view plane. Fragment culling is used to discard fragments
which fall outside of the tile, or outside of the slice thickness. To ensure
that geometry always produces fragments even when the polygons are oriented
perpendicular to the camera view direction, the geometry is rendered once with
the camera looking along X, once along Y, and once along Z. The results are then
blended together to produce the final output (sampling from each of the three
textures at the coordinates that, projected along the plane normal, intersect
the target voxel).

The main shaders are:

- `sliced_rendering.vertex`: Vertex shader which transforms the input geometry
  so that a slice of the geometry intersects the camera view plane. Outputs
  texture coordinates for sampling from the input texture, and outputs
  fragment coordinates for culling. This shader must be run instanced for each
  Nx3 slice,orientation pair.
- `sliced_rendering.fragment`: Fragment shader which samples from the input
  to compute the tiled texture fragment colors.
- `sample_to_3d_tex.fragment`: Fragment shader which converts the input 2D tiled
  texture into a 3D texture by sampling from the appropriate tiles.
