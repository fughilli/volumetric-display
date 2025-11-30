// GLSL shader source code for the sliced rendering pipeline

pub const SLICED_RENDERING_VERTEX: &str = r#"
#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoord;
layout(location = 3) in vec3 aColor;

out vec2 vUV;
out vec3 vWorldPos;
out vec3 vColor;

uniform ivec2 tileCountXY;
uniform int tileCount;
uniform float minZ;
uniform float maxZ;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

flat out int instanceID;

mat3 rotateXtoZ = mat3(
    0.0,  0.0,  1.0,
    0.0,  1.0,  0.0,
   -1.0,  0.0,  0.0
);

mat3 rotateYtoZ = mat3(
    1.0,  0.0,  0.0,
    0.0,  0.0,  1.0,
    0.0, -1.0,  0.0
);

mat3 rotateZtoZ = mat3(
    1.0,  0.0,  0.0,
    0.0,  1.0,  0.0,
    0.0,  0.0,  1.0
);

void main() {
  vColor = aColor;
  vec3 pos = aPos;
  vec4 worldSpacePos = model * vec4(pos, 1.0);

  // Get instance ID from gl_InstanceID
  int instance = gl_InstanceID / 3;
  instanceID = gl_InstanceID;

  mat3 rotationMatrix = rotateZtoZ;

  if (instanceID % 3 == 0) {
    rotationMatrix = rotateXtoZ;
  } else if (instanceID % 3 == 1) {
    rotationMatrix = rotateYtoZ;
  } else {
    rotationMatrix = rotateZtoZ;
  }

  worldSpacePos = vec4(rotationMatrix * worldSpacePos.xyz, 1.0);

  float z_displace = mix(minZ, maxZ, float(instance) / float(tileCount));
  worldSpacePos.z += z_displace;

  worldSpacePos.z /= tileCountXY.x;

  worldSpacePos.x /= 2.0;
  worldSpacePos.y /= 2.0;

  worldSpacePos.x = (float(instanceID % tileCountXY.x) + worldSpacePos.x + 0.5) /
                    float(tileCountXY.x);
  worldSpacePos.y = (float(instanceID / tileCountXY.x) + worldSpacePos.y + 0.5) /
                    float(tileCountXY.y);

  worldSpacePos.x = worldSpacePos.x * 2.0 - 1.0;
  worldSpacePos.y = worldSpacePos.y * 2.0 - 1.0;

  vWorldPos = worldSpacePos.xyz;
  vUV = aTexCoord;

  gl_Position = projection * view * worldSpacePos;
}
"#;

pub const SLICED_RENDERING_FRAGMENT: &str = r#"
#version 330 core

out vec4 fragColor;

// Slicing and tiling controls
uniform float uSliceThickness; // Thickness of the slice

uniform sampler2D inTex;

uniform ivec2 tileCountXY;

in vec3 vWorldPos;
in vec2 vUV;
in vec3 vColor;

flat in int instanceID;

vec4 shift_hue(vec4 color, float hue) {
  const vec3 k = vec3(0.57735, 0.57735, 0.57735);
  float cosine_hue = cos(hue);
  return vec4(color.xyz * cosine_hue + cross(k, color.xyz) * sin(hue) +
                  k * dot(k, color.xyz) * (1.0 - cosine_hue),
              color.w);
}

void main() {
  // Discard fragments outside the slicing plane
  if (vWorldPos.z > uSliceThickness / 2.0 || vWorldPos.z < -uSliceThickness / 2.0) {
    discard;
  }

  // Discard fragments outside of the tile
  vec2 tileBoundsMin =
      vec2(float(instanceID % tileCountXY.x) / float(tileCountXY.x),
           float(instanceID / tileCountXY.x) / float(tileCountXY.y));
  vec2 tileBoundsMax = tileBoundsMin + vec2(1.0 / float(tileCountXY.x),
                                            1.0 / float(tileCountXY.y));
  tileBoundsMin = tileBoundsMin * 2.0 - 1.0;
  tileBoundsMax = tileBoundsMax * 2.0 - 1.0;

  if (vWorldPos.x < tileBoundsMin.x || vWorldPos.x > tileBoundsMax.x ||
      vWorldPos.y < tileBoundsMin.y || vWorldPos.y > tileBoundsMax.y) {
    discard;
  } else {
    vec4 color =
        shift_hue(texture(inTex, vUV),
                  3.1415926 * 2.0 * 7.1 * (vColor.z - vColor.x) / vColor.y);
    fragColor = color;
  }
}
"#;

pub const SAMPLE_TO_3D_TEX_FRAGMENT: &str = r#"
#version 330 core

out vec4 fragColor;

// Input uniforms
uniform sampler2D tiledTexture;
uniform ivec2 tileCountXY; // Number of tiles in X (tileCountXY.x) and Y (tileCountXY.y)
uniform ivec3 volumeResolution; // Resolution of the 3D volume (x, y, z)
uniform int uZSlice; // Current Z slice being rendered (0 to volumeResolution.z - 1)

in vec2 vUV;

vec4 sampleTile(int tileIndex, vec2 uv) {
  vec2 tileBaseOffset =
      vec2(tileIndex % tileCountXY.x, tileIndex / tileCountXY.x) /
      vec2(tileCountXY.x, tileCountXY.y);

  vec2 sampleCoord = tileBaseOffset + uv / vec2(tileCountXY.x, tileCountXY.y);

  return texture(tiledTexture, sampleCoord);
}

// Assumes inputs are linear RGB. If you're in sRGB, linearize first (see
// below).
vec4 compositeXYZ(vec4 X, vec4 Y, vec4 Z) {
  vec3 rgb = max(X.rgb, max(Y.rgb, Z.rgb));
  float a = max(X.a, max(Y.a, Z.a));
  return vec4(rgb, a);
}

vec4 sample3d(ivec3 coord, ivec3 res) {
  int nx = res.x - 1 - coord.x;
  int ny = res.y - 1 - coord.y;
  int nz = res.z - 1 - coord.z;
  int x = coord.x;
  int y = coord.y;
  int z = coord.z;
  vec4 xsample =
      sampleTile(nx * 3, (vec2(z, y) + vec2(0.5)) / vec2(res.z, res.y));
  vec4 ysample =
      sampleTile(ny * 3 + 1, (vec2(x, z) + vec2(0.5)) / vec2(res.x, res.z));
  vec4 zsample =
      sampleTile(z * 3 + 2, (vec2(x, y) + vec2(0.5)) / vec2(res.x, res.y));

  return compositeXYZ(xsample, ysample, zsample);
}

void main() {
  ivec3 res = volumeResolution;
  // Get the current fragment coordinates from UV
  // UV maps to [0,1] range, multiply by resolution to get integer coordinates
  vec2 uv = vUV;
  // Use the Z slice uniform to determine which slice we're rendering
  ivec3 fragCoord = ivec3(int(uv.x * float(res.x)), int(uv.y * float(res.y)), uZSlice);

  vec4 color = sample3d(fragCoord, res);

  fragColor = color;
}
"#;
