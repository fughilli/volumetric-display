#ifndef DISPLAY_CONFIG_H
#define DISPLAY_CONFIG_H

#include <glm/glm.hpp>
#include <string>
#include <vector>

// Defines a single ArtNet listener (IP and port)
struct ArtNetListenerConfig {
  std::string ip;
  int port;
  std::vector<int> z_indices;
};

// Defines a scatter gather channel sample (strand with coordinates)
struct ScatterGatherChannelSample {
  int universe;
  std::vector<glm::ivec3> coords;  // [x, y, z] coordinates for each pixel
};

// Defines a scatter gather ArtNet listener (IP, port, and channel samples)
struct ScatterGatherListenerConfig {
  std::string ip;
  int port;
  std::vector<ScatterGatherChannelSample> channel_samples;
};

// Defines a single cube, including its position in world space
// and all the ArtNet listeners that feed it data.
struct CubeConfig {
  glm::vec3 position;
  int width = 20;  // Default cube dimensions
  int height = 20;
  int length = 20;
  std::vector<std::string> orientation = {"-Z", "Y", "X"};       // Default sampling orientation
  std::vector<std::string> world_orientation = {"X", "Y", "Z"};  // Default world orientation
  std::vector<ArtNetListenerConfig> listeners;
};

// Defines a scatter gather cube configuration
struct ScatterGatherCubeConfig {
  std::vector<ScatterGatherListenerConfig> listeners;
};

#endif  // DISPLAY_CONFIG_H
