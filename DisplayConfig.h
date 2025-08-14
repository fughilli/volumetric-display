#ifndef DISPLAY_CONFIG_H
#define DISPLAY_CONFIG_H

#include <string>
#include <vector>
#include <glm/glm.hpp>

// Defines a single ArtNet listener (IP and port)
struct ArtNetListenerConfig {
    std::string ip;
    int port;
};

// Defines a single cube, including its position in world space
// and all the ArtNet listeners that feed it data.
struct CubeConfig {
    glm::vec3 position;
    std::vector<ArtNetListenerConfig> listeners;
};

#endif // DISPLAY_CONFIG_H
