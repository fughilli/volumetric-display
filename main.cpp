#include "VolumetricDisplay.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/log.h"
#include "resources/icon.h"
#include <glm/glm.hpp>
#include <iostream>
#include <sstream>
#include <string>

// Define command-line flags using Abseil
ABSL_FLAG(std::string, geometry, "16x16x16",
          "Width, height, and length of the display");
ABSL_FLAG(std::string, ip, "127.0.0.1",
          "IP address to listen for ArtNet packets");
ABSL_FLAG(int, port, 6454, "Port to listen for ArtNet packets");
ABSL_FLAG(int, universes_per_layer, 6, "Number of universes per layer");
ABSL_FLAG(float, alpha, 0.5, "Alpha value for voxel colors");
ABSL_FLAG(int, layer_span, 1, "Layer span (1 for 1:1 mapping)");
ABSL_FLAG(std::string, rotate_rate, "0,0,0",
          "Continuous rotation rate in degrees/sec for X,Y,Z axes (e.g., "
          "\"10,0,5\")");

// Entry point
int main(int argc, char *argv[]) {
  // Parse command-line arguments using Abseil
  absl::ParseCommandLine(argc, argv);

  try {
    // Extract parsed arguments
    std::string geometry = absl::GetFlag(FLAGS_geometry);
    std::string ip = absl::GetFlag(FLAGS_ip);
    int port = absl::GetFlag(FLAGS_port);
    int universes_per_layer = absl::GetFlag(FLAGS_universes_per_layer);
    int layer_span = absl::GetFlag(FLAGS_layer_span);

    float alpha = absl::GetFlag(FLAGS_alpha);
    std::string rotate_rate_str = absl::GetFlag(FLAGS_rotate_rate);

    // Parse geometry dimensions
    int width, height, length;
    if (sscanf(geometry.c_str(), "%dx%dx%d", &width, &height, &length) != 3) {
      throw std::runtime_error(
          "Invalid geometry format. Use WIDTHxHEIGHTxLENGTH (e.g., 16x16x16).");
    }

    // Parse rotation rate
    glm::vec3 rotation_rate(0.0f);
    std::stringstream ss(rotate_rate_str);
    std::string segment;
    int i = 0;
    while (std::getline(ss, segment, ',') && i < 3) {
      try {
        rotation_rate[i++] = std::stof(segment);
      } catch (const std::invalid_argument &ia) {
        throw std::runtime_error(
            "Invalid rotate_rate format. Use comma-separated floats (e.g., "
            "10,0,5).");
      } catch (const std::out_of_range &oor) {
        throw std::runtime_error("Rotation rate value out of range.");
      }
    }
    if (i != 3 || std::getline(ss, segment, ',')) {
      throw std::runtime_error("Invalid rotate_rate format. Must provide "
                               "exactly three comma-separated floats (e.g., "
                               "10,0,5).");
    }

    LOG(INFO) << "Starting Volumetric Display with the following parameters:";
    LOG(INFO) << "Geometry: " << width << "x" << height << "x" << length;
    LOG(INFO) << "IP: " << ip;
    LOG(INFO) << "Port: " << port;
    LOG(INFO) << "Universes per layer: " << universes_per_layer;
    LOG(INFO) << "Rotation Rate (deg/s): X=" << rotation_rate.x
              << " Y=" << rotation_rate.y << " Z=" << rotation_rate.z;

    // Create and run the volumetric display
    VolumetricDisplay display(width, height, length, ip, port,
                              universes_per_layer, layer_span, alpha,
                              rotation_rate);

    // Configure icon
    SetIcon(argv[0]);

    display.run();

  } catch (const std::exception &ex) {
    std::cerr << "Error: " << ex.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
