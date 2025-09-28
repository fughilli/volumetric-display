#include "VolumetricDisplay.h"
#include "DisplayConfig.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/log.h"
#include "resources/icon.h"
#include <fstream> // Required for file operations
#include <glm/glm.hpp>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// Include the nlohmann/json library
#include "nlohmann/json.hpp"

// For convenience
using json = nlohmann::json;

// Define command-line flags using Abseil
ABSL_FLAG(std::string, config, "sim_config.json", "Path to the simulation configuration file");
ABSL_FLAG(float, alpha, 0.5, "Alpha value for voxel colors");
ABSL_FLAG(int, layer_span, 1, "Layer span (1 for 1:1 mapping)");
ABSL_FLAG(std::string, rotate_rate, "0,0,0",
          "Continuous rotation rate in degrees/sec for X,Y,Z axes (e.g., "
          "\"10,0,5\")");
ABSL_FLAG(bool, color_correction, false, "Enable color correction");
ABSL_FLAG(float, voxel_scale, 0.15f, "Scaling factor for individual voxels (e.g., 0.8 for smaller voxels with gaps)");
ABSL_FLAG(int, universes_per_layer, 3, "Number of universes per layer");

// Entry point
int main(int argc, char *argv[]) {
  // Parse command-line arguments using Abseil
  absl::ParseCommandLine(argc, argv);

  try {
    // Read and parse the configuration file
    std::string config_path = absl::GetFlag(FLAGS_config);
    std::ifstream config_file(config_path);
    if (!config_file.is_open()) {
        throw std::runtime_error("Could not open config file: " + config_path);
    }
    json config;
    config_file >> config;

    // Extract values from the JSON object

    // Parse geometry dimensions - support both old and new config formats
    std::string geometry_str;
    if (config.contains("world_geometry")) {
        geometry_str = config["world_geometry"];
    } else if (config.contains("cube_geometry")) {
        geometry_str = config["cube_geometry"];
    } else {
        throw std::runtime_error("Config must contain either 'world_geometry' or 'cube_geometry' field.");
    }

    int width, height, length;
    if (sscanf(geometry_str.c_str(), "%dx%dx%d", &width, &height, &length) != 3) {
      throw std::runtime_error(
          "Invalid geometry format in config. Use WIDTHxHEIGHTxLENGTH (e.g., 20x20x20).");
    }

    // Cube configs & artnet mappings
    std::vector<CubeConfig> cube_configs;
        for (const auto& cube_json : config["cubes"]) {
            CubeConfig current_cube;
            current_cube.position = glm::vec3(cube_json["position"][0], cube_json["position"][1], cube_json["position"][2]);

            // Parse individual cube dimensions (new format) or use global geometry (old format)
            if (cube_json.contains("dimensions")) {
                std::string cube_geometry_str = cube_json["dimensions"];
                int cube_width, cube_height, cube_length;
                if (sscanf(cube_geometry_str.c_str(), "%dx%dx%d", &cube_width, &cube_height, &cube_length) != 3) {
                    throw std::runtime_error("Invalid cube dimensions format. Use WIDTHxHEIGHTxLENGTH (e.g., 20x20x20).");
                }
                current_cube.width = cube_width;
                current_cube.height = cube_height;
                current_cube.length = cube_length;
            } else {
                // Fall back to global geometry for backward compatibility
                current_cube.width = width;
                current_cube.height = height;
                current_cube.length = length;
            }

            for (const auto& mapping_json : cube_json["artnet_mappings"]) {
                ArtNetListenerConfig listener;
                listener.ip = mapping_json["ip"];
                // The port might be a string in JSON, so we handle it safely
                if (mapping_json["port"].is_string()) {
                    listener.port = std::stoi(mapping_json["port"].get<std::string>());
                } else {
                    listener.port = mapping_json["port"];
                }

                // Parse z_idx array
                for (const auto& z : mapping_json["z_idx"]) {
                    listener.z_indices.push_back(z);
                }
                current_cube.listeners.push_back(listener);
            }
            cube_configs.push_back(current_cube);
        }

        if (cube_configs.empty()) {
            throw std::runtime_error("No cubes defined in the configuration file.");
        }

    // Other flags
    int universes_per_layer = absl::GetFlag(FLAGS_universes_per_layer);
    int layer_span = absl::GetFlag(FLAGS_layer_span);
    float alpha = absl::GetFlag(FLAGS_alpha);
    std::string rotate_rate_str = absl::GetFlag(FLAGS_rotate_rate);
    const bool color_correction_enabled = absl::GetFlag(FLAGS_color_correction);
    float voxel_scale = absl::GetFlag(FLAGS_voxel_scale);

    // Rotation rate
    glm::vec3 rotation_rate(0.0f);
    std::stringstream ss(rotate_rate_str);
    std::string segment;
    int i = 0;
    while (std::getline(ss, segment, ',') && i < 3) {
      rotation_rate[i++] = std::stof(segment);
    }

    LOG(INFO) << "Starting Volumetric Display with the following parameters:";
    LOG(INFO) << "Cube Geometry: " << width << "x" << height << "x" << length;
    LOG(INFO) << "Number of Cubes: " << cube_configs.size();

    // Create and run the volumetric display with the new parameters
    VolumetricDisplay display(width, height, length, universes_per_layer,
                                      layer_span, alpha, rotation_rate,
                                      color_correction_enabled, cube_configs, voxel_scale);

    // Configure icon and run
    SetIcon(argv[0]);
    display.run();

  } catch (const json::parse_error& e) {
    std::cerr << "JSON parsing error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  } catch (const std::exception &ex) {
    std::cerr << "Error: " << ex.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
