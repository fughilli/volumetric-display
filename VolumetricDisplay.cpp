#include "VolumetricDisplay.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <boost/asio.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

VolumetricDisplay::VolumetricDisplay(int width, int height, int length,
                                     const std::string &ip, int port,
                                     int universes_per_layer)
    : width(width), height(height), length(length), ip(ip), port(port),
      universes_per_layer(universes_per_layer),
      socket(io_service, boost::asio::ip::udp::endpoint(
                             boost::asio::ip::address::from_string(ip), port)) {

  if (universes_per_layer > MAX_UNIVERSES_PER_LAYER) {
    throw std::runtime_error("Layer size too large for ArtNet limitations");
  }

  pixels.resize(width * height * length, {0, 0, 0});

  running = true;
  needs_update = false;

  rotation_matrix = glm::mat4(1.0f);
  temp_matrix = glm::mat4(1.0f);

  setupOpenGL();
  setupVBO();

  artnet_thread = std::thread(&VolumetricDisplay::listenArtNet, this);
}

VolumetricDisplay::~VolumetricDisplay() { cleanup(); }

void VolumetricDisplay::setupOpenGL() {
  if (!glfwInit()) {
    throw std::runtime_error("Failed to initialize GLFW");
  }

  GLFWwindow *window =
      glfwCreateWindow(800, 600, "Volumetric Display", nullptr, nullptr);
  if (!window) {
    glfwTerminate();
    throw std::runtime_error("Failed to create GLFW window");
  }
  glfwMakeContextCurrent(window);
  glewInit();

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_LIGHTING);
  glEnable(GL_COLOR_MATERIAL);
  glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
  glLightModelfv(GL_LIGHT_MODEL_AMBIENT, (GLfloat[]){1.0, 1.0, 1.0, 1.0});
}

void VolumetricDisplay::setupVBO() {
  std::vector<GLfloat> vertices;
  std::vector<GLfloat> colors;

  vertex_count = width * height * length *
                 24; // 24 vertices per voxel (6 faces * 4 vertices)

  for (int x = 0; x < width; ++x) {
    for (int y = 0; y < height; ++y) {
      for (int z = 0; z < length; ++z) {
        GLfloat size = 0.1f;

        // Front face (as an example)
        vertices.insert(vertices.end(), {
                                            x - size,
                                            y - size,
                                            z + size,
                                            x + size,
                                            y - size,
                                            z + size,
                                            x + size,
                                            y + size,
                                            z + size,
                                            x - size,
                                            y + size,
                                            z + size,
                                        });

        // Initialize colors (e.g., white for now)
        for (int i = 0; i < 24; ++i) {
          colors.push_back(1.0f); // R
          colors.push_back(1.0f); // G
          colors.push_back(1.0f); // B
        }
      }
    }
  }

  // Generate and bind Vertex Buffer Object
  glGenBuffers(1, &vbo_vertices);
  glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices);
  glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(GLfloat),
               vertices.data(), GL_STATIC_DRAW);

  // Generate and bind Color Buffer Object
  glGenBuffers(1, &vbo_colors);
  glBindBuffer(GL_ARRAY_BUFFER, vbo_colors);
  glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(GLfloat), colors.data(),
               GL_DYNAMIC_DRAW);
}

void VolumetricDisplay::listenArtNet() {
  std::cout << "Listening for Art-Net on " << ip << ":" << port << std::endl;
  while (running) {
    std::cout << "Receive?" << std::endl;
    std::array<char, 1024> buffer;
    boost::asio::ip::udp::endpoint sender_endpoint;
    size_t length =
        socket.receive_from(boost::asio::buffer(buffer), sender_endpoint);

    std::cout << "Received " << length << " bytes" << std::endl;

    if (strncmp(buffer.data(), "Art-Net\0", 8) != 0) {
      std::cout << "Received non-Art-Net packet" << std::endl;
      continue;
    }

    uint16_t opcode = *reinterpret_cast<uint16_t *>(&buffer[8]);
    if (opcode == 0x5000) { // DMX Data
      uint16_t universe = *reinterpret_cast<uint16_t *>(&buffer[14]);
      uint16_t length = ntohs(*reinterpret_cast<uint16_t *>(&buffer[16]));

      int layer = universe / universes_per_layer;
      int universe_in_layer = universe % universes_per_layer;
      int start_pixel = universe_in_layer * 170;

      std::cout << "Received DMX data for universe " << universe << " (layer "
                << layer << ", universe in layer " << universe_in_layer << ")"
                << ", start pixel " << start_pixel << ", length " << length
                << std::endl;

      int updated_pixels = 0;

      for (int i = 0; i < length && (start_pixel + i / 3) < width * height;
           i += 3) {
        int idx = start_pixel + i / 3;
        int x = idx % width;
        int y = idx / width;
        int pixel_index = x + y * width + layer * width * height;
        if (pixel_index >= pixels.size()) {
          continue;
        }
        pixels[pixel_index] = {(unsigned char)buffer[18 + i],
                               (unsigned char)buffer[18 + i + 1],
                               (unsigned char)buffer[18 + i + 2]};
        updated_pixels++;
      }
      std::cout << "Updated " << updated_pixels << " pixels" << std::endl;
    } else if (opcode == 0x5200) {
      std::cout << "Received ArtPoll" << std::endl;
      needs_update = true;
    } else {
      std::cout << "Received unknown opcode: " << opcode << std::endl;
    }
  }
}

void VolumetricDisplay::updateColors() {
  glBindBuffer(GL_ARRAY_BUFFER, vbo_colors);
  std::vector<GLfloat> colors;

  for (const auto &pixel : pixels) {
    GLfloat r = pixel[0] / 255.0f;
    GLfloat g = pixel[1] / 255.0f;
    GLfloat b = pixel[2] / 255.0f;
    for (int i = 0; i < 24; ++i) {
      colors.push_back(r);
      colors.push_back(g);
      colors.push_back(b);
    }
  }

  glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(GLfloat), colors.data(),
               GL_DYNAMIC_DRAW);
}

void VolumetricDisplay::run() {
  while (running) {
    render();
    glfwPollEvents();
  }
}

void VolumetricDisplay::cleanup() {
  running = false;
  artnet_thread.join();
  socket.close();
  glfwTerminate();
}

void VolumetricDisplay::render() {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  gluPerspective(45.0f, 800.0f / 600.0f, 0.1f, 100.0f);
  glTranslatef(0, 0, -std::max(width, std::max(height, length)) * 2.5f);
  glMultMatrixf(glm::value_ptr(rotation_matrix));
  glTranslatef(-width / 2.0f, -height / 2.0f, -length / 2.0f);

  glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices);
  glVertexPointer(3, GL_FLOAT, 0, nullptr);
  glEnableClientState(GL_VERTEX_ARRAY);

  glBindBuffer(GL_ARRAY_BUFFER, vbo_colors);
  glColorPointer(3, GL_FLOAT, 0, nullptr);
  glEnableClientState(GL_COLOR_ARRAY);

  glDrawArrays(GL_QUADS, 0, vertex_count);

  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_COLOR_ARRAY);

  glfwSwapBuffers(glfwGetCurrentContext());
}
