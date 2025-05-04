#include "VolumetricDisplay.h"
#include "absl/log/log.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <boost/asio.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "color_correction.h"

VolumetricDisplay::VolumetricDisplay(int width, int height, int length,
                                     const std::string &ip, int port,
                                     int universes_per_layer, int layer_span,
                                     float alpha,
                                     const glm::vec3 &initial_rotation_rate)
    : left_mouse_button_pressed(false), right_mouse_button_pressed(false),
      width(width), height(height), length(length), ip(ip), port(port),
      universes_per_layer(universes_per_layer), layer_span(layer_span),
      alpha(alpha), rotation_rate(initial_rotation_rate), show_axis(false),
      show_wireframe(false), needs_update(false),
      socket(io_service, boost::asio::ip::udp::endpoint(
                             boost::asio::ip::address::from_string(ip), port)) {

  if (universes_per_layer > MAX_UNIVERSES_PER_LAYER) {
    throw std::runtime_error("Layer size too large for ArtNet limitations");
  }

  LOG(INFO) << "Color corrector brightness R="
            << util::kColorCorrectorWs2812bOptions.brightness[0]
            << " G=" << util::kColorCorrectorWs2812bOptions.brightness[1]
            << " B=" << util::kColorCorrectorWs2812bOptions.brightness[2];

  pixels.resize(width * height * (length / layer_span), {0, 0, 0});

  running = true;
  needs_update = false;

  rotation_matrix = glm::mat4(1.0f);
  temp_matrix = glm::mat4(1.0f);

  setupOpenGL();
  camera_position = glm::vec3(0.0f, 0.0f, 0.0f);
  camera_orientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
  camera_distance = std::max(width, std::max(height, length)) * 2.5f;
  left_mouse_button_pressed = false;
  right_mouse_button_pressed = false;
  last_mouse_x = 0.0;
  last_mouse_y = 0.0;
  setupVBO();

  artnet_thread = std::thread(&VolumetricDisplay::listenArtNet, this);
}

VolumetricDisplay::~VolumetricDisplay() { cleanup(); }

void VolumetricDisplay::drawWireframeCube() {
  glPushMatrix();
  glTranslatef(-0.5f, -0.5f, -0.5f);
  glColor3f(1.0f, 1.0f, 1.0f);
  glBegin(GL_LINES);
  // Bottom face
  glVertex3f(0.0f, 0.0f, 0.0f);
  glVertex3f(width, 0.0f, 0.0f);
  glVertex3f(width, 0.0f, 0.0f);
  glVertex3f(width, height, 0.0f);
  glVertex3f(width, height, 0.0f);
  glVertex3f(0.0f, height, 0.0f);
  glVertex3f(0.0f, height, 0.0f);
  glVertex3f(0.0f, 0.0f, 0.0f);
  // Top face
  glVertex3f(0.0f, 0.0f, length);
  glVertex3f(width, 0.0f, length);
  glVertex3f(width, 0.0f, length);
  glVertex3f(width, height, length);
  glVertex3f(width, height, length);
  glVertex3f(0.0f, height, length);
  glVertex3f(0.0f, height, length);
  glVertex3f(0.0f, 0.0f, length);
  // Vertical lines
  glVertex3f(0.0f, 0.0f, 0.0f);
  glVertex3f(0.0f, 0.0f, length);
  glVertex3f(width, 0.0f, 0.0f);
  glVertex3f(width, 0.0f, length);
  glVertex3f(width, height, 0.0f);
  glVertex3f(width, height, length);
  glVertex3f(0.0f, height, 0.0f);
  glVertex3f(0.0f, height, length);
  glEnd();
  glPopMatrix();
}

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

  glfwSetWindowUserPointer(window, this);
  glfwSetWindowCloseCallback(window, [](GLFWwindow *window) {
    static_cast<VolumetricDisplay *>(glfwGetWindowUserPointer(window))
        ->windowCloseCallback(window);
  });
  glfwSetMouseButtonCallback(
      window, [](GLFWwindow *window, int button, int action, int mods) {
        static_cast<VolumetricDisplay *>(glfwGetWindowUserPointer(window))
            ->mouseButtonCallback(window, button, action, mods);
      });
  glfwSetCursorPosCallback(
      window, [](GLFWwindow *window, double xpos, double ypos) {
        static_cast<VolumetricDisplay *>(glfwGetWindowUserPointer(window))
            ->cursorPositionCallback(window, xpos, ypos);
      });
  glfwSetScrollCallback(
      window, [](GLFWwindow *window, double xoffset, double yoffset) {
        static_cast<VolumetricDisplay *>(glfwGetWindowUserPointer(window))
            ->scrollCallback(window, xoffset, yoffset);
      });

  glfwSetKeyCallback(window, [](GLFWwindow *window, int key, int scancode,
                                int action, int mods) {
    static_cast<VolumetricDisplay *>(glfwGetWindowUserPointer(window))
        ->keyCallback(window, key, scancode, action, mods);
  });

  glfwSetFramebufferSizeCallback(
      window, [](GLFWwindow *window, int width, int height) {
        static_cast<VolumetricDisplay *>(glfwGetWindowUserPointer(window))
            ->framebufferSizeCallback(window, width, height);
      });

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_LIGHTING);
  glEnable(GL_COLOR_MATERIAL);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);

  const GLfloat kLightColor[] = {1.0f, 1.0f, 1.0f, 1.0f};
  glLightModelfv(GL_LIGHT_MODEL_AMBIENT, kLightColor);

  glfwGetFramebufferSize(window, &viewport_width, &viewport_height);
  glViewport(0, 0, viewport_width, viewport_height);
  viewport_aspect =
      static_cast<float>(viewport_width) / static_cast<float>(viewport_height);
}

void VolumetricDisplay::setupVBO() {
  std::vector<GLfloat> vertices;
  std::vector<GLfloat> colors;
  std::vector<GLuint> indices;

  vertex_count =
      width * height * (length / layer_span) *
      36; // 36 indices per voxel (6 faces * 2 triangles * 3 vertices)

  for (int x = 0; x < width; ++x) {
    for (int y = 0; y < height; ++y) {
      for (int z = 0; z < length * layer_span; z += layer_span) {
        GLfloat size = 0.1f;

        // Define the 8 corners of the cube
        std::array<GLfloat, 24> cube_vertices = {
            x - size, y - size, z - size, x + size, y - size, z - size,
            x + size, y + size, z - size, x - size, y + size, z - size,
            x - size, y - size, z + size, x + size, y - size, z + size,
            x + size, y + size, z + size, x - size, y + size, z + size,
        };

        vertices.insert(vertices.end(), cube_vertices.begin(),
                        cube_vertices.end());

        // Initialize colors (e.g., white for now)
        for (int i = 0; i < 8; ++i) {
          colors.push_back(1.0f);  // R
          colors.push_back(1.0f);  // G
          colors.push_back(1.0f);  // B
          colors.push_back(alpha); // A (alpha value for transparency)
        }

        // Define the 12 triangles (36 indices) for the 6 faces of the cube
        std::array<GLuint, 36> cube_indices = {
            0, 1, 2, 2, 3, 0, // Front face
            4, 5, 6, 6, 7, 4, // Back face
            0, 1, 5, 5, 4, 0, // Bottom face
            2, 3, 7, 7, 6, 2, // Top face
            0, 3, 7, 7, 4, 0, // Left face
            1, 2, 6, 6, 5, 1  // Right face
        };

        GLuint base_index = static_cast<GLuint>(vertices.size() / 3 - 8);
        for (auto index : cube_indices) {
          indices.push_back(base_index + index);
        }
      }
    }
  }

  // Generate and bind Vertex Buffer Object
  glGenBuffers(1, &vbo_vertices);
  glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices);
  glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(GLfloat),
               vertices.data(), GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0); // Unbind the VBO

  // Generate and bind Color Buffer Object
  glGenBuffers(1, &vbo_colors);
  glBindBuffer(GL_ARRAY_BUFFER, vbo_colors);
  glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(GLfloat), colors.data(),
               GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0); // Unbind the CBO

  // Generate and bind Index Buffer Object
  glGenBuffers(1, &vbo_indices);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_indices);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint),
               indices.data(), GL_STATIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0); // Unbind the IBO
}

void VolumetricDisplay::listenArtNet() {
  LOG(INFO) << "Listening for Art-Net on " << ip << ":" << port;
  while (running) {
    std::array<char, 1024> buffer;
    boost::asio::ip::udp::endpoint sender_endpoint;
    boost::system::error_code ec;
    size_t total_length = socket.receive_from(boost::asio::buffer(buffer),
                                              sender_endpoint, 0, ec);

    if (ec == boost::asio::error::operation_aborted) {
      LOG(WARNING) << "Receive operation cancelled";
      continue;
    } else if (ec) {
      if (!running) {
        // Assume that the error is due to the socket being closed.
        break;
      }
      LOG(ERROR) << "Error receiving data: " << ec.message();
      break;
    }

    if (total_length < 10) {
      LOG(ERROR) << "Received packet too short (" << total_length << " B)";
      continue;
    }

    if (strncmp(buffer.data(), "Art-Net\0", 8) != 0) {
      LOG(ERROR) << "Received non-Art-Net packet";
      continue;
    }

    uint16_t opcode = *reinterpret_cast<uint16_t *>(&buffer[8]);

    if (opcode == 0x5000) { // DMX Data
      uint16_t universe = *reinterpret_cast<uint16_t *>(&buffer[14]);
      uint16_t length = ntohs(*reinterpret_cast<uint16_t *>(&buffer[16]));

      VLOG(1) << "Received DMX Data packet of length " << length
              << " for universe " << universe;
      int layer = universe / universes_per_layer;
      int universe_in_layer = universe % universes_per_layer;
      int start_pixel = universe_in_layer * 170;

      auto lg = std::lock_guard(pixels_mu);
      for (size_t i = 0;
           i < length &&
           (start_pixel + static_cast<int>(i) / 3) < width * height &&
           (18 + i + 2 < total_length);
           i += 3) {
        int idx = start_pixel + i / 3;
        int x = idx % width;
        int y = idx / width;
        size_t pixel_index =
            static_cast<size_t>(x + y * width + layer * width * height);
        if (pixel_index >= pixels.size()) {
          continue;
        }
        pixels[pixel_index] = {(unsigned char)buffer[18 + i],
                               (unsigned char)buffer[18 + i + 1],
                               (unsigned char)buffer[18 + i + 2]};
      }
      view_update.notify_all();
    } else if (opcode == 0x5200) {
      VLOG(1) << "Received sync packet";
      needs_update = true;
    } else {
      LOG(ERROR) << "Received unknown opcode: " << opcode;
    }
  }
}

void VolumetricDisplay::updateColors() {
  glBindBuffer(GL_ARRAY_BUFFER, vbo_colors);
  std::vector<GLfloat> colors;

  auto lg = std::unique_lock(pixels_mu);

  view_update.wait_for(lg, std::chrono::milliseconds(100));

  for (auto pixel : pixels) {
    color_corrector_.ReverseCorrectInPlace(pixel.data());
    GLfloat r = pixel[0] / 255.0f;
    GLfloat g = pixel[1] / 255.0f;
    GLfloat b = pixel[2] / 255.0f;
    GLfloat a = alpha;

    // If the color is black, make it transparent
    if (r == 0.0f && g == 0.0f && b == 0.0f) {
      a = 0.0f;
    }
    for (int i = 0; i < 8; ++i) {
      colors.push_back(r);
      colors.push_back(g);
      colors.push_back(b);
      colors.push_back(a);
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
  socket.close(); // Close socket first to unblock the receive operation
  if (artnet_thread.joinable()) {
    artnet_thread.join();
  }
  glfwTerminate();
}

void VolumetricDisplay::framebufferSizeCallback(GLFWwindow *window, int width,
                                                int height) {
  glViewport(0, 0, width, height);
  viewport_width = width;
  viewport_height = height;
  viewport_aspect =
      static_cast<float>(viewport_width) / static_cast<float>(viewport_height);

  view_update.notify_all();
}

void VolumetricDisplay::updateCamera() {
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  gluPerspective(45.0f, viewport_aspect, 0.1f, 100.0f);
  glTranslatef(0, 0, -camera_distance);
  glTranslatef(camera_position.x, camera_position.y, camera_position.z);
  glm::mat4 rotation_matrix = glm::toMat4(camera_orientation);
  glMultMatrixf(glm::value_ptr(rotation_matrix));
  glTranslatef(-width / 2.0f, -height / 2.0f, -length / 2.0f);
}

void VolumetricDisplay::render() {
  // Calculate delta time
  double current_time = glfwGetTime();
  double delta_time = current_time - last_frame_time;
  last_frame_time = current_time;

  // Apply continuous rotation
  if (glm::length(rotation_rate) > 0.0f) { // Only rotate if rate is non-zero
    float angle_x =
        glm::radians(rotation_rate.x * static_cast<float>(delta_time));
    float angle_y =
        glm::radians(rotation_rate.y * static_cast<float>(delta_time));
    float angle_z =
        glm::radians(rotation_rate.z * static_cast<float>(delta_time));

    // Create rotation quaternions for each axis
    glm::quat rot_x = glm::angleAxis(angle_x, glm::vec3(1.0f, 0.0f, 0.0f));
    glm::quat rot_y = glm::angleAxis(angle_y, glm::vec3(0.0f, 1.0f, 0.0f));
    glm::quat rot_z = glm::angleAxis(angle_z, glm::vec3(0.0f, 0.0f, 1.0f));

    // Combine rotations and apply to camera orientation
    // Apply in ZYX order to match common conventions, though order might need
    // adjustment
    camera_orientation = rot_z * rot_y * rot_x * camera_orientation;
    camera_orientation =
        glm::normalize(camera_orientation); // Normalize to prevent drift
  }

  updateColors();

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  gluPerspective(45.0f, viewport_aspect, 0.1f, 100.0f);
  updateCamera();

  glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices);
  glEnableClientState(GL_VERTEX_ARRAY);
  glVertexPointer(3, GL_FLOAT, 0, nullptr);

  glBindBuffer(GL_ARRAY_BUFFER, vbo_colors);
  glEnableClientState(GL_COLOR_ARRAY);
  glColorPointer(4, GL_FLOAT, 0, nullptr);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_indices);
  glDrawElements(GL_TRIANGLES, vertex_count, GL_UNSIGNED_INT, nullptr);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

  if (show_axis) {
    // Push the transform
    glPushMatrix();
    glTranslatef(-1, -1, -1);
    glBegin(GL_LINES);
    // X axis (red)
    glColor3f(10.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(5.0f, 0.0f, 0.0f);
    // Y axis (green)
    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 5.0f, 0.0f);
    // Z axis (blue)
    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 5.0f);
    glEnd();
    // Pop the transform
    glPopMatrix();
  }

  if (show_wireframe) {
    drawWireframeCube();
  }

  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_COLOR_ARRAY);

  glfwSwapBuffers(glfwGetCurrentContext());
}

void VolumetricDisplay::keyCallback(GLFWwindow *window, int key, int scancode,
                                    int action, int mods) {
  if (key == GLFW_KEY_A && action == GLFW_PRESS) {
    show_axis = !show_axis;
  }

  if (key == GLFW_KEY_B && action == GLFW_PRESS) {
    show_wireframe = !show_wireframe;
  }

  view_update.notify_all();
}

void VolumetricDisplay::rotate(float angle, float x, float y, float z) {
  glm::vec3 axis(x, y, z);
  glm::quat rotation = glm::angleAxis(glm::radians(angle), axis);
  camera_orientation = rotation * camera_orientation;
}

void VolumetricDisplay::windowCloseCallback(GLFWwindow *window) {
  VLOG(0) << "Window closed";
  running = false;
  socket.close(); // Close socket first to unblock the receive operation
  if (artnet_thread.joinable()) {
    artnet_thread.join();
  }

  view_update.notify_all();
}

void VolumetricDisplay::mouseButtonCallback(GLFWwindow *window, int button,
                                            int action, int mods) {
  if (action == GLFW_PRESS) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
      if (mods & GLFW_MOD_SHIFT) {
        right_mouse_button_pressed = true;
      } else {
        left_mouse_button_pressed = true;
      }
    }
  } else if (action == GLFW_RELEASE) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
      right_mouse_button_pressed = false;
      left_mouse_button_pressed = false;
    }
  }

  view_update.notify_all();
}

void VolumetricDisplay::cursorPositionCallback(GLFWwindow *window, double xpos,
                                               double ypos) {
  if (left_mouse_button_pressed) {
    float dx = static_cast<float>(xpos - last_mouse_x);
    float dy = static_cast<float>(ypos - last_mouse_y);
    rotate(dx * 0.2f, 0.0f, 1.0f, 0.0f);
    rotate(dy * 0.2f, 1.0f, 0.0f, 0.0f);
  } else if (right_mouse_button_pressed) {
    float dx = static_cast<float>(xpos - last_mouse_x);
    float dy = static_cast<float>(ypos - last_mouse_y);
    camera_position -= glm::vec3(-dx * 0.05f, dy * 0.05f, 0.0f);
  }
  last_mouse_x = xpos;
  last_mouse_y = ypos;

  view_update.notify_all();
}

void VolumetricDisplay::scrollCallback(GLFWwindow *window, double xoffset,
                                       double yoffset) {
  camera_distance -= static_cast<float>(yoffset);
  if (camera_distance < 1.0f) {
    camera_distance = 1.0f;
  }

  view_update.notify_all();
}
