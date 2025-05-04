#ifndef VOLUMETRIC_DISPLAY_H
#define VOLUMETRIC_DISPLAY_H

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <array>
#include <atomic>
#include <boost/asio.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include <mutex>
#include <thread>
#include <vector>

#include "color_correction.h"

// Define the maximum universes per layer
constexpr int MAX_UNIVERSES_PER_LAYER = 10;

class VolumetricDisplay {
public:
  VolumetricDisplay(int width, int height, int length, const std::string &ip,
                    int port, int universes_per_layer, int layer_span,
                    float alpha, const glm::vec3 &initial_rotation_rate);
  ~VolumetricDisplay();

  void run();
  void cleanup();

private:
  void setupOpenGL();
  void setupVBO();
  void listenArtNet();
  void updateColors();
  void render();
  void processInput(GLFWwindow *window);
  void windowCloseCallback(GLFWwindow *window);
  void mouseButtonCallback(GLFWwindow *window, int button, int action,
                           int mods);
  void cursorPositionCallback(GLFWwindow *window, double xpos, double ypos);
  void scrollCallback(GLFWwindow *window, double xoffset, double yoffset);
  void rotate(float angle, float x, float y, float z);
  void keyCallback(GLFWwindow *window, int key, int scancode, int action,
                   int mods);
  void framebufferSizeCallback(GLFWwindow *window, int width, int height);
  void updateCamera();
  void drawWireframeCube();

  glm::vec3 camera_position;
  glm::quat camera_orientation;
  float camera_distance;
  bool left_mouse_button_pressed;
  bool right_mouse_button_pressed;
  double last_mouse_x;
  double last_mouse_y;

  double last_frame_time = 0.0;

  int viewport_width, viewport_height;
  float viewport_aspect = 1.0f;
  int width, height, length;
  std::string ip;
  int port;
  int universes_per_layer;
  int layer_span;

  std::mutex pixels_mu;
  std::condition_variable view_update;
  std::vector<std::array<unsigned char, 3>> pixels; // RGB for each voxel
  float alpha;
  std::atomic<bool> running;
  bool show_axis;
  bool show_wireframe;
  std::atomic<bool> needs_update;
  std::thread artnet_thread;

  GLuint vbo_vertices;
  GLuint vbo_colors;
  size_t vertex_count;
  GLuint vbo_indices;

  glm::mat4 rotation_matrix;
  glm::mat4 temp_matrix;
  glm::vec3 rotation_rate; // Degrees per second for X, Y, Z

  boost::asio::io_service io_service;
  boost::asio::ip::udp::socket socket;

  util::ReverseColorCorrector<3> color_corrector_{
      util::kColorCorrectorWs2812bOptions};
};

#endif // VOLUMETRIC_DISPLAY_H
