#ifndef VOLUMETRIC_DISPLAY_H
#define VOLUMETRIC_DISPLAY_H

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <array>
#include <atomic>
#include <boost/asio.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "DisplayConfig.h"
#include "color_correction.h"

struct VoxelColor { unsigned char r, g, b; };

struct ListenerThreadInfo {
    std::string ip;
    int port;
    int cube_index;
    std::vector<int> z_indices;
};

class VolumetricDisplay {
public:
    VolumetricDisplay(int width, int height, int length,
                      int universes_per_layer, int layer_span, float alpha,
                      const glm::vec3& initial_rotation_rate, bool color_correction_enabled,
                      const std::vector<CubeConfig>& cubes_config,
                      float voxel_scale);
    ~VolumetricDisplay();
    void run();

private:
    // Setup & Cleanup
    void setupOpenGL();
    void setupShaders();
    void setupWireframeShader();
    void setupAxesShader();
    void setupVBO();
    void cleanup();

    GLuint compileShader(GLenum type, const char* source);

    // Main Loop & Rendering Helpers
    void render();
    void updateColors();
    void drawWireframeCubes();
    void drawAxes();
    void rotate(float angle, float x, float y, float z);
    glm::vec3 calculateSceneCenter();  // ADD THIS LINE HERE

    // Networking
    void listenArtNet(int listener_index);

    // Transform matrix computation functions for cube orientation
    glm::mat4 computeCubeLocalTransformMatrix(const std::vector<std::string>& world_orientation, const glm::vec3& size);
    glm::mat4 computeCubeToWorldTransformMatrix(const std::vector<std::string>& world_orientation, const glm::vec3& cube_position);

    // GLFW Callbacks (must be static)
    void framebufferSizeCallback(GLFWwindow* window, int width, int height);
    void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    void cursorPositionCallback(GLFWwindow* window, double xpos, double ypos);
    void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
    void windowCloseCallback(GLFWwindow* window);

    // Member Variables
    GLFWwindow* window_;
    int width, height, length;
    int universes_per_layer, layer_span;
    size_t num_voxels;

    GLuint vao, vbo_vertices, vbo_indices, vbo_instance_positions, vbo_instance_colors;
    GLuint wireframe_vao, wireframe_vbo, wireframe_ebo;
    GLuint axis_vao, axis_vbo;
    GLuint shader_program, wireframe_shader_program, axis_shader_program;
    size_t vertex_count;
    float alpha, voxel_scale;
    bool show_axis = false;
    bool show_wireframe = false;
    std::atomic<bool> needs_update;

    glm::vec3 camera_position{0.0f, 0.0f, 0.0f};
    glm::mat4 rotation_matrix;
    glm::mat4 temp_matrix;
    glm::vec3 rotation_rate;
    glm::quat camera_orientation;
    float camera_distance;
    bool left_mouse_button_pressed = false;
    bool right_mouse_button_pressed = false;
    double last_mouse_x, last_mouse_y;
    int viewport_width, viewport_height;
    float viewport_aspect = 1.0f;
    double last_frame_time = 0.0;

    std::atomic<bool> running;
    std::mutex pixels_mu;
    std::condition_variable view_update;
    std::vector<VoxelColor> pixels;

    boost::asio::io_context io_context;
    std::vector<std::thread> artnet_threads;
    std::vector<std::unique_ptr<boost::asio::ip::udp::socket>> sockets;
    std::vector<CubeConfig> cubes_config_;
    std::vector<ListenerThreadInfo> listener_info_;

    bool color_correction_enabled_;
    util::ReverseColorCorrector<3> color_corrector_{util::kColorCorrectorWs2812bOptions};

    // Transform matrices for each cube (computed once and reused)
    std::vector<glm::mat4> cube_local_transforms_;
    std::vector<glm::mat4> cube_world_transforms_;
};

#endif // VOLUMETRIC_DISPLAY_H
