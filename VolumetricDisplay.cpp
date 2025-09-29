#include "VolumetricDisplay.h"
#include "absl/log/log.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <algorithm>
#include <array>
#include <arpa/inet.h>
#include <stdexcept>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>

// Voxel Shaders
const char* vertex_shader_source = R"glsl(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aInstancePosition;
    layout (location = 2) in vec4 aInstanceColor;

    out vec4 fColor;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    // Voxel scale is now 1.0, but we keep the uniform for flexibility.
    uniform float voxel_scale;

    void main()
    {
        // A voxel is a 1x1x1 cube centered at its instance position.
        vec3 scaled_pos = aPos * voxel_scale;
        gl_Position = projection * view * model * vec4(scaled_pos + aInstancePosition, 1.0);
        fColor = aInstanceColor;
    }
)glsl";

const char* fragment_shader_source = R"glsl(
    #version 330 core
    in vec4 fColor;
    out vec4 FragColor;

    void main()
    {
        if(fColor.a == 0.0)
            discard; // Discard transparent fragments
        FragColor = fColor;
    }
)glsl";

// Wireframe Shaders
const char* wireframe_vertex_shader_source = R"glsl(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    void main()
    {
        gl_Position = projection * view * model * vec4(aPos, 1.0);
    }
)glsl";

const char* wireframe_fragment_shader_source = R"glsl(
    #version 330 core
    out vec4 FragColor;
    uniform vec3 color;
    void main()
    {
        FragColor = vec4(color, 1.0f);
    }
)glsl";

// Axis Shaders
const char* simple_vertex_shader_source = R"glsl(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aColor;

    out vec3 fColor;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    void main()
    {
        gl_Position = projection * view * model * vec4(aPos, 1.0);
        fColor = aColor;
    }
)glsl";

const char* simple_fragment_shader_source = R"glsl(
    #version 330 core
    in vec3 fColor;
    out vec4 FragColor;

    void main()
    {
        FragColor = vec4(fColor, 1.0);
    }
)glsl";

VolumetricDisplay::VolumetricDisplay(int width, int height, int length,
                                     int universes_per_layer, int layer_span,
                                     float alpha,
                                     const glm::vec3 &initial_rotation_rate, bool color_correction_enabled,
                                     const std::vector<CubeConfig>& cubes_config,
                                     float voxel_scale)
    : width(width), height(height), length(length),
      universes_per_layer(universes_per_layer), layer_span(layer_span),
      alpha(alpha), voxel_scale(voxel_scale),
      show_axis(false), show_wireframe(false), needs_update(false),
      rotation_rate(initial_rotation_rate),
      running(false),
      pixels(),
      cubes_config_(cubes_config),
      color_correction_enabled_(color_correction_enabled) {

    if (cubes_config_.empty()) {
        throw std::runtime_error("Cube configuration cannot be empty.");
    }

    // Calculate total voxels using individual cube dimensions
    num_voxels = 0;
    for (const auto& cube_cfg : cubes_config_) {
        num_voxels += static_cast<size_t>(cube_cfg.width) * cube_cfg.height * (cube_cfg.length / layer_span);
    }
    pixels.resize(num_voxels, {0, 0, 0});

    running = true;
    rotation_matrix = glm::mat4(1.0f);
    temp_matrix = glm::mat4(1.0f);

    setupOpenGL();

    glm::quat rot_x = glm::angleAxis(glm::radians(45.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    glm::quat rot_y = glm::angleAxis(glm::radians(-35.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    camera_orientation = rot_y * rot_x;
    camera_position = glm::vec3(0.0f, 0.0f, 0.0f);
    camera_distance = std::max({(float)width, (float)height, (float)length}) * 3.0f;
    last_mouse_x = 0.0;
    last_mouse_y = 0.0;

    setupShaders();
    setupWireframeShader();
    setupAxesShader();
    setupVBO();

    LOG(INFO) << "Initializing " << listener_info_.size() << " Art-Net listener threads...";
    for (size_t i = 0; i < cubes_config_.size(); ++i) {
        for (const auto& listener_cfg : cubes_config_[i].listeners) {
            listener_info_.push_back({
            listener_cfg.ip,
            listener_cfg.port,
            static_cast<int>(i),
            listener_cfg.z_indices
        });
        }
    }

    for (size_t i = 0; i < listener_info_.size(); ++i) {
        const auto& info = listener_info_[i];
        try {
            auto socket = std::make_unique<boost::asio::ip::udp::socket>(io_context);
            socket->open(boost::asio::ip::udp::v4());
            socket->bind(boost::asio::ip::udp::endpoint(boost::asio::ip::make_address(info.ip), info.port));
            sockets.push_back(std::move(socket));
            artnet_threads.emplace_back(&VolumetricDisplay::listenArtNet, this, i);
        } catch (const boost::system::system_error& e) {
            LOG(FATAL) << "Failed to bind socket to " << info.ip << ":" << info.port << " - " << e.what();
            throw;
        }
    }
}


VolumetricDisplay::~VolumetricDisplay() { cleanup(); }

GLuint VolumetricDisplay::compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    int success;
    char infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        LOG(FATAL) << "Shader compilation failed: " << infoLog;
    }
    return shader;
}

void VolumetricDisplay::setupShaders() {
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertex_shader_source);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragment_shader_source);
    shader_program = glCreateProgram();
    glAttachShader(shader_program, vertexShader);
    glAttachShader(shader_program, fragmentShader);
    glLinkProgram(shader_program);
    int success;
    glGetProgramiv(shader_program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(shader_program, 512, NULL, infoLog);
        LOG(FATAL) << "Shader linking failed: " << infoLog;
    }
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

void VolumetricDisplay::setupWireframeShader() {
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, wireframe_vertex_shader_source);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, wireframe_fragment_shader_source);
    wireframe_shader_program = glCreateProgram();
    glAttachShader(wireframe_shader_program, vertexShader);
    glAttachShader(wireframe_shader_program, fragmentShader);
    glLinkProgram(wireframe_shader_program);
    int success;
    glGetProgramiv(wireframe_shader_program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(wireframe_shader_program, 512, NULL, infoLog);
        LOG(FATAL) << "Wireframe shader linking failed: " << infoLog;
    }
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

void VolumetricDisplay::setupAxesShader() {
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, simple_vertex_shader_source);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, simple_fragment_shader_source);
    axis_shader_program = glCreateProgram();
    glAttachShader(axis_shader_program, vertexShader);
    glAttachShader(axis_shader_program, fragmentShader);
    glLinkProgram(axis_shader_program);
    int success;
    glGetProgramiv(axis_shader_program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(axis_shader_program, 512, NULL, infoLog);
        LOG(FATAL) << "Axis shader linking failed: " << infoLog;
    }
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

void VolumetricDisplay::drawWireframeCubes() {
    glUseProgram(wireframe_shader_program);

    // Calculate the center for consistent view matrix
    glm::vec3 scene_center = calculateSceneCenter();

    glm::mat4 view = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -camera_distance)) *
                     glm::translate(glm::mat4(1.0f), camera_position) *
                     glm::toMat4(camera_orientation) *
                     glm::translate(glm::mat4(1.0f), -scene_center);

    glm::mat4 projection = glm::perspective(glm::radians(45.0f), viewport_aspect, 0.1f, 500.0f);

    glUniformMatrix4fv(glGetUniformLocation(wireframe_shader_program, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(wireframe_shader_program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
    glUniform3f(glGetUniformLocation(wireframe_shader_program, "color"), 1.0f, 1.0f, 1.0f);

    glBindVertexArray(wireframe_vao);

    for (const auto& cube_cfg : cubes_config_) {
        glm::mat4 scale_matrix = glm::scale(glm::mat4(1.0f), glm::vec3(cube_cfg.width, cube_cfg.height, cube_cfg.length));
        glm::vec3 center_offset(cube_cfg.width / 2.0f, cube_cfg.height / 2.0f, cube_cfg.length / 2.0f);
        glm::mat4 trans_matrix = glm::translate(glm::mat4(1.0f), cube_cfg.position + center_offset);

        glm::mat4 model = trans_matrix * scale_matrix;

        glUniformMatrix4fv(glGetUniformLocation(wireframe_shader_program, "model"), 1, GL_FALSE, glm::value_ptr(model));
        glDrawElements(GL_LINES, 24, GL_UNSIGNED_INT, 0);
    }
    glBindVertexArray(0);
}

void VolumetricDisplay::drawAxes() {
    glUseProgram(axis_shader_program);
    glLineWidth(2.0f);

    glm::vec3 scene_center = calculateSceneCenter();

    // Use the same view matrix as wireframes so axes move with the scene
    glm::mat4 view = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -camera_distance)) *
                     glm::translate(glm::mat4(1.0f), camera_position) *
                     glm::toMat4(camera_orientation) *
                     glm::translate(glm::mat4(1.0f), -scene_center);

    glm::mat4 projection = glm::perspective(glm::radians(45.0f), viewport_aspect, 0.1f, 500.0f);

    glUniformMatrix4fv(glGetUniformLocation(axis_shader_program, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(axis_shader_program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

    glBindVertexArray(axis_vao);

    // First, draw the world coordinate axis widget
    glm::vec3 min_bounds = cubes_config_[0].position;
    glm::vec3 max_bounds = cubes_config_[0].position + glm::vec3(cubes_config_[0].width, cubes_config_[0].height, cubes_config_[0].length);

    for (const auto& cube_cfg : cubes_config_) {
        glm::vec3 cube_min = cube_cfg.position;
        glm::vec3 cube_max = cube_cfg.position + glm::vec3(cube_cfg.width, cube_cfg.height, cube_cfg.length);

        min_bounds = glm::min(min_bounds, cube_min);
        max_bounds = glm::max(max_bounds, cube_max);
    }

    // Calculate world axis length and offset
    float world_axis_length = std::min({max_bounds.x - min_bounds.x, max_bounds.y - min_bounds.y, max_bounds.z - min_bounds.z}) * 0.3f;
    float world_offset = world_axis_length * 0.5f;
    glm::vec3 world_axis_position = min_bounds - glm::vec3(world_offset, world_offset, world_offset);

    // Draw world coordinate axis widget
    glm::mat4 world_model = glm::translate(glm::mat4(1.0f), world_axis_position) *
                            glm::scale(glm::mat4(1.0f), glm::vec3(world_axis_length));
    glUniformMatrix4fv(glGetUniformLocation(axis_shader_program, "model"), 1, GL_FALSE, glm::value_ptr(world_model));
    glDrawArrays(GL_LINES, 0, 6); // 3 lines (X, Y, Z) * 2 vertices each

    // Then, draw per-cube axis widgets using the same transforms as voxels
    for (size_t cube_idx = 0; cube_idx < cubes_config_.size(); ++cube_idx) {
        const auto& cube_cfg = cubes_config_[cube_idx];

        // Calculate axis length based on cube size
        float axis_length = std::min({(float)cube_cfg.width, (float)cube_cfg.height, (float)cube_cfg.length}) * 0.2f;

        // Get the transform matrices for this cube
        glm::mat4 local_transform = cube_local_transforms_[cube_idx];
        glm::mat4 world_transform = cube_world_transforms_[cube_idx];

        // Define the axis widget points in local space (unit vectors along X, Y, Z axes)
        // Each axis is defined by two points: origin and the axis direction
        std::vector<glm::vec3> local_axis_points = {
            glm::vec3(0.0f, 0.0f, 0.0f), // X-axis start
            glm::vec3(1.0f, 0.0f, 0.0f), // X-axis end
            glm::vec3(0.0f, 0.0f, 0.0f), // Y-axis start
            glm::vec3(0.0f, 1.0f, 0.0f), // Y-axis end
            glm::vec3(0.0f, 0.0f, 0.0f), // Z-axis start
            glm::vec3(0.0f, 0.0f, 1.0f)  // Z-axis end
        };

        // Transform each axis point using the same transforms as voxels
        std::vector<glm::vec3> transformed_axis_points;
        for (const auto& local_point : local_axis_points) {
            // Apply local transform first, then world transform
            glm::vec4 transformed_local = local_transform * glm::vec4(local_point, 1.0f);
            glm::vec4 world_point = world_transform * transformed_local;
            transformed_axis_points.push_back(glm::vec3(world_point));
        }

        // Scale the transformed points by axis length
        for (auto& point : transformed_axis_points) {
            point *= axis_length;
        }

        // Position the axis widget with a small offset from the cube's origin
        // The world transform already includes the cube position, so we just add a small offset
        float offset = axis_length * 0.3f;
        glm::vec3 axis_offset = glm::vec3(-offset, -offset, -offset);

        // Apply the offset to all points
        for (auto& point : transformed_axis_points) {
            point += axis_offset;
        }

        // Create a model matrix that just applies the final positioning
        glm::mat4 model = glm::mat4(1.0f);

        glUniformMatrix4fv(glGetUniformLocation(axis_shader_program, "model"), 1, GL_FALSE, glm::value_ptr(model));

        // Draw the transformed axis widget by drawing individual line segments
        // We need to draw 3 lines: X, Y, Z axes
        for (int axis = 0; axis < 3; ++axis) {
            int start_idx = axis * 2;
            int end_idx = start_idx + 1;

            // Create temporary vertices for this line segment
            GLfloat line_vertices[] = {
                transformed_axis_points[start_idx].x, transformed_axis_points[start_idx].y, transformed_axis_points[start_idx].z,
                transformed_axis_points[end_idx].x, transformed_axis_points[end_idx].y, transformed_axis_points[end_idx].z
            };

            // Create temporary colors for this line segment
            GLfloat line_colors[] = {
                (axis == 0) ? 1.0f : 0.0f, (axis == 1) ? 1.0f : 0.0f, (axis == 2) ? 1.0f : 0.0f, // Start color
                (axis == 0) ? 1.0f : 0.0f, (axis == 1) ? 1.0f : 0.0f, (axis == 2) ? 1.0f : 0.0f  // End color
            };

            // Create temporary VAO for this line
            GLuint temp_vao, temp_vbo;
            glGenVertexArrays(1, &temp_vao);
            glGenBuffers(1, &temp_vbo);

            glBindVertexArray(temp_vao);
            glBindBuffer(GL_ARRAY_BUFFER, temp_vbo);
            glBufferData(GL_ARRAY_BUFFER, sizeof(line_vertices) + sizeof(line_colors), nullptr, GL_STATIC_DRAW);
            glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(line_vertices), line_vertices);
            glBufferSubData(GL_ARRAY_BUFFER, sizeof(line_vertices), sizeof(line_colors), line_colors);

            // Position attribute
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)0);
            glEnableVertexAttribArray(0);
            // Color attribute
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)sizeof(line_vertices));
            glEnableVertexAttribArray(1);

            // Draw the line
            glDrawArrays(GL_LINES, 0, 2);

            // Clean up temporary VAO
            glDeleteVertexArrays(1, &temp_vao);
            glDeleteBuffers(1, &temp_vbo);
        }
    }

    glBindVertexArray(0);
}

void VolumetricDisplay::setupOpenGL() {
  if (!glfwInit()) {
    throw std::runtime_error("Failed to initialize GLFW");
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

  GLFWwindow *window =
      glfwCreateWindow(800, 800, "Volumetric Display", nullptr, nullptr);
  if (!window) {
    glfwTerminate();
    throw std::runtime_error("Failed to create GLFW window");
  }
  glfwMakeContextCurrent(window);
  glewExperimental = GL_TRUE;
  if (glewInit() != GLEW_OK) {
      throw std::runtime_error("Failed to initialize GLEW");
  }

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

  glfwGetFramebufferSize(window, &viewport_width, &viewport_height);
  glViewport(0, 0, viewport_width, viewport_height);
  viewport_aspect =
      static_cast<float>(viewport_width) / static_cast<float>(viewport_height);

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void VolumetricDisplay::setupVBO() {
    // VOXEL SETUP
    // Calculate total voxels using individual cube dimensions
    num_voxels = 0;
    for (const auto& cube_cfg : cubes_config_) {
        num_voxels += static_cast<size_t>(cube_cfg.width) * cube_cfg.height * (cube_cfg.length / layer_span);
    }

    GLfloat vertices[] = {
        -0.5f, -0.5f,  0.5f, 0.5f, -0.5f,  0.5f, 0.5f,  0.5f,  0.5f, -0.5f,  0.5f,  0.5f,
        -0.5f, -0.5f, -0.5f, 0.5f, -0.5f, -0.5f, 0.5f,  0.5f, -0.5f, -0.5f,  0.5f, -0.5f,
    };
    GLuint indices[] = {
        0, 1, 2, 2, 3, 0, 1, 5, 6, 6, 2, 1, 5, 4, 7, 7, 6, 5,
        4, 0, 3, 3, 7, 4, 3, 2, 6, 6, 7, 3, 4, 5, 1, 1, 0, 4
    };
    vertex_count = 36;

    // For GPU rendering, we'll store the transform matrices per cube
    // and let the vertex shader apply the transforms
    std::vector<glm::vec3> instance_positions(num_voxels);

    // Compute transform matrices for each cube and store as class members
    cube_local_transforms_.clear();
    cube_world_transforms_.clear();
    for (const auto& cube_cfg : cubes_config_) {
        glm::mat4 local_transform = computeCubeLocalTransformMatrix(cube_cfg.world_orientation, glm::vec3(cube_cfg.width, cube_cfg.height, cube_cfg.length));
        glm::mat4 world_transform = computeCubeToWorldTransformMatrix(cube_cfg.world_orientation, cube_cfg.position);

        cube_local_transforms_.push_back(local_transform);
        cube_world_transforms_.push_back(world_transform);
    }

    // For now, compute positions on CPU (will be moved to GPU later)
    size_t i = 0;
    int cube_index = 0;
    for (const auto& cube_cfg : cubes_config_) {
        glm::mat4 local_transform = cube_local_transforms_[cube_index];
        glm::mat4 world_transform = cube_world_transforms_[cube_index];

        for (int z = 0; z < cube_cfg.length; z += layer_span) {
            for (int y = 0; y < cube_cfg.height; ++y) {
                for (int x = 0; x < cube_cfg.width; ++x) {
                    if (i < num_voxels) {
                        // Start with local voxel position
                        glm::vec3 local_pos = glm::vec3(x + 0.5f, y + 0.5f, z + 0.5f);

                        // Apply transforms using matrices
                        glm::vec4 transformed_local_pos = local_transform * glm::vec4(local_pos, 1.0f);
                        glm::vec4 world_pos = world_transform * transformed_local_pos;

                        instance_positions[i++] = glm::vec3(world_pos);
                    }
                }
            }
        }
        cube_index++;
    }

    std::vector<glm::vec4> instance_colors(num_voxels, glm::vec4(0.0f, 0.0f, 0.0f, 0.0f)); // Use vec4 for RGBA

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &vbo_vertices);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)0);
    glEnableVertexAttribArray(0);

    glGenBuffers(1, &vbo_indices);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_indices);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glGenBuffers(1, &vbo_instance_positions);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_instance_positions);
    glBufferData(GL_ARRAY_BUFFER, num_voxels * sizeof(glm::vec3), &instance_positions[0], GL_STATIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribDivisor(1, 1);

    glGenBuffers(1, &vbo_instance_colors);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_instance_colors);
    glBufferData(GL_ARRAY_BUFFER, num_voxels * sizeof(glm::vec4), &instance_colors[0], GL_DYNAMIC_DRAW);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (void*)0); // Changed size to 4
    glEnableVertexAttribArray(2);
    glVertexAttribDivisor(2, 1);

    // WIREFRAME SETUP
    GLfloat wireframe_vertices[] = {
        -0.5f, -0.5f, -0.5f,  0.5f, -0.5f, -0.5f,
         0.5f,  0.5f, -0.5f, -0.5f,  0.5f, -0.5f,
        -0.5f, -0.5f,  0.5f,  0.5f, -0.5f,  0.5f,
         0.5f,  0.5f,  0.5f, -0.5f,  0.5f,  0.5f,
    };

    GLuint wireframe_indices[] = {
        0, 1, 1, 2, 2, 3, 3, 0, // Bottom face
        4, 5, 5, 6, 6, 7, 7, 4, // Top face
        0, 4, 1, 5, 2, 6, 3, 7  // Connecting lines
    };

    glGenVertexArrays(1, &wireframe_vao);
    glBindVertexArray(wireframe_vao);

    glGenBuffers(1, &wireframe_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, wireframe_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(wireframe_vertices), wireframe_vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)0);
    glEnableVertexAttribArray(0);

    glGenBuffers(1, &wireframe_ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, wireframe_ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(wireframe_indices), wireframe_indices, GL_STATIC_DRAW);

    // AXES SETUP
    GLfloat axis_vertices[] = {
        // Position      // Color
        0.0f, 0.0f, 0.0f,  1.0f, 0.0f, 0.0f, // X-axis Start (Red)
        1.0f, 0.0f, 0.0f,  1.0f, 0.0f, 0.0f, // X-axis End

        0.0f, 0.0f, 0.0f,  0.0f, 1.0f, 0.0f, // Y-axis Start (Green)
        0.0f, 1.0f, 0.0f,  0.0f, 1.0f, 0.0f, // Y-axis End

        0.0f, 0.0f, 0.0f,  0.0f, 0.0f, 1.0f, // Z-axis Start (Blue)
        0.0f, 0.0f, 1.0f,  0.0f, 0.0f, 1.0f, // Z-axis End
    };

    glGenVertexArrays(1, &axis_vao);
    glGenBuffers(1, &axis_vbo);

    glBindVertexArray(axis_vao);
    glBindBuffer(GL_ARRAY_BUFFER, axis_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(axis_vertices), axis_vertices, GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (void*)0);
    glEnableVertexAttribArray(0);
    // Color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (void*)(3* sizeof(GLfloat)));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void VolumetricDisplay::listenArtNet(int listener_index) {
    // This thread is responsible for ONE specific socket (IP:Port).
    // It looks up its own configuration using its index.
    const auto& info = listener_info_[listener_index];
    LOG(INFO) << "Thread started for cube " << info.cube_index << " on " << info.ip << ":" << info.port;

    // This listener only ever writes to the pixel buffer for its assigned cube.
    // Calculate pixels per cube using individual cube dimensions
    size_t pixels_per_cube = 0;
    size_t pixel_buffer_offset = 0;

    // Calculate offset by summing up all previous cubes' pixel counts
    for (int i = 0; i < info.cube_index; i++) {
        const auto& cube_cfg = cubes_config_[i];
        pixels_per_cube += static_cast<size_t>(cube_cfg.width) * cube_cfg.height * cube_cfg.length;
    }
    pixel_buffer_offset = pixels_per_cube;

    // Get the current cube's dimensions
    const auto& current_cube_cfg = cubes_config_[info.cube_index];
    pixels_per_cube = static_cast<size_t>(current_cube_cfg.width) * current_cube_cfg.height * current_cube_cfg.length;

    while (running) {
        std::array<char, 1024> buffer;
        boost::asio::ip::udp::endpoint sender_endpoint;
        boost::system::error_code ec;

        // Use the correct socket from the vector
        size_t total_length = sockets[listener_index]->receive_from(boost::asio::buffer(buffer),
                                                      sender_endpoint, 0, ec);

        if (ec == boost::asio::error::operation_aborted || !running) {
            break; // Exit thread if socket is closed or app is shutting down
        } else if (ec) {
            LOG(ERROR) << "Receive error on " << info.ip << ":" << info.port << ": " << ec.message();
            continue;
        }

        if (total_length < 18 || strncmp(buffer.data(), "Art-Net\0", 8) != 0) {
            continue; // Not a valid Art-Net packet
        }

        uint16_t opcode = *reinterpret_cast<uint16_t *>(&buffer[8]);
        if (opcode == 0x5000) { // ArtDmx opcode
            uint16_t universe = *reinterpret_cast<uint16_t *>(&buffer[14]);
            uint16_t dmx_length = ntohs(*reinterpret_cast<uint16_t *>(&buffer[16]));
            dmx_length = std::min(dmx_length, (uint16_t)512);

            // Map universe to specific Z-index within this controller's range
            int layer = universe / universes_per_layer;

            // Check if this layer is within this controller's range
            if (layer >= info.z_indices.size()) {
                LOG(WARNING) << "Port " << info.port << " received layer " << layer
                            << " but only has " << info.z_indices.size() << " z_indices";
                continue;
            }

            int actual_z = info.z_indices[layer];  // Map to actual Z-index
            int universe_in_layer = universe % universes_per_layer;
            int start_pixel_in_layer = universe_in_layer * 170;

            auto lg = std::lock_guard(pixels_mu);
            for (size_t i = 0; i < dmx_length; i += 3) {
                if (18 + i + 2 >= total_length) break;

                int idx_in_layer = start_pixel_in_layer + i / 3;
                if (idx_in_layer >= current_cube_cfg.width * current_cube_cfg.height) {
                    continue; // Skip overflow pixels
                }

                int x = idx_in_layer % current_cube_cfg.width;
                int y = idx_in_layer / current_cube_cfg.width;

                // Write to ONE specific Z-index, not all of them
                size_t pixel_index = pixel_buffer_offset + static_cast<size_t>(x + y * current_cube_cfg.width + actual_z * current_cube_cfg.width * current_cube_cfg.height);

                if (pixel_index < pixels.size()) {
                    pixels[pixel_index] = {
                        (unsigned char)buffer[18 + i],
                        (unsigned char)buffer[18 + i + 1],
                        (unsigned char)buffer[18 + i + 2]
                    };
                }
            }
            view_update.notify_all();
        }
    }
    LOG(INFO) << "Thread stopped for " << info.ip << ":" << info.port;
}

void VolumetricDisplay::updateColors() {
    std::vector<glm::vec4> instance_colors(num_voxels);
    {
        auto lg = std::lock_guard(pixels_mu);
        for (size_t i = 0; i < num_voxels; ++i) {
            VoxelColor pixel = pixels[i]; // Make a copy

            if (color_correction_enabled_) {
                // Create a temporary array for the corrector to modify.
                std::array<unsigned char, 3> color_data = {pixel.r, pixel.g, pixel.b};
                color_corrector_.ReverseCorrectInPlace(color_data.data());
                pixel = {color_data[0], color_data[1], color_data[2]};
            }

            // Use the .r, .g, .b members of the struct.
            float r = pixel.r / 255.0f;
            float g = pixel.g / 255.0f;
            float b = pixel.b / 255.0f;

            float current_alpha = (r == 0.0f && g == 0.0f && b == 0.0f) ? 0.0f : alpha;
            instance_colors[i] = glm::vec4(r, g, b, current_alpha);
        }
    }
    glBindBuffer(GL_ARRAY_BUFFER, vbo_instance_colors);
    glBufferSubData(GL_ARRAY_BUFFER, 0, num_voxels * sizeof(glm::vec4), &instance_colors[0]);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void VolumetricDisplay::run() {
  while (running && !glfwWindowShouldClose(glfwGetCurrentContext())) {
    render();
    glfwPollEvents();
  }
}

void VolumetricDisplay::cleanup() {
  running = false;
  io_context.stop();
  for (auto& socket : sockets) {
      if (socket && socket->is_open()) {
          socket->close();
      }
  }
  for (auto& thread : artnet_threads) {
      if (thread.joinable()) {
          thread.join();
      }
  }

  if (window_) {
      glfwDestroyWindow(window_);
      window_ = nullptr;
  }
  glfwTerminate();

  /*glDeleteVertexArrays(1, &vao);
  glDeleteBuffers(1, &vbo_vertices);
  glDeleteBuffers(1, &vbo_indices);
  glDeleteBuffers(1, &vbo_instance_positions);
  glDeleteBuffers(1, &vbo_instance_colors);
  glDeleteProgram(shader_program);

  glDeleteVertexArrays(1, &wireframe_vao);
  glDeleteBuffers(1, &wireframe_vbo);
  glDeleteBuffers(1, &wireframe_ebo);
  glDeleteProgram(wireframe_shader_program);

  glfwTerminate();*/
}

void VolumetricDisplay::framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    VolumetricDisplay* display = static_cast<VolumetricDisplay*>(glfwGetWindowUserPointer(window));
    if (display) {
        glViewport(0, 0, width, height);
        display->viewport_width = width;
        display->viewport_height = height;
        display->viewport_aspect = (height == 0) ? 1.0f : static_cast<float>(width) / static_cast<float>(height);
    }
}

void VolumetricDisplay::render() {
  double current_time = glfwGetTime();
  double delta_time = current_time - last_frame_time;
  last_frame_time = current_time;

  if (glm::length(rotation_rate) > 0.0f) {
    float angle_x = glm::radians(rotation_rate.x * static_cast<float>(delta_time));
    float angle_y = glm::radians(rotation_rate.y * static_cast<float>(delta_time));
    float angle_z = glm::radians(rotation_rate.z * static_cast<float>(delta_time));
    glm::quat rot_x = glm::angleAxis(angle_x, glm::vec3(1.0f, 0.0f, 0.0f));
    glm::quat rot_y = glm::angleAxis(angle_y, glm::vec3(0.0f, 1.0f, 0.0f));
    glm::quat rot_z = glm::angleAxis(angle_z, glm::vec3(0.0f, 0.0f, 1.0f));
    camera_orientation = rot_y * rot_x * rot_z * camera_orientation;
    camera_orientation = glm::normalize(camera_orientation);
  }

  updateColors();

  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // Calculate the center of all cubes for proper rotation
  glm::vec3 scene_center = calculateSceneCenter();

  // Define View and Projection matrices with proper centering
  glm::mat4 view = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -camera_distance)) *
                   glm::translate(glm::mat4(1.0f), camera_position) *
                   glm::toMat4(camera_orientation) *
                   glm::translate(glm::mat4(1.0f), -scene_center);

  glm::mat4 projection = glm::perspective(glm::radians(45.0f), viewport_aspect, 0.1f, 500.0f);

  // Draw the voxels
  glUseProgram(shader_program);
  glUniform1f(glGetUniformLocation(shader_program, "voxel_scale"), this->voxel_scale);
  glm::mat4 model = glm::mat4(1.0f);
  glUniformMatrix4fv(glGetUniformLocation(shader_program, "model"), 1, GL_FALSE, glm::value_ptr(model));
  glUniformMatrix4fv(glGetUniformLocation(shader_program, "view"), 1, GL_FALSE, glm::value_ptr(view));
  glUniformMatrix4fv(glGetUniformLocation(shader_program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

  glBindVertexArray(vao);
  glDrawElementsInstanced(GL_TRIANGLES, vertex_count, GL_UNSIGNED_INT, 0, num_voxels);
  glBindVertexArray(0);

  if (show_wireframe) {
      drawWireframeCubes();
  }
  if (show_axis) {
      drawAxes();
  }

  glfwSwapBuffers(glfwGetCurrentContext());
}

glm::vec3 VolumetricDisplay::calculateSceneCenter() {
    if (cubes_config_.empty()) {
        return glm::vec3(0.0f);
    }

    // Calculate bounding box of all cubes using individual cube dimensions
    glm::vec3 min_bounds = cubes_config_[0].position;
    glm::vec3 max_bounds = cubes_config_[0].position + glm::vec3(cubes_config_[0].width, cubes_config_[0].height, cubes_config_[0].length);

    for (const auto& cube_cfg : cubes_config_) {
        glm::vec3 cube_min = cube_cfg.position;
        glm::vec3 cube_max = cube_cfg.position + glm::vec3(cube_cfg.width, cube_cfg.height, cube_cfg.length);

        min_bounds = glm::min(min_bounds, cube_min);
        max_bounds = glm::max(max_bounds, cube_max);
    }

    // Return the center of the bounding box
    return (min_bounds + max_bounds) * 0.5f;
}

void VolumetricDisplay::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
  if (key == GLFW_KEY_A && action == GLFW_PRESS) {
    show_axis = !show_axis;
  }

  if (key == GLFW_KEY_B && action == GLFW_PRESS) {
    show_wireframe = !show_wireframe;
  }

  view_update.notify_all();
}

void VolumetricDisplay::rotate(float angle, float x, float y, float z) {
  glm::vec3 axis = glm::normalize(glm::vec3(x, y, z));
  glm::quat rotation = glm::angleAxis(glm::radians(angle), axis);
  camera_orientation = rotation * camera_orientation;
}

void VolumetricDisplay::windowCloseCallback(GLFWwindow* window) {
  VLOG(0) << "Window closed";
  running = false;
  view_update.notify_all();
}

void VolumetricDisplay::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
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

void VolumetricDisplay::cursorPositionCallback(GLFWwindow* window, double xpos, double ypos) {
  if (left_mouse_button_pressed) {
    // Rotation
    float dx = static_cast<float>(xpos - last_mouse_x);
    float dy = static_cast<float>(ypos - last_mouse_y);

    glm::quat rot_x = glm::angleAxis(glm::radians(dy * 0.2f), glm::vec3(1.0f, 0.0f, 0.0f));
    glm::quat rot_y = glm::angleAxis(glm::radians(dx * 0.2f), glm::vec3(0.0f, 1.0f, 0.0f));

    camera_orientation = rot_y * rot_x * camera_orientation;
  } else if (right_mouse_button_pressed) {
    // Panning (SHIFT + mouse drag)
    float dx = static_cast<float>(xpos - last_mouse_x);
    float dy = static_cast<float>(ypos - last_mouse_y);
    camera_position += glm::vec3(dx * 0.05f, -dy * 0.05f, 0.0f);
  }

  last_mouse_x = xpos;
  last_mouse_y = ypos;

  view_update.notify_all();
}

void VolumetricDisplay::scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
  camera_distance -= static_cast<float>(yoffset) * 2.0f;
  if (camera_distance < 1.0f) {
    camera_distance = 1.0f;
  }

  view_update.notify_all();
}

// Transform matrix computation functions for cube orientation
glm::mat4 VolumetricDisplay::computeCubeLocalTransformMatrix(const std::vector<std::string>& world_orientation, const glm::vec3& size) {
    // TODO: Implement cube-local transform matrix
    // This should handle axis swaps and sign flip compensation
    // For now, just return identity matrix
    auto transform_matrix = glm::mat4(0.0f);
    for (int i = 0; i < 3; i++) {
        std::string world_axis = world_orientation[i];
        bool is_negative = world_axis.starts_with("-");
        if (is_negative) {
            world_axis = world_axis.substr(1);
        }

        float axis_coeff = is_negative ? -1.0f : 1.0f;

        if (world_axis == "X") {
            transform_matrix[0][i] = axis_coeff;
        } else if (world_axis == "Y") {
            transform_matrix[1][i] = axis_coeff;
        } else if (world_axis == "Z") {
            transform_matrix[2][i] = axis_coeff;
        }

        if (is_negative) {
            // Add an offset to the transform matrix
            transform_matrix[3][i] = size[i];
        }
    }
    transform_matrix[3][3] = 1.0f;
    return transform_matrix;
}

glm::mat4 VolumetricDisplay::computeCubeToWorldTransformMatrix(const std::vector<std::string>& world_orientation, const glm::vec3& cube_position) {
    // TODO: Implement cube-to-world transform matrix
    // This should handle orientation matrix and world translation
    // For now, just return translation matrix
    return glm::translate(glm::mat4(1.0f), cube_position);
}
