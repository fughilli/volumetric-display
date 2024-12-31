import argparse
import socket
import struct
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import threading


class VolumetricDisplay:

    def __init__(self, width, height, length, ip_address, port,
                 universes_per_layer):
        self.width = width
        self.height = height
        self.length = length
        self.ip_address = ip_address

        # VBO data
        self.vbo_vertices = None
        self.vbo_colors = None
        self.vertex_count = 0

        # Track rotation key states
        self.rotation_keys = {
            pygame.K_LEFT: False,
            pygame.K_RIGHT: False,
            pygame.K_UP: False,
            pygame.K_DOWN: False
        }

        self.rotation_x = np.identity(4, dtype=np.float32)
        self.rotation_y = np.identity(4, dtype=np.float32)
        self.rotation_z = np.identity(4, dtype=np.float32)
        self.temp_matrix = np.identity(4, dtype=np.float32)

        # Initialize 3D array to store RGB values
        self.pixels = np.zeros((width, height, length, 3), dtype=np.uint8)

        # Calculate number of universes needed per layer
        pixels_per_layer = width * height
        self.universes_per_layer = universes_per_layer
        if self.universes_per_layer > 10:
            raise ValueError("Layer size too large for ArtNet limitations")

        # ArtNet setup
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((ip_address, port))

        # Initialize VBOs
        self._setup_vbo()

        # Setup perspective and lighting
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        # Set minimal global ambient light
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, (1.0, 1.0, 1.0, 1.0))

        # Initialize rotation state
        self.rotation_matrix = np.identity(4, dtype=np.float32)

        # Start ArtNet listener thread
        self.running = True
        self.needs_update = False
        self.thread = threading.Thread(target=self._listen_artnet)
        self.thread.start()

    def _listen_artnet(self):
        while self.running:
            data, addr = self.sock.recvfrom(1024)

            # Check if it's an ArtNet packet
            if data[0:8] != b'Art-Net\0':
                continue

            opcode = struct.unpack('<H', data[8:10])[0]

            # Handle DMX data
            if opcode == 0x5000:  # OpDmx
                universe = struct.unpack('<H', data[14:16])[0]
                length = struct.unpack('>H', data[16:18])[0]
                dmx_data = data[18:18 + length]

                # Calculate which layer and section this universe belongs to
                layer = universe // self.universes_per_layer
                universe_in_layer = universe % self.universes_per_layer

                if layer >= self.length:
                    continue

                # Update pixel values
                start_pixel = universe_in_layer * 170
                for i in range(0, len(dmx_data), 3):  # Skip last 2 channels
                    pixel_index = start_pixel + (i // 3)
                    if pixel_index >= self.width * self.height:
                        break

                    x = pixel_index % self.width
                    y = pixel_index // self.width

                    self.pixels[x, y, layer] = [
                        dmx_data[i], dmx_data[i + 1], dmx_data[i + 2]
                    ]

            # Handle sync packet
            elif opcode == 0x5200:  # OpSync
                self.needs_update = True

    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Load identity matrix and apply our stored rotation
        glLoadIdentity()
        gluPerspective(45, (800 / 600), 0.1, 100.0)
        glTranslatef(0, 0, -max(self.width, self.height, self.length) * 2.5)
        glMultMatrixf(self.rotation_matrix)
        glTranslatef(-self.width / 2, -self.height / 2, -self.length / 2)

        # Only update colors when new data arrives
        if self.needs_update:
            self.update_colors()
            self.needs_update = False

        # Enable vertex and color arrays
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        # Bind vertex buffer
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_vertices)
        glVertexPointer(3, GL_FLOAT, 0, None)

        # Bind color buffer
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_colors)
        glColorPointer(3, GL_FLOAT, 0, None)

        # Draw all cubes at once
        glDrawArrays(GL_QUADS, 0, self.vertex_count)

        # Disable vertex and color arrays
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

        # Draw coordinate axes for reference
        glBegin(GL_LINES)
        # X axis in red
        glColor3f(1, 0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(self.width, 0, 0)
        # Y axis in green
        glColor3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, self.height, 0)
        # Z axis in blue
        glColor3f(0, 0, 1)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, self.length)
        glEnd()

        pygame.display.flip()

    def _setup_vbo(self):
        """Initialize the VBO with cube geometry for all possible positions"""
        size = 0.1
        vertices = []
        self.color_data = np.zeros(
            (self.width * self.height * self.length * 24, 3), dtype=np.float32)

        # Generate vertices for each possible cube position
        for x in range(self.width):
            for y in range(self.height):
                for z in range(self.length):
                    cube_vertices = [
                        # Front face
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
                        # Back face
                        x + size,
                        y - size,
                        z - size,
                        x - size,
                        y - size,
                        z - size,
                        x - size,
                        y + size,
                        z - size,
                        x + size,
                        y + size,
                        z - size,
                        # Left face
                        x - size,
                        y - size,
                        z - size,
                        x - size,
                        y - size,
                        z + size,
                        x - size,
                        y + size,
                        z + size,
                        x - size,
                        y + size,
                        z - size,
                        # Right face
                        x + size,
                        y - size,
                        z + size,
                        x + size,
                        y - size,
                        z - size,
                        x + size,
                        y + size,
                        z - size,
                        x + size,
                        y + size,
                        z + size,
                        # Bottom face
                        x - size,
                        y - size,
                        z - size,
                        x + size,
                        y - size,
                        z - size,
                        x + size,
                        y - size,
                        z + size,
                        x - size,
                        y - size,
                        z + size,
                        # Top face
                        x - size,
                        y + size,
                        z + size,
                        x + size,
                        y + size,
                        z + size,
                        x + size,
                        y + size,
                        z - size,
                        x - size,
                        y + size,
                        z - size,
                    ]
                    vertices.extend(cube_vertices)

        vertices = np.array(vertices, dtype=np.float32)

        # Create and bind vertex VBO
        self.vbo_vertices = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_vertices)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices,
                     GL_STATIC_DRAW)

        # Create and bind color VBO
        self.vbo_colors = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_colors)
        glBufferData(GL_ARRAY_BUFFER, self.color_data.nbytes, self.color_data,
                     GL_DYNAMIC_DRAW)

        self.vertex_count = len(vertices) // 3

    def update_colors(self):
        """Update the color VBO based on current pixel values"""
        color_index = 0
        for x in range(self.width):
            for y in range(self.height):
                for z in range(self.length):
                    color = self.pixels[x, y, z]
                    # Each cube has 24 vertices (6 faces * 4 vertices)
                    for _ in range(24):
                        self.color_data[color_index] = color / 255.0
                        color_index += 1

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_colors)
        glBufferSubData(GL_ARRAY_BUFFER, 0, self.color_data.nbytes,
                        self.color_data)

    def rotate(self, angle, x, y, z):
        """Apply a rotation to our stored rotation matrix"""
        # Convert angle to radians
        angle = np.radians(angle)
        c = np.cos(angle)
        s = np.sin(angle)

        # Reuse pre-allocated matrices
        if x:  # Rotate around X axis
            self.rotation_x.flat[[5, 6, 9, 10]] = [c, -s, s, c]
            np.matmul(self.rotation_matrix,
                      self.rotation_x,
                      out=self.temp_matrix)
        elif y:  # Rotate around Y axis
            self.rotation_y.flat[[0, 2, 8, 10]] = [c, s, -s, c]
            np.matmul(self.rotation_matrix,
                      self.rotation_y,
                      out=self.temp_matrix)
        elif z:  # Rotate around Z axis
            self.rotation_z.flat[[0, 1, 4, 5]] = [c, -s, s, c]
            np.matmul(self.rotation_matrix,
                      self.rotation_z,
                      out=self.temp_matrix)

        # Swap matrices instead of copying
        self.rotation_matrix, self.temp_matrix = self.temp_matrix, self.rotation_matrix

        # Renormalize the rotation matrix to prevent accumulation of numerical errors
        # Extract 3x3 rotation part
        rotation_3x3 = self.rotation_matrix[:3, :3]
        # Use SVD to orthogonalize
        u, _, vh = np.linalg.svd(rotation_3x3)
        rotation_3x3 = np.matmul(u, vh)
        # Put back into 4x4 matrix
        self.rotation_matrix[:3, :3] = rotation_3x3

    def run(self):
        """Main display loop"""
        clock = pygame.time.Clock()

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key in self.rotation_keys:
                        self.rotation_keys[event.key] = True
                elif event.type == pygame.KEYUP:
                    if event.key in self.rotation_keys:
                        self.rotation_keys[event.key] = False

            # Handle continuous rotation
            if self.rotation_keys[pygame.K_LEFT]:
                self.rotate(1, 0, 1, 0)
            if self.rotation_keys[pygame.K_RIGHT]:
                self.rotate(-1, 0, 1, 0)
            if self.rotation_keys[pygame.K_UP]:
                self.rotate(1, 1, 0, 0)
            if self.rotation_keys[pygame.K_DOWN]:
                self.rotate(-1, 1, 0, 0)

            # Always render every frame
            self.render()
            clock.tick(60)

    def cleanup(self):
        """Cleanup resources"""
        self.running = False
        self.thread.join()
        self.sock.close()
        pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--geometry',
                        type=str,
                        default="20x20x20",
                        help="Width, height, and length of the display")
    parser.add_argument('--ip',
                        type=str,
                        default="127.0.0.1",
                        help="IP address to listen for ArtNet packets")
    parser.add_argument('--port',
                        type=int,
                        default=6454,
                        help="Port to listen for ArtNet packets")
    parser.add_argument('--universes-per-layer',
                        type=int,
                        default=6,
                        help="Number of universes per layer")
    args = parser.parse_args()

    width, height, length = map(int, args.geometry.split('x'))

    # Visualization setup
    pygame.init()
    pygame.display.set_mode((800, 600), DOUBLEBUF | OPENGL)

    # Example usage
    display = VolumetricDisplay(width, height, length, args.ip, args.port,
                                args.universes_per_layer)
    try:
        display.run()
    finally:
        display.cleanup()
