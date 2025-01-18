from artnet import Scene, RGB
import math
import pygame
from pygame.locals import *

white = RGB(255, 255, 255)
red = RGB(255, 0, 0)
blue = RGB(0, 0, 255)
green = RGB(0, 255, 0)
black = RGB(0, 0, 0)

class PygameInputHandler:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((640, 480))
        pygame.display.set_caption('3D Snake Scene')

    def get_direction(self, current_direction):
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                exit()
            elif event.type == KEYDOWN:
                if current_direction == (1, 0, 0) or current_direction == (-1, 0, 0):  # Moving along x-axis
                    if event.key == K_UP:
                        return (0, 1, 0)  # Move along positive y-axis
                    elif event.key == K_DOWN:
                        return (0, -1, 0)  # Move along negative y-axis
                    elif event.key == K_LEFT:
                        return (0, 0, 1) if current_direction[0] == 1 else (0, 0, -1)  # Move along z-axis
                    elif event.key == K_RIGHT:
                        return (0, 0, -1) if current_direction[0] == 1 else (0, 0, 1)  # Move along z-axis
                elif current_direction == (0, 1, 0) or current_direction == (0, -1, 0):  # Moving along y-axis
                    if event.key == K_UP:
                        return (1, 0, 0)  # Move along positive x-axis
                    elif event.key == K_DOWN:
                        return (-1, 0, 0)  # Move along negative x-axis
                    elif event.key == K_LEFT:
                        return (0, 0, -1) if current_direction[1] == 1 else (0, 0, 1)  # Move along z-axis
                    elif event.key == K_RIGHT:
                        return (0, 0, 1) if current_direction[1] == 1 else (0, 0, -1)  # Move along z-axis
                elif current_direction == (0, 0, 1) or current_direction == (0, 0, -1):  # Moving along z-axis
                    if event.key == K_UP:
                        return (0, 1, 0)  # Move along positive y-axis
                    elif event.key == K_DOWN:
                        return (0, -1, 0)  # Move along negative y-axis
                    elif event.key == K_LEFT:
                        return (-1, 0, 0) if current_direction[2] == 1 else (1, 0, 0)  # Move along x-axis
                    elif event.key == K_RIGHT:
                        return (1, 0, 0) if current_direction[2] == 1 else (-1, 0, 0)  # Move along x-axis
        return current_direction

class SnakeScene(Scene):
    def __init__(self, width=20, height=20, length=20, frameRate=3, input_handler=None):
        super().__init__()
        self.width = width
        self.height = height
        self.length = length
        self.frameRate = frameRate # Hz
        self.snake = [(width//2, length//2, height//2)]  # Initial position of the snake
        self.direction = (1, 0, 0)  # Initial direction (moving along x-axis)
        self.snake_length = width//2  # Initial length of the snake
        self.input_handler = input_handler or PygameInputHandler()

        # Timer for controlling update frequency
        self.last_update_time = 0

    def valid(self, x, y, z):
        return 0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.length
    
    def update_snake(self):
        # Calculate new head position
        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1], head[2] + self.direction[2])
        if not self.valid(*new_head):
            return # this could be game over if we consider the sides as boundaries
        
        # Add new head to the snake
        self.snake.insert(0, new_head)

        # Ensure the snake length
        if len(self.snake) > self.snake_length:
            self.snake.pop()
        
    def render(self, raster, time):
        if time - self.last_update_time < 1.0/self.frameRate:
            return  # Skip update if less than 1/frameRate seconds has passed
        self.last_update_time = time

        self.direction = self.input_handler.get_direction(self.direction)
        self.update_snake()

        # Clear the raster
        for y in range(raster.height):
            for x in range(raster.width):
                for z in range(raster.length):
                    idx = y * raster.width + x + z * raster.width * raster.height
                    raster.data[idx] = black

        # Draw the snake
        for i, segment in enumerate(self.snake):
            x, y, z = segment
            if 0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.length:
                idx = y * raster.width + x + z * raster.width * raster.height
                if i == 0:
                    raster.data[idx] = red
                else:
                    raster.data[idx] = green

if __name__ == "__main__":
    width, height, length = 20, 20, 20  # Example dimensions
    scene = SnakeScene(width, height, length)
    # Example usage
    # You need to integrate this scene into your existing framework to run it
