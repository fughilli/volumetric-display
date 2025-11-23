from artnet import Scene, RGB, HSV
import math
import pygame
from pygame.locals import *
from enum import Enum
import random

white = RGB(255, 255, 255)
red = RGB(255, 0, 0)
blue = RGB(0, 0, 255)
green = RGB(0, 255, 0)
black = RGB(0, 0, 0)

digit_map = {
    '0': [(0,4), (1,4), (2,4), (0,3), (2,3), (0,2), (2,2), (0,1), (2,1), (0,0), (1,0), (2,0)],
    '1': [(1,4), (1,3), (1,2), (1,1), (1,0)],
    '2': [(0,4), (1,4), (2,4), (2,3), (0,2), (1,2), (2,2), (0,1), (0,0), (1,0), (2,0)],
    '3': [(0,4), (1,4), (2,4), (2,3), (1,2), (2,2), (2,1), (0,0), (1,0), (2,0)],
    '4': [(0,4), (2,4), (0,3), (2,3), (0,2), (1,2), (2,2), (2,1), (2,0)],
    '5': [(0,4), (1,4), (2,4), (0,3), (0,2), (1,2), (2,2), (2,1), (0,0), (1,0), (2,0)],
    '6': [(0,4), (1,4), (2,4), (0,3), (0,2), (1,2), (2,2), (0,1), (2,1), (0,0), (1,0), (2,0)],
    '7': [(0,4), (1,4), (2,4), (2,3), (2,2), (2,1), (2,0)],
    '8': [(0,4), (1,4), (2,4), (0,3), (2,3), (0,2), (1,2), (2,2), (0,1), (2,1), (0,0), (1,0), (2,0)],
    '9': [(0,4), (1,4), (2,4), (0,3), (2,3), (0,2), (1,2), (2,2), (2,1), (0,0), (1,0), (2,0)]
}

class Direction(Enum):
    LEFT = 1   # Negative X
    RIGHT = 2  # Positive X
    UP = 3     # Positive Y
    DOWN = 4   # Negative Y
    FRONT = 5  # Negative Z
    BACK = 6   # Positive Z

class PygameInputHandler:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((640, 480))
        pygame.display.set_caption('3D Snake Scene')

    def get_direction_key(self):
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                if event.key in (K_UP, K_w):
                    return Direction.UP
                elif event.key in (K_DOWN, K_s):
                    return Direction.DOWN
                elif event.key in (K_LEFT, K_a):
                    return Direction.LEFT
                elif event.key in (K_RIGHT, K_d):
                    return Direction.RIGHT
                elif event.key == K_q:
                    return Direction.BACK
                elif event.key == K_e:
                    return Direction.FRONT
        return None

class SnakeScene(Scene):
    def __init__(self, width=20, height=20, length=20, frameRate=3, input_handler=None):
        super().__init__()
        self.thickness = 2
        self.width = width // self.thickness
        self.height = height // self.thickness
        self.length = length // self.thickness
        self.frameRate = frameRate # Hz
        self.input_handler = input_handler or PygameInputHandler()
        
        self.reset_game()

        # Timer for controlling update frequency
        self.last_update_time = 0

    def valid(self, x, y, z):
        return 0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.length
    
    def update_direction(self, key):
        # Map directions to their vectors
        direction_vectors = {
            Direction.LEFT: (-1, 0, 0),
            Direction.RIGHT: (1, 0, 0),
            Direction.UP: (0, 1, 0),
            Direction.DOWN: (0, -1, 0),
            Direction.FRONT: (0, 0, -1),
            Direction.BACK: (0, 0, 1)
        }
        
        # Get the new direction vector
        new_direction = direction_vectors.get(key)
        if new_direction:
            # Check if it's the opposite direction (would be a 180-degree turn)
            is_opposite = (
                new_direction[0] == -self.direction[0] and
                new_direction[1] == -self.direction[1] and
                new_direction[2] == -self.direction[2]
            )
            
            # Update direction if it's not the opposite
            if not is_opposite:
                self.direction = new_direction

    def update_apple(self):
        head = self.snake[0]
        if head == self.apple:
            self.score += 1
            self.snake_length += 1
            self.explosionSource = head
            self.explosionTime = 0
            self.place_new_apple()
    
    def place_new_apple(self):
        self.apple = (random.randint(0, self.width-1), random.randint(0, self.length-1), random.randint(0, self.height-1))
        while self.apple in self.snake:
            self.apple = (random.randint(0, self.width-1), random.randint(0, self.length-1), random.randint(0, self.height-1))
    
    def reset_game(self):
        self.snake = [(self.width//2, self.length//2, self.height//2)]
        self.direction = (1, 0, 0)
        self.snake_length = self.width//2
        self.score = 0
        self.place_new_apple()
        self.explosionSource = None
        self.explosionTime = None
        self.game_started = False
        self.game_over = False
    
    def update_snake(self):
        # Calculate new head position
        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1], head[2] + self.direction[2])
        if not self.valid(*new_head) or new_head in self.snake:
            self.game_over = True
            return
        
        # Add new head to the snake
        self.snake.insert(0, new_head)

        # Ensure the snake length
        if len(self.snake) > self.snake_length:
            self.snake.pop()
        
    def render(self, raster, time):
        ######### Update game state ########
        if time - self.last_update_time >= 1.0/self.frameRate:
            self.last_update_time = time

            key = self.input_handler.get_direction_key()
            if key:
                self.update_direction(key)
                if self.game_over:
                    self.reset_game()
                self.game_started = True
            if self.game_started:
                self.update_snake()
                self.update_apple()

        ########## Draw the scene ##########

        # Clear the raster
        for y in range(raster.height):
            for x in range(raster.width):
                for z in range(raster.length):
                    idx = y * raster.width + x + z * raster.width * raster.height
                    raster.data[idx] = black
        
        # Draw the explosion
        if self.explosionTime is not None:
            self.explosionTime += 1 / self.frameRate / 2
            if self.explosionTime < 10:
                xs, ys, zs = self.explosionSource
                xs *= self.thickness
                ys *= self.thickness
                zs *= self.thickness
                for y in range(raster.height):
                    for x in range(raster.width):
                        for z in range(raster.length):
                            dx = x - xs
                            dy = y - ys
                            dz = z - zs
                            if -1 < (dx*dx + dy*dy + dz*dz)**0.5 - self.explosionTime < 1:
                                idx = y * raster.width + x + z * raster.width * raster.height
                                # raster.data[idx] = red
                                hue = ((x + y + z) * 4 + self.explosionTime * 50)
                    
                                raster.data[idx] = RGB.from_hsv(HSV(
                                    hue % 256,
                                    255,
                                    255
                                ))
            else:
                self.explosionTime = None
                self.explosionSource = None

        # Draw the snake
        for i, segment in enumerate(self.snake):
            x, y, z = segment
            if self.valid(x, y, z):
                x *= self.thickness
                y *= self.thickness
                z *= self.thickness
                for dx in range(self.thickness):
                    for dy in range(self.thickness):
                        for dz in range(self.thickness):
                            idx = (y+dy) * raster.width + (x+dx) + (z+dz) * raster.width * raster.height
                            if i == 0:
                                raster.data[idx] = red
                            else:
                                raster.data[idx] = green

        # Draw the apple
        x, y, z = self.apple
        x *= self.thickness
        y *= self.thickness
        z *= self.thickness
        for dx in range(self.thickness):
            for dy in range(self.thickness):
                for dz in range(self.thickness):
                    idx = (y+dy) * raster.width + (x+dx) + (z+dz) * raster.width * raster.height
                    raster.data[idx] = blue

        # Draw the score
        if self.game_over:
            score_str = str(self.score)
            letter_thickness = 4
            for i, digit in enumerate(score_str):
                x = raster.width // 2 - letter_thickness // 2 - (len(score_str)//2 * letter_thickness) + i * letter_thickness
                y = raster.height // 2 - letter_thickness // 2
                reverse_x = (raster.width - 1) - x - letter_thickness
                for dx, dy in digit_map[digit]:
                    # front face
                    idx = (y+dy) * raster.width + (x+dx)
                    raster.data[idx] = white
                    # left face
                    idx = (y+dy) * raster.width + (reverse_x + (letter_thickness-dx)) + (raster.width-1) * raster.width * raster.height
                    raster.data[idx] = white
                    # right face
                    idx = (y+dy) * raster.width + (reverse_x + (letter_thickness-dx)) * raster.width * raster.height
                    raster.data[idx] = white
                    # back face
                    idx = (y+dy) * raster.width + (raster.width-1) + (x + dx) * raster.width * raster.height
                    raster.data[idx] = white
                    
                        