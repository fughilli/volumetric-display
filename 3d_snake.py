from artnet import Scene, RGB, HSV
import math
import pygame
from pygame.locals import *
from enum import Enum
import random
import asyncio
import threading
import control_port # Assuming control_port.py is in the same directory or PYTHONPATH
from collections import deque # Import deque

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
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4

class PygameInputHandler:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((640, 480))
        pygame.display.set_caption('3D Snake Scene')

    def get_direction_key(self):
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                if event.key == K_UP:
                    return Direction.UP
                elif event.key == K_DOWN:
                    return Direction.DOWN
                elif event.key == K_LEFT:
                    return Direction.LEFT
                elif event.key == K_RIGHT:
                    return Direction.RIGHT
        return None

class ControllerInputHandler:
    def __init__(self):
        self.last_button_states = [False] * 5 # LEFT, UP, RIGHT, DOWN, SELECT
        self.cp = control_port.ControlPort()
        self.controller = None
        self.listen_task = None
        self._lock = threading.Lock()
        self.initialized = False
        self.event_queue = deque() # Queue for button down events
        self.init_event = threading.Event() # To signal initialization completion
        self.loop = None # Will be set by the asyncio thread
        self._init_task = None
        self._listen_task = None

    async def _async_initialize_and_listen(self):
        """Runs in the asyncio thread to initialize and start listening."""
        print("Enumerating controllers...")
        # Use get_running_loop() as ControlPort methods expect it
        try:
            controllers = await self.cp.enumerate(timeout=5.0)
            if not controllers:
                print("No controllers found.")
                self.controller = None
                self.initialized = False
            else:
                # Get the first controller found
                ip, state = list(controllers.items())[0]
                print(f"Found controller at {ip}")
                if await state.connect():
                    print(f"Connected to controller {ip}")
                    self.controller = state
                    self.controller.register_button_callback(self._button_callback)
                    # Start the actual listening task
                    self._listen_task = self.loop.create_task(self.controller._listen_buttons())
                    self.initialized = True
                else:
                    print(f"Failed to connect to controller {ip}")
                    self.controller = None
                    self.initialized = False
        except Exception as e:
            print(f"Error during controller async initialization: {e}")
            self.controller = None
            self.initialized = False
        finally:
            # Signal that initialization attempt is complete
            self.init_event.set()

    def _button_callback(self, buttons):
        # This is called from the asyncio thread
        with self._lock:
            # Compare with previous state to find button down events
            if buttons[0] and not self.last_button_states[0]: self.event_queue.append(Direction.LEFT)
            if buttons[1] and not self.last_button_states[1]: self.event_queue.append(Direction.UP)
            if buttons[2] and not self.last_button_states[2]: self.event_queue.append(Direction.RIGHT)
            if buttons[3] and not self.last_button_states[3]: self.event_queue.append(Direction.DOWN)
            # Store the new state for the next comparison
            self.last_button_states = list(buttons) # Store a copy
        # print(f"Buttons: {self.last_button_states}") # Debugging

    def get_direction_key(self):
        # This is called from the main game thread
        if not self.controller or not self.initialized:
            return None

        with self._lock:
            if self.event_queue: # Check if the queue has events
                return self.event_queue.popleft() # Return the oldest event

        return None # No events in the queue

    def start_initialization(self):
        """Starts the background thread and waits for initialization."""
        self.thread = threading.Thread(target=self._run_asyncio_loop, daemon=True)
        self.thread.start()
        # Wait for the init event to be set by the asyncio thread
        initialized = self.init_event.wait(timeout=7.0) # Wait up to 7 seconds
        if not initialized:
            print("Controller initialization timed out.")
            # Attempt to stop the thread if it's stuck
            self.stop()
            return False
        return self.initialized # Return the actual success/fail status

    def _run_asyncio_loop(self):
        # Set up the event loop for this thread
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        try:
            # Schedule the main async task for this loop
            self._init_task = self.loop.create_task(self._async_initialize_and_listen())
            self.loop.run_forever()
        finally:
            print("Asyncio loop stopping...")
            # Cleanup tasks on loop stop
            if self._listen_task and not self._listen_task.done():
                self._listen_task.cancel()
            if self._init_task and not self._init_task.done():
                self._init_task.cancel() # Should be done, but cancel defensively

            # Gather cancelled tasks to allow them to finish cancelling
            async def gather_cancelled():
                tasks = [t for t in asyncio.all_tasks(self.loop) if t.cancelled()] # or t.done() ?
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
            if self.loop.is_running():
                self.loop.run_until_complete(gather_cancelled())
                self.loop.run_until_complete(self.loop.shutdown_asyncgens())

            self.loop.close()
            print("Asyncio loop stopped.")

    def stop(self):
        print("Stopping controller input handler...")
        # Ensure loop exists before checking if running
        loop = getattr(self, 'loop', None)
        if loop and loop.is_running():
            # Disconnect needs to be async, schedule it
            if self.controller and self.controller._connected:
                # Use call_soon_threadsafe as we're stopping from another thread
                disconnect_future = asyncio.run_coroutine_threadsafe(self.controller.disconnect(), loop)
                try:
                    disconnect_future.result(timeout=2) # Wait briefly for disconnect
                    print("Controller disconnected.")
                except Exception as e:
                    print(f"Error during controller disconnect: {e}")

            # Cancel running tasks first
            if self._listen_task:
                loop.call_soon_threadsafe(self._listen_task.cancel)
            if self._init_task:
                loop.call_soon_threadsafe(self._init_task.cancel)

            # Stop the loop itself
            loop.call_soon_threadsafe(loop.stop)

        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=3.0) # Increased timeout slightly

class SnakeScene(Scene):
    def __init__(self, width=20, height=20, length=20, frameRate=3, input_handler_type='controller'):
        super().__init__()
        self.thickness = 2
        self.width = width // self.thickness
        self.height = height // self.thickness
        self.length = length // self.thickness
        self.frameRate = frameRate # Hz

        print(f"Initializing SnakeScene with input type: {input_handler_type}")
        if input_handler_type == 'controller':
            print("Attempting to initialize controller input...")
            controller_handler = ControllerInputHandler()
            if controller_handler.start_initialization():
                self.input_handler = controller_handler
                print("Controller input handler started.")
            else:
                print("Controller initialization failed, falling back to Pygame.")
                self.input_handler = PygameInputHandler()
        else:
            self.input_handler = PygameInputHandler()

        self.reset_game()

        # Timer for controlling update frequency
        self.last_update_time = 0

    def valid(self, x, y, z):
        return 0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.length
    
    def update_direction(self, key):
        if self.direction == (1, 0, 0) or self.direction == (-1, 0, 0):  # Moving along x-axis
            if key == Direction.UP:
                self.direction = (0, 1, 0)  # Move along positive y-axis
            elif key == Direction.DOWN:
                self.direction = (0, -1, 0)  # Move along negative y-axis
            elif key == Direction.LEFT:
                self.direction = (0, 0, 1) if self.direction[0] == 1 else (0, 0, -1)  # Move along z-axis
            elif key == Direction.RIGHT:
                self.direction = (0, 0, -1) if self.direction[0] == 1 else (0, 0, 1)  # Move along z-axis
        elif self.direction == (0, 1, 0) or self.direction == (0, -1, 0):  # Moving along y-axis
            if key == Direction.UP:
                self.direction = (0, 0, 1)  # Move along positive z-axis
            elif key == Direction.DOWN:
                self.direction = (0, 0, -1)  # Move along negative z-axis
            elif key == Direction.LEFT:
                self.direction = (-1, 0, 0) if self.direction[1] == 1 else (1, 0, 0)  # Move along x-axis
            elif key == Direction.RIGHT:
                self.direction = (1, 0, 0) if self.direction[1] == 1 else (-1, 0, 0)  # Move along x-axis
        elif self.direction == (0, 0, 1) or self.direction == (0, 0, -1):  # Moving along z-axis
            if key == Direction.UP:
                self.direction = (0, 1, 0)  # Move along positive y-axis
            elif key == Direction.DOWN:
                self.direction = (0, -1, 0)  # Move along negative y-axis
            elif key == Direction.LEFT:
                self.direction = (-1, 0, 0) if self.direction[2] == 1 else (1, 0, 0)  # Move along x-axis
            elif key == Direction.RIGHT:
                self.direction = (1, 0, 0) if self.direction[2] == 1 else (-1, 0, 0)  # Move along x-axis

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
            for i, digit in enumerate(str(self.score)):
                x = raster.width // 2 - 2 + i * 4
                y = raster.height // 2 - 2
                for dx, dy in digit_map[digit]:
                    idx = (y+dy) * raster.width + (x+dx)
                    raster.data[idx] = white
                    
    def cleanup(self): # Add cleanup method
        print("Cleaning up SnakeScene...")
        if isinstance(self.input_handler, ControllerInputHandler):
            self.input_handler.stop()
                        