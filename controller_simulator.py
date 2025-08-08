import argparse
import asyncio
import json
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import pygame
import pygame.freetype

# Initialize Pygame (only if in main thread)
pygame_initialized = False
if threading.current_thread() is threading.main_thread():
    try:
        pygame.init()
        pygame.freetype.init()
        pygame_initialized = True
    except Exception as e:
        print(f"Error initializing Pygame: {e}")
        # Potentially exit or handle non-GUI mode
else:
    print("Warning: Pygame initialized outside main thread. GUI might not work.")

# Constants
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 600
BUTTON_SIZE = 30
BUTTON_SPACING = 15
LCD_WIDTH = 200
LCD_HEIGHT = 80
LCD_CHAR_WIDTH = 20
LCD_CHAR_HEIGHT = 4
FONT_SIZE = 12

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
DARK_GRAY = (64, 64, 64)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
LCD_BG = (0, 32, 0)
LCD_FG = (0, 255, 0)


class Button(Enum):
    UP = 0
    LEFT = 1
    DOWN = 2
    RIGHT = 3
    SELECT = 4


@dataclass
class VirtualControllerState:
    dip: int
    port: int
    buttons: List[bool]
    lcd_lines: List[str]
    rect: Optional[pygame.Rect] = None  # GUI element
    button_rects: Optional[Dict[Button, pygame.Rect]] = None  # GUI element
    lcd_rect: Optional[pygame.Rect] = None  # GUI element
    server_task: Optional[asyncio.Task] = None
    client_writer: Optional[asyncio.StreamWriter] = None
    last_button_sent: Optional[List[bool]] = None


class ControllerSimulator:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, config_path: Optional[str] = None):
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            self.running = True
            self.controllers: Dict[int, VirtualControllerState] = {}
            self.screen = None
            self.font = None
            self.lcd_font = None
            self._config_path = config_path
            self.asyncio_thread = None
            self.loop = None
            self.servers = []  # To keep server references

            self._load_config()

            # Initialize Pygame components only if Pygame was successfully initialized
            if pygame_initialized:
                self._init_pygame_components()

            self._initialized = True

    def _load_config(self):
        if not self._config_path:
            print("Error: Config path not provided.")
            return
        try:
            with open(self._config_path, "r") as f:
                config_data = json.load(f)

            for ctrl_config in config_data.get("controllers", []):
                dip = ctrl_config.get("dip")
                port = ctrl_config.get("port")
                if dip is not None and port is not None:
                    self.controllers[dip] = VirtualControllerState(
                        dip=dip,
                        port=port,
                        buttons=[False] * 5,
                        lcd_lines=[" " * LCD_CHAR_WIDTH for _ in range(LCD_CHAR_HEIGHT)],
                        last_button_sent=[False] * 5,  # Initialize last sent state
                    )
        except FileNotFoundError:
            print(f"Error: Config file not found at {self._config_path}")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {self._config_path}")
        except Exception as e:
            print(f"Error loading config: {e}")

    def _init_pygame_components(self):
        """Initialize Pygame specific components."""
        if not pygame_initialized:
            return
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Controller Simulator")
        try:
            self.font = pygame.freetype.SysFont("monospace", FONT_SIZE)
            self.lcd_font = pygame.freetype.SysFont("monospace", 15)
        except Exception as e:
            print(f"Error loading fonts: {e}. Using default font.")
            self.font = pygame.freetype.Font(None, FONT_SIZE)  # Default font
            self.lcd_font = pygame.freetype.Font(None, 15)  # Default font

        self.setup_gui_layout()

    def setup_gui_layout(self):
        """Initialize the layout of all controllers in the GUI"""
        if not pygame_initialized or not self.screen:
            return

        controller_width = LCD_WIDTH + 2 * BUTTON_SPACING
        controller_height = LCD_HEIGHT + 3 * BUTTON_SIZE + 5 * BUTTON_SPACING

        positions = [
            (
                WINDOW_WIDTH // 4 - controller_width // 2,
                WINDOW_HEIGHT // 4 - controller_height // 2,
            ),
            (
                3 * WINDOW_WIDTH // 4 - controller_width // 2,
                WINDOW_HEIGHT // 4 - controller_height // 2,
            ),
            (
                WINDOW_WIDTH // 4 - controller_width // 2,
                3 * WINDOW_HEIGHT // 4 - controller_height // 2,
            ),
            (
                3 * WINDOW_WIDTH // 4 - controller_width // 2,
                3 * WINDOW_HEIGHT // 4 - controller_height // 2,
            ),
        ]

        for i, dip in enumerate(sorted(self.controllers.keys())):
            if i >= len(positions):
                break  # Only display first 4

            controller = self.controllers[dip]
            x, y = positions[i]

            controller.rect = pygame.Rect(x, y, controller_width, controller_height)
            controller.lcd_rect = pygame.Rect(
                x + BUTTON_SPACING, y + BUTTON_SPACING, LCD_WIDTH, LCD_HEIGHT
            )

            button_center_x = x + controller_width // 2
            button_start_y = y + LCD_HEIGHT + 2 * BUTTON_SPACING

            controller.button_rects = {
                Button.UP: pygame.Rect(
                    button_center_x - BUTTON_SIZE // 2,
                    button_start_y,
                    BUTTON_SIZE,
                    BUTTON_SIZE,
                ),
                Button.LEFT: pygame.Rect(
                    button_center_x - 3 * BUTTON_SIZE // 2 - BUTTON_SPACING,
                    button_start_y + BUTTON_SIZE + BUTTON_SPACING,
                    BUTTON_SIZE,
                    BUTTON_SIZE,
                ),
                Button.RIGHT: pygame.Rect(
                    button_center_x + BUTTON_SIZE // 2 + BUTTON_SPACING,
                    button_start_y + BUTTON_SIZE + BUTTON_SPACING,
                    BUTTON_SIZE,
                    BUTTON_SIZE,
                ),
                Button.DOWN: pygame.Rect(
                    button_center_x - BUTTON_SIZE // 2,
                    button_start_y + 2 * BUTTON_SIZE + 2 * BUTTON_SPACING,
                    BUTTON_SIZE,
                    BUTTON_SIZE,
                ),
                Button.SELECT: pygame.Rect(
                    button_center_x + BUTTON_SIZE // 2 + BUTTON_SPACING,
                    button_start_y,
                    BUTTON_SIZE,
                    BUTTON_SIZE,
                ),
            }
            # Key names for display
            self.key_names = {  # TODO: Refactor this - maybe load from config?
                0: {
                    Button.UP: "2",
                    Button.LEFT: "Q",
                    Button.DOWN: "W",
                    Button.RIGHT: "E",
                    Button.SELECT: "3",
                },
                1: {
                    Button.UP: "S",
                    Button.LEFT: "Z",
                    Button.DOWN: "X",
                    Button.RIGHT: "C",
                    Button.SELECT: "D",
                },
                2: {
                    Button.UP: "J",
                    Button.LEFT: "N",
                    Button.DOWN: "M",
                    Button.RIGHT: ",",
                    Button.SELECT: "K",
                },
                3: {
                    Button.UP: "7",
                    Button.LEFT: "Y",
                    Button.DOWN: "U",
                    Button.RIGHT: "I",
                    Button.SELECT: "8",
                },
            }
            self.key_mappings = {  # TODO: Refactor this
                0: {
                    pygame.K_2: Button.UP,
                    pygame.K_q: Button.LEFT,
                    pygame.K_w: Button.DOWN,
                    pygame.K_e: Button.RIGHT,
                    pygame.K_3: Button.SELECT,
                },
                1: {
                    pygame.K_s: Button.UP,
                    pygame.K_z: Button.LEFT,
                    pygame.K_x: Button.DOWN,
                    pygame.K_c: Button.RIGHT,
                    pygame.K_d: Button.SELECT,
                },
                2: {
                    pygame.K_j: Button.UP,
                    pygame.K_n: Button.LEFT,
                    pygame.K_m: Button.DOWN,
                    pygame.K_COMMA: Button.RIGHT,
                    pygame.K_k: Button.SELECT,
                },
                3: {
                    pygame.K_7: Button.UP,
                    pygame.K_y: Button.LEFT,
                    pygame.K_u: Button.DOWN,
                    pygame.K_i: Button.RIGHT,
                    pygame.K_8: Button.SELECT,
                },
            }

    def handle_events(self) -> bool:
        """Handle pygame events. Returns False if the simulator should quit."""
        if not pygame_initialized:
            return True

        button_state_changed = False
        with self._lock:  # Lock access to controller buttons
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return False

                # Handle mouse clicks
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    for controller in self.controllers.values():
                        if controller.button_rects:
                            for button, rect in controller.button_rects.items():
                                if rect.collidepoint(pos):
                                    if not controller.buttons[button.value]:
                                        controller.buttons[button.value] = True
                                        button_state_changed = True

                elif event.type == pygame.MOUSEBUTTONUP:
                    pos = pygame.mouse.get_pos()
                    for controller in self.controllers.values():
                        if controller.button_rects:
                            for button, rect in controller.button_rects.items():
                                if rect.collidepoint(pos):
                                    if controller.buttons[button.value]:
                                        controller.buttons[button.value] = False
                                        button_state_changed = True

                # Handle keyboard
                elif event.type == pygame.KEYDOWN:
                    for dip, mapping in self.key_mappings.items():
                        if dip in self.controllers:
                            if event.key in mapping:
                                button = mapping[event.key]
                                if not self.controllers[dip].buttons[button.value]:
                                    self.controllers[dip].buttons[button.value] = True
                                    button_state_changed = True

                elif event.type == pygame.KEYUP:
                    for dip, mapping in self.key_mappings.items():
                        if dip in self.controllers:
                            if event.key in mapping:
                                button = mapping[event.key]
                                if self.controllers[dip].buttons[button.value]:
                                    self.controllers[dip].buttons[button.value] = False
                                    button_state_changed = True

        # If button state changed, notify asyncio thread to send updates
        if button_state_changed and self.loop:
            for dip in self.controllers:
                asyncio.run_coroutine_threadsafe(self.send_button_update(dip), self.loop)

        return True

    def draw(self):
        """Draw the simulator interface"""
        if not pygame_initialized or not self.screen:
            return

        self.screen.fill(WHITE)

        for dip, controller in self.controllers.items():
            if not controller.rect:
                continue  # Skip if GUI elements not set up

            # Draw controller background
            pygame.draw.rect(self.screen, GRAY, controller.rect)

            # Draw LCD
            pygame.draw.rect(self.screen, LCD_BG, controller.lcd_rect)

            # Draw LCD text
            with self._lock:  # Lock for reading LCD lines
                lcd_lines_copy = controller.lcd_lines[:]
            for i, line in enumerate(lcd_lines_copy):
                try:
                    text_surface, _ = self.lcd_font.render(
                        line[:LCD_CHAR_WIDTH].ljust(LCD_CHAR_WIDTH), LCD_FG
                    )
                    self.screen.blit(
                        text_surface,
                        (
                            controller.lcd_rect.x + 5,
                            controller.lcd_rect.y + 5 + i * (LCD_HEIGHT // LCD_CHAR_HEIGHT),
                        ),
                    )
                except Exception as e:
                    print(f"Error rendering LCD text for controller {dip}: {e}")

            # Draw buttons
            with self._lock:  # Lock for reading button states
                buttons_copy = controller.buttons[:]

            for button, rect in controller.button_rects.items():
                color = RED if buttons_copy[button.value] else DARK_GRAY
                pygame.draw.rect(self.screen, color, rect)

                # Draw button labels
                label = button.name
                try:
                    text_surface, _ = self.font.render(label, WHITE)
                    text_rect = text_surface.get_rect(center=rect.center)
                    self.screen.blit(text_surface, text_rect)

                    # Draw key binding above button (ensure dip exists in key_names)
                    if dip in self.key_names and button in self.key_names[dip]:
                        key_name = self.key_names[dip][button]
                        key_text_surface, _ = self.font.render(key_name, BLACK)
                        key_text_rect = key_text_surface.get_rect(
                            centerx=rect.centerx, bottom=rect.top - 2
                        )
                        self.screen.blit(key_text_surface, key_text_rect)
                except Exception as e:
                    print(f"Error rendering button text for controller {dip}: {e}")

            # Draw controller number
            try:
                text_surface, _ = self.font.render(f"Controller {dip}", BLUE)
                text_rect = text_surface.get_rect(
                    centerx=controller.rect.centerx, bottom=controller.rect.top - 5
                )
                self.screen.blit(text_surface, text_rect)
            except Exception as e:
                print(f"Error rendering controller number {dip}: {e}")

        pygame.display.flip()

    # --- Methods for TCP Server Interaction (Thread-safe) ---

    def set_lcd_line(self, dip: int, x: int, y: int, text: str):
        if dip in self.controllers and 0 <= x < LCD_CHAR_WIDTH and 0 <= y < LCD_CHAR_HEIGHT:
            with self._lock:
                lcd_lines = self.controllers[dip].lcd_lines
                text_to_write = text[: LCD_CHAR_WIDTH - x]
                lcd_lines[y] = (
                    lcd_lines[y][:x] + text_to_write + lcd_lines[y][x + len(text_to_write) :]
                )

    def clear_lcd(self, dip: int):
        if dip in self.controllers:
            with self._lock:
                for i in range(LCD_CHAR_HEIGHT):
                    self.controllers[dip].lcd_lines[i] = " " * LCD_CHAR_WIDTH

    async def send_button_update(self, dip: int):
        if dip not in self.controllers:
            return

        controller = self.controllers[dip]
        with self._lock:
            current_buttons = controller.buttons[:]  # Make a copy
            writer = controller.client_writer
            last_sent = controller.last_button_sent

        if writer and current_buttons != last_sent:
            try:
                msg = json.dumps({"buttons": current_buttons}) + "\n"
                writer.write(msg.encode())
                await writer.drain()
                with self._lock:  # Update last sent state only after successful send
                    controller.last_button_sent = current_buttons
                # print(f"Sent button update for DIP {dip}: {current_buttons}") # Debug
            except ConnectionResetError:
                print(f"Client disconnected from DIP {dip} while sending button update.")
                with self._lock:
                    controller.client_writer = None  # Clear writer on error
            except Exception as e:
                print(f"Error sending button update for DIP {dip}: {e}")
                with self._lock:
                    controller.client_writer = None  # Clear writer on error

    def set_client_writer(self, dip: int, writer: Optional[asyncio.StreamWriter]):
        if dip in self.controllers:
            with self._lock:
                self.controllers[dip].client_writer = writer
                if writer is None:  # Client disconnected, reset last sent state
                    self.controllers[dip].last_button_sent = self.controllers[dip].buttons[:]

    # --- Server and Main Loop ---

    async def handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, dip: int
    ):
        peername = writer.get_extra_info("peername")
        print(f"Client {peername} connected to controller DIP {dip}")
        self.set_client_writer(dip, writer)

        # Send initial button state
        await self.send_button_update(dip)

        # Create separate tasks for reading and writing
        read_task = asyncio.create_task(self._read_loop(reader, dip, peername))
        write_task = asyncio.create_task(self._write_loop(writer, dip))

        try:
            # Wait for the read task to complete - this happens when the client disconnects
            # The write task will continue running until then
            await read_task
        except Exception as e:
            print(f"Error in client handler for DIP {dip}: {e}")
        finally:
            # Cancel the write task when the read task completes
            write_task.cancel()
            try:
                await write_task
            except asyncio.CancelledError:
                pass

    async def _read_loop(self, reader, dip, peername):
        buffer = b""
        # Get writer from controller state
        writer = self.controllers[dip].client_writer if dip in self.controllers else None

        try:
            while self.running:
                try:
                    data = await reader.read(1024)
                    if not data:
                        print(f"Client {peername} (DIP {dip}) disconnected.")
                        break

                    buffer += data
                    buffer = buffer.replace(b"\r", b"")  # Clean up line endings

                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        line_str = line.decode()
                        if not line_str:
                            continue

                        print(f"Received from DIP {dip}: '{line_str}'")  # Debug
                        # Pass writer to _handle_command
                        await self._handle_command(dip, line_str, writer)

                except ConnectionResetError:
                    print(f"Client {peername} (DIP {dip}) reset connection.")
                    break
                except asyncio.CancelledError:
                    print(f"Read loop for DIP {dip} cancelled.")
                    break
                except Exception as e:
                    print(f"Error in read loop for DIP {dip}: {e}")
                    break
        finally:
            # Clean up connection when client disconnects
            print(f"Closing connection for DIP {dip}")
            self.set_client_writer(dip, None)
            if writer:
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception as e:
                    print(f"Error during writer close for DIP {dip}: {e}")

    async def _write_loop(self, writer, dip):
        try:
            while self.running and not writer.is_closing():
                # Handle any pending writes
                if dip in self.controllers:
                    await self.send_button_update(dip)
                await asyncio.sleep(0.1)  # Prevent busy-waiting

        except asyncio.CancelledError:
            print(f"Write loop for DIP {dip} cancelled.")
        except Exception as e:
            print(f"Error in write loop for DIP {dip}: {e}")

    async def _handle_command(self, dip, line_str, writer):
        parts = line_str.split(":")
        command = parts[0]

        if command == "enum":
            response = json.dumps({"type": "controller", "dip": dip}) + "\n"
            if writer:
                writer.write(response.encode())
                await writer.drain()
        elif command == "lcd" and len(parts) >= 4:
            try:
                x = int(parts[1])
                y = int(parts[2])
                text = ":".join(parts[3:])
                self.set_lcd_line(dip, x, y, text)
            except (ValueError, IndexError) as e:
                print(f"Error parsing LCD command for DIP {dip}: {line_str} - {e}")
        elif command == "lcd" and parts[1] == "clear":
            self.clear_lcd(dip)
        elif command == "noop":
            pass  # Keep connection alive
        else:
            print(f"Unknown command from DIP {dip}: {line_str}")

        # Don't close the connection after handling commands - connection should stay open

    async def start_server_for_controller(self, dip: int, port: int):
        try:
            server = await asyncio.start_server(
                lambda r, w: self.handle_client(r, w, dip), "0.0.0.0", port
            )
            self.servers.append(server)  # Keep reference
            addr = server.sockets[0].getsockname()
            print(f"Controller DIP {dip} listening on {addr}")
            async with server:
                await server.serve_forever()
        except asyncio.CancelledError:
            print(f"Server for DIP {dip} stopping...")
        except Exception as e:
            print(f"Error starting server for DIP {dip} on port {port}: {e}")
        finally:
            if "server" in locals() and server:
                server.close()
                await server.wait_closed()
                print(f"Server for DIP {dip} closed.")

    async def run_asyncio_servers(self):
        tasks = []
        with self._lock:
            controllers_to_start = list(self.controllers.values())

        for controller in controllers_to_start:
            tasks.append(self.start_server_for_controller(controller.dip, controller.port))

        if tasks:
            await asyncio.gather(*tasks)
        print("All asyncio servers finished.")

    def start_asyncio_thread(self):
        def run_loop():
            try:
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
                self.loop.run_until_complete(self.run_asyncio_servers())
            except Exception as e:
                print(f"Exception in asyncio thread: {e}")
            finally:
                if self.loop:
                    self.loop.close()
                print("Asyncio thread finished.")

        self.asyncio_thread = threading.Thread(target=run_loop, daemon=True)
        self.asyncio_thread.start()

    def run_pygame_loop(self):
        """Main loop for the Pygame GUI (must be called from main thread)"""
        if not pygame_initialized:
            print("Pygame not initialized, cannot run GUI loop.")
            # Fallback for non-GUI mode or wait? For now, just exit.
            self.running = False
            return

        clock = pygame.time.Clock()
        while self.running:
            if not self.handle_events():
                self.running = False  # handle_events signals quit

            self.draw()
            clock.tick(60)  # Limit to 60 FPS

        print("Pygame loop finishing...")
        # Signal asyncio servers to stop
        if self.loop:
            tasks = asyncio.all_tasks(self.loop)
            for task in tasks:
                if task is not asyncio.current_task(self.loop):  # Don't cancel self
                    task.cancel()
            # Give tasks time to finish cancelling
            # self.loop.run_until_complete(asyncio.sleep(0.1)) # May need careful handling

        pygame.quit()
        print("Pygame quit.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Controller Simulator")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to controller configuration JSON file",
    )
    args = parser.parse_args()

    # Ensure we are in the main thread for Pygame
    if threading.current_thread() is not threading.main_thread():
        print("Error: Simulator must be started from the main thread for Pygame GUI.")
        exit(1)

    if not pygame_initialized:
        print("Error: Pygame failed to initialize. Cannot start simulator.")
        exit(1)

    # Create and initialize the simulator
    simulator = ControllerSimulator(config_path=args.config)

    # Start the asyncio TCP servers in a background thread
    simulator.start_asyncio_thread()

    # Run the Pygame GUI loop in the main thread
    try:
        simulator.run_pygame_loop()
    except KeyboardInterrupt:
        print("Keyboard interrupt received, shutting down.")
        simulator.running = False  # Signal loops to stop

    # Wait briefly for threads to clean up (optional)
    if simulator.asyncio_thread:
        print("Waiting for asyncio thread to finish...")
        simulator.asyncio_thread.join(timeout=2.0)

    print("Simulator finished.")
