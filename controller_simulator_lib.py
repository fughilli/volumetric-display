"""
Controller Simulator Library for Testing

This module provides a reusable controller simulator that can be used in tests
to simulate the behavior of physical controllers without requiring GUI components.
"""

import asyncio
import json
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


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
    server_task: Optional[asyncio.Task] = None
    client_writer: Optional[asyncio.StreamWriter] = None
    last_button_sent: Optional[List[bool]] = None
    lcd_callback: Optional[Callable[[int, int, int, str], None]] = None
    button_callback: Optional[Callable[[int, List[bool]], None]] = None


class ControllerSimulator:
    """
    A headless controller simulator that provides TCP server functionality
    for testing control port connections.
    """

    def __init__(self):
        self.running = True
        self.controllers: Dict[int, VirtualControllerState] = {}
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.servers: List[asyncio.Server] = []
        self._lock = threading.Lock()
        self._asyncio_thread: Optional[threading.Thread] = None

    def add_controller(
        self,
        dip: int,
        port: int,
        lcd_callback: Optional[Callable[[int, int, int, str], None]] = None,
        button_callback: Optional[Callable[[int, List[bool]], None]] = None,
    ) -> None:
        """Add a controller to the simulator."""
        with self._lock:
            self.controllers[dip] = VirtualControllerState(
                dip=dip,
                port=port,
                buttons=[False] * 5,
                lcd_lines=[" " * 20 for _ in range(4)],  # 20x4 LCD
                last_button_sent=[False] * 5,
                lcd_callback=lcd_callback,
                button_callback=button_callback,
            )

    def set_button_state(self, dip: int, button: Button, pressed: bool) -> None:
        """Set the state of a button for a controller."""
        if dip in self.controllers:
            with self._lock:
                self.controllers[dip].buttons[button.value] = pressed
                # Notify asyncio thread to send updates
                if self.loop:
                    asyncio.run_coroutine_threadsafe(self.send_button_update(dip), self.loop)

    def get_button_state(self, dip: int) -> List[bool]:
        """Get the current button state for a controller."""
        if dip in self.controllers:
            with self._lock:
                return self.controllers[dip].buttons[:]
        return [False] * 5

    def get_lcd_content(self, dip: int) -> List[str]:
        """Get the current LCD content for a controller."""
        if dip in self.controllers:
            with self._lock:
                return self.controllers[dip].lcd_lines[:]
        return [" " * 20 for _ in range(4)]

    def set_lcd_line(self, dip: int, x: int, y: int, text: str) -> None:
        """Set a line of LCD text for a controller."""
        if dip in self.controllers and 0 <= x < 20 and 0 <= y < 4:
            with self._lock:
                lcd_lines = self.controllers[dip].lcd_lines
                text_to_write = text[: 20 - x]
                lcd_lines[y] = (
                    lcd_lines[y][:x] + text_to_write + lcd_lines[y][x + len(text_to_write) :]
                )

                # Call the callback if provided
                if self.controllers[dip].lcd_callback:
                    self.controllers[dip].lcd_callback(dip, x, y, text_to_write)

    def clear_lcd(self, dip: int) -> None:
        """Clear the LCD for a controller."""
        if dip in self.controllers:
            with self._lock:
                for i in range(4):
                    self.controllers[dip].lcd_lines[i] = " " * 20

    async def send_button_update(self, dip: int) -> None:
        """Send button state update to connected client."""
        if dip not in self.controllers:
            return

        controller = self.controllers[dip]
        with self._lock:
            current_buttons = controller.buttons[:]
            writer = controller.client_writer
            last_sent = controller.last_button_sent

        if writer and current_buttons != last_sent:
            try:
                msg = json.dumps({"buttons": current_buttons}) + "\n"
                writer.write(msg.encode())
                await writer.drain()
                with self._lock:
                    controller.last_button_sent = current_buttons

                # Call the button callback if provided
                if controller.button_callback:
                    controller.button_callback(dip, current_buttons)

            except Exception as e:
                print(f"Error sending button update for DIP {dip}: {e}")
                with self._lock:
                    controller.client_writer = None

    def set_client_writer(self, dip: int, writer: Optional[asyncio.StreamWriter]) -> None:
        """Set the client writer for a controller."""
        if dip in self.controllers:
            with self._lock:
                self.controllers[dip].client_writer = writer
                if writer is None:
                    self.controllers[dip].last_button_sent = self.controllers[dip].buttons[:]

    async def handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, dip: int
    ) -> None:
        """Handle a client connection for a specific controller."""
        peername = writer.get_extra_info("peername")
        print(f"Client {peername} connected to controller DIP {dip}")
        self.set_client_writer(dip, writer)

        # Send initial button state
        await self.send_button_update(dip)

        # Create separate tasks for reading and writing
        read_task = asyncio.create_task(self._read_loop(reader, dip, peername))
        write_task = asyncio.create_task(self._write_loop(writer, dip))

        try:
            await read_task
        except Exception as e:
            print(f"Error in client handler for DIP {dip}: {e}")
        finally:
            write_task.cancel()
            try:
                await write_task
            except asyncio.CancelledError:
                pass

    async def _read_loop(self, reader: asyncio.StreamReader, dip: int, peername: Any) -> None:
        """Read loop for handling incoming commands from clients."""
        buffer = b""
        writer = self.controllers[dip].client_writer if dip in self.controllers else None

        try:
            while self.running:
                try:
                    data = await reader.read(1024)
                    if not data:
                        print(f"Client {peername} (DIP {dip}) disconnected.")
                        break

                    buffer += data
                    buffer = buffer.replace(b"\r", b"")

                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        line_str = line.decode()
                        if not line_str:
                            continue

                        print(f"Received from DIP {dip}: '{line_str}'")
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
            print(f"Closing connection for DIP {dip}")
            self.set_client_writer(dip, None)
            if writer:
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception as e:
                    print(f"Error during writer close for DIP {dip}: {e}")

    async def _write_loop(self, writer: asyncio.StreamWriter, dip: int) -> None:
        """Write loop for handling outgoing messages to clients."""
        try:
            while self.running and not writer.is_closing():
                if dip in self.controllers:
                    await self.send_button_update(dip)
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            print(f"Write loop for DIP {dip} cancelled.")
        except Exception as e:
            print(f"Error in write loop for DIP {dip}: {e}")

    async def _handle_command(
        self, dip: int, line_str: str, writer: Optional[asyncio.StreamWriter]
    ) -> None:
        """Handle incoming commands from clients."""
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

    async def start_server_for_controller(self, dip: int, port: int) -> None:
        """Start a TCP server for a specific controller."""
        try:
            server = await asyncio.start_server(
                lambda r, w: self.handle_client(r, w, dip), "127.0.0.1", port
            )
            self.servers.append(server)
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

    async def run_asyncio_servers(self) -> None:
        """Run all asyncio servers."""
        tasks = []
        with self._lock:
            controllers_to_start = list(self.controllers.values())

        for controller in controllers_to_start:
            tasks.append(self.start_server_for_controller(controller.dip, controller.port))

        if tasks:
            await asyncio.gather(*tasks)
        print("All asyncio servers finished.")

    def start_asyncio_thread(self) -> None:
        """Start the asyncio event loop in a background thread."""

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

        self._asyncio_thread = threading.Thread(target=run_loop, daemon=True)
        self._asyncio_thread.start()

    def stop(self) -> None:
        """Stop the simulator and clean up resources."""
        self.running = False

        # Signal asyncio servers to stop
        if self.loop:
            tasks = asyncio.all_tasks(self.loop)
            for task in tasks:
                if task is not asyncio.current_task(self.loop):
                    task.cancel()

    def wait_for_shutdown(self, timeout: float = 2.0) -> None:
        """Wait for the simulator to shut down."""
        if self._asyncio_thread:
            self._asyncio_thread.join(timeout=timeout)
