import asyncio
import base64
import json
import platform
import socket
import subprocess
import time

ENUM_COMMAND = b"enum\n"
BUTTON_TIMEOUT = 0.1
CONNECTION_TIMEOUT = 2.0
PING_TIMEOUT = 1.0  # seconds


class ControllerState:
    def __init__(self, ip, dip, port, width=20, height=4):
        self.ip = ip
        self.dip = dip
        self.port = port
        self.button_callback = None
        self._listen_task = None
        self._heartbeat_task = None
        self._socket = None
        self._connected = False
        self._buffer = b""
        self._last_message_time = 0  # Track last message time
        # Store display dimensions
        self._display_width = width
        self._display_height = height
        # Initialize front and back buffers as 2D arrays of characters
        self._front_buffer = [[" " for _ in range(width)] for _ in range(height)]
        self._back_buffer = [[" " for _ in range(width)] for _ in range(height)]
        self._lcd_cache = {}  # Cache for LCD content

    async def connect(self):
        if self._connected:
            return True
        loop = asyncio.get_running_loop()
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(CONNECTION_TIMEOUT)
            await loop.sock_connect(self._socket, (self.ip, self.port))
            self._socket.setblocking(False)
            self._connected = True
            self._last_message_time = time.monotonic()
            # Start heartbeat task
            self._heartbeat_task = loop.create_task(self._send_heartbeats())

            # Restore LCD state after reconnection
            await self._send("lcd:clear\n".encode())
            # Send the entire front buffer
            for y in range(self._display_height):
                line = "".join(self._front_buffer[y])
                if line.strip():  # Only send non-empty lines
                    await self._send(f"lcd:0:{y}:{line}\n".encode())

            return True
        except Exception as e:
            print(f"Failed to connect to {self.ip}: {e}")
            self._connected = False
            return False

    def disconnect(self):
        if self._heartbeat_task:
            try:
                loop = self._heartbeat_task.get_loop()
                if loop.is_running() and not loop.is_closed():
                    self._heartbeat_task.cancel()
            except RuntimeError:
                # Loop already closed; ignore
                pass
            self._heartbeat_task = None
        if self._socket:
            try:
                self._socket.close()
            # Note: without a bare except here, the controllers fail to
            # enumerate.
            except:  # noqa: E722
                pass
        self._socket = None
        self._connected = False

    def clear(self):
        """Clear the back buffer by filling it with spaces."""
        for y in range(self._display_height):
            for x in range(self._display_width):
                self._back_buffer[y][x] = " "

    def write_lcd(self, x, y, text):
        """Write text to the back buffer at position (x,y)."""
        if not (0 <= y < self._display_height and 0 <= x < self._display_width):
            return
        for i, char in enumerate(text):
            if x + i < self._display_width:  # Ensure we don't write past the display width
                self._back_buffer[y][x + i] = char

    def _find_contiguous_changes(self, y):
        """Find contiguous regions of changes in a line."""
        changes = []
        start = None
        last_change_end = None

        for x in range(self._display_width):
            if self._front_buffer[y][x] != self._back_buffer[y][x]:
                if start is None:
                    # If within 3 chars of previous change, extend previous change
                    if last_change_end is not None and x - last_change_end <= 3:
                        changes[-1] = (changes[-1][0], x + 1)
                        last_change_end = x + 1
                    else:
                        start = x
                        last_change_end = None
            elif start is not None:
                changes.append((start, x))
                last_change_end = x
                start = None

        if start is not None:
            changes.append((start, self._display_width))

        return changes

    async def commit(self):
        """Compare back buffer to front buffer and send only the changes."""
        if not self._connected:
            if not await self.connect():
                return

        # If back buffer is empty (all spaces), send clear command
        if all(all(c == " " for c in row) for row in self._back_buffer):
            await self._send("lcd:clear\n".encode())
            self._front_buffer = [
                [" " for _ in range(self._display_width)] for _ in range(self._display_height)
            ]
            return

        # Find and send changes line by line
        for y in range(self._display_height):
            changes = self._find_contiguous_changes(y)
            for start, end in changes:
                # Get the new text for this region
                new_text = "".join(self._back_buffer[y][start:end])
                # Send the update command
                msg = f"lcd:{start}:{y}:{new_text}\n".encode()
                await self._send(msg)
                # Update the front buffer
                for x in range(start, end):
                    self._front_buffer[y][x] = self._back_buffer[y][x]

    async def clear_lcd(self):
        """Clear the LCD display."""
        self.clear()
        await self.commit()

    async def set_lcd(self, x, y, text):
        """Write text to the LCD display at position (x,y) and commit immediately."""
        self.write_lcd(x, y, text)
        await self.commit()

    async def set_backlights(self, states):
        payload = ":".join(["1" if s else "0" for s in states])
        msg = f"backlight:{payload}\n".encode()
        await self._send(msg)

    async def set_leds(self, rgb_values):
        """Set LED colors from a list of (r,g,b) tuples."""
        # Create payload: [num_leds (16-bit LE), r0,g0,b0, r1,g1,b1, ...]
        num_leds = len(rgb_values)
        payload = bytearray([num_leds & 0xFF, (num_leds >> 8) & 0xFF])
        for r, g, b in rgb_values:
            payload.extend([r & 0xFF, g & 0xFF, b & 0xFF])
        msg = f"led:{base64.b64encode(bytes(payload)).decode()}\n".encode()
        await self._send(msg)

    def register_button_callback(self, callback):
        self.button_callback = callback
        if not self._listen_task:
            self._listen_task = asyncio.get_running_loop().create_task(self._listen_buttons())

    async def _send(self, msg):
        if not self._connected:
            if not await self.connect():
                return
        loop = asyncio.get_running_loop()
        try:
            await loop.sock_sendall(self._socket, msg)
        except Exception as e:
            print(f"Error sending to {self.ip}: {e}")
            self.disconnect()

    async def _send_heartbeats(self):
        """Send noop messages every second to keep connection alive."""
        while True:
            try:
                if self._connected:
                    await self._send(b"noop\n")
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error sending heartbeat to {self.ip}: {e}")
                self.disconnect()
                break

    async def _listen_buttons(self):
        while True:
            if not self._connected:
                if not await self.connect():
                    await asyncio.sleep(1.0)
                    continue

            try:
                loop = asyncio.get_running_loop()
                data = await loop.sock_recv(self._socket, 1024)
                if not data:  # Connection closed
                    self.disconnect()
                    continue

                self._last_message_time = time.monotonic()  # Update last message time
                self._buffer += data
                # Remove all \r characters and split on \n
                self._buffer = self._buffer.replace(b"\r", b"")
                parts = self._buffer.split(b"\n")

                # Keep the last part if it's not a complete message
                self._buffer = parts[-1] if not self._buffer.endswith(b"\n") else b""

                # Process all complete messages
                messages_to_process = parts[:-1] if not self._buffer else parts

                for part in messages_to_process:
                    if not part:
                        continue
                    part_str = part.strip().decode()
                    if not part_str:
                        continue
                    try:
                        msg = json.loads(part_str)
                        if msg.get("type") == "heartbeat":
                            # Respond to heartbeat with noop
                            await self._send(b"noop\n")
                        elif "buttons" in msg and self.button_callback:
                            self.button_callback(msg["buttons"])
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e} on data: '{part_str}'")
                    except Exception as e:
                        print(f"Error processing message: {e}")

                # Check for timeout (5 seconds without messages)
                if time.monotonic() - self._last_message_time > 5.0:
                    print(f"No messages from {self.ip} for 5 seconds, disconnecting...")
                    self.disconnect()
                    continue

            except ConnectionResetError:
                print(f"Connection reset by {self.ip}. Reconnecting...")
                self.disconnect()
                await asyncio.sleep(1.0)
            except Exception as e:
                print(f"Error reading from socket {self.ip}: {e}")
                self.disconnect()
                await asyncio.sleep(1.0)


class ControlPort:
    def __init__(self, hosts_and_ports=[], loop=None):
        self.hosts_and_ports = hosts_and_ports
        self.loop = loop or asyncio.get_event_loop()
        self.controllers = {}

    async def ping_host(self, ip):
        """Ping a host and return True if it responds."""
        loop = asyncio.get_running_loop()
        param = "-n" if platform.system().lower() == "windows" else "-c"
        command = ["ping", param, "1", "-W", str(int(PING_TIMEOUT * 1000)), ip]
        try:
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    command, capture_output=True, text=True, timeout=PING_TIMEOUT
                ),
            )
            return result.returncode == 0
        except Exception as e:
            print(f"Error pinging {ip}: {e}")
            return False

    async def check_port(self, ip, port):
        """Check if the port is open using a socket connection."""
        loop = asyncio.get_running_loop()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.0)  # Short timeout for port check
        try:
            await loop.sock_connect(sock, (ip, port))
            print(f"Port {port} is open on {ip}")
            return True
        except Exception as e:
            print(f"Port {port} is closed on {ip}: {e}")
            return False
        finally:
            sock.close()

    async def discover_hosts(self):
        """Discover reachable hosts using ICMP ping and port check."""
        print("Discovering reachable hosts...")
        tasks = []
        for host_and_port in self.hosts_and_ports:
            ip, port = host_and_port
            tasks.append(self.ping_host(ip))

        results = await asyncio.gather(*tasks)
        reachable_hosts_and_ports = []
        for i, is_reachable in enumerate(results):
            ip, port = self.hosts_and_ports[i]
            if is_reachable:
                print(f"Host {ip} is reachable")
                # Check if the port is open
                if await self.check_port(ip, port):
                    reachable_hosts_and_ports.append((ip, port))
            else:
                print(f"Host {ip} is not reachable")
        return reachable_hosts_and_ports

    async def enumerate(self, timeout=2.0):
        # First discover reachable hosts
        reachable_hosts_and_ports = await self.discover_hosts()
        if not reachable_hosts_and_ports:
            print("No reachable hosts found")
            return self.controllers

        # Then try to enumerate only the reachable hosts
        tasks = []
        print(
            f"DEBUG: ControlPort.enumerate: reachable_hosts_and_ports = "
            f"{reachable_hosts_and_ports}"
        )
        for host_and_port_tuple in reachable_hosts_and_ports:
            ip_addr, port_num = host_and_port_tuple
            print(
                f"DEBUG: ControlPort.enumerate: Creating task for _query_controller "
                f"with ip={ip_addr}, port={port_num}"
            )
            tasks.append(self._query_controller(ip_addr, port_num, timeout / 2))

        try:
            results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout)
        except asyncio.TimeoutError:
            print("Global enumeration timeout")
            results = []

        for result in results:
            if result:
                ip, port, dip = result
                self.controllers[dip] = ControllerState(ip, dip, port)
        return self.controllers

    async def _query_controller(self, ip, port, timeout):
        print(f"\nAttempting to enumerate {ip}:{port}")
        loop = asyncio.get_running_loop()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            print(f"1. Connecting to {ip}:{port}")
            await loop.sock_connect(sock, (ip, port))
            print("2. Connection successful, setting non-blocking mode")
            sock.setblocking(False)

            print("3. Sending enum command")
            await loop.sock_sendall(sock, ENUM_COMMAND)
            print("4. Command sent, waiting for response")

            # Keep reading until we get a valid response or timeout
            buffer = b""
            start_time = time.monotonic()
            while time.monotonic() - start_time < timeout:
                try:
                    data = await loop.sock_recv(sock, 1024)
                    if not data:
                        break

                    buffer += data
                    # Remove all \r characters and split on \n
                    buffer = buffer.replace(b"\r", b"")
                    parts = buffer.split(b"\n")

                    # Keep the last part if it's not a complete message
                    buffer = parts[-1] if not buffer.endswith(b"\n") else b""

                    # Process all complete messages
                    messages_to_process = parts[:-1] if not buffer else parts

                    for part in messages_to_process:
                        if not part:
                            continue
                        part_str = part.strip().decode()
                        if not part_str:
                            continue
                        try:
                            msg = json.loads(part_str)
                            if msg.get("type") == "heartbeat":
                                # Respond to heartbeat with noop
                                await loop.sock_sendall(sock, b"noop\n")
                            elif msg.get("type") == "controller" and "dip" in msg:
                                print(
                                    f"6. Successfully enumerated controller with DIP={msg['dip']}"
                                )
                                return (ip, port, msg["dip"])
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON: {e} on data: '{part_str}'")
                            continue
                except asyncio.TimeoutError:
                    break
                except Exception as e:
                    print(f"Error reading from socket: {e}")
                    break

            print("6. No valid controller response received")
            return None
        except socket.timeout:
            print(f"Timeout while communicating with {ip}")
            return None
        except ConnectionRefusedError:
            print(f"Connection refused by {ip}")
            return None
        except ConnectionResetError:
            print(f"Connection reset by {ip}")
            return None
        except Exception as e:
            print(f"Error communicating with {ip}: {type(e).__name__}: {e}")
            return None
        finally:
            sock.close()
