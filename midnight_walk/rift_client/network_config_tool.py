#!/usr/bin/env python3
"""
Network Configuration Tool for Eth Cube Controllers

This utility allows you to:
1. Discover available controllers on the network
2. Configure network settings (base IP, subnet mask, gateway, target IP)
3. Monitor controller state in real-time for debugging
"""

import argparse
import asyncio
import base64
import json
import socket
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

CONTROLLER_PORT = 51333
CONNECTION_TIMEOUT = 1.0  # Reduced for faster discovery
DISCOVERY_TIMEOUT = 5.0


class ControllerConnection:
    """Manages a connection to a single controller."""

    def __init__(self, ip: str, port: int = CONTROLLER_PORT):
        self.ip = ip
        self.port = port
        self._socket: Optional[socket.socket] = None
        self._connected = False

    async def connect(self, loop=None):
        """Connect to the controller."""
        if self._connected:
            return True

        if loop is None:
            loop = asyncio.get_running_loop()

        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(CONNECTION_TIMEOUT)
            await loop.sock_connect(self._socket, (self.ip, self.port))
            self._socket.setblocking(False)
            self._connected = True
            return True
        except Exception:
            # Silently handle connection errors during discovery
            self._connected = False
            return False

    def disconnect(self):
        """Disconnect from the controller."""
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
        self._socket = None
        self._connected = False

    async def send_command(
        self, loop=None, command: str = "", timeout: float = 2.0
    ) -> Optional[str]:
        """Send a command and optionally wait for response."""
        if loop is None:
            loop = asyncio.get_running_loop()

        if not self._connected:
            if not await self.connect(loop):
                return None

        try:
            msg = f"{command}\n".encode()
            await loop.sock_sendall(self._socket, msg)

            # Wait for response (with timeout)
            try:
                data = await asyncio.wait_for(loop.sock_recv(self._socket, 4096), timeout=timeout)
                if data:
                    return data.decode().strip()
            except asyncio.TimeoutError:
                return None

        except Exception as e:
            # Only print errors for non-discovery operations
            if timeout > 1.5:  # Regular operations have longer timeouts
                print(f"Error sending command to {self.ip}: {e}")
            self.disconnect()
            return None

    async def set_leds(self, rgb_values: List[Tuple[int, int, int]], loop=None) -> bool:
        """Set LED colors from a list of (r,g,b) tuples."""
        if loop is None:
            loop = asyncio.get_running_loop()

        if not self._connected:
            if not await self.connect(loop):
                return False

        try:
            # Create payload: [num_leds (16-bit LE), r0,g0,b0, r1,g1,b1, ...]
            num_leds = len(rgb_values)
            payload = bytearray([num_leds & 0xFF, (num_leds >> 8) & 0xFF])
            for r, g, b in rgb_values:
                payload.extend([r & 0xFF, g & 0xFF, b & 0xFF])

            # Encode as base64 and send command
            encoded = base64.b64encode(bytes(payload)).decode()
            msg = f"led:{encoded}\n".encode()
            await loop.sock_sendall(self._socket, msg)
            return True
        except Exception as e:
            print(f"Error setting LEDs on {self.ip}: {e}")
            self.disconnect()
            return False


class NetworkConfigTool:
    """Main tool for managing network configurations."""

    def __init__(
        self, base_ip: str = "2.0.99", start: int = 150, end: int = 165, max_concurrent: int = 200
    ):
        self.base_ip = base_ip
        self.start = start
        self.end = end
        self.max_concurrent = max_concurrent

    async def discover_controllers(self) -> Dict[str, dict]:
        """Discover available controllers on the network."""
        total_ips = self.end - self.start + 1
        print(f"\n🔍 Discovering controllers in range {self.base_ip}.{self.start}-{self.end}...")
        print(
            f"    Scanning {total_ips} IP addresses in parallel (max {self.max_concurrent} concurrent)"
        )
        print("=" * 70)

        controllers = {}

        # Create semaphore to limit concurrent connections (prevents overwhelming system)
        semaphore = asyncio.Semaphore(self.max_concurrent)

        # Create all probe tasks - all will run in parallel (limited by semaphore)
        tasks = []
        for i in range(self.start, self.end + 1):
            ip = f"{self.base_ip}.{i}"
            tasks.append(self._probe_controller_with_semaphore(ip, semaphore))

        # Execute all tasks concurrently and collect results
        # return_exceptions=True ensures one failure doesn't stop others
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for result in results:
            if result and not isinstance(result, Exception):
                ip, info = result
                controllers[ip] = info
                print(f"✓ Found controller at {ip} (DIP: {info.get('dip', 'unknown')})")

        print("=" * 70)
        print(f"Found {len(controllers)} controller(s) out of {total_ips} scanned\n")
        return controllers

    async def _probe_controller_with_semaphore(
        self, ip: str, semaphore: asyncio.Semaphore
    ) -> Optional[tuple]:
        """Probe a controller with semaphore for rate limiting."""
        async with semaphore:
            return await self._probe_controller(ip)

    async def _probe_controller(self, ip: str) -> Optional[tuple]:
        """Probe a single IP address for a controller."""
        conn = ControllerConnection(ip)

        try:
            # Use shorter timeout for discovery to speed up scanning
            if await conn.connect():
                # Send enum command with shorter timeout
                response = await conn.send_command(command="enum", timeout=0.5)

                if response:
                    try:
                        data = json.loads(response)
                        if data.get("type") == "controller":
                            return (ip, data)
                    except json.JSONDecodeError:
                        pass
        except asyncio.TimeoutError:
            # Silently ignore timeouts during discovery
            pass
        except Exception:
            # Silently ignore other connection errors during discovery
            pass
        finally:
            conn.disconnect()

        return None

    async def get_network_config(self, ip: str) -> Optional[dict]:
        """Get the current network configuration from a controller."""
        conn = ControllerConnection(ip)

        try:
            if await conn.connect():
                response = await conn.send_command(command="netconfig:get")

                if response:
                    try:
                        # Response might contain multiple lines, find JSON
                        for line in response.split("\n"):
                            if line.strip().startswith("{"):
                                return json.loads(line.strip())
                    except json.JSONDecodeError as e:
                        print(f"Error parsing config response: {e}")
                        print(f"Response was: {response}")
        finally:
            conn.disconnect()

        return None

    async def set_network_config(self, ip: str, field: str, value: str) -> bool:
        """Set a network configuration field on a controller."""
        conn = ControllerConnection(ip)

        try:
            if await conn.connect():
                command = f"netconfig:set:{field}:{value}"
                response = await conn.send_command(command=command)

                if response:
                    print(f"Response: {response}")
                    return "saved" in response.lower() or "status" in response.lower()
        finally:
            conn.disconnect()

        return False

    async def reset_network_config(self, ip: str) -> bool:
        """Reset network configuration to defaults."""
        conn = ControllerConnection(ip)

        try:
            if await conn.connect():
                response = await conn.send_command(command="netconfig:reset")

                if response:
                    print(f"Response: {response}")
                    return "reset" in response.lower()
        finally:
            conn.disconnect()

        return False

    async def set_leds(self, ip: str, rgb_values: List[Tuple[int, int, int]]) -> bool:
        """Set LED colors on a controller."""
        conn = ControllerConnection(ip)

        try:
            if await conn.connect():
                return await conn.set_leds(rgb_values)
        finally:
            conn.disconnect()

        return False

    async def monitor_controller(self, ip: str, duration: int = 0):
        """Monitor a controller's state in real-time."""
        print(f"\n📡 Monitoring controller at {ip}")
        print("Press Ctrl+C to stop\n")
        print("=" * 70)

        conn = ControllerConnection(ip)
        loop = asyncio.get_running_loop()

        try:
            if not await conn.connect():
                print("Failed to connect to controller")
                return

            # Set up receive buffer
            receive_buffer = b""
            start_time = time.time()

            while True:
                # Check duration
                if duration > 0 and (time.time() - start_time) >= duration:
                    break

                try:
                    data = await asyncio.wait_for(loop.sock_recv(conn._socket, 4096), timeout=1.0)

                    if not data:
                        print("Connection closed by controller")
                        break

                    receive_buffer += data

                    # Process complete messages
                    while b"\n" in receive_buffer:
                        message, receive_buffer = receive_buffer.split(b"\n", 1)

                        try:
                            msg_str = message.decode().strip()
                            if msg_str:
                                # Try to parse as JSON
                                try:
                                    msg_json = json.loads(msg_str)
                                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

                                    # Special formatting for NFC tag messages
                                    if msg_json.get("type") == "nfc_tag":
                                        uid = msg_json.get("uid", "unknown")
                                        uid_length = msg_json.get("uid_length", 0)
                                        tag_data = msg_json.get("data", "")
                                        url = msg_json.get("url", "")

                                        print(f"[{timestamp}] 🏷️  NFC Tag Detected:")
                                        print(f"    UID: {uid} (length: {uid_length})")
                                        if tag_data:
                                            print(f"    Data (hex): {tag_data}")
                                            # Convert hex string to ASCII string
                                            try:
                                                # Convert hex string to bytes, then to ASCII string
                                                hex_bytes = bytes.fromhex(tag_data)
                                                # Filter out non-printable characters and null bytes
                                                ascii_str = "".join(
                                                    chr(b) if 32 <= b < 127 else "."
                                                    for b in hex_bytes
                                                )
                                                # Remove trailing null bytes and padding
                                                ascii_str = ascii_str.rstrip("\x00").rstrip(".")
                                                if ascii_str:
                                                    print(f"    Data (ASCII): {ascii_str}")
                                            except ValueError:
                                                pass  # Invalid hex, skip ASCII conversion
                                        if url:
                                            print(f"    URL: {url}")
                                    else:
                                        # Default JSON formatting for other message types
                                        print(f"[{timestamp}] {json.dumps(msg_json, indent=2)}")
                                except json.JSONDecodeError:
                                    # Not JSON, just print it
                                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                                    print(f"[{timestamp}] {msg_str}")
                        except UnicodeDecodeError:
                            pass

                except asyncio.TimeoutError:
                    # No data received, continue monitoring
                    pass

        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        except Exception as e:
            print(f"Error during monitoring: {e}")
        finally:
            conn.disconnect()
            print("=" * 70)


async def main():
    parser = argparse.ArgumentParser(
        description="Network Configuration Tool for Eth Cube Controllers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Discover all controllers
  %(prog)s discover

  # Get network config from a controller
  %(prog)s get 2.0.99.150

  # Set base IP on a controller
  %(prog)s set 2.0.99.150 base_ip 10.0.0.100

  # Set subnet mask
  %(prog)s set 2.0.99.150 subnet_mask 255.255.255.0

  # Set target IP
  %(prog)s set 2.0.99.150 target_ip 10.0.0.1

  # Reset to defaults
  %(prog)s reset 2.0.99.150

  # Monitor controller state
  %(prog)s monitor 2.0.99.150

  # Set LED colors (single RGB for all LEDs)
  %(prog)s led 2.0.99.150 255 0 0

  # Set LED colors (multiple RGB values, one per LED)
  %(prog)s led 2.0.99.150 255 0 0 0 255 0 0 0 255

  # Turn off all LEDs
  %(prog)s led 2.0.99.150 0 0 0
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Discover command
    discover_parser = subparsers.add_parser("discover", help="Discover controllers on the network")
    discover_parser.add_argument(
        "--base-ip", default="2.0.99", help="Base IP address (default: 2.0.99)"
    )
    discover_parser.add_argument(
        "--start", type=int, default=150, help="Start of IP range (default: 150)"
    )
    discover_parser.add_argument(
        "--end", type=int, default=165, help="End of IP range (default: 165)"
    )
    discover_parser.add_argument(
        "--max-concurrent", type=int, default=200, help="Max concurrent connections (default: 200)"
    )

    # Get config command
    get_parser = subparsers.add_parser("get", help="Get network configuration from a controller")
    get_parser.add_argument("ip", help="Controller IP address")

    # Set config command
    set_parser = subparsers.add_parser("set", help="Set network configuration on a controller")
    set_parser.add_argument("ip", help="Controller IP address")
    set_parser.add_argument(
        "field",
        choices=["base_ip", "subnet_mask", "gateway", "target_ip", "local_port", "target_port"],
        help="Configuration field to set",
    )
    set_parser.add_argument("value", help="New value")

    # Reset command
    reset_parser = subparsers.add_parser("reset", help="Reset network configuration to defaults")
    reset_parser.add_argument("ip", help="Controller IP address")

    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Monitor controller state in real-time")
    monitor_parser.add_argument("ip", help="Controller IP address")
    monitor_parser.add_argument(
        "--duration", type=int, default=0, help="Duration in seconds (0 = infinite)"
    )

    # LED command
    led_parser = subparsers.add_parser("led", help="Control LED colors on a controller")
    led_parser.add_argument("ip", help="Controller IP address")
    led_parser.add_argument(
        "rgb",
        nargs="+",
        type=int,
        help=(
            "RGB values: either 3 values (R G B) for all LEDs, "
            "or multiple sets of 3 values (R1 G1 B1 R2 G2 B2 ...) for individual LEDs"
        ),
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Create tool instance
    if args.command == "discover":
        tool = NetworkConfigTool(args.base_ip, args.start, args.end, args.max_concurrent)
        controllers = await tool.discover_controllers()

        if controllers:
            print("\n📋 Controller Summary:")
            print("=" * 70)
            for ip, info in controllers.items():
                print(f"  IP: {ip}")
                print(f"  DIP: {info.get('dip', 'unknown')}")
                print()

    elif args.command == "get":
        tool = NetworkConfigTool()
        config = await tool.get_network_config(args.ip)

        if config:
            print(f"\n⚙️  Network Configuration for {args.ip}:")
            print("=" * 70)
            print(json.dumps(config, indent=2))
            print()
        else:
            print(f"Failed to get configuration from {args.ip}")
            sys.exit(1)

    elif args.command == "set":
        tool = NetworkConfigTool()
        print(f"\n⚙️  Setting {args.field} = {args.value} on {args.ip}...")

        if await tool.set_network_config(args.ip, args.field, args.value):
            print(f"✓ Successfully updated {args.field}")
            print("\n⚠️  Please reboot the controller for changes to take effect")
        else:
            print(f"✗ Failed to update {args.field}")
            sys.exit(1)

    elif args.command == "reset":
        tool = NetworkConfigTool()
        print(f"\n⚠️  Resetting network configuration on {args.ip} to defaults...")

        if await tool.reset_network_config(args.ip):
            print("✓ Successfully reset to defaults")
            print("\n⚠️  Please reboot the controller for changes to take effect")
        else:
            print("✗ Failed to reset configuration")
            sys.exit(1)

    elif args.command == "monitor":
        tool = NetworkConfigTool()
        await tool.monitor_controller(args.ip, args.duration)

    elif args.command == "led":
        tool = NetworkConfigTool()

        # Parse RGB values
        rgb_values = []
        rgb_args = args.rgb

        # Validate that we have a multiple of 3 values
        if len(rgb_args) % 3 != 0:
            print("Error: RGB values must be provided in sets of 3 (R G B)")
            print(f"Received {len(rgb_args)} values, which is not a multiple of 3")
            sys.exit(1)

        # Parse RGB tuples
        for i in range(0, len(rgb_args), 3):
            r, g, b = rgb_args[i], rgb_args[i + 1], rgb_args[i + 2]

            # Validate range
            if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
                print("Error: RGB values must be between 0 and 255")
                print(f"Received: R={r}, G={g}, B={b}")
                sys.exit(1)

            rgb_values.append((r, g, b))

        # If only one RGB value provided, use it for all LEDs (or just one LED)
        # The controller will handle the number of LEDs it has
        if len(rgb_values) == 1:
            print(
                f"Setting all LEDs on {args.ip} to RGB({rgb_values[0][0]}, {rgb_values[0][1]}, {rgb_values[0][2]})"
            )
        else:
            print(f"Setting {len(rgb_values)} LED(s) on {args.ip}:")
            for i, (r, g, b) in enumerate(rgb_values):
                print(f"  LED {i}: RGB({r}, {g}, {b})")

        if await tool.set_leds(args.ip, rgb_values):
            print("✓ LED colors set successfully")
        else:
            print("✗ Failed to set LED colors")
            sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
