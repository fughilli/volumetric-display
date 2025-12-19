#!/usr/bin/env python3
"""
NFC Rift Gate Listener

This script monitors controllers for NFC tag scans, extracts beacon UUIDs,
and triggers encounters at the rift gate by calling the server API.

Designed to run as a service on a Raspberry Pi.
"""

import argparse
import asyncio
import json
import logging
import socket
import sys
import time
from typing import Dict, Optional
from urllib.parse import parse_qs, urlparse

try:
    import requests
except ImportError:
    print("Error: requests library not available. Install with: pip install requests")
    sys.exit(1)

CONTROLLER_PORT = 51333
CONNECTION_TIMEOUT = 1.0


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


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class NFCListener:
    """Listens for NFC tags from controllers and triggers encounters."""

    def __init__(
        self,
        controller_ips: list[str],
        server_url: str = "https://midnight.fughil.li",
        reconnect_delay: float = 5.0,
        debounce_seconds: float = 2.0,
    ):
        """
        Initialize the NFC listener.

        Args:
            controller_ips: List of controller IP addresses to monitor
            server_url: Base URL of the Midnight Walk server
            reconnect_delay: Seconds to wait before reconnecting after a disconnect
            debounce_seconds: Minimum seconds between processing the same UUID
        """
        self.controller_ips = controller_ips
        self.server_url = server_url.rstrip("/")
        self.reconnect_delay = reconnect_delay
        self.debounce_seconds = debounce_seconds
        self.connections: Dict[str, ControllerConnection] = {}
        self.last_processed: Dict[str, float] = {}  # UUID -> timestamp
        self.running = False

    def extract_uuid_from_url(self, url: str) -> Optional[str]:
        """Extract beacon UUID from the NFC tag URL."""
        try:
            parsed = urlparse(url)
            query_params = parse_qs(parsed.query)
            beacon_id = query_params.get("id", [None])[0]
            return beacon_id
        except Exception as e:
            logger.warning(f"Failed to extract UUID from URL {url}: {e}")
            return None

    async def trigger_encounter(self, beacon_uid: str) -> bool:
        """
        Trigger an encounter by calling the server API.

        Returns True if successful, False otherwise.
        """
        # Debounce: don't process the same UUID too frequently
        now = time.time()
        if beacon_uid in self.last_processed:
            time_since = now - self.last_processed[beacon_uid]
            if time_since < self.debounce_seconds:
                logger.debug(f"Debouncing UUID {beacon_uid} (last processed {time_since:.1f}s ago)")
                return False

        self.last_processed[beacon_uid] = now

        try:
            url = f"{self.server_url}/api/nfc-lookup"
            params = {"beacon_uid": beacon_uid, "start_encounter": True}

            logger.info(f"Triggering encounter for beacon UUID: {beacon_uid}")
            response = requests.get(url, params=params, timeout=5.0)

            if response.status_code == 404:
                logger.warning(f"Beacon not found: {beacon_uid}")
                return False

            if not response.ok:
                logger.error(
                    f"Failed to trigger encounter: HTTP {response.status_code} - {response.text}"
                )
                return False

            data = response.json()

            if data.get("error"):
                error_msg = data.get("message", "Unknown error")
                if data.get("error") == "active_encounter":
                    logger.info(f"Another encounter is active: {error_msg}")
                else:
                    logger.warning(f"Encounter error: {error_msg}")
                return False

            spirit_name = data.get("spirit_name", "Unknown Spirit")
            logger.info(f"✓ Encounter started with {spirit_name} (UUID: {beacon_uid})")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error triggering encounter: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error triggering encounter: {e}", exc_info=True)
            return False

    async def monitor_controller(self, ip: str):
        """Monitor a single controller for NFC tags."""
        logger.info(f"Starting monitor for controller at {ip}")
        loop = asyncio.get_running_loop()

        while self.running:
            conn = ControllerConnection(ip)
            self.connections[ip] = conn

            try:
                if not await conn.connect():
                    logger.warning(
                        f"Failed to connect to {ip}, retrying in {self.reconnect_delay}s..."
                    )
                    await asyncio.sleep(self.reconnect_delay)
                    continue

                logger.info(f"Connected to controller at {ip}")
                receive_buffer = b""

                while self.running:
                    try:
                        data = await asyncio.wait_for(
                            loop.sock_recv(conn._socket, 4096), timeout=1.0
                        )

                        if not data:
                            logger.warning(f"Connection closed by controller {ip}")
                            break

                        receive_buffer += data

                        # Process complete messages
                        while b"\n" in receive_buffer:
                            message, receive_buffer = receive_buffer.split(b"\n", 1)

                            try:
                                msg_str = message.decode().strip()
                                if not msg_str:
                                    continue

                                try:
                                    msg_json = json.loads(msg_str)

                                    # Process NFC tag messages
                                    if msg_json.get("type") == "nfc_tag":
                                        uid = msg_json.get("uid", "unknown")
                                        url = msg_json.get("url", "")

                                        logger.debug(
                                            f"NFC tag detected on {ip}: UID={uid}, URL={url}"
                                        )

                                        if url:
                                            beacon_uid = self.extract_uuid_from_url(url)
                                            if beacon_uid:
                                                await self.trigger_encounter(beacon_uid)
                                            else:
                                                logger.warning(
                                                    f"Could not extract UUID from URL: {url}"
                                                )
                                        else:
                                            logger.debug(f"NFC tag on {ip} has no URL (UID: {uid})")

                                except json.JSONDecodeError:
                                    # Not JSON, ignore
                                    pass

                            except UnicodeDecodeError:
                                pass

                    except asyncio.TimeoutError:
                        # No data received, continue monitoring
                        continue
                    except Exception as e:
                        logger.error(f"Error reading from {ip}: {e}", exc_info=True)
                        break

            except Exception as e:
                logger.error(f"Error monitoring {ip}: {e}", exc_info=True)
            finally:
                conn.disconnect()
                logger.warning(
                    f"Disconnected from {ip}, reconnecting in {self.reconnect_delay}s..."
                )
                await asyncio.sleep(self.reconnect_delay)

    async def start(self):
        """Start monitoring all controllers."""
        self.running = True
        logger.info("Starting NFC Rift Gate Listener")
        logger.info(
            f"Monitoring {len(self.controller_ips)} controller(s): {', '.join(self.controller_ips)}"
        )
        logger.info(f"Server URL: {self.server_url}")
        logger.info("=" * 70)

        # Start monitoring tasks for all controllers
        tasks = [self.monitor_controller(ip) for ip in self.controller_ips]

        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
        finally:
            self.running = False
            # Disconnect all connections
            for conn in self.connections.values():
                conn.disconnect()
            logger.info("NFC Rift Gate Listener stopped")


async def main():
    parser = argparse.ArgumentParser(
        description="NFC Rift Gate Listener - Monitors controllers and triggers encounters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Monitor a single controller
  %(prog)s --controller 2.0.99.150

  # Monitor multiple controllers
  %(prog)s --controller 2.0.99.150 --controller 2.0.99.151

  # Use custom server URL
  %(prog)s --controller 2.0.99.150 --server-url http://localhost:8000

  # Adjust debounce time (default: 2 seconds)
  %(prog)s --controller 2.0.99.150 --debounce 5.0
        """,
    )

    parser.add_argument(
        "--controller",
        action="append",
        dest="controllers",
        required=True,
        help="Controller IP address (can be specified multiple times)",
    )
    parser.add_argument(
        "--server-url",
        default="https://midnight.fughil.li",
        help="Base URL of the Midnight Walk server (default: https://midnight.fughil.li)",
    )
    parser.add_argument(
        "--reconnect-delay",
        type=float,
        default=5.0,
        help="Seconds to wait before reconnecting after disconnect (default: 5.0)",
    )
    parser.add_argument(
        "--debounce",
        type=float,
        default=2.0,
        help="Minimum seconds between processing the same UUID (default: 2.0)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    listener = NFCListener(
        controller_ips=args.controllers,
        server_url=args.server_url,
        reconnect_delay=args.reconnect_delay,
        debounce_seconds=args.debounce,
    )

    try:
        await listener.start()
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
