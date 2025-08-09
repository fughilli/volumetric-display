"""
Rust-based control port implementation with Python wrapper.

This module provides a drop-in replacement for control_port.py using the
high-performance Rust implementation with async sockets and web monitoring.
"""

import asyncio
import json
from typing import Any, Callable, Dict, List

try:
    from artnet_rs import ControllerManager, ControllerState

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    print("Warning: Rust control port not available, falling back to Python implementation")
    import control_port as fallback_control_port


class ControllerStateWrapper:
    """Python wrapper for Rust ControllerState that maintains API compatibility."""

    def __init__(self, rust_controller: "ControllerState"):
        self.rust_controller = rust_controller
        self.button_callback = None
        self._button_receiver = None

        # Compatibility attributes
        self.dip = rust_controller.dip
        self._connected = True  # Will be updated via properties

        # Display dimensions (matching original)
        self._display_width = 20
        self._display_height = 4

    @property
    def ip(self):
        """Get controller IP address."""
        return (
            self.rust_controller.config.ip if hasattr(self.rust_controller, "config") else "unknown"
        )

    @property
    def port(self):
        """Get controller port."""
        return self.rust_controller.config.port if hasattr(self.rust_controller, "config") else 0

    @property
    def _connected(self):
        """Check if controller is connected."""
        try:
            return self.rust_controller.connected
        except:  # noqa: E722
            return False

    async def connect(self):
        """Connect to the controller. In Rust implementation, this is handled automatically."""
        return self._connected

    def disconnect(self):
        """Disconnect from controller. In Rust implementation, this is handled automatically."""
        pass

    def clear(self):
        """Clear the back buffer by filling it with spaces."""
        self.rust_controller.clear_display()

    def write_lcd(self, x: int, y: int, text: str):
        """Write text to the back buffer at position (x,y)."""
        self.rust_controller.write_display(x, y, text)

    async def commit(self):
        """Compare back buffer to front buffer and send only the changes."""
        self.rust_controller.commit_display()

    async def clear_lcd(self):
        """Clear the LCD display."""
        self.clear()
        await self.commit()

    async def set_lcd(self, x: int, y: int, text: str):
        """Write text to the LCD display at position (x,y) and commit immediately."""
        self.write_lcd(x, y, text)
        await self.commit()

    async def set_backlights(self, states: List[bool]):
        """Set backlight states."""
        self.rust_controller.set_backlights(states)

    async def set_leds(self, rgb_values: List[tuple]):
        """Set LED colors from a list of (r,g,b) tuples."""
        self.rust_controller.set_leds(rgb_values)

    def register_button_callback(self, callback: Callable):
        """Register a callback for button events."""
        self.button_callback = callback
        if self._button_receiver is None:
            try:
                self._button_receiver = self.rust_controller.register_button_callback(
                    self._button_wrapper
                )
                self._button_receiver.start_listening()
            except Exception as e:
                print(f"Failed to register button callback: {e}")

    def _button_wrapper(self, buttons: List[bool]):
        """Wrapper to call the registered button callback."""
        if self.button_callback:
            try:
                self.button_callback(buttons)
            except Exception as e:
                print(f"Error in button callback: {e}")


class ControlPortRust:
    """Rust-based control port implementation."""

    def __init__(self, config_data: Dict[str, Any], web_monitor_port: int = 8080):
        if not RUST_AVAILABLE:
            raise RuntimeError("Rust control port implementation not available")

        self.config_data = config_data
        self.web_monitor_port = web_monitor_port
        self.controllers = {}
        self._manager = None
        self._initialized = False

    async def initialize(self):
        """Initialize the Rust controller manager."""
        if self._initialized:
            return self.controllers

        try:
            # Convert config to JSON string for Rust
            config_json = json.dumps(self.config_data)
            self._manager = ControllerManager(config_json)
            self._manager.initialize()

            # Start web monitor
            self._manager.start_web_monitor(self.web_monitor_port)

            # Create controller wrappers
            for dip in self.config_data.get("controller_addresses", {}).keys():
                rust_controller = self._manager.get_controller(dip)
                if rust_controller:
                    self.controllers[dip] = ControllerStateWrapper(rust_controller)

            self._initialized = True
            print(
                f"Initialized {len(self.controllers)} controllers with web monitor on port {self.web_monitor_port}"
            )

        except Exception as e:
            print(f"Failed to initialize Rust control port: {e}")
            raise

        return self.controllers

    async def enumerate(self, timeout: float = 2.0):
        """
        Enumerate controllers. In the Rust implementation, this uses fixed configuration
        rather than network discovery.
        """
        return await self.initialize()

    def get_stats(self):
        """Get controller statistics."""
        if self._manager:
            try:
                return self._manager.get_controller_stats()
            except Exception as e:
                print(f"Failed to get controller stats: {e}")
                return []
        return []

    def shutdown(self):
        """Shutdown the control port."""
        if self._manager:
            try:
                self._manager.shutdown()
            except Exception as e:
                print(f"Error during shutdown: {e}")
        self._initialized = False


class ControlPort:
    """
    Drop-in replacement for the original ControlPort class that uses Rust implementation
    when available, falls back to Python implementation otherwise.
    """

    def __init__(
        self,
        hosts_and_ports: List[tuple] = None,
        config_data: Dict[str, Any] = None,
        web_monitor_port: int = 8080,
        loop=None,
    ):
        self.loop = loop or asyncio.get_event_loop()
        self.controllers = {}

        if RUST_AVAILABLE and config_data:
            self._impl = ControlPortRust(config_data, web_monitor_port)
            self._use_rust = True
            print("Using Rust-based control port implementation")
        else:
            if hosts_and_ports is None:
                hosts_and_ports = []
            self._impl = fallback_control_port.ControlPort(hosts_and_ports, loop)
            self._use_rust = False
            print("Using Python-based control port implementation")

    async def enumerate(self, timeout: float = 2.0):
        """Enumerate available controllers."""
        self.controllers = await self._impl.enumerate(timeout)
        return self.controllers

    async def ping_host(self, ip: str):
        """Ping a host and return True if it responds."""
        if self._use_rust:
            # Rust implementation handles connectivity automatically
            return True
        else:
            return await self._impl.ping_host(ip)

    async def check_port(self, ip: str, port: int):
        """Check if the port is open using a socket connection."""
        if self._use_rust:
            # Rust implementation handles connectivity automatically
            return True
        else:
            return await self._impl.check_port(ip, port)

    async def discover_hosts(self):
        """Discover reachable hosts."""
        if self._use_rust:
            # Rust implementation uses fixed configuration
            return list(self.controllers.keys())
        else:
            return await self._impl.discover_hosts()

    def get_stats(self):
        """Get controller statistics (Rust implementation only)."""
        if self._use_rust:
            return self._impl.get_stats()
        else:
            return []

    def shutdown(self):
        """Shutdown the control port."""
        if hasattr(self._impl, "shutdown"):
            self._impl.shutdown()


def create_control_port_from_config(config_path: str, web_monitor_port: int = 8080):
    """
    Factory function to create a ControlPort instance from a configuration file.

    Args:
        config_path: Path to the JSON configuration file
        web_monitor_port: Port for the web monitoring interface

    Returns:
        ControlPort instance configured for the given config
    """
    try:
        with open(config_path, "r") as f:
            config_data = json.load(f)

        return ControlPort(config_data=config_data, web_monitor_port=web_monitor_port)

    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        raise
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in config file: {e}")
        raise
    except Exception as e:
        print(f"Error loading config: {e}")
        raise


# For backwards compatibility, also export the individual classes
ControllerState = ControllerStateWrapper  # noqa: F811

# Usage example and compatibility info
if __name__ == "__main__":
    print(f"Rust control port available: {RUST_AVAILABLE}")
    print("To use with existing code, simply replace:")
    print("  from control_port import ControlPort")
    print("with:")
    print("  from control_port_rust import ControlPort")
    print("")
    print("Or use the factory function for config-based initialization:")
    print("  cp = create_control_port_from_config('config.json', web_monitor_port=8080)")
    print("")
    print("Web monitor will be available at http://localhost:8080")
