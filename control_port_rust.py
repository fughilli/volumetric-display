"""
Python wrapper for the Rust-based ControlPortManager.

This module provides a clean Python interface to the Rust implementation
of the control port system, which manages multiple control ports and
provides web monitoring capabilities.
"""

from typing import Any, Callable, Dict, List, Optional

from src.control_port.control_port_rs import ControlPortManager as ControlPortManagerRs


class ControlPortManager:
    """
    Python wrapper for the Rust ControlPortManager.

    This class manages multiple control ports and provides web monitoring
    capabilities. It's designed to integrate with game state and other
    Python applications.
    """

    def __init__(self, config_path: str):
        """
        Initialize the ControlPortManager with a configuration file.

        Args:
            config_path: Path to the JSON configuration file
        """
        with open(config_path, "r") as f:
            config_json = f.read()

        self._rust_manager = ControlPortManagerRs(config_json)
        self._control_ports: Dict[str, ControlPort] = {}
        self._web_monitor_started = False

    def initialize(self) -> None:
        """Initialize all control ports and establish connections."""
        self._rust_manager.initialize()

        # Create Python wrapper objects for each control port
        for dip in self._get_configured_dips():
            if self._rust_manager.get_control_port(dip):
                self._control_ports[dip] = ControlPort(self._rust_manager.get_control_port(dip))

    def start_web_monitor(
        self, port: int = 8080, log_buffer_size: int = 1000, bind_address: str = "0.0.0.0"
    ) -> None:
        """
        Start the web monitoring interface.

        Args:
            port: Port number for the web server (default: 8080)
            log_buffer_size: Number of log entries to keep in buffer (default: 1000)
            bind_address: Bind address for the web server (default: "0.0.0.0" for all interfaces)
        """
        if not self._web_monitor_started:
            self._rust_manager.start_web_monitor_with_full_config(
                port, log_buffer_size, bind_address
            )
            self._web_monitor_started = True
            if bind_address == "0.0.0.0":
                print("🌐 Web monitor started on:")
                print(f"   Local: http://localhost:{port}")
                print(f"   Network: http://0.0.0.0:{port}")
            else:
                print(f"🌐 Web monitor started on http://{bind_address}:{port}")
            print(f"   Dashboard: http://localhost:{port}")
            print(f"   API: http://localhost:{port}/api/control_ports")
            print(f"   Log buffer size: {log_buffer_size} entries")

    def get_control_port(self, dip: str) -> Optional["ControlPort"]:
        """
        Get a control port by its DIP address.

        Args:
            dip: DIP address of the control port

        Returns:
            ControlPort instance if found, None otherwise
        """
        return self._control_ports.get(dip)

    def get_all_control_ports(self) -> Dict[str, "ControlPort"]:
        """
        Get all available control ports.

        Returns:
            Dictionary mapping DIP addresses to ControlPort instances
        """
        return self._control_ports.copy()

    def get_stats(self) -> List[Dict[str, Any]]:
        """
        Get statistics for all control ports.

        Returns:
            List of statistics dictionaries
        """
        return self._rust_manager.get_all_stats()

    def shutdown(self) -> None:
        """Shutdown all control ports and cleanup resources."""
        self._rust_manager.shutdown()
        self._control_ports.clear()
        self._web_monitor_started = False

    def _get_configured_dips(self) -> List[str]:
        """Get list of configured DIP addresses from the config."""
        # This is a simplified approach - in practice, you might want to
        # parse the config file directly or add a method to the Rust side
        stats = self._rust_manager.get_all_stats()
        return [stat["dip"] for stat in stats]


class ControlPort:
    """
    Python wrapper for a single control port.

    This class provides methods to interact with a specific controller,
    including display control, LED control, and button event handling.
    """

    def __init__(self, rust_control_port):
        """
        Initialize the ControlPort wrapper.

        Args:
            rust_control_port: The underlying Rust ControlPort instance
        """
        self._rust_port = rust_control_port
        self._button_callbacks: List[Callable[[List[bool]], None]] = []

    def clear_display(self) -> None:
        """Clear the LCD display."""
        self._rust_port.clear_display()

    def clear(self) -> None:
        """Clear the LCD display (compatibility method)."""
        self._rust_port.clear_display()

    def write_display(self, x: int, y: int, text: str) -> None:
        """
        Write text to the LCD display at specified coordinates.

        Args:
            x: X coordinate (column)
            y: Y coordinate (row)
            text: Text to display
        """
        self._rust_port.write_display(x, y, text)

    def write_lcd(self, x: int, y: int, text: str) -> None:
        """
        Write text to the LCD display at specified coordinates (compatibility method).

        Args:
            x: X coordinate (column)
            y: Y coordinate (row)
            text: Text to display
        """
        self._rust_port.write_display(x, y, text)

    async def commit_display(self) -> None:
        """Commit pending display changes to the controller."""
        # The Rust commit_display() method returns PyResult<()> which is Ok(()) on success
        # We need to call it and handle any potential errors
        try:
            if hasattr(self, "_rust_port") and self._rust_port is not None:
                result = self._rust_port.commit_display()
                # The Rust method returns Ok(()) on success, which Python sees as None
                # This is normal behavior for PyResult<()> - None means success
                if result is not None:
                    print(f"Warning: commit_display returned unexpected value: {result}")
                return result
            else:
                print("Warning: ControlPort._rust_port is None or invalid")
                return None
        except Exception as e:
            print(f"Error in ControlPort.commit_display(): {e}")
            return None

    async def commit(self):
        """Commit display changes (alias for commit_display)."""
        return await self.commit_display()

    def set_leds(self, rgb_values: List[tuple]) -> None:
        """
        Set LED colors.

        Args:
            rgb_values: List of (r, g, b) tuples for each LED
        """
        self._rust_port.set_leds(rgb_values)

    def set_backlights(self, states: List[bool]) -> None:
        """
        Set backlight states.

        Args:
            states: List of boolean values for each backlight
        """
        self._rust_port.set_backlights(states)

    def register_button_callback(self, callback: Callable[[List[bool]], None]) -> None:
        """
        Register a callback function for button events.

        Args:
            callback: Function to call when button state changes.
                     Takes a list of boolean values representing button states.
        """
        self._button_callbacks.append(callback)
        # Register with Rust side
        receiver = self._rust_port.register_button_callback(callback)
        receiver.start_listening()

    @property
    def dip(self) -> str:
        """Get the DIP address of this control port."""
        return self._rust_port.dip()

    @property
    def connected(self) -> bool:
        """Check if this control port is connected."""
        return self._rust_port.connected()

    @property
    def ip(self) -> str:
        """Get the IP address of this control port."""
        # This would need to be added to the Rust side
        return "unknown"

    @property
    def port(self) -> int:
        """Get the port number of this control port."""
        # This would need to be added to the Rust side
        return 0


def create_control_port_from_config(
    config_path: str,
    web_monitor_port: int = 8080,
    log_buffer_size: int = 1000,
    bind_address: str = "0.0.0.0",
) -> ControlPortManager:
    """
    Create and initialize a ControlPortManager from a configuration file.

    Args:
        config_path: Path to the JSON configuration file
        web_monitor_port: Port for the web monitoring interface
        log_buffer_size: Number of log entries to keep in buffer (default: 1000)
        bind_address: Bind address for the web server (default: "0.0.0.0" for all interfaces)

    Returns:
        Initialized ControlPortManager instance
    """
    manager = ControlPortManager(config_path)
    manager.initialize()
    manager.start_web_monitor(web_monitor_port, log_buffer_size, bind_address)
    return manager


# Convenience function for backward compatibility
def create_control_port_manager(
    config_path: str, web_monitor_port: int = 8080
) -> ControlPortManager:
    """
    Alias for create_control_port_from_config for backward compatibility.

    Args:
        config_path: Path to the JSON configuration file
        web_monitor_port: Port for the web monitoring interface

    Returns:
        Initialized ControlPortManager instance
    """
    return create_control_port_from_config(config_path, web_monitor_port)
