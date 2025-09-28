"""
Python wrapper for the Rust-based sender monitor.

This module provides monitoring capabilities for ArtNet sender operations,
including controller status tracking and system performance metrics.
"""

try:
    from src.sender_monitor.sender_monitor_rs import SenderMonitorManager

    SENDER_MONITOR_AVAILABLE = True
    print("Loaded Rust-based SenderMonitor")
except ImportError:
    SENDER_MONITOR_AVAILABLE = False
    print("Sender monitor not available - monitoring disabled")


def create_sender_monitor() -> "SenderMonitorManager | None":
    """
    Create a new sender monitor instance.

    Returns:
        SenderMonitorManager instance if available, None otherwise
    """
    if not SENDER_MONITOR_AVAILABLE:
        return None

    try:
        return SenderMonitorManager()
    except Exception as e:
        print(f"Warning: Failed to create sender monitor: {e}")
        return None


def create_sender_monitor_with_web_interface(
    port: int = 8081, bind_address: str = "0.0.0.0", cooldown_seconds: int = 30
) -> "SenderMonitorManager | None":
    """
    Create a new sender monitor instance with web interface.

    Args:
        port: Port for the web interface (default: 8081)
        bind_address: Bind address for the web interface (default: "0.0.0.0")
        cooldown_seconds: Cooldown period in seconds before marking failed controllers as routable (default: 30)

    Returns:
        SenderMonitorManager instance if available, None otherwise
    """
    monitor = create_sender_monitor()
    if monitor is None:
        return None

    try:
        # Configure cooldown duration
        monitor.set_cooldown_duration(cooldown_seconds)

        if bind_address != "0.0.0.0":
            monitor.start_web_monitor_with_bind_address(port, bind_address)
        else:
            monitor.start_web_monitor(port)
        return monitor
    except Exception as e:
        print(f"Warning: Failed to start web interface: {e}")
        return monitor


class SenderMonitorWrapper:
    """
    Wrapper class that provides a consistent interface whether the Rust implementation
    is available or not.
    """

    def __init__(self, monitor=None):
        self.monitor = monitor
        self._debug_mode = False
        self._is_paused = False
        self._debug_command = None

    def register_controller(self, ip: str, port: int) -> None:
        """Register a controller for monitoring."""
        if self.monitor:
            self.monitor.register_controller(ip, port)

    def report_controller_success(self, ip: str, port: int) -> None:
        """Report successful transmission to a controller."""
        if self.monitor:
            self.monitor.report_controller_success(ip, port)

    def report_controller_failure(self, ip: str, port: int, error: str) -> None:
        """Report failed transmission to a controller."""
        if self.monitor:
            self.monitor.report_controller_failure(ip, port, error)

    def report_frame(self) -> None:
        """Report a frame being processed."""
        if self.monitor:
            self.monitor.report_frame()

    def set_debug_mode(self, enabled: bool) -> None:
        """Enable or disable debug mode."""
        self._debug_mode = enabled
        if self.monitor:
            self.monitor.set_debug_mode(enabled)

    def set_debug_pause(self, paused: bool) -> None:
        """Pause or resume the render loop."""
        self._is_paused = paused
        if self.monitor:
            self.monitor.set_debug_pause(paused)

    def is_debug_mode(self) -> bool:
        """Check if debug mode is enabled."""
        if self.monitor:
            return self.monitor.is_debug_mode()
        return self._debug_mode

    def is_paused(self) -> bool:
        """Check if the render loop is paused."""
        if self.monitor:
            return self.monitor.is_paused()
        return self._is_paused

    def get_debug_command(self):
        """Get the current debug command if any."""
        if self.monitor:
            return self.monitor.get_debug_command()
        return self._debug_command

    def set_world_dimensions(self, width: int, height: int, length: int) -> None:
        """Set the world raster dimensions for the mapping tester."""
        if self.monitor:
            self.monitor.set_world_dimensions(width, height, length)

    def set_cube_list(self, cubes: list) -> None:
        """Set the list of available cubes for the mapping tester."""
        if self.monitor:
            self.monitor.set_cube_list(cubes)

    def shutdown(self) -> None:
        """Shutdown the monitor."""
        if self.monitor:
            self.monitor.shutdown()


def create_sender_monitor_with_web_interface_wrapped(
    port: int = 8081, bind_address: str = "0.0.0.0", cooldown_seconds: int = 30
) -> SenderMonitorWrapper:
    """
    Create a new sender monitor wrapper instance with web interface.

    Args:
        port: Port for the web interface (default: 8081)
        bind_address: Bind address for the web interface (default: "0.0.0.0")
        cooldown_seconds: Cooldown period in seconds before marking failed controllers as routable (default: 30)

    Returns:
        SenderMonitorWrapper instance
    """
    monitor = create_sender_monitor_with_web_interface(port, bind_address, cooldown_seconds)
    return SenderMonitorWrapper(monitor)
