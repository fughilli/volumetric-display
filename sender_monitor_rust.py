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
