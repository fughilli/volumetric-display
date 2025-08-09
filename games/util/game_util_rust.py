"""
Rust-based game utility replacement with enhanced performance and monitoring.

This module provides a drop-in replacement for the ControllerInputHandler
in game_util.py using the high-performance Rust control port implementation.
"""

import asyncio
import threading
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from control_port_rust import create_control_port_from_config

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    print("Warning: Rust control port not available, falling back to Python implementation")
    # Import the original classes for fallback

from games.util.game_util import (
    Button,
    ButtonState,
)
from games.util.game_util import (
    ControllerInputHandler as FallbackControllerInputHandler,
)


class DisplayManager:
    """Enhanced display manager with Rust backend."""

    def __init__(self):
        self.last_display_update = 0
        self.display_update_interval = 0.1  # Update displays every 100ms

    def _log_lcd_command(self, controller_id, line, text):
        """Log an LCD command being sent to a controller."""
        print(f"LCD[{controller_id}] Line {line}: {text}")

    async def update_displays(self, controllers, game_state):
        """Update all controller displays with current game state."""
        current_time = time.monotonic()
        if current_time - self.last_display_update < self.display_update_interval:
            return

        self.last_display_update = current_time

        async def update_single_controller(controller_state, player_id):
            try:
                # Use the game_state's update_controller_display_state method if it exists
                if hasattr(game_state, "update_controller_display_state") and callable(
                    game_state.update_controller_display_state
                ):
                    await game_state.update_controller_display_state(controller_state, player_id)
                elif hasattr(game_state, "update_display") and callable(game_state.update_display):
                    await game_state.update_display(controller_state, player_id)
                else:
                    controller_state.clear()
                    controller_state.write_lcd(0, 0, "ARTNET DISPLAY")
                    controller_state.write_lcd(0, 1, "Rust Enhanced")
                    if hasattr(game_state, "__class__") and hasattr(
                        game_state.__class__, "__name__"
                    ):
                        controller_state.write_lcd(0, 2, f"Class: {game_state.__class__.__name__}")
                    else:
                        controller_state.write_lcd(0, 2, "Unknown Class")
                    await controller_state.commit()
            except (BrokenPipeError, ConnectionError, RuntimeError, OSError) as e:
                print(f"Display update failed for controller: {e}")

        # Create and gather all controller update tasks
        update_tasks = [
            update_single_controller(controller_state, player_id)
            for controller_id, (controller_state, player_id) in controllers.items()
        ]
        if update_tasks:
            await asyncio.gather(*update_tasks, return_exceptions=True)


class ControllerInputHandlerRust:
    """
    Rust-based controller input handler with enhanced performance and monitoring.

    This class provides a drop-in replacement for the original ControllerInputHandler
    with the following enhancements:
    - Uses Rust async sockets for better performance
    - Fixed configuration instead of enumeration
    - Built-in web monitoring interface
    - Enhanced logging and statistics
    """

    def __init__(
        self,
        controller_mapping: Optional[Dict[str, Any]] = None,
        config_path: str = "config.json",
        web_monitor_port: int = 8080,
    ):
        self.controller_mapping = controller_mapping or {}
        self.config_path = config_path
        self.web_monitor_port = web_monitor_port

        # State management
        self.controllers = {}  # Maps controller_id to (controller_state, player_id)
        self.active_controllers = []  # List of active controller states
        self.initialized = False
        self.event_queue = deque()  # Queue for (player_id, direction) events
        self.init_event = threading.Event()

        # Button state tracking
        self.select_hold_data = {}  # Maps controller_id to hold state
        self.last_button_states = {}  # Maps controller_id to list of button states
        self.menu_selection_time = 0  # Time of last menu selection change
        self.menu_votes = {}  # Maps controller_id to their game vote
        self.voting_states = {}  # Maps controller_id to whether they have voted

        # New callback system
        self.button_callbacks = {}  # Maps controller_id to callback function

        # Control port instance
        self.cp = None
        self._lock = threading.Lock()
        self.loop = None
        self._init_task = None

        print(f"ControllerInputHandlerRust initialized with config: {config_path}")
        print(f"Web monitor will be available on port: {web_monitor_port}")

    def register_button_callback(self, controller_id: str, callback: Callable):
        """Register a callback for button events.

        The callback should have the signature:
        callback(player_id, button, button_state)

        Where:
        - player_id is the PlayerID enum
        - button is the Button enum
        - button_state is the ButtonState enum (PRESSED, RELEASED, HELD)
        """
        if controller_id in self.controllers:
            self.button_callbacks[controller_id] = callback
            return True
        return False

    def unregister_button_callback(self, controller_id: str):
        """Unregister a callback for button events."""
        if controller_id in self.button_callbacks:
            del self.button_callbacks[controller_id]
            return True
        return False

    async def _async_initialize_and_listen(self):
        """Runs in the asyncio thread to initialize and start listening."""
        print("ControllerInputHandlerRust: Starting async initialization...")
        try:
            # Create control port with fixed configuration
            self.cp = create_control_port_from_config(self.config_path, self.web_monitor_port)

            # Initialize controllers
            discovered_controllers = await self.cp.enumerate(timeout=5.0)

            if not discovered_controllers:
                print("ControllerInputHandlerRust: No controllers found in configuration.")
                self.initialized = False
                self.init_event.set()
                return

            # Map controllers to players
            for dip, controller_state in discovered_controllers.items():
                if dip in self.controller_mapping:
                    player_id = self.controller_mapping[dip]
                    self.controllers[dip] = (controller_state, player_id)
                    self.active_controllers.append(controller_state)

                    # Set up button callback
                    controller_state.register_button_callback(
                        lambda buttons, controller_dip=dip: self._button_callback(
                            buttons, controller_dip
                        )
                    )

                    # Initialize button state tracking
                    self.select_hold_data[dip] = {"start_time": 0, "is_counting_down": False}
                    self.last_button_states[dip] = [False] * 5

                    print(
                        f"ControllerInputHandlerRust: Registered controller DIP {dip} as player {player_id}"
                    )
                else:
                    print(
                        f"ControllerInputHandlerRust: Controller DIP {dip} not mapped to any player"
                    )

            self.initialized = len(self.controllers) > 0
            print(
                f"ControllerInputHandlerRust: Initialization complete. "
                f"Initialized: {self.initialized}, Controllers: {len(self.controllers)}"
            )

        except Exception as e:
            print(f"Error during controller async initialization: {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()
            self.initialized = False
        finally:
            self.init_event.set()

    def _button_callback(self, buttons: List[bool], controller_id: str):
        """Called when a controller's buttons change."""
        if controller_id not in self.controllers:
            return

        controller_state, player_id = self.controllers[controller_id]
        last_buttons = self.last_button_states.get(controller_id, [False] * 5)

        # Process each button to determine state changes
        for button in Button:
            button_idx = button.value

            # Check for button state changes
            if buttons[button_idx] and not last_buttons[button_idx]:
                # Button was just pressed
                self._handle_button_event(controller_id, player_id, button, ButtonState.PRESSED)

                # For SELECT button, also track press time for hold detection
                if button == Button.SELECT:
                    self.select_hold_data[controller_id] = {
                        "start_time": time.monotonic(),
                        "is_counting_down": True,
                    }

            elif not buttons[button_idx] and last_buttons[button_idx]:
                # Button was just released
                self._handle_button_event(controller_id, player_id, button, ButtonState.RELEASED)

                # For SELECT button, check for press-and-release (click) event
                if button == Button.SELECT:
                    self.select_hold_data[controller_id]["is_counting_down"] = False

            elif buttons[button_idx]:
                # Button is being held down
                self._handle_button_event(controller_id, player_id, button, ButtonState.HELD)

        # Store the new state
        self.last_button_states[controller_id] = list(buttons)

    def _handle_button_event(
        self, controller_id: str, player_id: Any, button: Button, button_state: ButtonState
    ):
        """Process a button event and call the registered callback if any."""
        # Invoke the callback if registered
        if controller_id in self.button_callbacks:
            try:
                callback = self.button_callbacks[controller_id]
                callback(player_id, button, button_state)
            except Exception as e:
                print(f"Error in button callback for controller {controller_id}: {e}")

    def get_direction_key(self) -> Optional[Tuple[Any, Button, ButtonState]]:
        """Called from the main game thread to get the next input event.

        Returns:
            tuple or None: A tuple containing (player_id, button, button_state) or None if no events
        """
        if not self.initialized:
            return None

        with self._lock:
            if self.event_queue:
                return self.event_queue.popleft()
        return None

    def check_for_restart_signal(self) -> bool:
        """Check if any controller has held SELECT for 5 seconds."""
        current_time = time.monotonic()
        for controller_id, hold_data in self.select_hold_data.items():
            if hold_data["is_counting_down"]:
                if current_time - hold_data["start_time"] >= 5.0:
                    return True
        return False

    def clear_all_select_holds(self):
        """Clear all SELECT hold states."""
        for hold_data in self.select_hold_data.values():
            hold_data["is_counting_down"] = False

    def clear_menu_votes(self):
        """Clear all menu votes and selections."""
        self.menu_votes.clear()
        self.voting_states.clear()

    def start_initialization(self) -> bool:
        """Starts the background thread and waits for initialization."""
        if not RUST_AVAILABLE:
            print("Rust implementation not available, cannot start")
            return False

        self.thread = threading.Thread(target=self._run_asyncio_loop, daemon=True)
        self.thread.start()
        initialized = self.init_event.wait(timeout=10.0)  # Longer timeout for Rust init
        if not initialized:
            print("Controller initialization timed out.")
            self.stop()
            return False
        return self.initialized

    def _run_asyncio_loop(self):
        """Run the asyncio event loop in a separate thread."""
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        try:
            self._init_task = self.loop.create_task(self._async_initialize_and_listen())
            self.loop.run_forever()
        finally:
            print("Asyncio loop stopping...")
            if self._init_task and not self._init_task.done():
                self._init_task.cancel()

            if self.loop.is_running():
                self.loop.run_until_complete(self.loop.shutdown_asyncgens())

            self.loop.close()
            print("Asyncio loop stopped.")

    def stop(self):
        """Clean up all controller connections and tasks."""
        print("Stopping Rust controller input handler...")

        if hasattr(self, "cp") and self.cp:
            self.cp.shutdown()

        loop = getattr(self, "loop", None)
        if loop and loop.is_running():
            if self._init_task:
                loop.call_soon_threadsafe(self._init_task.cancel)
            loop.call_soon_threadsafe(loop.stop)

        if hasattr(self, "thread") and self.thread.is_alive():
            self.thread.join(timeout=3.0)

    def get_stats(self) -> List[Dict[str, Any]]:
        """Get controller statistics (Rust implementation only)."""
        if self.cp:
            return self.cp.get_stats()
        return []

    def get_web_monitor_url(self) -> str:
        """Get the URL for the web monitoring interface."""
        return f"http://localhost:{self.web_monitor_port}"


# Compatibility class that chooses between Rust and Python implementations
class ControllerInputHandler:
    """
    Smart controller input handler that uses Rust when available, falls back to Python.
    """

    def __new__(
        cls,
        controller_mapping: Optional[Dict[str, Any]] = None,
        hosts_and_ports: Optional[List[Tuple[str, int]]] = None,
        config_path: str = "config.json",
        web_monitor_port: int = 8080,
        **kwargs,
    ):
        # Try to use Rust implementation first
        if RUST_AVAILABLE:
            try:
                return ControllerInputHandlerRust(
                    controller_mapping=controller_mapping,
                    config_path=config_path,
                    web_monitor_port=web_monitor_port,
                )
            except Exception as e:
                print(f"Failed to initialize Rust implementation: {e}")
                print("Falling back to Python implementation")

        # Fall back to Python implementation
        if hosts_and_ports is None:
            # Convert config-based setup to hosts_and_ports for compatibility
            try:
                import json

                with open(config_path, "r") as f:
                    config = json.load(f)

                hosts_and_ports = []
                for controller_config in config.get("controller_addresses", {}).values():
                    hosts_and_ports.append((controller_config["ip"], controller_config["port"]))

            except Exception as e:
                print(f"Could not load config file {config_path}: {e}")
                hosts_and_ports = []

        return FallbackControllerInputHandler(
            controller_mapping=controller_mapping, hosts_and_ports=hosts_and_ports, **kwargs
        )


# Usage example
if __name__ == "__main__":
    print(f"Rust control port available: {RUST_AVAILABLE}")
    print("")
    print("Usage:")
    print("  # Use with existing code (automatic fallback)")
    print("  handler = ControllerInputHandler(controller_mapping={'0': 'player1'})")
    print("")
    print("  # Use Rust implementation explicitly")
    print("  handler = ControllerInputHandlerRust(")
    print("      controller_mapping={'0': 'player1'},")
    print("      config_path='config.json',")
    print("      web_monitor_port=8080")
    print("  )")
    print("")
    print("Web monitor will be available at http://localhost:8080")
