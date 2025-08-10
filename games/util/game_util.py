import asyncio
import threading
import time
from collections import deque
from enum import Enum

import control_port_rust

# Attempt to import GameScene and BaseGame for type checking
# This might create a circular dependency if they also import from game_util
# A cleaner solution might involve a state type enum or attribute checking
try:
    from game_scene import GameScene
except ImportError:
    GameScene = (
        None  # Fallback if import fails, to avoid crashing if files are temporarily unavailable
    )

try:
    from games.util.base_game import BaseGame
except ImportError:
    BaseGame = None  # Fallback


class Button(Enum):
    UP = 0
    LEFT = 1
    DOWN = 2
    RIGHT = 3
    SELECT = 4


class ButtonState(Enum):
    """Enum representing button state events."""

    PRESSED = 0  # Button was just pressed (down event)
    RELEASED = 1  # Button was just released (up event)
    HELD = 2  # Button is being held down


class Direction(Enum):
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4


class DisplayManager:
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
                    controller_state.write_lcd(0, 1, "Display Error")
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
            await asyncio.gather(*update_tasks)


class ControllerInputHandler:
    def __init__(
        self,
        controller_mapping=None,
        hosts_and_ports: list[tuple[str, int]] | None = None,
    ):
        self.cp = control_port_rust.ControlPort(hosts_and_ports=hosts_and_ports)
        self.controllers = {}  # Maps controller_id to (controller_state, player_id)
        self.active_controllers = []  # List of active controller states
        self._lock = threading.Lock()
        self.initialized = False
        self.event_queue = deque()  # Queue for (player_id, direction) events
        self.init_event = threading.Event()
        self.loop = None
        self._init_task = None
        self._listen_tasks = {}  # Maps controller_id to its listen task
        self.select_hold_data = (
            {}
        )  # Maps controller_id to {'start_time': float, 'is_counting_down': bool}
        self.last_button_states = {}  # Maps controller_id to list of button states
        self.menu_selection_time = 0  # Time of last menu selection change
        self.menu_votes = {}  # Maps controller_id to their game vote
        self.voting_states = {}  # Maps controller_id to whether they have voted
        self.controller_mapping = controller_mapping or {}

        # New callback system
        self.button_callbacks = {}  # Maps controller_id to callback function

    def register_button_callback(self, controller_id, callback):
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

    def unregister_button_callback(self, controller_id):
        """Unregister a callback for button events."""
        if controller_id in self.button_callbacks:
            del self.button_callbacks[controller_id]
            return True
        return False

    async def _async_initialize_and_listen(self):
        """Runs in the asyncio thread to initialize and start listening."""
        print("ControllerInputHandler: Starting async initialization...")
        try:
            print("ControllerInputHandler: Calling cp.enumerate()...")
            discovered_controller_states = await self.cp.enumerate(timeout=5.0)
            print(
                f"ControllerInputHandler: cp.enumerate() returned: {discovered_controller_states}"
            )

            if not discovered_controller_states:
                print(
                    "ControllerInputHandler: No controllers found/returned by ControlPort.enumerate."
                )
                self.initialized = False
                self.init_event.set()
                return

            connect_tasks = []
            print("ControllerInputHandler: Iterating discovered_controller_states items...")
            for dip_key, state_from_cp in discovered_controller_states.items():
                print(
                    f"ControllerInputHandler: Processing discovered_controller_states "
                    f"item: dip_key={dip_key}, state_from_cp={state_from_cp}"
                )
                if state_from_cp.dip in self.controller_mapping:
                    player_id = self.controller_mapping[state_from_cp.dip]
                    print(
                        f"ControllerInputHandler: Controller DIP {state_from_cp.dip} "
                        f"maps to player_id={player_id}. Creating connect task."
                    )
                    connect_tasks.append(self._connect_and_register(state_from_cp, player_id))
                else:
                    print(
                        f"ControllerInputHandler: Discovered/queried controller "
                        f"{state_from_cp.ip}:{state_from_cp.port} (DIP: {state_from_cp.dip}) "
                        f"not assigned a role in mapping. Skipping."
                    )

            print(f"ControllerInputHandler: Built connect_tasks list: {connect_tasks}")
            if connect_tasks:
                print(
                    f"ControllerInputHandler: Calling asyncio.gather for {len(connect_tasks)} tasks..."
                )
                results = await asyncio.gather(*connect_tasks, return_exceptions=True)
                print(f"ControllerInputHandler: asyncio.gather results: {results}")

            self.initialized = len(self.controllers) > 0
            print(
                f"ControllerInputHandler: Initialization complete. "
                f"self.initialized = {self.initialized}, "
                f"self.controllers = {self.controllers}"
            )

        except Exception as e:
            print(f"Error during controller async initialization: {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()
            self.initialized = False
        finally:
            self.init_event.set()

    async def _connect_and_register(
        self, controller_state_instance: control_port_rust.ControllerState, player_id
    ):
        """Helper to connect a single controller (given as ControllerState instance) and register callback."""
        dip = controller_state_instance.dip

        if not controller_state_instance._connected:
            if not await controller_state_instance.connect():
                return

        print(
            f"ControllerInputHandler: Successfully connected/verified controller "
            f"DIP {dip} ({player_id.name})"
        )

        self.controllers[dip] = (controller_state_instance, player_id)
        if controller_state_instance not in self.active_controllers:
            self.active_controllers.append(controller_state_instance)

        controller_state_instance.register_button_callback(
            lambda buttons, controller_dip=dip: self._button_callback(buttons, controller_dip)
        )

        self.select_hold_data[dip] = {"start_time": 0, "is_counting_down": False}
        self.last_button_states[dip] = [False] * 5

    def _button_callback(self, buttons, controller_id):
        """Called from the asyncio thread when a controller's buttons change."""
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

                # NOTE: No longer adding to event queue - using callbacks exclusively

            elif not buttons[button_idx] and last_buttons[button_idx]:
                # Button was just released
                self._handle_button_event(controller_id, player_id, button, ButtonState.RELEASED)

                # For SELECT button, check for press-and-release (click) event
                if button == Button.SELECT:
                    self.select_hold_data[controller_id]["is_counting_down"] = False

                # NOTE: No longer adding to event queue - using callbacks exclusively

            elif buttons[button_idx]:
                # Button is being held down
                self._handle_button_event(controller_id, player_id, button, ButtonState.HELD)

                # We don't add HELD events to the queue

        # Store the new state
        self.last_button_states[controller_id] = list(buttons)

    def _handle_button_event(self, controller_id, player_id, button, button_state):
        """Process a button event and call the registered callback if any."""
        # Invoke the callback if registered
        if controller_id in self.button_callbacks:
            try:
                callback = self.button_callbacks[controller_id]
                callback(player_id, button, button_state)
            except Exception as e:
                print(f"Error in button callback for controller {controller_id}: {e}")

    def get_direction_key(self):
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

    def check_for_restart_signal(self):
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

    def start_initialization(self):
        """Starts the background thread and waits for initialization."""
        self.thread = threading.Thread(target=self._run_asyncio_loop, daemon=True)
        self.thread.start()
        initialized = self.init_event.wait(timeout=7.0)
        if not initialized:
            print("Controller initialization timed out.")
            self.stop()
            return False
        return self.initialized

    def _run_asyncio_loop(self):
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

            for task in self._listen_tasks.values():
                if not task.done():
                    task.cancel()

            async def gather_cancelled():
                tasks = [t for t in asyncio.all_tasks(self.loop) if t.cancelled()]
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

            if self.loop.is_running():
                self.loop.run_until_complete(gather_cancelled())
                self.loop.run_until_complete(self.loop.shutdown_asyncgens())

            self.loop.close()
            print("Asyncio loop stopped.")

    def stop(self):
        """Clean up all controller connections and tasks."""
        print("Stopping controller input handler...")
        loop = getattr(self, "loop", None)
        if loop and loop.is_running():
            for controller_id, (controller_state, _) in self.controllers.items():
                if controller_state._connected:
                    disconnect_future = asyncio.run_coroutine_threadsafe(
                        controller_state.disconnect(), loop
                    )
                    try:
                        disconnect_future.result(timeout=2)
                        print(f"Controller {controller_id} disconnected.")
                    except Exception as e:
                        print(f"Error disconnecting controller {controller_id}: {e}")

            if self._init_task:
                loop.call_soon_threadsafe(self._init_task.cancel)

            for task in self._listen_tasks.values():
                if not task.done():
                    loop.call_soon_threadsafe(task.cancel)

            loop.call_soon_threadsafe(loop.stop)

        if hasattr(self, "thread") and self.thread.is_alive():
            self.thread.join(timeout=3.0)
