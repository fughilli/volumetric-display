import asyncio
import importlib.util
import math
import os
import random
import time

from artnet import RGB, Scene
from games.util.base_game import BaseGame, PlayerID
from games.util.game_util import Button, ButtonState, ControllerInputHandler


class GameScene(Scene):

    def __init__(
        self, width=20, height=20, length=20, frameRate=30, config=None, control_port_manager=None
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.length = length
        self.frameRate = frameRate
        self.menu_frame_rate = 30
        self.config = config
        self.game_started = False  # Initialize game_started attribute
        self.button_pressed = False

        # Initialize menu-related attributes
        self.menu_selections = {}  # Maps controller_id to their current selection
        self.menu_votes = {}  # Maps controller_id to their game vote
        self.voting_states = {}  # Maps controller_id to whether they have voted

        # Store controller mapping from config
        self.controller_mapping = {}
        print(f"Debug: config = {config}")
        if config and "scene" in config:
            print(f"Debug: scene keys = {list(config['scene'].keys())}")
            if "3d_snake" in config["scene"]:
                scene_config = config["scene"]["3d_snake"]
                print(f"Debug: 3d_snake config = {scene_config}")
                if "controller_mapping" in scene_config:
                    for role, dip in scene_config["controller_mapping"].items():
                        try:
                            player_id = PlayerID[role.upper()]
                            self.controller_mapping[dip] = player_id
                            print(f"Mapped controller DIP {dip} to {player_id.name}")
                        except KeyError:
                            print(f"Warning: Unknown player role '{role}' in controller mapping")
                else:
                    print("Debug: No controller_mapping found in 3d_snake config")
            else:
                print("Debug: 3d_snake not found in scene config")
        else:
            print("Debug: No scene config found")
        print(f"Debug: Final controller_mapping = {self.controller_mapping}")

        # Initialize controller input handler
        # Prepare addresses_to_enumerate for ControllerInputHandler
        addresses_for_handler = None
        if self.config and "controller_addresses" in self.config:
            print("Found 'controller_addresses' in config.")
            controller_addr_config = self.config["controller_addresses"]
            if isinstance(
                controller_addr_config, dict
            ):  # Expecting dict like {"0": {"ip": ..., "port": ...}}
                addresses_for_handler = []
                for dip_str, addr_info in controller_addr_config.items():
                    if isinstance(addr_info, dict) and "ip" in addr_info and "port" in addr_info:
                        addresses_for_handler.append((addr_info["ip"], addr_info["port"]))
                    else:
                        print(
                            f"Warning: Invalid address info format for DIP {dip_str} "
                            f"in controller_addresses. Expected {{ip:str, port:int}}."
                        )
                if not addresses_for_handler:
                    print(
                        "Warning: 'controller_addresses' was present but parsed into an empty list."
                    )
                    addresses_for_handler = None  # Fallback to no specific addresses
            else:
                print(
                    "Warning: 'controller_addresses' is not a dictionary as expected. "
                    "Will not use for specific enumeration."
                )

        if addresses_for_handler:
            print(
                f"Passing specific addresses to ControllerInputHandler for enumeration: "
                f"{addresses_for_handler}"
            )
        else:
            print(
                "No specific controller_addresses to pass; ControllerInputHandler will "
                "use its default enumeration (if any)."
            )

        # Pass controller_mapping for role assignment AND control_port_manager
        self.input_handler = ControllerInputHandler(
            controller_mapping=self.controller_mapping,
            control_port_manager=control_port_manager,
        )
        if not self.input_handler.start_initialization():
            print("GameScene: ControllerInputHandler initialization failed.")
            self.input_handler = None  # Ensure it's None on failure
        else:
            print("GameScene: ControllerInputHandler initialized successfully.")

            # Register this GameScene to receive button callbacks
            for controller_id in self.input_handler.controllers:
                self.input_handler.register_button_callback(controller_id, self.handle_button_event)

        self.last_update_time = 0
        self.last_countdown_time = 0
        self.game_over_active = False
        self.game_over_flash_state = {
            "count": 0,
            "timer": 0,
            "interval": 0.2,
            "border_on": False,
        }
        self.menu_active = True
        self.countdown_active = False
        self.countdown_value = None
        self.difficulty = None

        # Load available games
        self.available_games = self._load_available_games()
        self.current_game = self

    def _load_available_games(self):
        """Load all available game modules."""
        games = {}
        games_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "games")
        print(f"Looking for game modules in directory: {games_dir}")

        for filename in os.listdir(games_dir):
            if filename.endswith("_game.py"):
                module_name = filename[:-3]  # Remove .py extension
                print(f"Found game module file: {filename}, module name: {module_name}")

                try:
                    print(f"Attempting to load module: {module_name}")
                    spec = importlib.util.spec_from_file_location(
                        module_name, os.path.join(games_dir, filename)
                    )
                    if spec is None:
                        print(f"Error: Could not create spec for module {module_name}")
                        continue

                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Find the game class (should be the only class that inherits from BaseGame)
                    game_class_found = False
                    for name, obj in module.__dict__.items():
                        if isinstance(obj, type) and issubclass(obj, BaseGame) and obj != BaseGame:
                            print(f"Found game class '{name}' in module {module_name}")
                            # Store both the game class and its display name
                            display_name = getattr(
                                obj, "DISPLAY_NAME", module_name.replace("_game", "").upper()
                            )
                            games[module_name] = {"class": obj, "display_name": display_name}
                            game_class_found = True
                            break

                    if not game_class_found:
                        print(f"Warning: No game class found in module {module_name}")
                        print(
                            f"Module contains these objects: "
                            f"{[name for name in module.__dict__ if not name.startswith('__')]}"
                        )

                except ImportError as e:
                    print(f"ImportError loading game module {filename}: {e}")
                    print(
                        f"Module might have missing dependencies: {e.__class__.__name__}: {str(e)}"
                    )
                except AttributeError as e:
                    print(f"AttributeError loading game module {filename}: {e}")
                    print(
                        f"Module might be missing expected attributes: {e.__class__.__name__}: {str(e)}"
                    )
                except TypeError as e:
                    print(f"TypeError loading game module {filename}: {e}")
                    print(f"Possible type mismatch in module: {e.__class__.__name__}: {str(e)}")
                except Exception as e:
                    print(f"Error loading game module {filename}: {e.__class__.__name__}: {str(e)}")
                    import traceback

                    traceback.print_exc()

        print(f"Successfully loaded {len(games)} game modules: {list(games.keys())}")
        return games

    def get_player_config(self, player_id):
        """Get the configuration for a player."""
        raise NotImplementedError("get_player_config method not implemented")

    def get_player_score(self, player_id):
        """Get the score for a player."""
        if self.current_game:
            return self.current_game.get_player_score(player_id)
        return 0

    def get_opponent_score(self, player_id):
        """Get the score for a player's opponent."""
        if self.current_game:
            return self.current_game.get_opponent_score(player_id)
        return 0

    def reset_game(self):
        """Reset the game state."""
        if self.current_game:
            if self.current_game.menu_active:
                self.current_game = self
                self.menu_active = True
                self.game_started = False
                self.countdown_active = False
                self.countdown_value = None
                # Reset all menu-related state
                self.menu_selections = {}
                self.menu_votes = {}
                self.voting_states = {}
                # Re-register button callbacks for the menu
                if self.input_handler:
                    for controller_id in self.input_handler.controllers:
                        self.input_handler.register_button_callback(
                            controller_id, self.handle_button_event
                        )
            else:
                self.current_game.reset_game()
        else:
            # Initialize with first available game
            if self.available_games:
                game_info = next(iter(self.available_games.values()))
                game_class = game_info["class"]
                self.current_game = game_class(
                    width=self.width,
                    height=self.height,
                    length=self.length,
                    frameRate=self.frameRate,
                    config=self.config,
                    input_handler=self.input_handler,
                )

    def process_player_input(self, player_id, button, button_state=None):
        """Process input from a player."""
        if self.current_game and self.current_game != self:
            # If button_state is provided, use handle_button_event if available
            if button_state is not None and hasattr(self.current_game, "handle_button_event"):
                self.current_game.handle_button_event(player_id, button, button_state)
            elif hasattr(self.current_game, "process_player_input"):
                # Otherwise fall back to old-style process_player_input
                # Try with both 2 and 3 arguments for backwards compatibility
                try:
                    if button_state is not None:
                        self.current_game.process_player_input(player_id, button, button_state)
                    else:
                        self.current_game.process_player_input(player_id, button)
                except TypeError:
                    # If the game doesn't accept button_state, call with just button
                    try:
                        self.current_game.process_player_input(player_id, button)
                    except Exception as e:
                        print(f"Error in process_player_input: {e}")

    async def update_controller_display_state(self, controller_state, player_id):
        """Update the controller's LCD display for this player."""

        # Clear the display first
        controller_state.clear()

        # Handling display for GameScene (Game Selection Menu & Countdown)
        if self.menu_active:
            # Get controller DIP from the controller state
            controller_dip = controller_state.dip if hasattr(controller_state, "dip") else None

            # Find matching controller ID based on player_id if dip isn't available
            if controller_dip is None and self.input_handler:
                for cid, (cstate, pid) in self.input_handler.controllers.items():
                    if pid == player_id and cstate == controller_state:
                        controller_dip = cid
                        break

            if controller_dip is None:
                # Fallback if we can't determine the controller DIP
                controller_state.write_lcd(0, 0, "GAME SELECT ERROR")
                controller_state.write_lcd(0, 1, "CONTROLLER DIP")
                controller_state.write_lcd(0, 2, "NOT IDENTIFIED")
                await controller_state.commit()
                return

            current_selection = self.menu_selections.get(controller_dip, 0)
            has_voted = self.voting_states.get(controller_dip, False)
            waiting_count = sum(1 for v_dip in self.voting_states if self.voting_states[v_dip])

            # Calculate total players
            total_players = 0
            if (
                self.input_handler
                and hasattr(self.input_handler, "controllers")
                and self.input_handler.controllers
            ):
                total_players = len(self.input_handler.controllers)

            game_votes = {game_name: 0 for game_name in self.available_games.keys()}
            for voted_game_name in self.menu_votes.values():
                if voted_game_name in game_votes:
                    game_votes[voted_game_name] += 1

            # Write header
            controller_state.write_lcd(0, 0, "SELECT GAME:")

            # Get sorted list of game names
            game_names = list(self.available_games.keys())
            num_games = len(game_names)

            # Calculate which games to display (show 3 at a time with scrolling)
            display_start = 0
            if num_games > 3:
                # If selection is 0 or 1, start from 0
                if current_selection <= 0:
                    display_start = 0
                # If selection is at the end, show the last 3 games
                elif current_selection >= num_games - 1:
                    display_start = max(0, num_games - 3)
                # Otherwise, center the selection
                else:
                    display_start = current_selection - 1

            # Display the visible games (up to 3)
            for i in range(3):
                if display_start + i < num_games:
                    game_index = display_start + i
                    game_name_key = game_names[game_index]
                    game_info = self.available_games[game_name_key]
                    game_display_name = game_info["display_name"]

                    # Determine if this game is selected or voted for
                    marker = " "
                    if current_selection == game_index:
                        marker = "<"
                    elif has_voted and self.menu_votes.get(controller_dip) == game_name_key:
                        marker = "X"

                    # Get vote count for this game
                    votes = game_votes.get(game_name_key, 0)

                    # Display game name (left-aligned) and marker (right-aligned)
                    controller_state.write_lcd(0, i + 1, game_display_name)
                    controller_state.write_lcd(19, i + 1, marker)

                    # Show vote count if there are votes
                    if votes > 0:
                        controller_state.write_lcd(17, i + 1, str(votes))

            # Display status at the bottom (line 4)
            status_text = (
                f"Wait: {total_players - waiting_count} more"
                if has_voted and total_players > 0
                else "SELECT to vote"
            )
            controller_state.write_lcd(0, 4, status_text)

        elif self.countdown_active:
            # GameScene: Countdown to start a selected game
            selected_game_name = "GAME"
            if self.current_game:
                # Try to get the display name from the game class
                selected_game_name = getattr(
                    self.current_game.__class__,
                    "DISPLAY_NAME",
                    self.current_game.__class__.__name__.replace("Game", "").upper(),
                )

            controller_state.write_lcd(0, 0, "STARTING:")
            controller_state.write_lcd(0, 1, selected_game_name)
            controller_state.write_lcd(0, 2, f"IN {self.countdown_value}...")
            controller_state.write_lcd(0, 3, "GET READY!")

        elif self.current_game and self.current_game != self:
            # If current_game is set and not pointing to self, delegate to it
            if hasattr(self.current_game, "update_controller_display_state"):
                await self.current_game.update_controller_display_state(controller_state, player_id)
            else:
                # Backwards compatibility for older games using update_display
                if hasattr(self.current_game, "update_display"):
                    await self.current_game.update_display(controller_state, player_id)
                else:
                    # Fallback if game has no display update method
                    controller_state.write_lcd(0, 0, "GAME ACTIVE")
                    controller_state.write_lcd(0, 1, f"{self.current_game.__class__.__name__}")
                    controller_state.write_lcd(0, 2, "NO DISPLAY METHOD")
                    controller_state.write_lcd(0, 3, "IMPLEMENTED")
                    await controller_state.commit()
            return  # Return early as the current_game will handle commit

        # Commit the display updates for GameScene's own displays
        await controller_state.commit()

    def update_game_state(self):
        """Update the game state."""
        pass

    def render_game_state(self, raster):
        """Render the game state to the volumetric raster."""
        # Clear the raster (black background)
        for x in range(self.width):
            for y in range(self.height):
                for z in range(self.length):
                    raster.set_pix(x, y, z, RGB(0, 0, 0))

        # When in menu mode, render a rotating cube in the center
        if self.menu_active or self.countdown_active and self.countdown_value > 3:
            if self.countdown_active:
                scale = max(self.scale_down_time + 1 - time.monotonic(), 0)
            else:
                scale = 1
            # Initialize rotation state if not exists
            if not hasattr(self, "_cube_rot_state"):
                self._cube_rot_state = {
                    "angles": [0, 0, 0],  # x, y, z angles
                    "velocities": [0.2, 0.3, 0.1],  # Angular velocities
                    "size": 5.0,  # Current size
                    "target_size": 5.0,  # Target size
                    "size_velocity": 0.0,  # Size change velocity
                    "last_time": time.monotonic(),
                }

            state = self._cube_rot_state
            current_time = time.monotonic()
            dt = current_time - state["last_time"]
            state["last_time"] = current_time

            # Update rotation angles with varying velocities
            for i in range(3):
                state["angles"][i] = (state["angles"][i] + state["velocities"][i] * dt) % (
                    2 * math.pi
                )
                # Slowly vary velocities with noise
                state["velocities"][i] += (random.random() - 0.5) * 0.1 * dt
                state["velocities"][i] = max(min(state["velocities"][i], 0.5), -0.5)

            # Spring physics for size
            spring_k = 30.0  # Spring constant
            damping = 4.0  # Damping factor

            # Calculate spring force
            spring_force = (state["target_size"] - state["size"]) * spring_k
            # Apply damping
            damping_force = -state["size_velocity"] * damping
            # Update size
            state["size_velocity"] += (spring_force + damping_force) * dt
            state["size"] += state["size_velocity"] * dt

            # Check for new inputs to add "kicks"
            if self.input_handler:
                for controller_id, (
                    controller_state,
                    _,
                ) in self.input_handler.controllers.items():
                    if self.button_pressed:
                        self.button_pressed = False
                        # Add rotational kick
                        for i in range(3):
                            state["velocities"][i] += (random.random() - 0.5) * 6.0
                        # Compress the cube
                        state["target_size"] = 4.0
                        state["size_velocity"] += 10.0
                    else:
                        state["target_size"] = 5.0

            center_x = self.width // 2
            center_y = self.height // 2
            center_z = self.length // 2

            # Define cube points
            points = []
            size = state["size"] * scale
            for x in range(-1, 2, 1):
                for y in range(-1, 2, 1):
                    for z in range(-1, 2, 1):
                        # Scale points by size
                        px, py, pz = x * size, y * size, z * size

                        # Apply all rotations using rotation matrices
                        for axis, angle in enumerate(state["angles"]):
                            if axis == 0:  # X rotation
                                py, pz = (
                                    py * math.cos(angle) - pz * math.sin(angle),
                                    py * math.sin(angle) + pz * math.cos(angle),
                                )
                            elif axis == 1:  # Y rotation
                                px, pz = (
                                    px * math.cos(angle) - pz * math.sin(angle),
                                    px * math.sin(angle) + pz * math.cos(angle),
                                )
                            else:  # Z rotation
                                px, py = (
                                    px * math.cos(angle) - py * math.sin(angle),
                                    px * math.sin(angle) + py * math.cos(angle),
                                )

                        # Add to center and convert to integer coordinates
                        screen_x = int(center_x + px)
                        screen_y = int(center_y + py)
                        screen_z = int(center_z + pz)

                        if (
                            0 <= screen_x < self.width
                            and 0 <= screen_y < self.height
                            and 0 <= screen_z < self.length
                        ):
                            points.append((screen_x, screen_y, screen_z))

            # Render cube with different colors based on vote counts
            if self.available_games:
                color_idx = 0
                colors = [
                    RGB(255, 0, 0),  # Red
                    RGB(0, 255, 0),  # Green
                    RGB(0, 0, 255),  # Blue
                    RGB(255, 255, 0),  # Yellow
                    RGB(255, 0, 255),  # Purple
                    RGB(0, 255, 255),  # Cyan
                ]

                # Calculate vote percentages
                total_votes = sum(1 for _ in self.voting_states if self.voting_states[_])
                if total_votes > 0:
                    color_idx = (
                        len(self.voting_states) * 100 // len(self.input_handler.controllers)
                    ) % len(colors)

                for point in points:
                    raster.set_pix(point[0], point[1], point[2], colors[color_idx])

        # When in countdown mode, render a countdown number
        elif self.countdown_active and self.countdown_value is not None:
            # Simplified digit rendering in the center
            digit = self.countdown_value
            center_x = self.width // 2
            center_y = self.height // 2
            center_z = self.length // 2

            # Define digit patterns (very simplified)
            if digit == 3:
                # Draw a "3" in bright blue
                for x in range(-2, 3):
                    for y in range(-5, 6):
                        if x == -2 or y == -5 or y == 5 or y == 0:
                            raster.set_pix(center_x + x, center_y, center_z + y, RGB(0, 128, 255))
            elif digit == 2:
                # Draw a "2" in bright green
                for x in range(-2, 3):
                    for y in range(-5, 6):
                        if y == -5 or y == 5 or y == 0 or (x == 2 and y < 0) or (x == -2 and y > 0):
                            raster.set_pix(center_x + x, center_y, center_z + y, RGB(0, 255, 128))
            elif digit == 1:
                # Draw a "1" in bright red
                for y in range(-5, 6):
                    raster.set_pix(center_x, center_y, center_z + y, RGB(255, 0, 0))

    def render(self, raster, current_time):
        """Render the game state."""
        # Update controller displays independently of game state
        if self.input_handler:
            # Create a new event loop for this thread to handle async calls
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Create update tasks for each controller
            update_tasks = []
            for controller_id, (
                controller_state,
                player_id,
            ) in self.input_handler.controllers.items():
                update_tasks.append(
                    self.update_controller_display_state(controller_state, player_id)
                )

            # Run tasks to completion if any exist
            if update_tasks:
                loop.run_until_complete(asyncio.gather(*update_tasks))

            # Clean up
            loop.close()

        # Use higher frame rate for menu and countdown
        current_frame_rate = (
            self.menu_frame_rate if (self.menu_active or self.countdown_active) else self.frameRate
        )

        # Update game state
        if current_time - self.last_update_time >= 1.0 / current_frame_rate:
            self.last_update_time = current_time

            if self.input_handler:
                # Handle menu and countdown
                if self.menu_active:
                    self.select_game(current_time)
                elif self.countdown_active:
                    # Decrement countdown every second
                    if current_time - self.last_countdown_time >= 1.0:
                        print(f"Countdown: {self.countdown_value}")
                        self.countdown_value -= 1
                        self.last_countdown_time = current_time
                        if self.countdown_value <= 0:
                            print("Countdown finished, starting game")
                            self.countdown_active = False
                            self.game_started = True

                # Update game state if started and not in menu/countdown
                if self.game_started and not self.game_over_active and self.current_game != self:
                    self.current_game.update_game_state()
                elif self.game_over_active or self.menu_active or self.countdown_active:
                    self.update_game_state()

                # Check for restart signal
                if self.input_handler.check_for_restart_signal():
                    self.reset_game()
                    self.input_handler.clear_all_select_holds()

        # Clear the raster
        raster.clear()

        # Render game state - make sure we handle the case where current_game is self
        if self.current_game != self and self.current_game and not self.countdown_active:
            self.current_game.render_game_state(raster)
        else:
            self.render_game_state(raster)

    def select_game(self, current_time):
        """Handle game selection and voting."""
        if not self.input_handler:
            return

        # Check if all active controllers have voted
        active_controllers = set(self.input_handler.controllers.keys())
        voted_controllers = set(self.voting_states.keys())

        # Only proceed if at least one player has voted
        if not voted_controllers:
            return  # No votes cast yet

        # Check if all active controllers have voted
        if not active_controllers.issubset(voted_controllers):
            return  # Not all controllers have voted yet

        # All players have voted
        vote_counts = {game: 0 for game in self.available_games}
        for vote in self.menu_votes.values():
            if vote is not None and vote in vote_counts:
                vote_counts[vote] += 1

        # Find highest vote count
        if not vote_counts:
            return  # No valid votes

        max_votes = max(vote_counts.values())
        if max_votes == 0:
            return  # No votes cast

        # Get all games with max votes
        max_games = [g for g, v in vote_counts.items() if v == max_votes]
        if not max_games:
            return  # No games with votes

        # Randomly select from tied games
        selected_game = random.choice(max_games)

        # Create new game instance
        try:
            game_info = self.available_games[selected_game]
            game_class = game_info["class"]
            self.current_game = game_class(
                width=self.width,
                height=self.height,
                length=self.length,
                frameRate=self.frameRate,
                config=self.config,
                input_handler=self.input_handler,
            )

            print(
                f"Selected game: {selected_game}, instantiated {self.current_game.__class__.__name__}"
            )

            # Set difficulty for games that support it
            if hasattr(self.current_game, "set_difficulty") and self.difficulty is not None:
                self.current_game.set_difficulty(self.difficulty)
                print(f"Set difficulty to {self.difficulty}")

            # Start countdown
            self.menu_active = False
            self.countdown_active = True
            self.countdown_value = 4
            self.last_countdown_time = current_time
            self.scale_down_time = time.monotonic()
            self.input_handler.clear_all_select_holds()

        except Exception as e:
            print(f"Error creating game instance: {e}")
            import traceback

            traceback.print_exc()

    def process_menu_input(self, player_id, action):
        """Process menu-related input."""
        if not self.input_handler:
            return

        # Find the controller DIP/ID for this player_id
        controller_dip = None
        for cid, (_, pid) in self.input_handler.controllers.items():
            if pid == player_id:
                controller_dip = cid
                break

        if controller_dip is None:
            print(f"Warning: Could not find controller for player {player_id}")
            return

        self.button_pressed = True

        if action == Button.SELECT:
            if controller_dip in self.voting_states and self.voting_states[controller_dip]:
                # If already voted, remove vote
                self.voting_states[controller_dip] = False
                self.menu_votes.pop(controller_dip, None)
            else:
                # Convert current selection to vote
                selection = self.menu_selections.get(controller_dip, 0)
                selected_game = list(self.available_games.keys())[selection]
                self.menu_votes[controller_dip] = selected_game
                self.voting_states[controller_dip] = True
        elif action == Button.UP:
            # Move selection up with wraparound
            current = self.menu_selections.get(controller_dip, 0)
            self.menu_selections[controller_dip] = (current - 1) % len(self.available_games)
            # If player was in voting state, remove their vote
            if controller_dip in self.voting_states and self.voting_states[controller_dip]:
                self.voting_states[controller_dip] = False
                self.menu_votes.pop(controller_dip, None)
        elif action == Button.DOWN:
            # Move selection down with wraparound
            current = self.menu_selections.get(controller_dip, 0)
            self.menu_selections[controller_dip] = (current + 1) % len(self.available_games)
            # If player was in voting state, remove their vote
            if controller_dip in self.voting_states and self.voting_states[controller_dip]:
                self.voting_states[controller_dip] = False
                self.menu_votes.pop(controller_dip, None)

    def cleanup(self):
        """Clean up resources."""
        print("Cleaning up game...")
        if self.input_handler:
            self.input_handler.stop()

    def handle_button_event(self, player_id, button, button_state):
        """Handle button events from the controllers.

        This is called via the callback system when buttons change state.
        """
        if self.menu_active and button_state == ButtonState.PRESSED:
            # Handle menu inputs only on button press
            self.process_menu_input(player_id, button)
        elif self.game_started and not self.game_over_active and self.current_game != self:
            # Forward to current game if we have one
            if hasattr(self.current_game, "handle_button_event"):
                self.current_game.handle_button_event(player_id, button, button_state)
            else:
                # Fall back to old-style process_player_input
                self.process_player_input(player_id, button, button_state)
