import math
import random
import time

from artnet import HSV
from games.util.base_game import RGB, BaseGame, Difficulty, PlayerID, TeamID
from games.util.game_util import Button, ButtonState

# Configuration mapping player roles to their team and view orientation
PLAYER_CONFIG = {
    PlayerID.P1: {
        "team": TeamID.BLUE,
        "view": (-1, 0, 0),  # -X view
        "left_dir": (0, -1, 0),  # -Y
        "right_dir": (0, 1, 0),  # +Y
        "up_dir": (0, 0, 1),  # +Z
        "down_dir": (0, 0, -1),  # -Z
    },
    PlayerID.P2: {
        "team": TeamID.BLUE,
        "view": (0, -1, 0),  # -Y view
        "left_dir": (1, 0, 0),  # +X
        "right_dir": (-1, 0, 0),  # -X
        "up_dir": (0, 0, 1),  # +Z
        "down_dir": (0, 0, -1),  # -Z
    },
    PlayerID.P3: {
        "team": TeamID.ORANGE,
        "view": (1, 0, 0),  # +X view
        "left_dir": (0, 1, 0),  # +Y
        "right_dir": (0, -1, 0),  # -Y
        "up_dir": (0, 0, 1),  # +Z
        "down_dir": (0, 0, -1),  # -Z
    },
    PlayerID.P4: {
        "team": TeamID.ORANGE,
        "view": (0, 1, 0),  # +Y view
        "left_dir": (-1, 0, 0),  # -X
        "right_dir": (1, 0, 0),  # +X
        "up_dir": (0, 0, 1),  # +Z
        "down_dir": (0, 0, -1),  # -Z
    },
}


class RainbowExplosion:
    def __init__(self, position, time):
        self.position = position
        self.birth_time = time
        self.lifetime = 1.0  # Explosion lasts 1 second
        self.radius = 0
        self.max_radius = 5

    def is_expired(self, current_time):
        return current_time - self.birth_time > self.lifetime

    def get_current_radius(self, current_time):
        age = current_time - self.birth_time
        if age < self.lifetime:
            # Expand from 0 to max_radius over lifetime
            return self.max_radius * (age / self.lifetime)
        return 0


class SnakeData:
    def __init__(self, id, color, start_pos, start_dir):
        self.id = id
        self.color = color
        self.body = [start_pos]
        self.direction = start_dir
        self.length = 3
        self.score = 0


class SnakeGame(BaseGame):
    DISPLAY_NAME = "Snake"

    def __init__(
        self,
        width=20,
        height=20,
        length=20,
        frameRate=30,
        config=None,
        input_handler=None,
        sound_manager=None,
    ):
        self.thickness = 2  # Each snake segment is 2x2x2 voxels
        self.width = width // self.thickness
        self.height = height // self.thickness
        self.length = length // self.thickness

        # Per-player control mode: True = L/R mode, False = IN/OUT mode
        # Initialize before super().__init__() because reset_game() is called there
        self.control_mode = {}  # Maps player_id to bool (True = L/R, False = IN/OUT)
        for player_id in PlayerID:
            self.control_mode[player_id] = True  # Default to L/R mode

        super().__init__(
            width,
            height,
            length,
            frameRate,
            config=config,
            input_handler=input_handler,
            sound_manager=sound_manager,
        )
        self.snakes = {}  # Maps player_id to snake body (deque of positions)
        self.apple = None
        self.apple_color = RGB(0, 255, 0)  # Green food
        self.snake_colors = {
            TeamID.BLUE: RGB(0, 0, 255),  # Blue snake
            TeamID.ORANGE: RGB(255, 165, 0),  # Orange snake
        }
        self.last_step_time = 0  # Track last game step update
        self.step_rate = 3  # Default to 3 steps per second (MEDIUM/HARD)
        self.max_score = 9

        # Menu-related attributes
        self.menu_active = True
        self.countdown_active = False
        self.countdown_value = None
        self.difficulty = None
        self.menu_selections = (
            {}
        )  # Maps controller_id to their current selection (0=EASY, 1=MEDIUM, 2=HARD)
        self.menu_votes = {}  # Maps controller_id to their difficulty vote
        self.voting_states = {}  # Maps controller_id to whether they have voted
        self.last_countdown_time = 0
        self.game_started = False  # Track whether the game has started

        self.explosions = []

        self.reset_game()

    def reset_game(self):
        """Reset the game state."""
        # Initialize blue snake
        blue_start = (self.width // 8, self.length * 3 // 8, self.height // 4)
        self.snakes = {TeamID.BLUE: SnakeData("blue", RGB(255, 0, 0), blue_start, (1, 0, 0))}

        # Initialize orange snake
        orange_start = (3 * self.width // 8, self.length // 8, self.height // 4)
        self.snakes[TeamID.ORANGE] = SnakeData("orange", RGB(255, 165, 0), orange_start, (-1, 0, 0))

        # Set initial lengths
        self.snakes[TeamID.BLUE].length = 3
        self.snakes[TeamID.ORANGE].length = 3

        # Reset scores
        self.snakes[TeamID.BLUE].score = 0
        self.snakes[TeamID.ORANGE].score = 0

        # Place first apple
        self.place_new_apple()

        self.game_over_active = False
        self.game_over_flash_state = {
            "count": 0,
            "timer": 0,
            "interval": 0.2,
            "border_on": False,
            "border_color": RGB(255, 0, 0),
        }
        self.last_step_time = 0
        self.step_rate = 3  # Reset to default step rate
        self.game_started = False  # Reset game started state
        self.menu_active = True  # Start in menu state
        self.countdown_active = False
        self.countdown_value = None
        self.difficulty = None
        self.menu_selections.clear()
        self.menu_votes.clear()
        self.voting_states.clear()

        # Reset control modes to default (L/R mode)
        for player_id in PlayerID:
            self.control_mode[player_id] = True

        # Spawn initial food
        self.place_new_apple()

    def get_player_score(self, player_id):
        """Get the score for a player."""
        return self.snakes[PLAYER_CONFIG[player_id]["team"]].score

    def get_opponent_score(self, player_id):
        """Get the score for a player's opponent."""
        config = PLAYER_CONFIG[player_id]
        team = config["team"]
        opponent_team = TeamID.ORANGE if team == TeamID.BLUE else TeamID.BLUE

        # Sum up scores of all players on the opponent's team
        total_score = 0
        for team_id, snake in self.snakes.items():
            if team_id == opponent_team:
                total_score += snake.score
        return total_score

    def select_difficulty(self):
        """Handle difficulty selection and voting."""
        if not self.input_handler or not hasattr(self.input_handler, "controllers"):
            return

        # Check if all active controllers have voted
        active_controllers = set(self.input_handler.controllers.keys())
        voted_controllers = set(self.voting_states.keys())

        # Only proceed if at least one controller has voted
        if not voted_controllers:
            return  # No votes cast yet

        # Check if all active controllers have voted
        if not active_controllers.issubset(voted_controllers):
            return  # Not all controllers have voted yet

        # All players have voted
        vote_counts = {d: 0 for d in Difficulty}
        for difficulty in self.menu_votes.values():
            if difficulty in vote_counts:
                vote_counts[difficulty] += 1

        # Find highest vote count
        max_votes = max(vote_counts.values())
        if max_votes == 0:
            return  # No votes cast

        # Get all difficulties with max votes
        max_difficulties = [d for d, v in vote_counts.items() if v == max_votes]
        if not max_difficulties:
            return  # No difficulties with votes

        # Randomly select from tied difficulties
        self.difficulty = random.choice(max_difficulties)
        print(f"Selected difficulty: {self.difficulty.name}")

        # Set game speed based on difficulty
        if self.difficulty == Difficulty.EASY:
            self.step_rate = 1  # 1 step per second
        elif self.difficulty == Difficulty.MEDIUM:
            self.step_rate = 2  # 2 steps per second
        else:  # HARD
            self.step_rate = 3  # 3 steps per second

        # Start countdown
        self.menu_active = False
        self.countdown_active = True
        self.countdown_value = 3
        self.last_countdown_time = time.monotonic()

    def process_menu_input(self, player_id, action):
        """Process menu-related input."""
        if not self.input_handler or not hasattr(self.input_handler, "controllers"):
            return

        # Find the controller DIP/ID for this player
        controller_dip = None
        for cid, (_, pid) in self.input_handler.controllers.items():
            if pid == player_id:
                controller_dip = cid
                break

        if controller_dip is None:
            print(f"Warning: Could not find controller for player {player_id}")
            return

        if action == Button.SELECT:
            if controller_dip in self.voting_states and self.voting_states[controller_dip]:
                # If already voted, remove vote
                self.voting_states[controller_dip] = False
                self.menu_votes.pop(controller_dip, None)
                print(f"Player {player_id} removed their vote")
            else:
                # Convert current selection to difficulty directly
                selection = self.menu_selections.get(controller_dip, 0)
                difficulties = list(Difficulty)
                if 0 <= selection < len(difficulties):
                    voted_difficulty = difficulties[selection]
                    self.menu_votes[controller_dip] = voted_difficulty
                    self.voting_states[controller_dip] = True
                    print(f"Player {player_id} voted for {voted_difficulty.name}")
                else:
                    print(f"Invalid selection {selection} for player {player_id}")
        elif action == Button.UP:
            # Move selection up with wraparound
            current = self.menu_selections.get(controller_dip, 0)
            self.menu_selections[controller_dip] = (current - 1) % len(Difficulty)
            # If player was in voting state, remove their vote
            if controller_dip in self.voting_states and self.voting_states[controller_dip]:
                self.voting_states[controller_dip] = False
                self.menu_votes.pop(controller_dip, None)
                print(f"Player {player_id} changed selection, vote removed")
        elif action == Button.DOWN:
            # Move selection down with wraparound
            current = self.menu_selections.get(controller_dip, 0)
            self.menu_selections[controller_dip] = (current + 1) % len(Difficulty)
            # If player was in voting state, remove their vote
            if controller_dip in self.voting_states and self.voting_states[controller_dip]:
                self.voting_states[controller_dip] = False
                self.menu_votes.pop(controller_dip, None)
                print(f"Player {player_id} changed selection, vote removed")

    def update_game_state(self):
        """Update the game state."""
        current_time = time.monotonic()

        # Handle menu and countdown
        if self.menu_active:
            self.select_difficulty()
        elif self.countdown_active:
            # Decrement countdown every second
            if current_time - self.last_countdown_time >= 1.0:
                self.countdown_value -= 1
                self.last_countdown_time = current_time
                if self.countdown_value <= 0:
                    self.countdown_active = False
                    self.game_started = True
                    print("Game started!")  # Debug log

        # Only update snake positions at the step rate if game has started
        if self.game_started and not self.game_over_active:
            if current_time - self.last_step_time >= 1.0 / self.step_rate:
                self.last_step_time = current_time

                self.update_snake(TeamID.BLUE)
                self.update_snake(TeamID.ORANGE)

        # Update game over flash state (this can run at frame rate)
        if self.game_over_active:
            if (
                current_time - self.game_over_flash_state["timer"]
                >= self.game_over_flash_state["interval"]
            ):
                self.game_over_flash_state["timer"] = current_time
                self.game_over_flash_state["border_on"] = not self.game_over_flash_state[
                    "border_on"
                ]
                if self.game_over_flash_state["count"] <= 0:
                    self.game_over_flash_state["border_on"] = False
                else:
                    self.game_over_flash_state["count"] -= 1

    def render_game_state(self, raster):
        """Render the game state to the raster."""
        # Clear the raster first
        for x in range(raster.width):
            for y in range(raster.height):
                for z in range(raster.length):
                    raster.set_pix(x, y, z, RGB(0, 0, 0))  # Black background

        current_time = time.monotonic()

        # Draw explosions
        active_explosions = []
        for explosion in self.explosions:
            if not explosion.is_expired(current_time):
                active_explosions.append(explosion)
                radius = explosion.get_current_radius(current_time)
                x, y, z = explosion.position
                x *= self.thickness
                y *= self.thickness
                z *= self.thickness

                # Draw rainbow shell
                for dx in range(-int(radius), int(radius) + 1):
                    for dy in range(-int(radius), int(radius) + 1):
                        for dz in range(-int(radius), int(radius) + 1):
                            # Only draw points on the shell
                            if abs(math.sqrt(dx * dx + dy * dy + dz * dz) - radius) < 0.5:
                                nx = x + dx
                                ny = y + dy
                                nz = z + dz
                                if (
                                    0 <= nx < raster.width
                                    and 0 <= ny < raster.height
                                    and 0 <= nz < raster.length
                                ):
                                    # Create rainbow color based on position
                                    hue = ((dx + dy + dz) * 4 + current_time * 50) % 256
                                    raster.set_pix(nx, ny, nz, RGB.from_hsv(HSV(hue, 255, 255)))
        self.explosions = active_explosions

        # Draw food
        if self.apple:
            x, y, z = self.apple
            x *= self.thickness
            y *= self.thickness
            z *= self.thickness
            for dx in range(self.thickness):
                for dy in range(self.thickness):
                    for dz in range(self.thickness):
                        if (
                            x + dx < raster.width
                            and y + dy < raster.height
                            and z + dz < raster.length
                        ):
                            raster.set_pix(x + dx, y + dy, z + dz, self.apple_color)

        # Draw snakes
        for team_id, snake in self.snakes.items():
            # Get player's team
            snake_color = self.snake_colors.get(team_id, RGB(255, 255, 255))

            # Draw all segments
            for segment in snake.body:
                x, y, z = segment
                x *= self.thickness
                y *= self.thickness
                z *= self.thickness
                for dx in range(self.thickness):
                    for dy in range(self.thickness):
                        for dz in range(self.thickness):
                            if (
                                x + dx < raster.width
                                and y + dy < raster.height
                                and z + dz < raster.length
                            ):
                                raster.set_pix(x + dx, y + dy, z + dz, snake_color)

        # Draw game over border
        if self.game_over_active and self.game_over_flash_state["border_on"]:
            border_color = self.game_over_flash_state["border_color"]
            for x in range(raster.width):
                for y in range(raster.height):
                    for z in range(raster.length):
                        if (
                            x == 0
                            or x == raster.width - 1
                            or y == 0
                            or y == raster.height - 1
                            or z == 0
                            or z == raster.length - 1
                        ):
                            raster.set_pix(x, y, z, border_color)

    async def update_controller_display_state(self, controller_state, player_id):
        """Update the controller display for this player."""
        # Clear the display first
        controller_state.clear()

        # Handle menu display
        if self.menu_active:
            # Get controller DIP from the controller state or find it by player_id
            controller_dip = controller_state.dip if hasattr(controller_state, "dip") else None

            # Find matching controller ID based on player_id if dip isn't available
            if controller_dip is None and self.input_handler:
                for cid, (cstate, pid) in self.input_handler.controllers.items():
                    if pid == player_id and cstate == controller_state:
                        controller_dip = cid
                        break

            if controller_dip is None:
                # Fallback if we can't determine the controller DIP
                controller_state.write_lcd(0, 0, "SNAKE: ERROR")
                controller_state.write_lcd(0, 1, "CONTROLLER DIP")
                controller_state.write_lcd(0, 2, "NOT IDENTIFIED")
                await controller_state.commit()
                return

            current_selection = self.menu_selections.get(controller_dip, 0)
            has_voted = self.voting_states.get(controller_dip, False)

            # Calculate total players and waiting count
            total_players = 0
            waiting_count = 0
            if self.input_handler and hasattr(self.input_handler, "controllers"):
                total_players = len(self.input_handler.controllers)
                waiting_count = sum(1 for v_dip in self.voting_states if self.voting_states[v_dip])

            # Calculate votes for each difficulty
            vote_counts = {d: 0 for d in Difficulty}
            for vote_difficulty in self.menu_votes.values():
                if vote_difficulty in vote_counts:
                    vote_counts[vote_difficulty] += 1

            # Write header
            controller_state.write_lcd(0, 0, "SNAKE: SELECT LEVEL")

            # Display difficulty options
            difficulties = list(Difficulty)
            for i, diff in enumerate(difficulties):
                marker = " "

                # Check if this is the current selection
                if i == current_selection:
                    marker = "<"

                # Check if this player has voted for this difficulty
                voted_difficulty = self.menu_votes.get(controller_dip)
                if has_voted and voted_difficulty == diff:
                    marker = "X"

                # Display difficulty name
                controller_state.write_lcd(0, i + 1, f"{diff.name}")

                # Display vote count for this difficulty
                vote_count = vote_counts.get(diff, 0)
                if vote_count > 0:
                    controller_state.write_lcd(15, i + 1, f"({vote_count})")

                # Display selection marker
                controller_state.write_lcd(19, i + 1, marker)

            # Display status
            status_text = (
                f"Wait: {total_players - waiting_count} more"
                if has_voted and total_players > 0
                else "SELECT to vote"
            )
            controller_state.write_lcd(0, 4, status_text)

        elif self.countdown_active:
            # Default countdown display
            difficulty_text = ""
            if hasattr(self, "difficulty") and self.difficulty:
                difficulty_text = f"{self.difficulty.name}"

            controller_state.write_lcd(0, 0, f"{self.__class__.__name__.replace('Game', '')}")
            if difficulty_text:
                controller_state.write_lcd(0, 1, difficulty_text)
            controller_state.write_lcd(0, 2, f"GET READY! {self.countdown_value}...")
            controller_state.write_lcd(0, 3, "Hold SELECT to EXIT")
        elif self.game_over_active:
            # Default game over display
            score = self.get_player_score(player_id)
            other_score = self.get_opponent_score(player_id)
            result = "DRAW"
            if score > other_score:
                result = "WIN! :)"
            elif score < other_score:
                result = "LOSE :("

            config = PLAYER_CONFIG[player_id]
            team_name = config["team"].name if config and "team" in config else "NO TEAM"

            controller_state.write_lcd(0, 0, f"GAME OVER! YOU {result}")
            controller_state.write_lcd(0, 1, f"TEAM {team_name}: {score}")
            controller_state.write_lcd(0, 2, f"OPPONENT: {other_score}")
            controller_state.write_lcd(0, 3, "Hold SELECT to EXIT")
        else:
            # Default in-game display
            config = PLAYER_CONFIG[player_id]
            team_name = config["team"].name if config and "team" in config else "NO TEAM"
            score = self.get_player_score(player_id)
            other_score = self.get_opponent_score(player_id)

            # Display current control mode
            mode = self.control_mode.get(player_id, True)
            mode_str = "L/R" if mode else "IN/OUT"

            controller_state.write_lcd(0, 0, f"TEAM: {team_name}")
            controller_state.write_lcd(0, 1, f"SCORE:    {score}")
            controller_state.write_lcd(0, 2, f"OPPONENT: {other_score}")
            controller_state.write_lcd(0, 3, f"MODE: {mode_str} (SELECT)")

        # Commit the changes
        await controller_state.commit()

    def process_player_input(self, player_id, action, button_state):
        """Process input from a player."""
        if self.game_over_active:
            return

        if button_state == ButtonState.RELEASED:
            # Ignore button releases
            return

        if self.menu_active:
            self.process_menu_input(player_id, action)
            return

        if not self.game_started:
            return

        print(f"Processing input for player {player_id}: {action}")

        config = PLAYER_CONFIG[player_id]
        team = config["team"]
        snake = self.snakes[team]

        # Handle SELECT button to toggle control mode
        if action == Button.SELECT and button_state == ButtonState.PRESSED:
            # Toggle control mode for this player
            self.control_mode[player_id] = not self.control_mode[player_id]
            mode_str = "L/R" if self.control_mode[player_id] else "IN/OUT"
            print(f"Player {player_id} control mode toggled to: {mode_str}")
            return

        # Map the action to a new direction based on player's view orientation
        if action == Button.LEFT:
            if self.control_mode[player_id]:
                # L/R mode: use left_dir
                new_dir = config["left_dir"]
            else:
                # IN/OUT mode: LEFT = OUT (opposite of view direction)
                view = config["view"]
                new_dir = (-view[0], -view[1], -view[2])
        elif action == Button.RIGHT:
            if self.control_mode[player_id]:
                # L/R mode: use right_dir
                new_dir = config["right_dir"]
            else:
                # IN/OUT mode: RIGHT = IN (same as view direction)
                view = config["view"]
                new_dir = view
        elif action == Button.UP:
            new_dir = config["up_dir"]
        elif action == Button.DOWN:
            new_dir = config["down_dir"]
        else:
            return

        # Prevent 180-degree turns
        if (
            snake.direction[0] != -new_dir[0]
            or snake.direction[1] != -new_dir[1]
            or snake.direction[2] != -new_dir[2]
        ):
            snake.direction = new_dir

    def valid(self, x, y, z):
        if self.difficulty in [Difficulty.EASY, Difficulty.MEDIUM]:
            # Wrap around in EASY and MEDIUM modes
            x = x % (self.width // self.thickness)
            y = y % (self.height // self.thickness)
            z = z % (self.length // self.thickness)
            return True
        return (
            0 <= x < (self.width // self.thickness)
            and 0 <= y < (self.height // self.thickness)
            and 0 <= z < (self.length // self.thickness)
        )

    def update_snake(self, snake_id):
        """Update a single snake's position and handle collisions."""
        snake = self.snakes[snake_id]
        other_snake_id = TeamID.ORANGE if snake_id == TeamID.BLUE else TeamID.BLUE
        other_snake = self.snakes[other_snake_id]

        # Calculate new head position
        head = snake.body[0]
        new_head = (
            head[0] + snake.direction[0],
            head[1] + snake.direction[1],
            head[2] + snake.direction[2],
        )

        # Handle wrapping in EASY and MEDIUM modes
        if self.difficulty in [Difficulty.EASY, Difficulty.MEDIUM]:
            new_head = (
                new_head[0] % (self.width // self.thickness),
                new_head[1] % (self.height // self.thickness),
                new_head[2] % (self.length // self.thickness),
            )

        # Check for collisions
        if (
            not self.valid(*new_head)  # Wall collision (only in HARD mode)
            or new_head in snake.body  # Self collision
            or new_head in other_snake.body
        ):
            snake.score -= 1
            snake.length -= 1
            if snake.length <= 0:
                snake.length = 0
                self.game_over_active = True
                # Play collision sound
                self.play_collision_sound()
                self.game_over_flash_state = {
                    "count": 20,  # 5 flashes (on/off)
                    "timer": 0,
                    "interval": 0.2,
                    "border_on": False,
                    "border_color": RGB(255, 0, 0),
                }
                return False
            else:
                # Respawn snake at a new position
                while True:
                    x = random.randint(0, self.width - 1)
                    y = random.randint(0, self.height - 1)
                    z = random.randint(0, self.length - 1)
                    pos = (x, y, z)
                    if pos not in snake.body and pos not in other_snake.body and pos != self.apple:
                        snake.body = [pos]
                        snake.direction = (1, 0, 0)  # Default direction
                        break
                return True

        # If a snake has exceeded the max score, set the game over active
        if snake.score >= self.max_score:
            self.game_over_active = True
            # Play victory sound
            self.play_game_sound("warble")
            self.game_over_flash_state = {
                "count": 20,  # 5 flashes (on/off)
                "timer": 0,
                "interval": 0.2,
                "border_on": False,
                "border_color": RGB(0, 255, 0),
            }
            return False

        # Move snake
        snake.body.insert(0, new_head)
        if len(snake.body) > snake.length:
            snake.body.pop()

        # Check for apple consumption
        if new_head == self.apple:
            snake.length += 1
            snake.score += 1  # Increment score when eating an apple
            # Play scoring sound
            self.play_score_sound()
            # Create explosion effect at apple position
            self.explosions.append(RainbowExplosion(self.apple, time.monotonic()))
            self.place_new_apple()

        return True

    def place_new_apple(self):
        """Place apple in a valid position not occupied by any snake."""
        while True:
            x = random.randint(0, self.width // self.thickness - 1)
            y = random.randint(0, self.height // self.thickness - 1)
            z = random.randint(0, self.length // self.thickness - 1)
            pos = (x, y, z)
            if (
                pos not in self.snakes[TeamID.BLUE].body
                and pos not in self.snakes[TeamID.ORANGE].body
            ):
                self.apple = pos
                break
