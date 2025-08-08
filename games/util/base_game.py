from enum import Enum

from artnet import RGB
from games.util.game_util import (
    ControllerInputHandler,
    DisplayManager,
)


class Difficulty(Enum):
    EASY = 1
    MEDIUM = 2
    HARD = 3


class PlayerID(Enum):
    P1 = 1  # Controls from -X view
    P2 = 2  # Controls from -Y view
    P3 = 3  # Controls from +X view
    P4 = 4  # Controls from +Y view


class TeamID(Enum):
    RED = 1
    ORANGE = 2
    YELLOW = 3
    GREEN = 4
    BLUE = 5
    PURPLE = 6

    def get_color(self, alternate=False):
        color = {
            TeamID.RED: RGB(255, 0, 0),
            TeamID.ORANGE: RGB(255, 128, 0),
            TeamID.YELLOW: RGB(255, 255, 0),
            TeamID.GREEN: RGB(0, 255, 0),
            TeamID.BLUE: RGB(0, 0, 255),
            TeamID.PURPLE: RGB(128, 0, 128),
        }[self]
        if alternate:
            return RGB(color.r ^ 32, color.g ^ 32, color.b ^ 32)
        return color


class BaseGame:
    # Default display name - subclasses should override this
    DISPLAY_NAME = "Unknown Game"
    
    def __init__(
        self,
        width=20,
        height=20,
        length=20,
        frameRate=3,
        config=None,
        input_handler=None,
    ):
        self.width = width
        self.height = height
        self.length = length
        self.frameRate = frameRate
        self.base_frame_rate = frameRate  # Store original frame rate
        self.config = config
        self.input_handler = input_handler

        # Initialize menu-related attributes
        self.menu_selections = {}  # Maps controller_id to their current selection
        self.menu_votes = {}  # Maps controller_id to their game vote
        self.voting_states = {}  # Maps controller_id to whether they have voted

        # Store controller mapping from config
        self.controller_mapping = {}
        if config and "scene" in config and "3d_snake" in config["scene"]:
            scene_config = config["scene"]["3d_snake"]
            if "controller_mapping" in scene_config:
                for role, dip in scene_config["controller_mapping"].items():
                    try:
                        role_upper = role.upper()
                        player_id = PlayerID[role_upper]
                        self.controller_mapping[dip] = player_id
                    except KeyError:
                        print(
                            f"BaseGame: Warning: Unknown player role '{role}' in controller mapping"
                        )

        self.display_manager = DisplayManager()
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

        # Register button callbacks for all controllers if input_handler is provided
        if self.input_handler and isinstance(self.input_handler, ControllerInputHandler):
            for controller_id in self.input_handler.controllers:
                self.input_handler.register_button_callback(controller_id, self.handle_button_event)

        self.reset_game()

    def handle_button_event(self, player_id, button, button_state):
        """Handle button events with state information.

        This is the new callback-based approach that provides both press and release events.
        Games should override this method to handle button events.

        Args:
            player_id: PlayerID enum value
            button: Button enum value
            button_state: ButtonState enum value (PRESSED, RELEASED, HELD)
        """
        # In game mode, directly pass all button events to process_player_input
        self.process_player_input(player_id, button, button_state)

    def get_player_score(self, player_id):
        """Get the score for a player."""
        raise NotImplementedError("Subclasses must implement get_player_score")

    def get_opponent_score(self, player_id):
        """Get the score for a player's opponent."""
        raise NotImplementedError("Subclasses must implement get_opponent_score")

    def reset_game(self):
        """Reset the game state."""
        raise NotImplementedError("Subclasses must implement reset_game")

    def process_player_input(self, player_id, button, button_state):
        """Process input from a player.

        Args:
            player_id: PlayerID enum value
            button: Button enum value (UP, DOWN, LEFT, RIGHT, SELECT)
            button_state: ButtonState enum value (PRESSED, RELEASED, HELD)
        """
        raise NotImplementedError("Subclasses must implement process_player_input")

    def update_game_state(self):
        """Update the game state."""
        raise NotImplementedError("Subclasses must implement update_game_state")

    def render_game_state(self, raster):
        """Render the game state to the raster."""
        raise NotImplementedError("Subclasses must implement render_game_state")

    async def update_controller_display_state(self, controller_state, player_id):
        """Update the controller's LCD display for this player.

        Args:
            controller_state: The ControllerState for the player's controller
            player_id: The PlayerID enum value for this player

        Note: Implement this in each game subclass to customize display content.
        Each implementation should clear and commit the display.
        """
        # Clear the display first
        controller_state.clear()
        controller_state.write_lcd(0, 0, "Game: {}".format(self.__class__.__name__))

        # Commit the changes
        await controller_state.commit()

    # Add backwards compatibility method
    async def update_display(self, controller_state, player_id):
        """Legacy method - delegates to update_controller_display_state"""
        return await self.update_controller_display_state(controller_state, player_id)

    def cleanup(self):
        """Clean up resources."""
        print("Cleaning up game...")
        if isinstance(self.input_handler, ControllerInputHandler):
            # Unregister callbacks
            for controller_id in self.input_handler.controllers:
                self.input_handler.unregister_button_callback(controller_id)
            # Stop the input handler
            self.input_handler.stop()
