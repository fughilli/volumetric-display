import random
import time

from games.util.base_game import RGB, BaseGame, PlayerID
from games.util.game_util import Button, ButtonState

PLAYER_TO_COLOR = {
    PlayerID.P1: RGB(0, 128, 255),
    PlayerID.P2: RGB(128, 0, 255),
    PlayerID.P3: RGB(255, 128, 0),
    PlayerID.P4: RGB(255, 0, 128),
}


class BlinkyGame(BaseGame):
    DISPLAY_NAME = "Blinky"

    def __init__(
        self,
        width=20,
        height=20,
        length=20,
        frameRate=30,
        config=None,
        input_handler=None,
    ):
        super().__init__(
            width, height, length, frameRate, config=config, input_handler=input_handler
        )
        self.cube = None
        self.cube_color = None
        self.cube_timer = 0
        self.cube_duration = 0.5  # How long the cube stays visible
        self.reset_game()

    def reset_game(self):
        """Reset the game state."""
        self.cube = None
        self.cube_color = None
        self.cube_timer = 0
        self.game_over_active = False
        self.game_over_flash_state = {
            "count": 0,
            "timer": 0,
            "interval": 0.2,
            "border_on": False,
        }

    def get_player_score(self, player_id):
        """Get the score for a player."""
        return 0  # Blinky game doesn't have scores

    def get_opponent_score(self, player_id):
        """Get the score for a player's opponent."""
        return 0  # Blinky game doesn't have scores

    def process_player_input(self, player_id, button, button_state):
        """Process input from a player."""
        if self.game_over_active:
            return

        if button_state == ButtonState.RELEASED:
            return

        if button == Button.SELECT:
            # Create a random cube
            size = random.randint(3, 8)
            center_x = random.randint(size, self.width - size - 1)
            center_y = random.randint(size, self.height - size - 1)
            center_z = random.randint(size, self.length - size - 1)

            # Create cube points
            self.cube = []
            for x in range(center_x - size, center_x + size + 1):
                for y in range(center_y - size, center_y + size + 1):
                    for z in range(center_z - size, center_z + size + 1):
                        # Only add points that form the cube's surface
                        if (
                            x == center_x - size
                            or x == center_x + size
                            or y == center_y - size
                            or y == center_y + size
                            or z == center_z - size
                            or z == center_z + size
                        ):
                            self.cube.append((x, y, z))

            self.cube_color = PLAYER_TO_COLOR[player_id]
            self.cube_timer = time.monotonic()

            # Play a pop sound when creating a cube
            self.play_game_sound("pop")

    def update_game_state(self):
        """Update the game state."""
        if self.game_over_active:
            # Update game over flash state
            current_time = time.monotonic()
            if (
                current_time - self.game_over_flash_state["timer"]
                >= self.game_over_flash_state["interval"]
            ):
                self.game_over_flash_state["timer"] = current_time
                self.game_over_flash_state["border_on"] = not self.game_over_flash_state[
                    "border_on"
                ]
                self.game_over_flash_state["count"] += 1
        else:
            # Check if cube should disappear
            if self.cube and time.monotonic() - self.cube_timer >= self.cube_duration:
                self.cube = None
                self.cube_color = None

    def render_game_state(self, raster):
        """Render the game state to the raster."""
        # Draw cube if it exists
        if self.cube and self.cube_color:
            for x, y, z in self.cube:
                if 0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.length:
                    raster.set_pix(x, y, z, self.cube_color)

        # Draw game over border
        if self.game_over_active and self.game_over_flash_state["border_on"]:
            border_color = RGB(255, 0, 0)  # Red border
            for x in range(self.width):
                for y in range(self.height):
                    for z in range(self.length):
                        if (
                            x == 0
                            or x == self.width - 1
                            or y == 0
                            or y == self.height - 1
                            or z == 0
                            or z == self.length - 1
                        ):
                            raster.set_pix(x, y, z, border_color)

    async def update_controller_display_state(self, controller_state, player_id):
        """Update the controller display for this player."""
        # Clear the display first
        controller_state.clear()

        # Game-specific display
        player_color = PLAYER_TO_COLOR.get(player_id, RGB(255, 255, 255))
        color_str = f"R{player_color.red} G{player_color.green} B{player_color.blue}"

        controller_state.write_lcd(0, 0, "BLINKY GAME")
        controller_state.write_lcd(0, 1, f"PLAYER: {player_id.name}")
        controller_state.write_lcd(0, 2, f"COLOR: {color_str}")

        # Show cube status
        if self.cube and self.cube_color:
            time_left = max(0, self.cube_duration - (time.monotonic() - self.cube_timer))
            time_left_percent = int(time_left / self.cube_duration * 100)
            controller_state.write_lcd(0, 3, f"CUBE: {time_left_percent}% left")
        else:
            controller_state.write_lcd(0, 3, "SELECT to cube")

        # Commit the changes
        await controller_state.commit()
