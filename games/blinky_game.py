import math
import random
import time

import numpy as np

from games.util.base_game import RGB, BaseGame, PlayerID
from games.util.game_util import Button, ButtonState

# Palettes with saturated colors for good additive blending
PLAYER_PALETTES = {
    PlayerID.P1: {
        "Deep Blue": RGB(0, 0, 200),
        "Cyan": RGB(0, 200, 200),
        "Teal": RGB(0, 128, 128),
        "Azure": RGB(0, 127, 255),
        "Spring Green": RGB(0, 255, 127),
    },
    PlayerID.P2: {
        "Red": RGB(200, 0, 0),
        "Orange": RGB(255, 100, 0),
        "Yellow": RGB(200, 200, 0),
        "Amber": RGB(255, 191, 0),
        "Scarlet": RGB(255, 36, 0),
    },
    PlayerID.P3: {
        "Magenta": RGB(200, 0, 200),
        "Violet": RGB(150, 0, 255),
        "Electric Indigo": RGB(111, 0, 255),
        "Rose": RGB(255, 0, 127),
        "Hot Pink": RGB(255, 105, 180),
    },
    PlayerID.P4: {
        "Lime Green": RGB(127, 255, 0),
        "Gold": RGB(255, 215, 0),
        "Crimson": RGB(220, 20, 60),
        "Turquoise": RGB(64, 224, 208),
        "Chartreuse": RGB(127, 255, 0),
    },
}

# Shape and parameter definitions
SHAPES = ["Cube", "Sphere", "Pyramid", "Plane"]
ORIENTATIONS = {
    "Cube": ["Flat", "X-Rot", "Y-Rot", "Z-Rot"],
    "Sphere": ["Default"],
    "Pyramid": ["Up", "Down", "Left", "Right"],
    "Plane": ["Horizontal", "Vertical-X", "Vertical-Z"],
}
SPEEDS = {"Static": 0, "Slow": 5, "Mid": 10, "Fast": 15}


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
        self.shapes = []
        self.shape_duration = 2.5
        self.fade_in_duration = 0.2
        self.fade_out_duration = 0.3

        self.last_update_time = time.monotonic()

        self.player_color_indices = {p: 0 for p in PlayerID}
        self.player_shape_indices = {p: 0 for p in PlayerID}
        self.player_orientation_indices = {p: 0 for p in PlayerID}
        self.player_speed_indices = {p: 0 for p in PlayerID}

        self.reset_game()

    def reset_game(self):
        """Reset the game state."""
        self.shapes = []
        self.game_over_active = False
        self.game_over_flash_state = {"count": 0, "timer": 0, "interval": 0.2, "border_on": False}
        self.player_color_indices = {p: 0 for p in PlayerID}
        self.player_shape_indices = {p: 0 for p in PlayerID}
        self.player_orientation_indices = {p: 0 for p in PlayerID}
        self.player_speed_indices = {p: 0 for p in PlayerID}

    def get_player_score(self, player_id):
        return 0

    def get_opponent_score(self, player_id):
        return 0

    def _rotate_point(self, x, y, z, rot_x, rot_y, rot_z):
        """Rotates a point around the origin (0,0,0)."""
        # Y-axis rotation
        x, z = x * math.cos(rot_y) - z * math.sin(rot_y), x * math.sin(rot_y) + z * math.cos(rot_y)
        # X-axis rotation
        y, z = y * math.cos(rot_x) - z * math.sin(rot_x), y * math.sin(rot_x) + z * math.cos(rot_x)
        # Z-axis rotation
        x, y = x * math.cos(rot_z) - y * math.sin(rot_z), x * math.sin(rot_z) + y * math.cos(rot_z)
        return x, y, z

    def _generate_shape_points(self, shape_data):
        """Generates the final list of points for a shape based on its current state."""
        points = []
        center_x, center_y, center_z = shape_data["center"]
        size = shape_data["size"]
        shape_name = shape_data["type"]
        orientation = shape_data["orientation"]

        rot_x, rot_y, rot_z = 0, 0, 0
        if shape_name == "Cube":
            angle = math.pi / 4
            if orientation == "X-Rot":
                rot_x = angle
            elif orientation == "Y-Rot":
                rot_y = angle
            elif orientation == "Z-Rot":
                rot_z = angle

        base_points = []
        if shape_name in ["Cube", "Sphere"]:
            for x in range(-size, size + 1):
                for y in range(-size, size + 1):
                    for z in range(-size, size + 1):
                        if shape_name == "Sphere" and x**2 + y**2 + z**2 > size**2:
                            continue
                        base_points.append((x, y, z))
        elif shape_name == "Pyramid":
            for y in range(-size, size + 1):
                level_height = y + size
                level_width = int(size * (1 - (level_height / (2 * size))))
                for x in range(-level_width, level_width + 1):
                    for z in range(-level_width, level_width + 1):
                        if orientation == "Up":
                            base_points.append((x, y, z))
                        elif orientation == "Down":
                            base_points.append((x, -y, z))
                        elif orientation == "Left":
                            base_points.append((-y, x, z))
                        elif orientation == "Right":
                            base_points.append((y, x, z))
        elif shape_name == "Plane":
            plane_size = max(self.width, self.length) // 2
            for i in range(-plane_size, plane_size):
                for j in range(-plane_size, plane_size):
                    if orientation == "Horizontal":
                        base_points.append((i, 0, j))
                    elif orientation == "Vertical-X":
                        base_points.append((i, j, 0))
                    elif orientation == "Vertical-Z":
                        base_points.append((0, i, j))

        for x, y, z in base_points:
            rx, ry, rz = self._rotate_point(x, y, z, rot_x, rot_y, rot_z)
            points.append((int(center_x + rx), int(center_y + ry), int(center_z + rz)))

        return points

    def process_player_input(self, player_id, button, button_state):
        if self.game_over_active or button_state == ButtonState.RELEASED:
            return

        # UP for Color
        if button == Button.UP:
            current_index = self.player_color_indices[player_id]
            self.player_color_indices[player_id] = (current_index + 1) % len(
                PLAYER_PALETTES[player_id]
            )
            return
        # DOWN for Shape
        if button == Button.DOWN:
            current_index = self.player_shape_indices[player_id]
            self.player_shape_indices[player_id] = (current_index + 1) % len(SHAPES)
            self.player_orientation_indices[player_id] = 0
            return

        # --- MODIFICATION START ---
        # LEFT for Orientation - now affects existing shape
        if button == Button.LEFT:
            shape_name = SHAPES[self.player_shape_indices[player_id]]
            num_orientations = len(ORIENTATIONS[shape_name])
            new_index = (self.player_orientation_indices[player_id] + 1) % num_orientations
            self.player_orientation_indices[player_id] = new_index
            new_orientation_name = ORIENTATIONS[shape_name][new_index]

            # Find and update the last shape from this player
            for shape in reversed(self.shapes):
                if shape["player_id"] == player_id:
                    # Only update if the shape type matches the current selection
                    if shape["type"] == shape_name:
                        shape["orientation"] = new_orientation_name
                    break
            return

        # RIGHT for Speed - now affects existing shape
        if button == Button.RIGHT:
            new_index = (self.player_speed_indices[player_id] + 1) % len(SPEEDS)
            self.player_speed_indices[player_id] = new_index
            new_speed_name = list(SPEEDS.keys())[new_index]
            new_speed_value = SPEEDS[new_speed_name]

            # Find and update the last shape from this player
            for shape in reversed(self.shapes):
                if shape["player_id"] == player_id:
                    shape["speed"] = new_speed_value
                    break
            return
        # --- MODIFICATION END ---

        if button == Button.SELECT:
            current_time = time.monotonic()

            for shape in self.shapes:
                if shape["player_id"] == player_id:
                    time_alive = current_time - shape["creation_time"]
                    if self.shape_duration - time_alive > self.fade_out_duration:
                        shape["creation_time"] = current_time - (
                            self.shape_duration - self.fade_out_duration
                        )

            size = random.randint(3, 7)
            center_x = random.randint(size, self.width - size - 1)
            center_y = random.randint(size, self.height - size - 1)
            center_z = random.randint(size, self.length - size - 1)

            shape_idx = self.player_shape_indices[player_id]
            shape_name = SHAPES[shape_idx]

            color_idx = self.player_color_indices[player_id]
            color_name = list(PLAYER_PALETTES[player_id].keys())[color_idx]

            orientation_idx = self.player_orientation_indices[player_id]
            orientation_name = ORIENTATIONS[shape_name][orientation_idx]

            speed_idx = self.player_speed_indices[player_id]
            speed_name = list(SPEEDS.keys())[speed_idx]

            new_shape = {
                "center": [center_x, center_y, center_z],
                "size": size,
                "type": shape_name,
                "color": PLAYER_PALETTES[player_id][color_name],
                "orientation": orientation_name,
                "speed": SPEEDS[speed_name],
                "creation_time": current_time,
                "player_id": player_id,
            }
            self.shapes.append(new_shape)

            # Play a pop sound when creating a cube
            self.play_game_sound("pop")

    def update_game_state(self):
        current_time = time.monotonic()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time

        if not self.game_over_active:
            for shape in self.shapes:
                if shape["speed"] > 0:
                    # Move along Z-axis (forward)
                    shape["center"][2] += shape["speed"] * dt
                    # Wrap around logic for all axes
                    shape["center"][0] %= self.width
                    shape["center"][1] %= self.height
                    shape["center"][2] %= self.length

            self.shapes = [
                s for s in self.shapes if current_time - s["creation_time"] < self.shape_duration
            ]

    def render_game_state(self, raster):
        raster.clear()
        current_time = time.monotonic()
        blend_buffer = np.zeros((self.length, self.height, self.width, 3), dtype=np.float32)

        for shape in self.shapes:
            time_alive = current_time - shape["creation_time"]
            time_left = self.shape_duration - time_alive
            brightness = 1.0
            if time_alive < self.fade_in_duration:
                brightness = time_alive / self.fade_in_duration
            elif time_left < self.fade_out_duration:
                brightness = time_left / self.fade_out_duration
            brightness = max(0.0, min(1.0, brightness))
            if brightness <= 0.01:
                continue

            shape_points = self._generate_shape_points(shape)
            original_color = shape["color"]
            r, g, b = (
                original_color.red * brightness,
                original_color.green * brightness,
                original_color.blue * brightness,
            )

            for x, y, z in shape_points:
                if 0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.length:
                    blend_buffer[z, y, x, 0] += r
                    blend_buffer[z, y, x, 1] += g
                    blend_buffer[z, y, x, 2] += b

        final_pixels = np.clip(blend_buffer, 0, 255).astype(np.uint8)
        for z, y, x in zip(*np.nonzero(np.any(final_pixels, axis=-1))):
            r, g, b = final_pixels[z, y, x]
            raster.set_pix(x, y, z, RGB(r, g, b))

    async def update_controller_display_state(self, controller_state, player_id):
        controller_state.clear()

        color_idx = self.player_color_indices[player_id]
        color_name = list(PLAYER_PALETTES[player_id].keys())[color_idx]

        shape_idx = self.player_shape_indices[player_id]
        shape_name = SHAPES[shape_idx]

        orientation_idx = self.player_orientation_indices[player_id]
        # Ensure orientation index is valid for the current shape
        if orientation_idx >= len(ORIENTATIONS[shape_name]):
            orientation_idx = 0
            self.player_orientation_indices[player_id] = 0
        orientation_name = ORIENTATIONS[shape_name][orientation_idx]

        speed_idx = self.player_speed_indices[player_id]
        speed_name = list(SPEEDS.keys())[speed_idx]

        controller_state.write_lcd(0, 0, "BLINKY VJ")
        controller_state.write_lcd(0, 1, f"SHP:{shape_name} ORT:{orientation_name}")
        controller_state.write_lcd(0, 2, f"CLR:{color_name} SPD:{speed_name}")

        player_shape = next((s for s in reversed(self.shapes) if s["player_id"] == player_id), None)

        if player_shape:
            time_left = max(
                0, self.shape_duration - (time.monotonic() - player_shape["creation_time"])
            )
            if time_left > 0:
                percent = int((time_left / self.shape_duration) * 100)
                controller_state.write_lcd(0, 3, f"LAST SHAPE: {percent}%")
            else:
                controller_state.write_lcd(0, 3, "SELECT to spawn")
        else:
            controller_state.write_lcd(0, 3, "SELECT to spawn")

        await controller_state.commit()
