import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Set

from artnet import RGB
from games.util.base_game import BaseGame, PlayerID, TeamID
from games.util.game_util import Button, ButtonState

# Game constants
SHIP_SIZE = 1.5
SHIP_SPEED = 15.0  # Increased from 8.0 for faster lateral movement
BULLET_SPEED = 12.0
BULLET_SIZE = 0.5
BLOCK_SIZE = 5.0  # Doubled from 2.5 - now much larger rectangular prisms
BLOCK_WIDTH = 6  # Width of rectangular prism (X axis) - much wider
BLOCK_HEIGHT = 6  # Height of rectangular prism (Y axis) - much taller
BLOCK_DEPTH = 3  # Depth of rectangular prism (Z axis) - slightly deeper
BLOCK_SPEED = 0.3  # Very slow initial speed
BLOCK_SPAWN_RATE = 0.2  # Very low initial spawn rate (1 block every 5 seconds)
BLOCK_SPEED_INCREASE = 0.05  # Speed increase per second
BLOCK_SPAWN_INCREASE = 0.02  # Spawn rate increase per second
BLOCK_HP = 3
DAMAGE_MATCHING = 2  # damage when color matches
DAMAGE_NON_MATCHING = 1  # damage when color doesn't match
FLASH_DURATION = 0.2  # seconds
PARTICLE_COUNT = 15
PARTICLE_LIFETIME = 2.0


@dataclass
class Spaceship:
    """Player spaceship at the bottom of the screen."""

    player_id: PlayerID
    x: float
    y: float
    z: float
    color: RGB
    team_id: TeamID
    held_buttons: set = None

    def __post_init__(self):
        if self.held_buttons is None:
            self.held_buttons = set()


@dataclass
class Bullet:
    """Bullet fired by player spaceships."""

    x: float
    y: float
    z: float
    vz: float  # velocity in Z direction (upwards)
    color: RGB
    player_id: PlayerID
    team_id: TeamID
    birth_time: float


@dataclass
class Block:
    """Falling blocks that players need to destroy."""

    x: float
    y: float
    z: float
    vz: float  # velocity in Z direction (downwards)
    color: RGB
    team_id: TeamID
    hp: int
    max_hp: int
    last_damage_time: float = 0.0
    damage_flash_active: bool = False


@dataclass
class Particle:
    """Particle for explosion effects."""

    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    birth_time: float
    lifetime: float
    color: RGB

    GRAVITY = 15.0
    AIR_DAMPING = 0.98

    def update(self, dt: float):
        # Apply gravity (negative Z direction)
        self.vz -= self.GRAVITY * dt
        # Apply air damping
        self.vx *= self.AIR_DAMPING
        self.vy *= self.AIR_DAMPING
        self.vz *= self.AIR_DAMPING
        # Update position
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt

    def is_expired(self, current_time: float) -> bool:
        return current_time - self.birth_time > self.lifetime


class SpaceInvadersGame(BaseGame):
    DISPLAY_NAME = "Space Invaders"

    def __init__(
        self,
        width=20,
        height=20,
        length=20,
        frameRate=30,
        config=None,
        input_handler=None,
    ):
        # Game state
        self.game_phase = "lobby"  # lobby, running, gameover
        self.join_deadline = time.monotonic() + 15.0  # 15 seconds to join
        self.min_lobby_time = time.monotonic() + 5.0  # Minimum 5 seconds in lobby
        self.start_game_votes = 0  # Number of players who want to start
        self.active_players: Set[PlayerID] = set()
        self.spaceships: Dict[PlayerID, Spaceship] = {}
        self.bullets: List[Bullet] = []
        self.blocks: List[Block] = []
        self.particles: List[Particle] = []

        # Team/color mapping for players
        self.player_teams = {
            PlayerID.P1: TeamID.RED,
            PlayerID.P2: TeamID.BLUE,
            PlayerID.P3: TeamID.GREEN,
            PlayerID.P4: TeamID.YELLOW,
        }

        # Timing
        self.last_block_spawn = 0.0
        self.game_start_time = 0.0

        # Scores and damage tracking
        self.player_scores: Dict[PlayerID, int] = {pid: 0 for pid in PlayerID}
        self.damage_dealt: Dict[PlayerID, int] = {
            pid: 0 for pid in PlayerID
        }  # Track damage for scoring

        # Progressive difficulty
        self.current_block_speed = BLOCK_SPEED
        self.current_spawn_rate = BLOCK_SPAWN_RATE

        super().__init__(width, height, length, frameRate, config, input_handler)

    def reset_game(self):
        """Reset the game state."""
        self.game_phase = "lobby"
        self.join_deadline = time.monotonic() + 15.0
        self.min_lobby_time = time.monotonic() + 5.0
        self.start_game_votes = 0
        self.active_players = set()
        self.spaceships = {}
        self.bullets = []
        self.blocks = []
        self.particles = []
        self.last_block_spawn = 0.0
        self.game_start_time = 0.0
        self.player_scores = {pid: 0 for pid in PlayerID}
        self.damage_dealt = {pid: 0 for pid in PlayerID}
        self.current_block_speed = BLOCK_SPEED
        self.current_spawn_rate = BLOCK_SPAWN_RATE

    def get_player_score(self, player_id):
        """Get the score for a player."""
        return self.player_scores.get(player_id, 0)

    def get_opponent_score(self, player_id):
        """Get the highest score among other players."""
        other_scores = [score for pid, score in self.player_scores.items() if pid != player_id]
        return max(other_scores) if other_scores else 0

    def process_player_input(self, player_id, button, button_state):
        """Process input from a player."""
        if self.game_phase == "lobby":
            if button == Button.SELECT and button_state == ButtonState.PRESSED:
                if player_id not in self.active_players:
                    self._join_player(player_id)
                else:
                    # Already joined - vote to start game
                    self._vote_start_game(player_id)
            return
        elif self.game_phase == "gameover":
            if button == Button.SELECT and button_state == ButtonState.PRESSED:
                self.reset_game()
            return

        if self.game_phase != "running":
            return

        if player_id not in self.spaceships:
            return

        spaceship = self.spaceships[player_id]

        # Track held buttons
        if button_state == ButtonState.PRESSED:
            spaceship.held_buttons.add(button)
        elif button_state == ButtonState.RELEASED:
            spaceship.held_buttons.discard(button)

        # Handle shooting (only on PRESSED)
        if button == Button.SELECT and button_state == ButtonState.PRESSED:
            self._fire_bullet(player_id)

    def _join_player(self, player_id):
        """Add a player to the game."""
        if player_id in self.active_players:
            return

        self.active_players.add(player_id)

        # Create spaceship for the player
        team_id = self.player_teams[player_id]
        color = team_id.get_color()

        # Position spaceships around the bottom of the screen
        positions = [
            (self.width * 0.25, self.height * 0.25, 2),  # P1
            (self.width * 0.75, self.height * 0.25, 2),  # P2
            (self.width * 0.75, self.height * 0.75, 2),  # P3
            (self.width * 0.25, self.height * 0.75, 2),  # P4
        ]

        player_index = list(PlayerID).index(player_id)
        x, y, z = positions[player_index]

        self.spaceships[player_id] = Spaceship(
            player_id=player_id, x=x, y=y, z=z, color=color, team_id=team_id
        )

    def _vote_start_game(self, player_id):
        """Player votes to start the game early."""
        # Simple implementation - any joined player pressing SELECT again votes to start
        self.start_game_votes += 1

    def _fire_bullet(self, player_id):
        """Fire a bullet from the player's spaceship."""
        if player_id not in self.spaceships:
            return

        spaceship = self.spaceships[player_id]
        team_id = self.player_teams[player_id]

        bullet = Bullet(
            x=spaceship.x,
            y=spaceship.y,
            z=spaceship.z + 1,  # Start slightly above the spaceship
            vz=BULLET_SPEED,
            color=spaceship.color,
            player_id=player_id,
            team_id=team_id,
            birth_time=time.monotonic(),
        )

        self.bullets.append(bullet)

    def _update_ship_movement(self, dt):
        """Update ship positions based on held buttons."""
        for spaceship in self.spaceships.values():
            # Handle movement based on held buttons
            if Button.LEFT in spaceship.held_buttons:
                spaceship.x = max(2, spaceship.x - SHIP_SPEED * dt)
            if Button.RIGHT in spaceship.held_buttons:
                spaceship.x = min(self.width - 3, spaceship.x + SHIP_SPEED * dt)
            if Button.UP in spaceship.held_buttons:
                spaceship.y = min(self.height - 3, spaceship.y + SHIP_SPEED * dt)
            if Button.DOWN in spaceship.held_buttons:
                spaceship.y = max(2, spaceship.y - SHIP_SPEED * dt)

    def _blocks_overlap(self, x1, y1, z1, x2, y2, z2):
        """Check if two blocks would overlap."""
        # Check if rectangular prisms overlap
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        dz = abs(z1 - z2)

        return dx < BLOCK_WIDTH + 1 and dy < BLOCK_HEIGHT + 1 and dz < BLOCK_DEPTH + 1

    def _spawn_block(self):
        """Spawn a new block at the top of the screen."""
        # Only spawn blocks if there are active players
        if not self.active_players:
            return

        # Choose a random color from active player teams
        active_teams = [self.player_teams[pid] for pid in self.active_players]
        team_id = random.choice(active_teams)
        color = team_id.get_color()

        # Try to find a non-overlapping position (try up to 10 times)
        for attempt in range(10):
            # Random position at the top (with margin for larger blocks)
            x = random.uniform(BLOCK_WIDTH, self.width - BLOCK_WIDTH)
            y = random.uniform(BLOCK_HEIGHT, self.height - BLOCK_HEIGHT)
            z = self.length - 3  # Start higher to ensure full block is visible

            # Check for overlaps with existing blocks
            overlaps = False
            for existing_block in self.blocks:
                if self._blocks_overlap(
                    x, y, z, existing_block.x, existing_block.y, existing_block.z
                ):
                    overlaps = True
                    break

            if not overlaps:
                # Found a good position
                block = Block(
                    x=x,
                    y=y,
                    z=z,
                    vz=-self.current_block_speed,  # Use current progressive speed
                    color=color,
                    team_id=team_id,
                    hp=BLOCK_HP,
                    max_hp=BLOCK_HP,
                )

                self.blocks.append(block)
                break

    def _update_bullets(self, dt):
        """Update bullet positions and remove out-of-bounds bullets."""
        new_bullets = []

        for bullet in self.bullets:
            bullet.z += bullet.vz * dt

            # Remove bullets that have gone off the top of the screen
            if bullet.z < self.length:
                new_bullets.append(bullet)

        self.bullets = new_bullets

    def _update_blocks(self, dt):
        """Update block positions and remove blocks that have fallen off."""
        current_time = time.monotonic()
        new_blocks = []

        for block in self.blocks:
            block.z += block.vz * dt

            # Update damage flash
            if block.damage_flash_active and current_time - block.last_damage_time > FLASH_DURATION:
                block.damage_flash_active = False

            # Remove blocks that have fallen off the bottom
            if block.z > 0:  # Keep blocks that haven't fallen off completely
                new_blocks.append(block)

        self.blocks = new_blocks

    def _check_collisions(self):
        """Check for collisions between bullets and blocks."""
        current_time = time.monotonic()

        for bullet in self.bullets[:]:  # Use slice to allow modification during iteration
            for block in self.blocks[:]:
                # Rectangular collision detection for bullet vs block
                dx = abs(bullet.x - block.x)
                dy = abs(bullet.y - block.y)
                dz = abs(bullet.z - block.z)

                # Check if bullet is within the rectangular block bounds
                if (
                    dx < BLOCK_WIDTH / 2 + BULLET_SIZE
                    and dy < BLOCK_HEIGHT / 2 + BULLET_SIZE
                    and dz < BLOCK_DEPTH / 2 + BULLET_SIZE
                ):
                    # Collision detected!
                    self._handle_collision(bullet, block, current_time)

                    # Remove the bullet
                    if bullet in self.bullets:
                        self.bullets.remove(bullet)
                    break

    def _handle_collision(self, bullet, block, current_time):
        """Handle collision between a bullet and a block."""
        # Determine damage based on color matching
        damage = DAMAGE_MATCHING if bullet.team_id == block.team_id else DAMAGE_NON_MATCHING

        # Apply damage
        block.hp -= damage
        block.last_damage_time = current_time
        block.damage_flash_active = True

        # Add score to the player
        self.player_scores[bullet.player_id] += damage

        # Check if block is destroyed
        if block.hp <= 0:
            self._destroy_block(block)
            # Bonus points for destroying a block
            self.player_scores[bullet.player_id] += 10

    def _destroy_block(self, block):
        """Destroy a block and create particle explosion."""
        # Remove the block
        if block in self.blocks:
            self.blocks.remove(block)

        # Create particle explosion
        self._create_explosion(block.x, block.y, block.z, block.color)

    def _create_explosion(self, x, y, z, base_color):
        """Create a particle explosion at the given location."""
        current_time = time.monotonic()

        for _ in range(PARTICLE_COUNT):
            # Random velocity in all directions
            vx = random.uniform(-5, 5)
            vy = random.uniform(-5, 5)
            vz = random.uniform(-2, 8)  # Mostly upward

            # Vary the color slightly
            r = max(0, min(255, base_color.red + random.randint(-50, 50)))
            g = max(0, min(255, base_color.green + random.randint(-50, 50)))
            b = max(0, min(255, base_color.blue + random.randint(-50, 50)))

            particle = Particle(
                x=x + random.uniform(-0.5, 0.5),
                y=y + random.uniform(-0.5, 0.5),
                z=z + random.uniform(-0.5, 0.5),
                vx=vx,
                vy=vy,
                vz=vz,
                birth_time=current_time,
                lifetime=PARTICLE_LIFETIME,
                color=RGB(r, g, b),
            )

            self.particles.append(particle)

    def _update_particles(self, dt):
        """Update particle positions and remove expired particles."""
        current_time = time.monotonic()
        new_particles = []

        for particle in self.particles:
            if not particle.is_expired(current_time):
                particle.update(dt)

                # Only keep particles that are still in bounds (roughly)
                if (
                    particle.x > -5
                    and particle.x < self.width + 5
                    and particle.y > -5
                    and particle.y < self.height + 5
                    and particle.z > -5
                ):
                    new_particles.append(particle)

        self.particles = new_particles

    def update_game_state(self):
        """Update the game state."""
        current_time = time.monotonic()
        dt = 1.0 / self.frameRate

        if self.game_phase == "lobby":
            # Check conditions to start the game
            min_time_passed = current_time > self.min_lobby_time
            deadline_passed = current_time > self.join_deadline
            enough_votes = (
                self.start_game_votes >= len(self.active_players) and len(self.active_players) > 0
            )

            # Start game if: minimum time passed AND (deadline passed OR enough players voted to start)
            if min_time_passed and (deadline_passed or enough_votes):
                if self.active_players:
                    self.game_phase = "running"
                    self.game_start_time = current_time
                    print(f"Starting game with {len(self.active_players)} players")
                else:
                    # No players joined, restart lobby
                    self.join_deadline = current_time + 15.0
                    self.min_lobby_time = current_time + 5.0
                    self.start_game_votes = 0

        elif self.game_phase == "running":
            # Update ship movement based on held buttons
            self._update_ship_movement(dt)

            # Progressive difficulty: increase speed and spawn rate over time
            elapsed_time = current_time - self.game_start_time
            self.current_block_speed = BLOCK_SPEED + (BLOCK_SPEED_INCREASE * elapsed_time)
            self.current_spawn_rate = BLOCK_SPAWN_RATE + (BLOCK_SPAWN_INCREASE * elapsed_time)

            # Spawn blocks
            if current_time - self.last_block_spawn > (1.0 / self.current_spawn_rate):
                self._spawn_block()
                self.last_block_spawn = current_time

            # Update game objects
            self._update_bullets(dt)
            self._update_blocks(dt)
            self._update_particles(dt)

            # Check collisions
            self._check_collisions()

            # Check for game over: any block reaches the bottom
            for block in self.blocks:
                if block.z <= 0:
                    self.game_phase = "gameover"
                    break

    def render_game_state(self, raster):
        """Render the game state to the raster."""
        current_time = time.monotonic()

        if self.game_phase == "lobby":
            self._render_lobby(raster, current_time)
        elif self.game_phase == "gameover":
            self._render_game_over(raster, current_time)
        else:
            self._render_game(raster, current_time)

    def _render_lobby(self, raster, current_time):
        """Render the lobby state."""
        # Simple pattern to indicate lobby
        center_x = self.width // 2
        center_y = self.height // 2
        center_z = self.length // 2

        # Pulsing center indicator
        pulse = int(abs(math.sin(current_time * 3) * 255))
        color = RGB(pulse, pulse, pulse)

        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    x = center_x + dx
                    y = center_y + dy
                    z = center_z + dz

                    if 0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.length:
                        raster.set_pix(x, y, z, color)

    def _render_game(self, raster, current_time):
        """Render the actual game."""
        # Render spaceships
        for spaceship in self.spaceships.values():
            self._render_spaceship(raster, spaceship)

        # Render bullets
        for bullet in self.bullets:
            self._render_bullet(raster, bullet)

        # Render blocks
        for block in self.blocks:
            self._render_block(raster, block, current_time)

        # Render particles
        for particle in self.particles:
            self._render_particle(raster, particle)

    def _render_spaceship(self, raster, spaceship):
        """Render a spaceship."""
        # Render spaceship as a small 3D cross pattern
        center_x = int(spaceship.x)
        center_y = int(spaceship.y)
        center_z = int(spaceship.z)

        positions = [
            (center_x, center_y, center_z),  # Center
            (center_x - 1, center_y, center_z),  # Left
            (center_x + 1, center_y, center_z),  # Right
            (center_x, center_y - 1, center_z),  # Front
            (center_x, center_y + 1, center_z),  # Back
            (center_x, center_y, center_z + 1),  # Top
        ]

        for x, y, z in positions:
            if 0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.length:
                raster.set_pix(x, y, z, spaceship.color)

    def _render_bullet(self, raster, bullet):
        """Render a bullet."""
        x = int(round(bullet.x))
        y = int(round(bullet.y))
        z = int(round(bullet.z))

        if 0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.length:
            raster.set_pix(x, y, z, bullet.color)

    def _render_block(self, raster, block, current_time):
        """Render a block with damage effects."""
        center_x = int(block.x)
        center_y = int(block.y)
        center_z = int(block.z)

        # Determine color based on damage state
        if block.damage_flash_active:
            # Flash white when taking damage
            color = RGB(255, 255, 255)
        else:
            # Calculate damage ratio for cracking effect
            damage_ratio = 1.0 - (block.hp / block.max_hp)

            if damage_ratio < 0.33:
                # Minimal damage - original color
                color = block.color
            elif damage_ratio < 0.66:
                # Medium damage - mix with some white/gray
                mix_factor = 0.3
                color = RGB(
                    int(block.color.red * (1 - mix_factor) + 128 * mix_factor),
                    int(block.color.green * (1 - mix_factor) + 128 * mix_factor),
                    int(block.color.blue * (1 - mix_factor) + 128 * mix_factor),
                )
            else:
                # Heavy damage - add random glitching
                if random.random() < 0.3:  # 30% chance of glitch pixel
                    color = RGB(
                        random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
                    )
                else:
                    color = block.color

        # Render block as a rectangular prism
        half_width = BLOCK_WIDTH // 2
        half_height = BLOCK_HEIGHT // 2
        half_depth = BLOCK_DEPTH // 2

        for dx in range(-half_width, half_width + 1):
            for dy in range(-half_height, half_height + 1):
                for dz in range(-half_depth, half_depth + 1):
                    x = center_x + dx
                    y = center_y + dy
                    z = center_z + dz

                    if 0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.length:
                        raster.set_pix(x, y, z, color)

    def _render_game_over(self, raster, current_time):
        """Render the game over screen with red border and final scores."""
        # Flash red border
        flash_intensity = int(abs(math.sin(current_time * 5) * 255))
        border_color = RGB(flash_intensity, 0, 0)

        # Draw red border around the entire display
        for i in range(self.width):
            for j in range(self.height):
                # Top and bottom faces
                if 0 <= i < self.width and 0 <= j < self.height:
                    raster.set_pix(i, j, 0, border_color)  # Bottom face
                    raster.set_pix(i, j, self.length - 1, border_color)  # Top face

        for i in range(self.width):
            for k in range(self.length):
                # Front and back faces
                if 0 <= i < self.width and 0 <= k < self.length:
                    raster.set_pix(i, 0, k, border_color)  # Front face
                    raster.set_pix(i, self.height - 1, k, border_color)  # Back face

        for j in range(self.height):
            for k in range(self.length):
                # Left and right faces
                if 0 <= j < self.height and 0 <= k < self.length:
                    raster.set_pix(0, j, k, border_color)  # Left face
                    raster.set_pix(self.width - 1, j, k, border_color)  # Right face

        # Show final scores in the center
        center_x = self.width // 2
        center_y = self.height // 2
        center_z = self.length // 2

        # Sort players by score
        sorted_players = sorted(self.player_scores.items(), key=lambda x: x[1], reverse=True)

        # Display top 3 scores
        for i, (player_id, score) in enumerate(sorted_players[:3]):
            if i == 0:
                color = RGB(255, 215, 0)  # Gold for 1st place
            elif i == 1:
                color = RGB(192, 192, 192)  # Silver for 2nd place
            else:
                color = RGB(205, 127, 50)  # Bronze for 3rd place

            # Display player and score
            y_offset = center_y - 2 + i
            if 0 <= y_offset < self.height:
                # Simple text representation (P1: 150, P2: 120, etc.)
                text = f"P{player_id.value}: {score}"
                # For now, just show a colored block representing the score
                for dx in range(min(len(text), 8)):  # Limit to 8 characters
                    x = center_x - 4 + dx
                    if 0 <= x < self.width:
                        raster.set_pix(x, y_offset, center_z, color)

    def _render_particle(self, raster, particle):
        """Render a particle."""
        x = int(round(particle.x))
        y = int(round(particle.y))
        z = int(round(particle.z))

        if 0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.length:
            raster.set_pix(x, y, z, particle.color)

    async def update_controller_display_state(self, controller_state, player_id):
        """Update the controller's LCD display for this player."""
        controller_state.clear()

        if self.game_phase == "lobby":
            controller_state.write_lcd(0, 0, "BREAKOUT")
            if player_id in self.active_players:
                controller_state.write_lcd(0, 1, f"Joined! ({len(self.active_players)} players)")
                controller_state.write_lcd(0, 2, "SELECT again to vote start")
                votes_needed = len(self.active_players) - self.start_game_votes
                controller_state.write_lcd(0, 3, f"Need {votes_needed} more votes")
            else:
                controller_state.write_lcd(0, 1, "Press SELECT to join")
                controller_state.write_lcd(0, 2, f"{len(self.active_players)} players joined")
                current_time = time.monotonic()
                time_left = max(0, int(self.join_deadline - current_time))
                controller_state.write_lcd(0, 3, f"Time left: {time_left}s")
        elif self.game_phase == "gameover":
            controller_state.write_lcd(0, 0, "GAME OVER")
            score = self.get_player_score(player_id)
            controller_state.write_lcd(0, 1, f"Final Score: {score}")

            # Show player ranking
            sorted_players = sorted(self.player_scores.items(), key=lambda x: x[1], reverse=True)
            player_rank = next(
                (i + 1 for i, (pid, _) in enumerate(sorted_players) if pid == player_id), 0
            )
            controller_state.write_lcd(0, 2, f"Rank: {player_rank}/{len(sorted_players)}")
            controller_state.write_lcd(0, 3, "Press SELECT to restart")
        else:
            controller_state.write_lcd(0, 0, "BREAKOUT")
            score = self.get_player_score(player_id)
            controller_state.write_lcd(0, 1, f"Score: {score}")
            if player_id in self.spaceships:
                controller_state.write_lcd(0, 2, "Arrows: move")
                controller_state.write_lcd(0, 3, "SELECT: shoot")
            else:
                controller_state.write_lcd(0, 2, "Spectating")

        await controller_state.commit()
