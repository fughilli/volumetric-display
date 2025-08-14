import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

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

# New game constants for enhanced gameplay
MAX_HEALTH = 100
HEALTH_DAMAGE_PER_INVADER = 15  # Health lost when invader reaches ground
POWERUP_SPAWN_RATE = 0.05  # Very low spawn rate (1 powerup every 20 seconds)
POWERUP_LIFETIME = 10.0  # How long powerups last
POWERUP_ROTATION_SPEED = 3.0  # Rotation speed for powerup animation
HEALTH_POWERUP_AMOUNT = 25  # Health restored by health powerup
POWER_SHOT_MULTIPLIER = 3  # Damage multiplier for power shot
EXPLOSIVE_RADIUS = 2.0  # Radius of explosive damage

# Enemy constants
DRONE_SPAWN_RATE = 0.03  # Very low spawn rate
WARRIOR_SPAWN_RATE = 0.02  # Even lower spawn rate
ELITE_SPAWN_RATE = 0.01  # Very rare spawn rate
DRONE_HP = 5
WARRIOR_HP = 8
ELITE_HP = 12
DRONE_BULLET_SPEED = 8.0
WARRIOR_BULLET_SPEED = 10.0
ELITE_BULLET_SPEED = 12.0
ENEMY_BULLET_DAMAGE = 10  # Damage to player health

# Boss constants
BOSS_SPAWN_TIME = 60.0  # Spawn boss after 60 seconds
BOSS_HP = 50
BOSS_BULLET_SPEED = 15.0
BOSS_BULLET_DAMAGE = 20
BOSS_SPAWN_RATE = 0.1  # Spawn rate during boss fight


# Game phases
class GamePhase(Enum):
    LOBBY = "lobby"
    RUNNING = "running"
    BOSS_INTRO = "boss_intro"  # New phase for boss intro animation
    BOSS_FIGHT = "boss_fight"
    GAME_OVER = "gameover"
    VICTORY = "victory"


# Enemy types
class EnemyType(Enum):
    BLOCK = "block"
    DRONE = "drone"
    WARRIOR = "warrior"
    ELITE = "elite"


# Power-up types
class PowerUpType(Enum):
    HEALTH = "health"
    POWER_SHOT = "power_shot"
    EXPLOSIVE_SHOT = "explosive_shot"


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
    vx: float = 0.0  # Current velocity in X direction
    vy: float = 0.0  # Current velocity in Y direction
    target_vx: float = 0.0  # Target velocity in X direction
    target_vy: float = 0.0  # Target velocity in Y direction
    tilt_x: float = 0.0  # Tilt angle in X direction (for visual effect)
    tilt_y: float = 0.0  # Tilt angle in Y direction (for visual effect)

    def __post_init__(self):
        if self.held_buttons is None:
            self.held_buttons = set()

    def update_movement(self, dt: float, target_speed: float, max_speed: float):
        """Update ship movement with easing and tilting."""
        # Calculate target velocities based on held buttons
        self.target_vx = 0.0
        self.target_vy = 0.0

        if Button.LEFT in self.held_buttons:
            self.target_vx = -target_speed
        if Button.RIGHT in self.held_buttons:
            self.target_vx = target_speed
        if Button.UP in self.held_buttons:
            self.target_vy = target_speed
        if Button.DOWN in self.held_buttons:
            self.target_vy = -target_speed

        # Apply easing to velocities (smooth acceleration/deceleration)
        acceleration = 8.0  # How quickly velocity changes
        self.vx += (self.target_vx - self.vx) * acceleration * dt
        self.vy += (self.target_vy - self.vy) * acceleration * dt

        # Clamp velocities to max speed
        speed = math.sqrt(self.vx * self.vx + self.vy * self.vy)
        if speed > max_speed:
            self.vx = (self.vx / speed) * max_speed
            self.vy = (self.vy / speed) * max_speed

        # Update position
        self.x += self.vx * dt
        self.y += self.vy * dt

        # Update tilt angles based on velocity (for visual effect)
        tilt_sensitivity = 0.1  # Reduced from 0.3 for more subtle tilt
        self.tilt_x = -self.vy * tilt_sensitivity  # Tilt forward when moving up
        self.tilt_y = self.vx * tilt_sensitivity  # Tilt right when moving right


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
    damage_multiplier: int = 1
    explosive: bool = False


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


@dataclass
class PowerUp:
    """Power-up that players can collect."""

    x: float
    y: float
    z: float
    vz: float  # velocity in Z direction (downwards)
    powerup_type: PowerUpType
    birth_time: float
    rotation_angle: float = 0.0

    def update(self, dt: float):
        self.z += self.vz * dt
        self.rotation_angle += POWERUP_ROTATION_SPEED * dt

    def get_color(self) -> RGB:
        if self.powerup_type == PowerUpType.HEALTH:
            return RGB(255, 0, 0)  # Red
        elif self.powerup_type == PowerUpType.POWER_SHOT:
            return RGB(255, 255, 0)  # Yellow
        else:  # EXPLOSIVE_SHOT
            return RGB(255, 0, 255)  # Magenta


@dataclass
class EnemyBullet:
    """Bullet fired by enemies."""

    x: float
    y: float
    z: float
    vx: float = 0.0  # velocity in X direction
    vy: float = 0.0  # velocity in Y direction
    vz: float = 0.0  # velocity in Z direction
    color: RGB = field(default_factory=lambda: RGB(255, 255, 255))
    damage: int = 10
    birth_time: float = 0.0


@dataclass
class Enemy:
    """Enhanced enemy with movement and attack patterns."""

    x: float
    y: float
    z: float
    vz: float  # velocity in Z direction (downwards)
    vx: float  # lateral velocity
    vy: float  # lateral velocity
    enemy_type: EnemyType
    color: RGB
    team_id: TeamID
    hp: int
    max_hp: int
    last_damage_time: float = 0.0
    damage_flash_active: bool = False
    last_shot_time: float = 0.0
    shot_cooldown: float = 0.0
    movement_phase: float = 0.0  # For complex movement patterns
    target_x: Optional[float] = None  # For elite targeting
    target_y: Optional[float] = None

    def update(self, dt: float, current_time: float, players: Dict[PlayerID, "Spaceship"]):
        # Update position based on enemy type
        if self.enemy_type == EnemyType.BLOCK:
            # Simple falling movement
            self.z += self.vz * dt
        elif self.enemy_type == EnemyType.DRONE:
            # Lissajous figure movement
            self.movement_phase += dt * 2.0
            self.x += math.sin(self.movement_phase) * dt * 3.0
            self.y += math.cos(self.movement_phase * 0.7) * dt * 2.0
            self.z += self.vz * dt
        elif self.enemy_type == EnemyType.WARRIOR:
            # Simple lateral movement with shooting
            self.movement_phase += dt * 1.5
            self.x += math.sin(self.movement_phase) * dt * 2.0
            self.z += self.vz * dt
        elif self.enemy_type == EnemyType.ELITE:
            # Deliberate movement toward players
            if players:
                # Find closest player
                closest_player = min(
                    players.values(), key=lambda p: abs(p.x - self.x) + abs(p.y - self.y)
                )

                # Move toward player
                dx = closest_player.x - self.x
                dy = closest_player.y - self.y
                dist = math.sqrt(dx * dx + dy * dy)

                if dist > 0:
                    self.x += (dx / dist) * dt * 2.0
                    self.y += (dy / dist) * dt * 2.0

                self.z += self.vz * dt

        # Update damage flash
        if self.damage_flash_active and current_time - self.last_damage_time > FLASH_DURATION:
            self.damage_flash_active = False

    def can_shoot(self, current_time: float) -> bool:
        if self.enemy_type in [EnemyType.WARRIOR, EnemyType.ELITE]:
            return current_time - self.last_shot_time > self.shot_cooldown
        return False

    def shoot(self, current_time: float) -> Optional[EnemyBullet]:
        if not self.can_shoot(current_time):
            return None

        self.last_shot_time = current_time

        if self.enemy_type == EnemyType.WARRIOR:
            bullet_speed = WARRIOR_BULLET_SPEED
            damage = ENEMY_BULLET_DAMAGE
        else:  # ELITE
            bullet_speed = ELITE_BULLET_SPEED
            damage = ENEMY_BULLET_DAMAGE * 2

        return EnemyBullet(
            x=self.x,
            y=self.y,
            z=self.z - 1,
            vz=bullet_speed,
            color=self.color,
            damage=damage,
            birth_time=current_time,
        )


@dataclass
class Boss:
    """Boss enemy with special abilities."""

    x: float
    y: float
    z: float
    vx: float  # lateral velocity
    vy: float  # lateral velocity
    vz: float  # vertical velocity (can go up and down)
    boss_type: str
    color: RGB
    team_id: TeamID
    hp: int
    max_hp: int
    weapon_type: str  # "sniper", "spray", "burst", "laser"
    last_damage_time: float = 0.0
    damage_flash_active: bool = False
    last_shot_time: float = 0.0
    shot_cooldown: float = 1.0
    movement_phase: float = 0.0
    special_attack_cooldown: float = 5.0
    last_special_attack: float = 0.0
    animation_phase: float = 0.0
    target_x: Optional[float] = None
    target_y: Optional[float] = None
    target_z: Optional[float] = None

    def update(
        self,
        dt: float,
        current_time: float,
        players: Dict[PlayerID, "Spaceship"],
        game_width: int,
        game_height: int,
        game_length: int,
    ):
        # Find closest player for targeting
        if players:
            closest_player = min(
                players.values(),
                key=lambda p: abs(p.x - self.x) + abs(p.y - self.y) + abs(p.z - self.z),
            )
            self.target_x = closest_player.x
            self.target_y = closest_player.y
            self.target_z = closest_player.z

        # Boss movement pattern - free movement in 3D space
        self.movement_phase += dt * 0.3

        # Different movement patterns based on boss type
        if self.boss_type == "DESTROYER":
            # Aggressive movement - moves toward players
            if self.target_x is not None:
                dx = self.target_x - self.x
                dy = self.target_y - self.y
                dz = self.target_z - self.z
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                if dist > 0:
                    self.x += (dx / dist) * dt * 2.0
                    self.y += (dy / dist) * dt * 2.0
                    self.z += (dz / dist) * dt * 1.5
        elif self.boss_type == "ANNIHILATOR":
            # Evasive movement - keeps distance while circling
            if self.target_x is not None:
                dx = self.x - self.target_x
                dy = self.y - self.target_y
                dz = self.z - self.target_z
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                if dist < 8:  # Too close, back away
                    self.x += (dx / dist) * dt * 1.5
                    self.y += (dy / dist) * dt * 1.5
                    self.z += (dz / dist) * dt * 1.0
                else:  # Circle around
                    self.x += math.sin(self.movement_phase) * dt * 2.0
                    self.y += math.cos(self.movement_phase * 0.7) * dt * 2.0
        else:  # APOCALYPSE
            # Erratic movement - unpredictable patterns
            self.x += math.sin(self.movement_phase * 1.5) * dt * 3.0
            self.y += math.cos(self.movement_phase * 0.8) * dt * 2.5
            self.z += math.sin(self.movement_phase * 0.6) * dt * 2.0

        # Keep boss within bounds
        self.x = max(3, min(game_width - 4, self.x))
        self.y = max(3, min(game_height - 4, self.y))
        self.z = max(2, min(game_length - 6, self.z))

        # Animation phase for visual effects
        self.animation_phase += dt * 4.0

        # Update damage flash
        if self.damage_flash_active and current_time - self.last_damage_time > FLASH_DURATION:
            self.damage_flash_active = False

    def can_shoot(self, current_time: float) -> bool:
        return current_time - self.last_shot_time > self.shot_cooldown

    def shoot(self, current_time: float) -> List[EnemyBullet]:
        if not self.can_shoot(current_time):
            return []

        self.last_shot_time = current_time
        bullets = []

        if self.weapon_type == "sniper":
            # Sniper weapon - shoots directly at closest player
            if self.target_x is not None:
                dx = self.target_x - self.x
                dy = self.target_y - self.y
                dz = self.target_z - self.z
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                if dist > 0:
                    vx = (dx / dist) * BOSS_BULLET_SPEED
                    vy = (dy / dist) * BOSS_BULLET_SPEED
                    vz = (dz / dist) * BOSS_BULLET_SPEED
                    bullets.append(
                        EnemyBullet(
                            x=self.x,
                            y=self.y,
                            z=self.z,
                            vx=vx,
                            vy=vy,
                            vz=vz,
                            color=self.color,
                            damage=BOSS_BULLET_DAMAGE * 2,
                            birth_time=current_time,
                        )
                    )

        elif self.weapon_type == "spray":
            # Spray weapon - bullet hell pattern
            for i in range(8):
                angle = i * math.pi / 4
                bullets.append(
                    EnemyBullet(
                        x=self.x + math.cos(angle) * 2,
                        y=self.y + math.sin(angle) * 2,
                        z=self.z,
                        vz=BOSS_BULLET_SPEED * 0.8,
                        color=RGB(255, 100, 100),  # Red bullets for spray
                        damage=BOSS_BULLET_DAMAGE,
                        birth_time=current_time,
                    )
                )

        elif self.weapon_type == "burst":
            # Burst weapon - rapid fire in direction of player
            if self.target_x is not None:
                dx = self.target_x - self.x
                dy = self.target_y - self.y
                dz = self.target_z - self.z
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                if dist > 0:
                    for i in range(3):  # Burst of 3 bullets
                        spread = (i - 1) * 0.2
                        vx = (dx / dist + spread) * BOSS_BULLET_SPEED
                        vy = (dy / dist + spread) * BOSS_BULLET_SPEED
                        vz = (dz / dist) * BOSS_BULLET_SPEED
                        bullets.append(
                            EnemyBullet(
                                x=self.x,
                                y=self.y,
                                z=self.z,
                                vx=vx,
                                vy=vy,
                                vz=vz,
                                color=RGB(255, 255, 100),  # Yellow bullets for burst
                                damage=BOSS_BULLET_DAMAGE,
                                birth_time=current_time,
                            )
                        )

        else:  # laser
            # Laser weapon - straight line of bullets
            if self.target_x is not None:
                dx = self.target_x - self.x
                dy = self.target_y - self.y
                dz = self.target_z - self.z
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                if dist > 0:
                    for i in range(5):  # Line of 5 bullets
                        offset = (i - 2) * 0.5
                        vx = (dx / dist) * BOSS_BULLET_SPEED * 1.2
                        vy = (dy / dist) * BOSS_BULLET_SPEED * 1.2
                        vz = (dz / dist) * BOSS_BULLET_SPEED * 1.2
                        bullets.append(
                            EnemyBullet(
                                x=self.x + offset,
                                y=self.y + offset,
                                z=self.z,
                                vx=vx,
                                vy=vy,
                                vz=vz,
                                color=RGB(100, 255, 255),  # Cyan bullets for laser
                                damage=BOSS_BULLET_DAMAGE,
                                birth_time=current_time,
                            )
                        )

        return bullets

    def can_special_attack(self, current_time: float) -> bool:
        return current_time - self.last_special_attack > self.special_attack_cooldown

    def special_attack(self, current_time: float) -> List[EnemyBullet]:
        if not self.can_special_attack(current_time):
            return []

        self.last_special_attack = current_time
        bullets = []

        # Circular pattern attack
        for i in range(8):
            angle = i * math.pi / 4
            bullets.append(
                EnemyBullet(
                    x=self.x + math.cos(angle) * 3,
                    y=self.y + math.sin(angle) * 3,
                    z=self.z - 1,
                    vz=BOSS_BULLET_SPEED * 0.7,
                    color=RGB(255, 0, 255),  # Special color for special attacks
                    damage=BOSS_BULLET_DAMAGE * 2,
                    birth_time=current_time,
                )
            )

        return bullets


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
        self.game_phase = GamePhase.LOBBY
        self.join_deadline = time.monotonic() + 15.0  # 15 seconds to join
        self.min_lobby_time = time.monotonic() + 5.0  # Minimum 5 seconds in lobby
        self.start_game_votes = 0  # Number of players who want to start
        self.active_players: Set[PlayerID] = set()
        self.spaceships: Dict[PlayerID, Spaceship] = {}
        self.bullets: List[Bullet] = []
        self.enemies: List[Enemy] = []
        self.enemy_bullets: List[EnemyBullet] = []
        self.powerups: List[PowerUp] = []
        self.particles: List[Particle] = []
        self.boss: Optional[Boss] = None

        # Team/color mapping for players
        self.player_teams = {
            PlayerID.P1: TeamID.RED,
            PlayerID.P2: TeamID.BLUE,
            PlayerID.P3: TeamID.GREEN,
            PlayerID.P4: TeamID.YELLOW,
        }

        # Timing
        self.last_enemy_spawn = 0.0
        self.last_powerup_spawn = 0.0
        self.game_start_time = 0.0
        self.boss_spawn_time = 0.0

        # Health and power-up system
        self.global_health = MAX_HEALTH
        self.player_powerups: Dict[PlayerID, Dict[PowerUpType, float]] = {
            pid: {} for pid in PlayerID
        }  # powerup_type -> expiration_time

        # Scores and damage tracking
        self.player_scores: Dict[PlayerID, int] = {pid: 0 for pid in PlayerID}
        self.damage_dealt: Dict[PlayerID, int] = {
            pid: 0 for pid in PlayerID
        }  # Track damage for scoring

        # Progressive difficulty
        self.current_enemy_speed = BLOCK_SPEED
        self.current_spawn_rate = BLOCK_SPAWN_RATE
        self.enemies_defeated = 0
        self.bosses_defeated = 0

        # Boss intro animation state
        self.boss_intro_start_time = 0.0
        self.boss_intro_duration = 3.0  # 3 seconds for intro animation
        self.boss_intro_phase = 0  # 0=white flash, 1=enemy fade, 2=warp, 3=spawn
        self.boss_intro_boss_type = ""
        self.boss_intro_weapon_type = ""

        super().__init__(width, height, length, frameRate, config, input_handler)

    def reset_game(self):
        """Reset the game state."""
        self.game_phase = GamePhase.LOBBY
        self.join_deadline = time.monotonic() + 15.0
        self.min_lobby_time = time.monotonic() + 5.0
        self.start_game_votes = 0
        self.active_players = set()
        self.spaceships = {}
        self.bullets = []
        self.enemies = []
        self.enemy_bullets = []
        self.powerups = []
        self.particles = []
        self.boss = None
        self.last_enemy_spawn = 0.0
        self.last_powerup_spawn = 0.0
        self.game_start_time = 0.0
        self.boss_spawn_time = 0.0
        self.global_health = MAX_HEALTH
        self.player_powerups = {pid: {} for pid in PlayerID}
        self.player_scores = {pid: 0 for pid in PlayerID}
        self.damage_dealt = {pid: 0 for pid in PlayerID}
        self.current_enemy_speed = BLOCK_SPEED
        self.current_spawn_rate = BLOCK_SPAWN_RATE
        self.enemies_defeated = 0
        self.bosses_defeated = 0
        self.boss_intro_start_time = 0.0
        self.boss_intro_phase = 0
        self.boss_intro_boss_type = ""
        self.boss_intro_weapon_type = ""

    def get_player_score(self, player_id):
        """Get the score for a player."""
        return self.player_scores.get(player_id, 0)

    def get_opponent_score(self, player_id):
        """Get the highest score among other players."""
        other_scores = [score for pid, score in self.player_scores.items() if pid != player_id]
        return max(other_scores) if other_scores else 0

    def process_player_input(self, player_id, button, button_state):
        """Process input from a player."""
        if self.game_phase == GamePhase.LOBBY:
            if button == Button.SELECT and button_state == ButtonState.PRESSED:
                if player_id not in self.active_players:
                    self._join_player(player_id)
                else:
                    # Already joined - vote to start game
                    self._vote_start_game(player_id)
            return
        elif self.game_phase in [GamePhase.GAME_OVER, GamePhase.VICTORY]:
            if button == Button.SELECT and button_state == ButtonState.PRESSED:
                self.reset_game()
            return

        if self.game_phase not in [GamePhase.RUNNING, GamePhase.BOSS_FIGHT, GamePhase.BOSS_INTRO]:
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
        current_time = time.monotonic()

        # Check for power-ups
        power_shot_active = current_time < self.player_powerups[player_id].get(
            PowerUpType.POWER_SHOT, 0
        )
        explosive_active = current_time < self.player_powerups[player_id].get(
            PowerUpType.EXPLOSIVE_SHOT, 0
        )

        # Determine damage multiplier
        damage_multiplier = POWER_SHOT_MULTIPLIER if power_shot_active else 1

        bullet = Bullet(
            x=spaceship.x,
            y=spaceship.y,
            z=spaceship.z + 1,  # Start slightly above the spaceship
            vz=BULLET_SPEED,
            color=spaceship.color,
            player_id=player_id,
            team_id=team_id,
            birth_time=current_time,
        )

        # Add power-up properties to bullet
        bullet.damage_multiplier = damage_multiplier
        bullet.explosive = explosive_active

        self.bullets.append(bullet)

    def _update_ship_movement(self, dt):
        """Update ship positions based on held buttons with smooth movement."""
        for spaceship in self.spaceships.values():
            # Update movement with easing and tilting
            spaceship.update_movement(dt, SHIP_SPEED, SHIP_SPEED * 1.2)

            # Clamp position to bounds
            spaceship.x = max(2, min(self.width - 3, spaceship.x))
            spaceship.y = max(2, min(self.height - 3, spaceship.y))

    def _blocks_overlap(self, x1, y1, z1, x2, y2, z2):
        """Check if two blocks would overlap."""
        # Check if rectangular prisms overlap
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        dz = abs(z1 - z2)

        return dx < BLOCK_WIDTH + 1 and dy < BLOCK_HEIGHT + 1 and dz < BLOCK_DEPTH + 1

    def _spawn_enemy(self):
        """Spawn a new enemy at the top of the screen."""
        # Only spawn enemies if there are active players
        if not self.active_players:
            return

        # Choose enemy type based on spawn rates
        rand = random.random()
        if rand < ELITE_SPAWN_RATE:
            enemy_type = EnemyType.ELITE
            hp = ELITE_HP
            shot_cooldown = 3.0
        elif rand < ELITE_SPAWN_RATE + WARRIOR_SPAWN_RATE:
            enemy_type = EnemyType.WARRIOR
            hp = WARRIOR_HP
            shot_cooldown = 2.0
        elif rand < ELITE_SPAWN_RATE + WARRIOR_SPAWN_RATE + DRONE_SPAWN_RATE:
            enemy_type = EnemyType.DRONE
            hp = DRONE_HP
            shot_cooldown = 0.0
        else:
            enemy_type = EnemyType.BLOCK
            hp = BLOCK_HP
            shot_cooldown = 0.0

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

            # Check for overlaps with existing enemies
            overlaps = False
            for existing_enemy in self.enemies:
                if self._enemies_overlap(
                    x, y, z, existing_enemy.x, existing_enemy.y, existing_enemy.z
                ):
                    overlaps = True
                    break

            if not overlaps:
                # Found a good position
                enemy = Enemy(
                    x=x,
                    y=y,
                    z=z,
                    vz=-self.current_enemy_speed,
                    vx=0.0,
                    vy=0.0,
                    enemy_type=enemy_type,
                    color=color,
                    team_id=team_id,
                    hp=hp,
                    max_hp=hp,
                    shot_cooldown=shot_cooldown,
                )

                self.enemies.append(enemy)
                break

    def _spawn_powerup(self):
        """Spawn a new power-up at the top of the screen."""
        if not self.active_players:
            return

        # Choose power-up type
        powerup_type = random.choice(list(PowerUpType))

        # Random position at the top
        x = random.uniform(2, self.width - 3)
        y = random.uniform(2, self.height - 3)
        z = self.length - 2

        powerup = PowerUp(
            x=x,
            y=y,
            z=z,
            vz=-1.0,  # Slow fall
            powerup_type=powerup_type,
            birth_time=time.monotonic(),
        )

        self.powerups.append(powerup)

    def _spawn_boss(self):
        """Spawn a boss enemy."""
        if self.boss is not None:
            return

        # Choose boss type and weapon type
        boss_types = ["DESTROYER", "ANNIHILATOR", "APOCALYPSE"]
        weapon_types = ["sniper", "spray", "burst", "laser"]
        boss_type = random.choice(boss_types)
        weapon_type = random.choice(weapon_types)

        # Choose a random team for the boss
        active_teams = [self.player_teams[pid] for pid in self.active_players]
        team_id = random.choice(active_teams)

        # Rainbow color for boss
        color = RGB(255, 0, 255)  # Magenta base

        # Position boss in the middle area of the game space
        x = self.width / 2
        y = self.height / 2
        z = self.length / 2  # Start in the middle, not at the top

        self.boss = Boss(
            x=x,
            y=y,
            z=z,
            vx=0.0,
            vy=0.0,
            vz=0.0,  # No initial downward movement
            boss_type=boss_type,
            color=color,
            team_id=team_id,
            weapon_type=weapon_type,
            hp=BOSS_HP,
            max_hp=BOSS_HP,
        )

        print(f"BOSS SPAWNED: {boss_type} with {weapon_type} weapon")

    def _start_boss_intro(self):
        """Start the boss intro sequence."""
        # Choose boss type and weapon type
        boss_types = ["DESTROYER", "ANNIHILATOR", "APOCALYPSE"]
        weapon_types = ["sniper", "spray", "burst", "laser"]
        boss_type = random.choice(boss_types)
        weapon_type = random.choice(weapon_types)

        # Store boss info for intro
        self.boss_intro_boss_type = boss_type
        self.boss_intro_weapon_type = weapon_type
        self.boss_intro_start_time = time.monotonic()
        self.boss_intro_phase = 0

        # Clear all enemies and enemy bullets
        self.enemies.clear()
        self.enemy_bullets.clear()

        # Switch to intro phase
        self.game_phase = GamePhase.BOSS_INTRO
        print(f"BOSS INTRO STARTING: {boss_type} with {weapon_type} weapon")

    def _update_boss_intro(self, current_time, dt):
        """Update boss intro animation."""
        elapsed = current_time - self.boss_intro_start_time
        phase_duration = self.boss_intro_duration / 4  # 4 phases, 0.75 seconds each

        if elapsed < phase_duration:
            # Phase 0: White flash border
            self.boss_intro_phase = 0
        elif elapsed < phase_duration * 2:
            # Phase 1: Enemy fade out (already done)
            self.boss_intro_phase = 1
        elif elapsed < phase_duration * 3:
            # Phase 2: Space warp animation
            self.boss_intro_phase = 2
        elif elapsed < phase_duration * 4:
            # Phase 3: Boss spawn
            self.boss_intro_phase = 3
            self._spawn_boss_with_type(self.boss_intro_boss_type, self.boss_intro_weapon_type)
            self.game_phase = GamePhase.BOSS_FIGHT
            print(f"BOSS BATTLE STARTED: {self.boss_intro_boss_type}")

    def _spawn_boss_with_type(self, boss_type, weapon_type):
        """Spawn a boss with specific type and weapon."""
        # Choose a random team for the boss
        active_teams = [self.player_teams[pid] for pid in self.active_players]
        team_id = random.choice(active_teams)

        # Rainbow color for boss
        color = RGB(255, 0, 255)  # Magenta base

        # Position boss in the middle area of the game space
        x = self.width / 2
        y = self.height / 2
        z = self.length / 2  # Start in the middle, not at the top

        self.boss = Boss(
            x=x,
            y=y,
            z=z,
            vx=0.0,
            vy=0.0,
            vz=0.0,  # No initial downward movement
            boss_type=boss_type,
            color=color,
            team_id=team_id,
            weapon_type=weapon_type,
            hp=BOSS_HP,
            max_hp=BOSS_HP,
        )

    def _enemies_overlap(self, x1, y1, z1, x2, y2, z2):
        """Check if two enemies would overlap."""
        # Check if rectangular prisms overlap
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        dz = abs(z1 - z2)

        return dx < BLOCK_WIDTH + 1 and dy < BLOCK_HEIGHT + 1 and dz < BLOCK_DEPTH + 1

    def _update_bullets(self, dt):
        """Update bullet positions and remove out-of-bounds bullets."""
        new_bullets = []

        for bullet in self.bullets:
            bullet.z += bullet.vz * dt

            # Remove bullets that have gone off the top of the screen
            if bullet.z < self.length:
                new_bullets.append(bullet)

        self.bullets = new_bullets

    def _update_enemies(self, dt):
        """Update enemy positions and remove enemies that have fallen off."""
        current_time = time.monotonic()
        new_enemies = []

        for enemy in self.enemies:
            enemy.update(dt, current_time, self.spaceships)

            # Check if enemy can shoot
            if enemy.can_shoot(current_time):
                bullet = enemy.shoot(current_time)
                if bullet:
                    self.enemy_bullets.append(bullet)

            # Remove enemies that have fallen off the bottom
            if enemy.z > 0:  # Keep enemies that haven't fallen off completely
                new_enemies.append(enemy)
            else:
                # Enemy reached the ground - damage health
                self.global_health = max(0, self.global_health - HEALTH_DAMAGE_PER_INVADER)

        self.enemies = new_enemies

    def _update_enemy_bullets(self, dt):
        """Update enemy bullet positions and remove out-of-bounds bullets."""
        new_bullets = []

        for bullet in self.enemy_bullets:
            bullet.x += bullet.vx * dt
            bullet.y += bullet.vy * dt
            bullet.z += bullet.vz * dt

            # Remove bullets that have gone off the bottom of the screen
            if bullet.z > 0:
                new_bullets.append(bullet)

        self.enemy_bullets = new_bullets

    def _update_powerups(self, dt):
        """Update power-up positions and remove out-of-bounds power-ups."""
        new_powerups = []

        for powerup in self.powerups:
            powerup.update(dt)

            # Remove powerups that have fallen off the bottom
            if powerup.z > 0:
                new_powerups.append(powerup)

        self.powerups = new_powerups

    def _update_boss(self, dt):
        """Update boss position and behavior."""
        if self.boss is None:
            return

        current_time = time.monotonic()
        self.boss.update(dt, current_time, self.spaceships, self.width, self.height, self.length)

        # Check if boss can shoot
        if self.boss.can_shoot(current_time):
            bullets = self.boss.shoot(current_time)
            self.enemy_bullets.extend(bullets)

        # Check if boss can do special attack
        if self.boss.can_special_attack(current_time):
            bullets = self.boss.special_attack(current_time)
            self.enemy_bullets.extend(bullets)

        # Bosses don't fall off the bottom - they move freely in the game area
        # Only remove boss if health reaches 0 (handled in collision detection)

    def _check_collisions(self):
        """Check for collisions between bullets and enemies, power-ups, and player ships."""
        current_time = time.monotonic()

        # Check bullet vs enemy collisions
        for bullet in self.bullets[:]:  # Use slice to allow modification during iteration
            # Check against regular enemies
            for enemy in self.enemies[:]:
                if self._bullet_enemy_collision(bullet, enemy):
                    self._handle_bullet_enemy_collision(bullet, enemy, current_time)
                    if bullet in self.bullets:
                        self.bullets.remove(bullet)
                    break

            # Check against boss
            if self.boss and bullet in self.bullets:
                if self._bullet_boss_collision(bullet, self.boss):
                    self._handle_bullet_boss_collision(bullet, self.boss, current_time)
                    if bullet in self.bullets:
                        self.bullets.remove(bullet)

        # Check player vs power-up collisions
        for spaceship in self.spaceships.values():
            for powerup in self.powerups[:]:
                if self._player_powerup_collision(spaceship, powerup):
                    self._handle_powerup_collection(spaceship.player_id, powerup)
                    if powerup in self.powerups:
                        self.powerups.remove(powerup)
                    break

        # Check bullet vs power-up collisions
        for bullet in self.bullets[:]:
            for powerup in self.powerups[:]:
                if self._bullet_powerup_collision(bullet, powerup):
                    self._handle_powerup_collection(bullet.player_id, powerup)
                    if powerup in self.powerups:
                        self.powerups.remove(powerup)
                    if bullet in self.bullets:
                        self.bullets.remove(bullet)
                    break

        # Check enemy bullets vs player collisions
        for bullet in self.enemy_bullets[:]:
            for spaceship in self.spaceships.values():
                if self._bullet_player_collision(bullet, spaceship):
                    self._handle_player_damage(spaceship.player_id, bullet.damage)
                    if bullet in self.enemy_bullets:
                        self.enemy_bullets.remove(bullet)
                    break

    def _bullet_enemy_collision(self, bullet, enemy):
        """Check collision between bullet and enemy."""
        dx = abs(bullet.x - enemy.x)
        dy = abs(bullet.y - enemy.y)
        dz = abs(bullet.z - enemy.z)

        return (
            dx < BLOCK_WIDTH / 2 + BULLET_SIZE
            and dy < BLOCK_HEIGHT / 2 + BULLET_SIZE
            and dz < BLOCK_DEPTH / 2 + BULLET_SIZE
        )

    def _bullet_boss_collision(self, bullet, boss):
        """Check collision between bullet and boss."""
        dx = abs(bullet.x - boss.x)
        dy = abs(bullet.y - boss.y)
        dz = abs(bullet.z - boss.z)

        # Boss is larger than regular enemies
        return (
            dx < BLOCK_WIDTH + BULLET_SIZE
            and dy < BLOCK_HEIGHT + BULLET_SIZE
            and dz < BLOCK_DEPTH + BULLET_SIZE
        )

    def _player_powerup_collision(self, spaceship, powerup):
        """Check collision between player and power-up."""
        dx = abs(spaceship.x - powerup.x)
        dy = abs(spaceship.y - powerup.y)
        dz = abs(spaceship.z - powerup.z)

        return dx < 3 and dy < 3 and dz < 3

    def _bullet_powerup_collision(self, bullet, powerup):
        """Check collision between bullet and power-up."""
        dx = abs(bullet.x - powerup.x)
        dy = abs(bullet.y - powerup.y)
        dz = abs(bullet.z - powerup.z)

        return dx < 3 and dy < 3 and dz < 3

    def _bullet_player_collision(self, bullet, spaceship):
        """Check collision between enemy bullet and player."""
        dx = abs(bullet.x - spaceship.x)
        dy = abs(bullet.y - spaceship.y)
        dz = abs(bullet.z - spaceship.z)

        return dx < 2 and dy < 2 and dz < 2

    def _handle_bullet_enemy_collision(self, bullet, enemy, current_time):
        """Handle collision between a bullet and an enemy."""
        # Determine damage based on color matching and power-ups
        base_damage = DAMAGE_MATCHING if bullet.team_id == enemy.team_id else DAMAGE_NON_MATCHING
        total_damage = base_damage * bullet.damage_multiplier

        # Apply damage
        enemy.hp -= total_damage
        enemy.last_damage_time = current_time
        enemy.damage_flash_active = True

        # Add score to the player
        self.player_scores[bullet.player_id] += total_damage

        # Handle explosive bullets
        if bullet.explosive:
            self._handle_explosive_damage(bullet, enemy, current_time)

        # Check if enemy is destroyed
        if enemy.hp <= 0:
            self._destroy_enemy(enemy)
            self.enemies_defeated += 1
            # Bonus points for destroying an enemy
            self.player_scores[bullet.player_id] += 10

    def _handle_bullet_boss_collision(self, bullet, boss, current_time):
        """Handle collision between a bullet and a boss."""
        # Determine damage based on color matching and power-ups
        base_damage = DAMAGE_MATCHING if bullet.team_id == boss.team_id else DAMAGE_NON_MATCHING
        total_damage = base_damage * bullet.damage_multiplier

        # Apply damage
        boss.hp -= total_damage
        boss.last_damage_time = current_time
        boss.damage_flash_active = True

        # Add score to the player
        self.player_scores[bullet.player_id] += total_damage

        # Handle explosive bullets
        if bullet.explosive:
            self._handle_explosive_damage(bullet, boss, current_time)

        # Check if boss is destroyed
        if boss.hp <= 0:
            self._destroy_boss(boss)
            self.bosses_defeated += 1
            # Major bonus points for destroying a boss
            self.player_scores[bullet.player_id] += 100

    def _handle_explosive_damage(self, bullet, target, current_time):
        """Handle area damage from explosive bullets."""
        # Damage all enemies within explosion radius
        for enemy in self.enemies[:]:
            dx = abs(bullet.x - enemy.x)
            dy = abs(bullet.y - enemy.y)
            dz = abs(bullet.z - enemy.z)
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)

            if distance <= EXPLOSIVE_RADIUS:
                damage = DAMAGE_MATCHING if bullet.team_id == enemy.team_id else DAMAGE_NON_MATCHING
                enemy.hp -= damage
                enemy.last_damage_time = current_time
                enemy.damage_flash_active = True

                if enemy.hp <= 0:
                    self._destroy_enemy(enemy)
                    self.enemies_defeated += 1

        # Damage boss if within range
        if self.boss:
            dx = abs(bullet.x - self.boss.x)
            dy = abs(bullet.y - self.boss.y)
            dz = abs(bullet.z - self.boss.z)
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)

            if distance <= EXPLOSIVE_RADIUS:
                damage = (
                    DAMAGE_MATCHING if bullet.team_id == self.boss.team_id else DAMAGE_NON_MATCHING
                )
                self.boss.hp -= damage
                self.boss.last_damage_time = current_time
                self.boss.damage_flash_active = True

                if self.boss.hp <= 0:
                    self._destroy_boss(self.boss)
                    self.bosses_defeated += 1

    def _handle_powerup_collection(self, player_id, powerup):
        """Handle power-up collection by a player."""
        current_time = time.monotonic()
        expiration_time = current_time + POWERUP_LIFETIME

        if powerup.powerup_type == PowerUpType.HEALTH:
            # Restore health
            self.global_health = min(MAX_HEALTH, self.global_health + HEALTH_POWERUP_AMOUNT)
        else:
            # Add power-up to player
            self.player_powerups[player_id][powerup.powerup_type] = expiration_time

        # Add score for collecting power-up
        self.player_scores[player_id] += 5

    def _handle_player_damage(self, player_id, damage):
        """Handle damage to player health."""
        self.global_health = max(0, self.global_health - damage)

    def _destroy_enemy(self, enemy):
        """Destroy an enemy and create particle explosion."""
        # Remove the enemy
        if enemy in self.enemies:
            self.enemies.remove(enemy)

        # Create particle explosion
        self._create_explosion(enemy.x, enemy.y, enemy.z, enemy.color)

    def _destroy_boss(self, boss):
        """Destroy a boss and create massive particle explosion."""
        # Remove the boss
        self.boss = None

        # Create massive particle explosion
        for i in range(3):  # Multiple explosions
            self._create_explosion(
                boss.x + random.uniform(-2, 2),
                boss.y + random.uniform(-2, 2),
                boss.z + random.uniform(-2, 2),
                boss.color,
            )

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

        # Update power-up expiration
        self._update_powerup_expiration(current_time)

        if self.game_phase == GamePhase.LOBBY:
            # Check conditions to start the game
            min_time_passed = current_time > self.min_lobby_time
            deadline_passed = current_time > self.join_deadline
            enough_votes = (
                self.start_game_votes >= len(self.active_players) and len(self.active_players) > 0
            )

            # Start game if: minimum time passed AND (deadline passed OR enough players voted to start)
            if min_time_passed and (deadline_passed or enough_votes):
                if self.active_players:
                    self.game_phase = GamePhase.RUNNING
                    self.game_start_time = current_time
                    print(f"Starting game with {len(self.active_players)} players")
                else:
                    # No players joined, restart lobby
                    self.join_deadline = current_time + 15.0
                    self.min_lobby_time = current_time + 5.0
                    self.start_game_votes = 0

        elif self.game_phase == GamePhase.RUNNING:
            # Update ship movement based on held buttons
            self._update_ship_movement(dt)

            # Progressive difficulty: increase speed and spawn rate over time
            elapsed_time = current_time - self.game_start_time
            self.current_enemy_speed = BLOCK_SPEED + (BLOCK_SPEED_INCREASE * elapsed_time)
            self.current_spawn_rate = BLOCK_SPAWN_RATE + (BLOCK_SPAWN_INCREASE * elapsed_time)

            # Spawn enemies
            if current_time - self.last_enemy_spawn > (1.0 / self.current_spawn_rate):
                self._spawn_enemy()
                self.last_enemy_spawn = current_time

            # Spawn power-ups
            if current_time - self.last_powerup_spawn > (1.0 / POWERUP_SPAWN_RATE):
                self._spawn_powerup()
                self.last_powerup_spawn = current_time

            # Check for boss spawn
            if elapsed_time > BOSS_SPAWN_TIME and self.boss is None:
                self._start_boss_intro()

            # Update game objects
            self._update_bullets(dt)
            self._update_enemies(dt)
            self._update_enemy_bullets(dt)
            self._update_powerups(dt)
            self._update_particles(dt)

            # Check collisions
            self._check_collisions()

            # Check for game over: health depleted
            if self.global_health <= 0:
                self.game_phase = GamePhase.GAME_OVER

        elif self.game_phase == GamePhase.BOSS_INTRO:
            # Handle boss intro animation
            self._update_boss_intro(current_time, dt)

        elif self.game_phase == GamePhase.BOSS_FIGHT:
            # Update ship movement based on held buttons
            self._update_ship_movement(dt)

            # Spawn additional enemies during boss fight
            if current_time - self.last_enemy_spawn > (1.0 / BOSS_SPAWN_RATE):
                self._spawn_enemy()
                self.last_enemy_spawn = current_time

            # Update game objects
            self._update_bullets(dt)
            self._update_enemies(dt)
            self._update_enemy_bullets(dt)
            self._update_powerups(dt)
            self._update_boss(dt)
            self._update_particles(dt)

            # Check collisions
            self._check_collisions()

            # Check for game over: health depleted
            if self.global_health <= 0:
                self.game_phase = GamePhase.GAME_OVER

            # Check for victory: boss defeated
            if self.boss is None:
                if self.bosses_defeated >= 3:  # Defeat 3 bosses to win
                    self.game_phase = GamePhase.VICTORY
                else:
                    # Return to normal gameplay
                    self.game_phase = GamePhase.RUNNING

    def _update_powerup_expiration(self, current_time):
        """Update power-up expiration times."""
        for player_id in self.player_powerups:
            expired_powerups = []
            for powerup_type, expiration_time in self.player_powerups[player_id].items():
                if current_time > expiration_time:
                    expired_powerups.append(powerup_type)

            for powerup_type in expired_powerups:
                del self.player_powerups[player_id][powerup_type]

    def render_game_state(self, raster):
        """Render the game state to the raster."""
        current_time = time.monotonic()

        if self.game_phase == GamePhase.LOBBY:
            self._render_lobby(raster, current_time)
        elif self.game_phase == GamePhase.BOSS_INTRO:
            self._render_boss_intro(raster, current_time)
        elif self.game_phase == GamePhase.GAME_OVER:
            self._render_game_over(raster, current_time)
        elif self.game_phase == GamePhase.VICTORY:
            self._render_victory(raster, current_time)
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
        # Render health warning border if health is low
        if self.global_health < MAX_HEALTH * 0.2:
            self._render_health_warning_border(raster, current_time)

        # Render spaceships
        for spaceship in self.spaceships.values():
            self._render_spaceship(raster, spaceship)

        # Render bullets
        for bullet in self.bullets:
            self._render_bullet(raster, bullet)

        # Render enemy bullets
        for bullet in self.enemy_bullets:
            self._render_enemy_bullet(raster, bullet)

        # Render enemies
        for enemy in self.enemies:
            self._render_enemy(raster, enemy, current_time)

        # Render power-ups
        for powerup in self.powerups:
            self._render_powerup(raster, powerup, current_time)

        # Render boss
        if self.boss:
            self._render_boss(raster, self.boss, current_time)

        # Render particles
        for particle in self.particles:
            self._render_particle(raster, particle)

    def _render_spaceship(self, raster, spaceship):
        """Render a spaceship with tilting effect."""
        # Calculate tilted positions based on ship's tilt angles
        center_x = int(spaceship.x)
        center_y = int(spaceship.y)
        center_z = int(spaceship.z)

        # Apply tilt transformation to create a larger, more dynamic ship
        # Base ship shape (larger cross pattern)
        base_positions = [
            (0, 0, 0),  # Center
            (-1, 0, 0),
            (1, 0, 0),  # Left/Right
            (0, -1, 0),
            (0, 1, 0),  # Front/Back
            (0, 0, 1),  # Top
            # Additional points for larger ship
            (-2, 0, 0),
            (2, 0, 0),  # Extended left/right
            (0, -2, 0),
            (0, 2, 0),  # Extended front/back
            (-1, -1, 0),
            (1, -1, 0),
            (-1, 1, 0),
            (1, 1, 0),  # Corners
            (0, 0, 2),  # Extended top
        ]

        # Apply tilt transformation
        for dx, dy, dz in base_positions:
            # Apply X-axis tilt (forward/backward)
            tilted_y = dy * math.cos(spaceship.tilt_x) - dz * math.sin(spaceship.tilt_x)
            tilted_z = dy * math.sin(spaceship.tilt_x) + dz * math.cos(spaceship.tilt_x)

            # Apply Y-axis tilt (left/right)
            tilted_x = dx * math.cos(spaceship.tilt_y) - tilted_z * math.sin(spaceship.tilt_y)
            final_z = dx * math.sin(spaceship.tilt_y) + tilted_z * math.cos(spaceship.tilt_y)

            # Add to final positions
            x = center_x + int(tilted_x)
            y = center_y + int(tilted_y)
            z = center_z + int(final_z)

            if 0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.length:
                raster.set_pix(x, y, z, spaceship.color)

        # Add engine glow effect based on movement
        speed = math.sqrt(spaceship.vx * spaceship.vx + spaceship.vy * spaceship.vy)
        if speed > 0.1:  # Only show engine glow when moving
            # Engine glow behind the ship
            glow_intensity = min(100, int(speed * 30))  # Reduced intensity and multiplier
            glow_color = RGB(
                min(255, spaceship.color.red + glow_intensity),
                min(255, spaceship.color.green + glow_intensity),
                min(255, spaceship.color.blue + glow_intensity),
            )

            # Calculate engine position (behind the ship based on movement direction)
            engine_x = center_x
            engine_y = center_y
            engine_z = center_z

            if abs(spaceship.vx) > abs(spaceship.vy):
                # Moving horizontally
                engine_x = center_x - int(math.copysign(1, spaceship.vx))  # Just one pixel behind
            else:
                # Moving vertically
                engine_y = center_y - int(math.copysign(1, spaceship.vy))  # Just one pixel behind

            # Ensure engine glow is within bounds
            if (
                0 <= engine_x < self.width
                and 0 <= engine_y < self.height
                and 0 <= engine_z < self.length
            ):
                raster.set_pix(engine_x, engine_y, engine_z, glow_color)

    def _render_bullet(self, raster, bullet):
        """Render a bullet."""
        x = int(round(bullet.x))
        y = int(round(bullet.y))
        z = int(round(bullet.z))

        if 0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.length:
            raster.set_pix(x, y, z, bullet.color)

    def _render_health_warning_border(self, raster, current_time):
        """Render pulsing red border when health is low."""
        pulse_intensity = int(abs(math.sin(current_time * 5) * 255))
        border_color = RGB(pulse_intensity, 0, 0)

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

    def _render_enemy(self, raster, enemy, current_time):
        """Render an enemy with damage effects."""
        center_x = int(enemy.x)
        center_y = int(enemy.y)
        center_z = int(enemy.z)

        # Determine color based on damage state and enemy type
        if enemy.damage_flash_active:
            # Flash white when taking damage
            color = RGB(255, 255, 255)
        else:
            # Calculate damage ratio for cracking effect
            damage_ratio = 1.0 - (enemy.hp / enemy.max_hp)

            if damage_ratio < 0.33:
                # Minimal damage - original color
                color = enemy.color
            elif damage_ratio < 0.66:
                # Medium damage - mix with some white/gray
                mix_factor = 0.3
                color = RGB(
                    int(enemy.color.red * (1 - mix_factor) + 128 * mix_factor),
                    int(enemy.color.green * (1 - mix_factor) + 128 * mix_factor),
                    int(enemy.color.blue * (1 - mix_factor) + 128 * mix_factor),
                )
            else:
                # Heavy damage - add random glitching
                if random.random() < 0.3:  # 30% chance of glitch pixel
                    color = RGB(
                        random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
                    )
                else:
                    color = enemy.color

        # Render enemy as a rectangular prism with type-specific effects
        half_width = BLOCK_WIDTH // 2
        half_height = BLOCK_HEIGHT // 2
        half_depth = BLOCK_DEPTH // 2

        # Add special effects based on enemy type
        if enemy.enemy_type == EnemyType.DRONE:
            # Drones have pulsing effect
            pulse = int(abs(math.sin(current_time * 4 + enemy.movement_phase) * 50))
            color = RGB(
                min(255, color.red + pulse),
                min(255, color.green + pulse),
                min(255, color.blue + pulse),
            )
        elif enemy.enemy_type == EnemyType.WARRIOR:
            # Warriors have weapon glow
            weapon_glow = int(abs(math.sin(current_time * 6) * 100))
            color = RGB(
                min(255, color.red + weapon_glow),
                min(255, color.green),
                min(255, color.blue + weapon_glow),
            )
        elif enemy.enemy_type == EnemyType.ELITE:
            # Elites have targeting indicator
            target_glow = int(abs(math.sin(current_time * 8) * 150))
            color = RGB(
                min(255, color.red + target_glow),
                min(255, color.green + target_glow),
                min(255, color.blue),
            )

        for dx in range(-half_width, half_width + 1):
            for dy in range(-half_height, half_height + 1):
                for dz in range(-half_depth, half_depth + 1):
                    x = center_x + dx
                    y = center_y + dy
                    z = center_z + dz

                    if 0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.length:
                        raster.set_pix(x, y, z, color)

    def _render_powerup(self, raster, powerup, current_time):
        """Render a power-up with rotation and flashing effects."""
        center_x = int(powerup.x)
        center_y = int(powerup.y)
        center_z = int(powerup.z)

        # Get base color
        color = powerup.get_color()

        # Add flashing effect
        flash = int(abs(math.sin(current_time * 8) * 100))
        color = RGB(
            min(255, color.red + flash), min(255, color.green + flash), min(255, color.blue + flash)
        )

        # Render as large rotating plus sign (4 voxels wide)
        angle = powerup.rotation_angle
        size = 4  # Make it 4 voxels wide

        # Large plus sign pattern
        positions = []

        # Horizontal line of the plus
        for i in range(-size // 2, size // 2 + 1):
            positions.append((i, 0))

        # Vertical line of the plus
        for i in range(-size // 2, size // 2 + 1):
            if i != 0:  # Avoid duplicating the center
                positions.append((0, i))

        for dx, dy in positions:
            # Apply rotation
            rx = int(dx * math.cos(angle) - dy * math.sin(angle))
            ry = int(dx * math.sin(angle) + dy * math.cos(angle))

            x = center_x + rx
            y = center_y + ry
            z = center_z

            if 0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.length:
                raster.set_pix(x, y, z, color)

        # Add a second layer slightly above for more visibility
        for dx, dy in positions:
            # Apply rotation
            rx = int(dx * math.cos(angle) - dy * math.sin(angle))
            ry = int(dx * math.sin(angle) + dy * math.cos(angle))

            x = center_x + rx
            y = center_y + ry
            z = center_z + 1

            if 0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.length:
                raster.set_pix(x, y, z, color)

    def _render_boss(self, raster, boss, current_time):
        """Render a boss with special effects."""
        center_x = int(boss.x)
        center_y = int(boss.y)
        center_z = int(boss.z)

        # Determine color based on damage state
        if boss.damage_flash_active:
            # Flash white when taking damage
            color = RGB(255, 255, 255)
        else:
            # Animated rainbow color for boss
            hue = (current_time * 0.5 + boss.animation_phase) % (2 * math.pi)
            r = int(255 * (1 + math.sin(hue)) / 2)
            g = int(255 * (1 + math.sin(hue + 2 * math.pi / 3)) / 2)
            b = int(255 * (1 + math.sin(hue + 4 * math.pi / 3)) / 2)
            color = RGB(r, g, b)

        # Render boss as a MUCH larger, more complex shape
        size = 6  # Boss is significantly larger than regular enemies

        # Massive boss shape - multiple layers and complex structure
        positions = []

        # Core body (large cube)
        for dx in range(-size // 2, size // 2 + 1):
            for dy in range(-size // 2, size // 2 + 1):
                for dz in range(-size // 2, size // 2 + 1):
                    positions.append((dx, dy, dz))

        # Extended arms/weapons
        for i in range(-size, size + 1):
            # Horizontal arms
            positions.append((i, -size - 1, 0))
            positions.append((i, size + 1, 0))
            positions.append((-size - 1, i, 0))
            positions.append((size + 1, i, 0))

            # Vertical extensions
            positions.append((0, 0, i))
            if i % 2 == 0:  # Alternating vertical spikes
                positions.append((size // 2, 0, i))
                positions.append((-size // 2, 0, i))
                positions.append((0, size // 2, i))
                positions.append((0, -size // 2, i))

        # Weapon-specific visual effects
        if boss.weapon_type == "sniper":
            # Sniper scope effect
            for i in range(3):
                positions.append((size + 2 + i, 0, 0))  # Extended barrel
        elif boss.weapon_type == "spray":
            # Multiple weapon barrels
            for angle in range(0, 360, 45):
                rad = math.radians(angle)
                x = int(size * math.cos(rad))
                y = int(size * math.sin(rad))
                positions.append((x, y, 0))
        elif boss.weapon_type == "burst":
            # Burst weapon effect
            for i in range(3):
                positions.append((size + 1 + i, 0, 0))
                positions.append((size + 1 + i, 1, 0))
                positions.append((size + 1 + i, -1, 0))
        else:  # laser
            # Laser weapon effect
            for i in range(5):
                positions.append((size + 1 + i, 0, 0))

        # Render all boss voxels
        for dx, dy, dz in positions:
            x = center_x + dx
            y = center_y + dy
            z = center_z + dz

            if 0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.length:
                # Use weapon-specific colors for weapon parts
                if boss.weapon_type == "sniper" and dx > size // 2:
                    raster.set_pix(x, y, z, RGB(255, 255, 255))  # White scope
                elif boss.weapon_type == "spray" and abs(dx) == size and abs(dy) == size:
                    raster.set_pix(x, y, z, RGB(255, 100, 100))  # Red barrels
                elif boss.weapon_type == "burst" and dx > size:
                    raster.set_pix(x, y, z, RGB(255, 255, 100))  # Yellow burst
                elif boss.weapon_type == "laser" and dx > size:
                    raster.set_pix(x, y, z, RGB(100, 255, 255))  # Cyan laser
                else:
                    raster.set_pix(x, y, z, color)

        # Render boss health bar (larger and more prominent)
        health_ratio = boss.hp / boss.max_hp
        bar_width = 12  # Wider health bar
        bar_height = 2  # Taller health bar
        bar_x = center_x - bar_width // 2
        bar_y = center_y - size - 2
        bar_z = center_z + size + 1

        # Health bar background (red)
        for i in range(bar_width):
            for j in range(bar_height):
                x = bar_x + i
                y = bar_y + j
                if 0 <= x < self.width and 0 <= y < self.height and 0 <= bar_z < self.length:
                    raster.set_pix(x, y, bar_z, RGB(255, 0, 0))

        # Health bar fill (green)
        fill_width = int(bar_width * health_ratio)
        for i in range(fill_width):
            for j in range(bar_height):
                x = bar_x + i
                y = bar_y + j
                if 0 <= x < self.width and 0 <= y < self.height and 0 <= bar_z < self.length:
                    raster.set_pix(x, y, bar_z, RGB(0, 255, 0))

    def _render_enemy_bullet(self, raster, bullet):
        """Render an enemy bullet."""
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
        """Render the game over screen with flashing red border over frozen game state."""
        # First render the current game state (frozen)
        self._render_game(raster, current_time)

        # Then overlay the flashing red border
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

    def _render_victory(self, raster, current_time):
        """Render the victory screen with green border and celebration."""
        # Flash green border
        flash_intensity = int(abs(math.sin(current_time * 5) * 255))
        border_color = RGB(0, flash_intensity, 0)

        # Draw green border around the entire display
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

        # Show victory message and final scores in the center
        center_x = self.width // 2
        center_y = self.height // 2
        center_z = self.length // 2

        # Victory celebration effect
        celebration_color = RGB(
            int(255 * (1 + math.sin(current_time * 3)) / 2),
            int(255 * (1 + math.sin(current_time * 3 + 2 * math.pi / 3)) / 2),
            int(255 * (1 + math.sin(current_time * 3 + 4 * math.pi / 3)) / 2),
        )

        # Sort players by score
        sorted_players = sorted(self.player_scores.items(), key=lambda x: x[1], reverse=True)

        # Display "YOU WIN!" and top score
        if sorted_players:
            top_player, top_score = sorted_players[0]

            # Display top score with celebration color
            y_offset = center_y
            if 0 <= y_offset < self.height:
                for dx in range(8):  # Display score
                    x = center_x - 4 + dx
                    if 0 <= x < self.width:
                        raster.set_pix(x, y_offset, center_z, celebration_color)

    def _render_particle(self, raster, particle):
        """Render a particle."""
        x = int(round(particle.x))
        y = int(round(particle.y))
        z = int(round(particle.z))

        if 0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.length:
            raster.set_pix(x, y, z, particle.color)

    def _render_boss_intro(self, raster, current_time):
        """Render the boss intro animation."""

        if self.boss_intro_phase == 0:
            # Phase 0: White flash border
            flash_intensity = int(abs(math.sin(current_time * 20) * 255))  # Very fast flash
            border_color = RGB(flash_intensity, flash_intensity, flash_intensity)

            # Draw white border around the entire display
            for i in range(self.width):
                for j in range(self.height):
                    if 0 <= i < self.width and 0 <= j < self.height:
                        raster.set_pix(i, j, 0, border_color)  # Bottom face
                        raster.set_pix(i, j, self.length - 1, border_color)  # Top face

            for i in range(self.width):
                for k in range(self.length):
                    if 0 <= i < self.width and 0 <= k < self.length:
                        raster.set_pix(i, 0, k, border_color)  # Front face
                        raster.set_pix(i, self.height - 1, k, border_color)  # Back face

            for j in range(self.height):
                for k in range(self.length):
                    if 0 <= j < self.height and 0 <= k < self.length:
                        raster.set_pix(0, j, k, border_color)  # Left face
                        raster.set_pix(self.width - 1, j, k, border_color)  # Right face

        elif self.boss_intro_phase == 1:
            # Phase 1: Enemy fade out (already done, just show empty space)
            # Render only players and particles
            for spaceship in self.spaceships.values():
                self._render_spaceship(raster, spaceship)
            for particle in self.particles:
                self._render_particle(raster, particle)

        elif self.boss_intro_phase == 2:
            # Phase 2: Space warp animation - falling white bolts
            # Render players
            for spaceship in self.spaceships.values():
                self._render_spaceship(raster, spaceship)

            # Create hyperspace warp effect
            warp_color = RGB(255, 255, 255)
            num_bolts = 20

            for i in range(num_bolts):
                # Create falling bolts at random positions
                x = random.randint(0, self.width - 1)
                y = random.randint(0, self.height - 1)
                z = int((current_time * 10 + i * 2) % self.length)  # Falling effect

                # Make bolts longer (multiple voxels)
                for dz in range(3):
                    bolt_z = (z + dz) % self.length
                    if 0 <= x < self.width and 0 <= y < self.height and 0 <= bolt_z < self.length:
                        raster.set_pix(x, y, bolt_z, warp_color)

        elif self.boss_intro_phase == 3:
            # Phase 3: Boss spawn - show boss appearing
            # Render players
            for spaceship in self.spaceships.values():
                self._render_spaceship(raster, spaceship)

            # Boss should be spawned by now, render it
            if self.boss:
                self._render_boss(raster, self.boss, current_time)

    async def update_controller_display_state(self, controller_state, player_id):
        """Update the controller's LCD display for this player."""
        controller_state.clear()

        if self.game_phase == GamePhase.LOBBY:
            controller_state.write_lcd(0, 0, "SPACE INVADERS")
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
        elif self.game_phase == GamePhase.GAME_OVER:
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
        elif self.game_phase == GamePhase.BOSS_INTRO:
            controller_state.write_lcd(0, 0, "BOSS BATTLE!")
            controller_state.write_lcd(0, 1, f"Boss: {self.boss_intro_boss_type}")
            controller_state.write_lcd(0, 2, f"Weapon: {self.boss_intro_weapon_type}")
            controller_state.write_lcd(0, 3, "Prepare for battle...")
        elif self.game_phase == GamePhase.VICTORY:
            controller_state.write_lcd(0, 0, "YOU WIN!")
            score = self.get_player_score(player_id)
            controller_state.write_lcd(0, 1, f"Final Score: {score}")
            controller_state.write_lcd(0, 2, f"Bosses Defeated: {self.bosses_defeated}")
            controller_state.write_lcd(0, 3, "Press SELECT to restart")
        else:
            # Game is running or boss fight
            controller_state.write_lcd(0, 0, "SPACE INVADERS")

            # Show health
            health_percent = int((self.global_health / MAX_HEALTH) * 100)
            controller_state.write_lcd(0, 1, f"Health: {health_percent}%")

            # Show score
            score = self.get_player_score(player_id)
            controller_state.write_lcd(0, 2, f"Score: {score}")

            # Show active power-ups
            current_time = time.monotonic()
            active_powerups = []
            for powerup_type, expiration_time in self.player_powerups[player_id].items():
                if current_time < expiration_time:
                    time_left = int(expiration_time - current_time)
                    if powerup_type == PowerUpType.POWER_SHOT:
                        active_powerups.append(f"PWR:{time_left}s")
                    elif powerup_type == PowerUpType.EXPLOSIVE_SHOT:
                        active_powerups.append(f"EXP:{time_left}s")

            if active_powerups:
                controller_state.write_lcd(
                    0, 3, " ".join(active_powerups[:2])
                )  # Show up to 2 power-ups
            elif self.game_phase == GamePhase.BOSS_FIGHT and self.boss:
                # Show boss health during boss fight
                boss_health = int((self.boss.hp / self.boss.max_hp) * 100)
                controller_state.write_lcd(0, 3, f"Boss: {boss_health}%")
            else:
                controller_state.write_lcd(0, 3, "Arrows: move, SELECT: shoot")

        await controller_state.commit()
