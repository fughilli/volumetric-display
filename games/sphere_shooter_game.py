import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Set

from games.util.base_game import RGB, BaseGame, PlayerID, TeamID
from games.util.game_util import Button, ButtonState

TOP_SCORE = 10
TIME_LIMIT = 180

# Configuration mapping player roles to their team and view orientation
PLAYER_CONFIG = {
    PlayerID.P1: {
        "team": TeamID.RED,
        "view": (1, 0, 0),  # -X view
        "left_dir": (0, 1, 0),  # -Y
        "right_dir": (0, -1, 0),  # +Y
        "up_dir": (0, 0, 1),  # +Z
        "down_dir": (0, 0, -1),  # -Z
    },
    PlayerID.P2: {
        "team": TeamID.GREEN,
        "view": (0, 1, 0),  # -Y view
        "left_dir": (-1, 0, 0),  # +X
        "right_dir": (1, 0, 0),  # -X
        "up_dir": (0, 0, 1),  # +Z
        "down_dir": (0, 0, -1),  # -Z
    },
    PlayerID.P3: {
        "team": TeamID.ORANGE,
        "view": (-1, 0, 0),  # +X view
        "left_dir": (0, -1, 0),  # +Y
        "right_dir": (0, 1, 0),  # -Y
        "up_dir": (0, 0, 1),  # +Z
        "down_dir": (0, 0, -1),  # -Z
    },
    PlayerID.P4: {
        "team": TeamID.BLUE,
        "view": (0, -1, 0),  # +Y view
        "left_dir": (1, 0, 0),  # -X
        "right_dir": (-1, 0, 0),  # +X
        "up_dir": (0, 0, 1),  # +Z
        "down_dir": (0, 0, -1),  # -Z
    },
}

# Time in seconds to reach a full power shot when holding SELECT
FULL_CHARGE_TIME = 1.5


@dataclass
class Sphere:
    x: float  # position
    y: float
    z: float
    vx: float  # velocity
    vy: float
    vz: float
    radius: float
    birth_time: float
    mass: float
    lifetime: float
    color: RGB
    team: TeamID  # Track which team shot this sphere
    owner: PlayerID  # Which player fired this sphere
    bounce_count: int = 0  # How many times the sphere has bounced off a wall/floor/ceiling
    floor_bounced: bool = False  # Track if sphere has already bounced on the floor once

    # Physics constants
    GRAVITY = 100.0  # Gravity acceleration (reduced)
    ELASTICITY = 0.95  # Bounce elasticity (1.0 = perfect bounce)
    AIR_DAMPING = 0.999  # Air resistance (velocity multiplier per update)
    GROUND_FRICTION = 0.95  # Additional friction when touching ground
    MINIMUM_SPEED = 0.01  # Speed below which we stop movement
    MAX_BOUNCES = 5  # Expire after this many bounces

    def update(self, dt: float, bounds: tuple[float, float, float]):
        # Apply gravity
        self.vz -= self.GRAVITY * dt

        # Apply air resistance
        self.vx *= self.AIR_DAMPING
        self.vy *= self.AIR_DAMPING
        self.vz *= self.AIR_DAMPING

        # Apply additional ground friction when touching floor BEFORE first bounce occurs
        if not self.floor_bounced and self.z - self.radius <= 0:
            self.vx *= self.GROUND_FRICTION
            self.vy *= self.GROUND_FRICTION

        # Stop very slow movement
        speed = math.sqrt(self.vx * self.vx + self.vy * self.vy + self.vz * self.vz)
        if speed < self.MINIMUM_SPEED:
            self.vx = 0
            self.vy = 0
            self.vz = 0

        # Update position
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt

        # Bounce off walls with energy loss
        width, height, length = bounds
        bounced = False  # Track if we bounced this update
        if self.x - self.radius < 0:
            self.x = self.radius
            self.vx = abs(self.vx) * self.ELASTICITY
            bounced = True
        elif self.x + self.radius > width - 1:
            self.x = width - 1 - self.radius
            self.vx = -abs(self.vx) * self.ELASTICITY
            bounced = True

        if self.y - self.radius < 0:
            self.y = self.radius
            self.vy = abs(self.vy) * self.ELASTICITY
            bounced = True
        elif self.y + self.radius > height - 1:
            self.y = height - 1 - self.radius
            self.vy = -abs(self.vy) * self.ELASTICITY
            bounced = True

        if self.z - self.radius < 0:
            if not self.floor_bounced:
                # First time hitting floor: bounce normally
                self.z = self.radius
                self.vz = abs(self.vz) * self.ELASTICITY
                bounced = True
                self.floor_bounced = True
            # After first bounce, no further collision response; sphere may fall below floor
        elif self.z + self.radius > length - 1:
            self.z = length - 1 - self.radius
            self.vz = -abs(self.vz) * self.ELASTICITY
            bounced = True

        # Increment bounce counter if we hit anything
        if bounced:
            self.bounce_count += 1

    def collide_with(self, other: "Sphere"):
        """Handle elastic collision with another sphere"""
        # Calculate distance between sphere centers
        dx = other.x - self.x
        dy = other.y - self.y
        dz = other.z - self.z
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)

        # Check if spheres are overlapping
        if distance < self.radius + other.radius:
            # Normal vector of the collision
            nx = dx / distance
            ny = dy / distance
            nz = dz / distance

            # Relative velocity
            rvx = other.vx - self.vx
            rvy = other.vy - self.vy
            rvz = other.vz - self.vz

            # Relative velocity along normal
            normal_vel = rvx * nx + rvy * ny + rvz * nz

            # Only collide if spheres are moving toward each other
            if normal_vel < 0:
                # Calculate the impulse scalar
                impulse = -(1 + self.ELASTICITY) * normal_vel / (1 / self.mass + 1 / other.mass)

                # Update velocities using conservation of momentum
                self.vx -= (impulse / self.mass) * nx
                self.vy -= (impulse / self.mass) * ny
                self.vz -= (impulse / self.mass) * nz
                other.vx += (impulse / other.mass) * nx
                other.vy += (impulse / other.mass) * ny
                other.vz += (impulse / other.mass) * nz

                # Separate spheres to prevent sticking
                overlap = (self.radius + other.radius - distance) / 2
                self.x -= nx * overlap
                self.y -= ny * overlap
                self.z -= nz * overlap
                other.x += nx * overlap
                other.y += ny * overlap
                other.z += nz * overlap

    def is_expired(self, current_time: float) -> bool:
        # Expire after lifetime OR after too many bounces
        return (current_time - self.birth_time > self.lifetime) or (
            self.bounce_count >= self.MAX_BOUNCES
        )


@dataclass
class Cannon:
    x: float  # Position on the face
    y: float
    face: str  # Which face of the cube ('x', '-x', 'y', '-y')
    team: TeamID
    color: RGB
    owner: PlayerID
    select_hold_start: float = None  # When SELECT was pressed
    radius: float = 1.0
    charging: bool = False  # Whether the cannon is charging
    held_dirs: Set[Button] = field(default_factory=set)  # Directions currently held down
    draw_radius: float = 2.0  # Visual radius when rendering
    cooldown_remaining: float = 0.0
    cooldown_total: float = 0.0


# ------------------------
# Hoop representation
# ------------------------


@dataclass
class Hoop:
    """A moving hoop positioned at (x, y_level, z) where y_level is its height above the floor (z-axis in this game)."""

    x: float  # position along width (x-axis)
    z: float  # position along depth/height plane (y-axis of arena)
    radius: float
    level: float = 0.0  # vertical position along z-axis (0 = floor)
    color: RGB = field(default_factory=lambda: RGB(255, 255, 255))  # default white
    flash_color: RGB | None = None
    flash_timer: float = 0.0  # seconds remaining for flash


# ------------------------
# Particle representation for explosion
# ------------------------


@dataclass
class Particle:
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    birth_time: float
    lifetime: float
    color: RGB
    radius: float = 0.5

    GRAVITY = 100.0
    AIR_DAMPING = 0.99

    def update(self, dt: float):
        # Gravity along -z
        self.vz -= self.GRAVITY * dt
        # Damping
        self.vx *= self.AIR_DAMPING
        self.vy *= self.AIR_DAMPING
        self.vz *= self.AIR_DAMPING
        # Position
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt

    def is_expired(self, current_time: float) -> bool:
        return current_time - self.birth_time > self.lifetime


class SphereShooterGame(BaseGame):
    DISPLAY_NAME = "Hoops"

    def __init__(
        self,
        width=20,
        height=20,
        length=20,
        frameRate=30,
        config=None,
        input_handler=None,
    ):
        self.team_colors = {
            TeamID.RED: RGB(255, 0, 0),  # Red team
            TeamID.ORANGE: RGB(255, 165, 0),  # Orange team
            TeamID.YELLOW: RGB(255, 255, 0),  # Yellow team
            TeamID.GREEN: RGB(0, 255, 0),  # Green team
            TeamID.BLUE: RGB(0, 0, 255),  # Blue team
            TeamID.PURPLE: RGB(255, 0, 255),  # Purple team
        }

        # Hoop parameters – must exist before BaseGame.__init__ triggers reset_game
        self.hoop_angle = 0.0
        hoop_radius = 3.0
        self.hoop = Hoop(width / 2, height / 2, hoop_radius, level=0.0)

        # Hoop motion state variables
        self.hoop_moving = False
        self.hoop_dwell_timer = random.uniform(1.0, 2.0)
        self.hoop_move_progress = 0.0

        # Particle system
        self.particles: List[Particle] = []

        # Track recent scoring timestamps per player for ON FIRE status
        self.score_times: Dict[PlayerID, List[float]] = {pid: [] for pid in PlayerID}
        self.on_fire_until: Dict[PlayerID, float] = {pid: 0.0 for pid in PlayerID}

        # Game timing
        self.game_start_time = time.monotonic()
        self.winner_players: List[PlayerID] = []

        # Call parent constructor (this invokes reset_game)
        super().__init__(width, height, length, frameRate, config, input_handler)

        # Player score map (reset_game will have created it; keep for clarity)
        if not hasattr(self, "player_scores"):
            self.player_scores: Dict[PlayerID, int] = {pid: 0 for pid in PlayerID}

        self.last_update_time = time.monotonic()

    def reset_game(self):
        """Reset the game state."""
        self.spheres = []
        self.cannons = {}
        self.player_scores = {pid: 0 for pid in PlayerID}
        self.score_times = {pid: [] for pid in PlayerID}
        self.on_fire_until = {pid: 0.0 for pid in PlayerID}
        self.game_start_time = time.monotonic()
        self.winner_players = []

        # Reset hoop
        self.hoop_angle = 0.0
        self.hoop.x = self.width / 2
        self.hoop.z = self.height / 2
        self.hoop.level = 0.0

        self.game_over_active = False
        self.game_over_flash_state = {
            "count": 0,
            "timer": 0,
            "interval": 0.2,
            "border_on": False,
        }

        # Initialize cannons for each player
        for player_id in PlayerID:
            config = PLAYER_CONFIG[player_id]
            team = config["team"]
            view = config["view"]

            # Determine which face the cannon is on based on view direction
            if view[0] != 0:  # X view
                face = "-x" if view[0] < 0 else "x"
                x = self.height // 2  # Position on face
                y = self.length // 2
            else:  # Y view
                face = "-y" if view[1] < 0 else "y"
                x = self.width // 2
                y = self.length // 2

            self.cannons[player_id] = Cannon(
                x=x,
                y=y,
                face=face,
                team=team,
                color=self.team_colors[team],
                owner=player_id,
            )

        # Reset hoop motion state
        self.hoop_moving = False
        self.hoop_dwell_timer = random.uniform(1.0, 2.0)
        self.hoop_move_progress = 0.0

        # Clear particles
        self.particles = []

    def get_player_score(self, player_id):
        """Get the score for a player."""
        return self.player_scores.get(player_id, 0)

    def get_opponent_score(self, player_id):
        """Get the score for a player's opponent."""
        return max(score for pid, score in self.player_scores.items() if pid != player_id)

    def process_player_input(self, player_id, button, button_state):
        """Process input from a player."""
        if self.game_over_active or player_id not in self.cannons:
            return

        cannon = self.cannons[player_id]

        # Handle SELECT button (charging and firing)
        if button == Button.SELECT:
            if button_state == ButtonState.PRESSED:
                # Respect cooldown
                if cannon.cooldown_remaining > 0:
                    return
                # Start charging when SELECT is pressed
                cannon.select_hold_start = time.monotonic()
                cannon.charging = True
            elif button_state == ButtonState.RELEASED and cannon.charging:
                # Fire the shot when SELECT is released
                if cannon.select_hold_start is not None:
                    charge_time = time.monotonic() - cannon.select_hold_start
                    self.launch_sphere(cannon, charge_time)
                # Reset charging state
                cannon.select_hold_start = None
                cannon.charging = False
            return

        # Track directional button holds
        if button in {Button.LEFT, Button.RIGHT, Button.UP, Button.DOWN}:
            if button_state == ButtonState.PRESSED:
                cannon.held_dirs.add(button)
            elif button_state == ButtonState.RELEASED:
                cannon.held_dirs.discard(button)

    def update_game_state(self):
        """Update the game state."""
        current_time = time.monotonic()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time

        # ---------- Move hoop with random smoothstep ----------
        if not hasattr(self, "hoop_moving"):
            self.hoop_moving = False
            self.hoop_dwell_timer = random.uniform(1.0, 2.0)

        if not self.hoop_moving:
            # Waiting period before picking a new target
            self.hoop_dwell_timer -= dt
            if self.hoop_dwell_timer <= 0.0:
                # Pick new random target position on floor within bounds
                margin = self.hoop.radius + 1
                target_x = random.uniform(margin, self.width - margin)
                target_z = random.uniform(margin, self.height - margin)
                target_level = random.uniform(
                    0, self.length / 2
                )  # up to halfway in vertical (z-axis length)

                self.hoop_start_x = self.hoop.x
                self.hoop_start_z = self.hoop.z
                self.hoop_start_level = self.hoop.level

                self.hoop_target_x = target_x
                self.hoop_target_z = target_z
                self.hoop_target_level = target_level

                self.hoop_move_duration = random.uniform(2.0, 4.0)  # seconds to move
                self.hoop_move_progress = 0.0
                self.hoop_moving = True
        else:
            # Progress the move
            self.hoop_move_progress += dt / self.hoop_move_duration
            if self.hoop_move_progress >= 1.0:
                self.hoop_move_progress = 1.0
                self.hoop_moving = False
                self.hoop_dwell_timer = random.uniform(1.0, 2.0)

            # Smoothstep interpolation
            t = self.hoop_move_progress
            t_smooth = t * t * (3 - 2 * t)
            self.hoop.x = self.hoop_start_x + (self.hoop_target_x - self.hoop_start_x) * t_smooth
            self.hoop.z = self.hoop_start_z + (self.hoop_target_z - self.hoop_start_z) * t_smooth
            self.hoop.level = (
                self.hoop_start_level + (self.hoop_target_level - self.hoop_start_level) * t_smooth
            )

        # ---------- Move cannons based on held directions ----------
        cannon_speed = 5.0  # voxels per second
        move_amt = cannon_speed * dt
        for cannon in self.cannons.values():
            if cannon.cooldown_remaining > 0:
                cannon.cooldown_remaining = max(0.0, cannon.cooldown_remaining - dt)

            # Movement in face plane
            if Button.LEFT in cannon.held_dirs:
                if cannon.face in ["x", "-y"]:
                    cannon.x -= move_amt
                else:
                    cannon.x += move_amt
            if Button.RIGHT in cannon.held_dirs:
                if cannon.face in ["x", "-y"]:
                    cannon.x += move_amt
                else:
                    cannon.x -= move_amt
            if Button.UP in cannon.held_dirs:
                cannon.y = min(self.length - 1 - cannon.radius, cannon.y + move_amt)
            if Button.DOWN in cannon.held_dirs:
                cannon.y = max(cannon.radius, cannon.y - move_amt)

            # Clamp within bounds of face
            cannon.x = max(cannon.radius, min(self.height - 1 - cannon.radius, cannon.x))

        # ---------- Update sphere physics, check scoring ----------
        bounds = (self.width, self.height, self.length)
        new_spheres = []

        # Update hoop flash timer
        if self.hoop.flash_timer > 0:
            self.hoop.flash_timer = max(0.0, self.hoop.flash_timer - dt)

        for sphere in self.spheres:
            if sphere.is_expired(current_time):
                continue

            # Update physics
            sphere.update(dt, bounds)

            # Collision with other spheres
            for other in self.spheres:
                if sphere != other and not other.is_expired(current_time):
                    sphere.collide_with(other)

            # Score/rim logic only when centre is at or below hoop plane
            if sphere.z <= self.hoop.level and sphere.vz < 0:
                dx = sphere.x - self.hoop.x
                dy_plane = sphere.y - self.hoop.z  # hoop.z is Y coordinate in plane
                dist_plane = math.sqrt(dx * dx + dy_plane * dy_plane)
                if dist_plane <= self.hoop.radius:
                    # Score for owner
                    self.player_scores[sphere.owner] += 1

                    # Record score time
                    self.score_times[sphere.owner].append(current_time)

                    # Trim to last 6s
                    self.score_times[sphere.owner] = [
                        t for t in self.score_times[sphere.owner] if current_time - t <= 6.0
                    ]
                    if len(self.score_times[sphere.owner]) > 3:
                        self.on_fire_until[sphere.owner] = current_time + 5.0  # ON FIRE lasts 5s

                    # Hoop flash
                    self.hoop.flash_color = sphere.color
                    self.hoop.flash_timer = 0.5

                    # Particle explosion
                    self.spawn_particle_explosion(sphere)

                    continue  # Do not keep this sphere
                else:
                    # RIM COLLISION CHECK (use slightly larger virtual rim)
                    rim_radius = self.hoop.radius + 1.0  # virtual rim size for bounce
                    if (
                        dist_plane <= rim_radius + sphere.radius
                        and dist_plane >= self.hoop.radius - sphere.radius
                    ):
                        # Sphere hits rim if moving toward it
                        if dist_plane != 0:
                            nx = dx / dist_plane
                            ny = dy_plane / dist_plane
                            # Velocity component along rim normal (horizontal plane)
                            vel_normal = sphere.vx * nx + sphere.vy * ny
                            if vel_normal < 0:
                                sphere.vx -= (1 + sphere.ELASTICITY) * vel_normal * nx
                                sphere.vy -= (1 + sphere.ELASTICITY) * vel_normal * ny
                                sphere.bounce_count += 1

            # Remove spheres that have fallen outside the cube volume entirely
            if (
                sphere.x < -sphere.radius
                or sphere.x > self.width - 1 + sphere.radius
                or sphere.y < -sphere.radius
                or sphere.y > self.height - 1 + sphere.radius
                or sphere.z < -sphere.radius
                or sphere.z > self.length - 1 + sphere.radius
            ):
                # Sphere is out of play – do not keep it
                continue

            new_spheres.append(sphere)

        self.spheres = new_spheres

        # ---------- Update particles ----------
        new_particles = []
        for p in self.particles:
            if p.is_expired(current_time):
                continue
            p.update(dt)
            # Cull if out of bounds
            if p.x < 0 or p.x >= self.width or p.y < 0 or p.y >= self.height or p.z < 0:
                continue
            new_particles.append(p)
        self.particles = new_particles

        # ---------- Win condition check ----------
        if not self.game_over_active:
            # Score win
            top_score = max(self.player_scores.values())
            if top_score >= TOP_SCORE:
                winners = [pid for pid, s in self.player_scores.items() if s == top_score]
            else:
                # Time win
                elapsed = current_time - self.game_start_time
                winners = []
                if elapsed >= TIME_LIMIT:
                    top_score = max(self.player_scores.values())
                    winners = [pid for pid, s in self.player_scores.items() if s == top_score]

            if winners:
                self.game_over_active = True
                self.winner_players = winners
                # Border color: single winner's team color else white
                if len(winners) == 1:
                    team = PLAYER_CONFIG[winners[0]]["team"]
                    self.game_over_flash_state["border_color"] = self.team_colors[team]
                else:
                    self.game_over_flash_state["border_color"] = RGB(255, 255, 255)
                self.game_over_flash_state["timer"] = current_time
                self.game_over_flash_state["border_on"] = True

    def launch_sphere(self, cannon: Cannon, charge_time: float):
        """Launch a sphere from a cannon."""
        # Calculate velocity based on charge time (1-3 seconds)
        base_speed = 10.0  # Base speed (reduced)
        max_speed = 100.0  # Maximum speed (reduced)
        speed = base_speed + (max_speed - base_speed) * min(charge_time / FULL_CHARGE_TIME, 1.0)

        # Set initial position based on cannon face
        if cannon.face == "x":
            x = self.width - 1
            y = cannon.x
            z = cannon.y
            vx = -speed  # Shoot towards -x
            vy = 0
            vz = 0
        elif cannon.face == "-x":
            x = 0
            y = cannon.x
            z = cannon.y
            vx = speed  # Shoot towards +x
            vy = 0
            vz = 0
        elif cannon.face == "y":
            x = cannon.x
            y = self.height - 1
            z = cannon.y
            vx = 0
            vy = -speed  # Shoot towards -y
            vz = 0
        else:  # '-y'
            x = cannon.x
            y = 0
            z = cannon.y
            vx = 0
            vy = speed  # Shoot towards +y
            vz = 0

        # Add some random spread
        spread = 2.0
        vx += random.uniform(-spread, spread)
        vy += random.uniform(-spread, spread)
        vz += random.uniform(-spread, spread)

        # Create the sphere
        sphere = Sphere(
            x=x,
            y=y,
            z=z,
            vx=vx,
            vy=vy,
            vz=vz,
            radius=2.0,
            birth_time=time.monotonic(),
            lifetime=15.0,  # Spheres last 15 seconds
            color=cannon.color,
            team=cannon.team,
            mass=1.0,
            owner=cannon.owner,
        )
        self.spheres.append(sphere)

        # Set cooldown for cannon based on ON FIRE status
        current_time = time.monotonic()
        is_fire = self._is_on_fire(cannon.owner, current_time)
        cannon.cooldown_total = 0.25 if is_fire else 0.5
        cannon.cooldown_remaining = cannon.cooldown_total

    def spawn_particle_explosion(self, sphere: Sphere, count: int = 30):
        """Spawn particles at the sphere's location after scoring."""
        for _ in range(count):
            speed = random.uniform(10, 40)
            theta = random.uniform(0, 2 * math.pi)
            phi = random.uniform(0, math.pi / 2)  # upward hemisphere
            vx = speed * math.cos(theta) * math.sin(phi)
            vy = speed * math.sin(theta) * math.sin(phi)
            vz = speed * math.cos(phi)  # vertical component upward

            p = Particle(
                x=sphere.x,
                y=sphere.y,
                z=sphere.z,
                vx=vx,
                vy=vy,
                vz=vz,
                birth_time=time.monotonic(),
                lifetime=3.0,
                color=sphere.color,
            )
            self.particles.append(p)

    def render_game_state(self, raster):
        """Render the game state to the raster."""
        current_time = time.monotonic()
        # Draw spheres
        for sphere in self.spheres:
            # Determine the bounding box for the sphere
            min_x = math.floor(sphere.x - sphere.radius)
            max_x = math.ceil(sphere.x + sphere.radius)
            min_y = math.floor(sphere.y - sphere.radius)
            max_y = math.ceil(sphere.y + sphere.radius)
            min_z = math.floor(sphere.z - sphere.radius)
            max_z = math.ceil(sphere.z + sphere.radius)

            for vx in range(min_x, max_x + 1):
                for vy in range(min_y, max_y + 1):
                    for vz in range(min_z, max_z + 1):
                        # Voxel center
                        voxel_center_x = vx + 0.5
                        voxel_center_y = vy + 0.5
                        voxel_center_z = vz + 0.5

                        # Distance from voxel center to sphere center
                        dist_sq = (
                            (voxel_center_x - sphere.x) ** 2
                            + (voxel_center_y - sphere.y) ** 2
                            + (voxel_center_z - sphere.z) ** 2
                        )

                        if dist_sq <= sphere.radius**2:
                            if (
                                0 <= vx < self.width
                                and 0 <= vy < self.height
                                and 0 <= vz < self.length
                            ):
                                raster.set_pix(vx, vy, vz, sphere.color)

        # Draw cannons
        for cannon in self.cannons.values():
            # Calculate cannon color (include charging pulse as before)
            color = cannon.color
            if cannon.charging and cannon.select_hold_start:
                charge_time = time.monotonic() - cannon.select_hold_start
                charge_percentage = min(1.0, charge_time / FULL_CHARGE_TIME)
                pulse_speed = 5 + charge_percentage * 15
                pulse = (math.sin(charge_time * pulse_speed) + 1) / 2
                brightness = 1.0 + charge_percentage * 1.5
                color = RGB(
                    min(255, int(color.red * brightness * (1 + pulse * 0.5))),
                    min(255, int(color.green * brightness * (1 + pulse * 0.5))),
                    min(255, int(color.blue * brightness * (1 + pulse * 0.5))),
                )

            # Draw cannon as filled circle on its face
            for u in range(-int(cannon.draw_radius), int(cannon.draw_radius) + 1):
                for v in range(-int(cannon.draw_radius), int(cannon.draw_radius) + 1):
                    if u * u + v * v > cannon.draw_radius * cannon.draw_radius:
                        continue
                    if cannon.face == "x":
                        xx = self.width - 1
                        yy = int(cannon.x + u)
                        zz = int(cannon.y + v)
                        if 0 <= yy < self.height and 0 <= zz < self.length:
                            raster.set_pix(xx, yy, zz, color)
                    elif cannon.face == "-x":
                        xx = 0
                        yy = int(cannon.x + u)
                        zz = int(cannon.y + v)
                        if 0 <= yy < self.height and 0 <= zz < self.length:
                            raster.set_pix(xx, yy, zz, color)
                    elif cannon.face == "y":
                        yy = self.height - 1
                        xx = int(cannon.x + u)
                        zz = int(cannon.y + v)
                        if 0 <= xx < self.width and 0 <= zz < self.length:
                            raster.set_pix(xx, yy, zz, color)
                    else:  # '-y'
                        yy = 0
                        xx = int(cannon.x + u)
                        zz = int(cannon.y + v)
                        if 0 <= xx < self.width and 0 <= zz < self.length:
                            raster.set_pix(xx, yy, zz, color)

        # Draw particles
        for p in self.particles:
            vx = int(round(p.x))
            vy = int(round(p.y))
            vz = int(round(p.z))
            if 0 <= vx < self.width and 0 <= vy < self.height and 0 <= vz < self.length:
                raster.set_pix(vx, vy, vz, p.color)

        # Draw hoop (ring)
        ring_thickness = 0.5
        hoop_color = (
            self.hoop.color
            if self.hoop.flash_timer <= 0
            else (self.hoop.flash_color or self.hoop.color)
        )
        for xx in range(self.width):
            for yy in range(self.height):
                dx = xx + 0.5 - self.hoop.x
                dy = yy + 0.5 - self.hoop.z
                dist = math.sqrt(dx * dx + dy * dy)
                if abs(dist - self.hoop.radius) <= ring_thickness:
                    z_level = int(round(self.hoop.level))
                    if 0 <= z_level < self.length:
                        raster.set_pix(xx, yy, z_level, hoop_color)

        # Draw game over border
        if self.game_over_active:
            # toggle border flash
            if (
                current_time - self.game_over_flash_state["timer"]
                >= self.game_over_flash_state["interval"]
            ):
                self.game_over_flash_state["timer"] = current_time
                self.game_over_flash_state["border_on"] = not self.game_over_flash_state[
                    "border_on"
                ]
            if self.game_over_flash_state["border_on"]:
                border_color = self.game_over_flash_state["border_color"]
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
        if self.game_over_active:
            # Game over screen
            controller_state.clear()
            winners_names = ",".join([pid.name for pid in self.winner_players])
            header = "WINNERS" if len(self.winner_players) > 1 else "WINNER"
            controller_state.write_lcd(0, 0, "GAME OVER")
            controller_state.write_lcd(0, 1, f"{header}: ")
            controller_state.write_lcd(0, 2, winners_names[:20])
            controller_state.write_lcd(0, 3, "Hold SELECT to EXIT")
            await controller_state.commit()
            return

        config = PLAYER_CONFIG[player_id]
        team_name = config["team"].name

        # Get the player's cannon
        cannon = self.cannons.get(player_id)

        controller_state.clear()

        current_time = time.monotonic()
        on_fire = self._is_on_fire(player_id, current_time)

        on_fire_string = " (ON FIRE)" if on_fire else ""
        controller_state.write_lcd(0, 0, f"PLAYER {team_name}{on_fire_string}")
        if cannon and cannon.charging and cannon.select_hold_start:
            # Show charging animation when SELECT is held
            charge_time = time.monotonic() - cannon.select_hold_start
            charge_percentage = min(1.0, charge_time / FULL_CHARGE_TIME)

            # Create a charging progress bar
            charge_bar = ""
            bar_length = 18
            filled = int(charge_percentage * bar_length)
            charge_bar = "[" + "#" * filled + "-" * (bar_length - filled) + "]"

            controller_state.write_lcd(0, 2, f"CHARGING: {charge_percentage * 100:.0f}%")
            controller_state.write_lcd(0, 3, charge_bar)
        else:

            my_score = self.get_player_score(player_id)
            opponent_score = self.get_opponent_score(player_id)

            controller_state.write_lcd(0, 1, f"     YOU: {my_score}")
            controller_state.write_lcd(0, 2, f"BEST OPP: {opponent_score}")
            if on_fire:
                remaining = max(0.0, self.on_fire_until[player_id] - current_time)
                pct = remaining / 5.0
                bar_len = 18
                filled = int(pct * bar_len)
                bar = "[" + "#" * filled + "-" * (bar_len - filled) + "]"
                if cannon.cooldown_remaining > 0:
                    controller_state.write_lcd(0, 3, "COOL")
                else:
                    controller_state.write_lcd(0, 3, bar)
            else:
                if cannon.cooldown_remaining > 0:
                    status_line = "COOL"  # indicates cooling down
                else:
                    status_line = "READY"
                controller_state.write_lcd(0, 3, status_line)

        await controller_state.commit()

    def _is_on_fire(self, player_id: PlayerID, current_time: float) -> bool:
        return current_time < self.on_fire_until.get(player_id, 0.0)
