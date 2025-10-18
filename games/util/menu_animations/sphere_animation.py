import math
import random
from dataclasses import dataclass
from typing import List

from artnet import HSV, RGB, Raster
from games.util.menu_animations import MenuAnimation


@dataclass
class Sphere:
    """A bouncing sphere with physics."""

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

    # Physics constants
    GRAVITY = 20.0  # Gravity acceleration
    ELASTICITY = 0.95  # Bounce elasticity (1.0 = perfect bounce)
    AIR_DAMPING = 0.999  # Air resistance
    GROUND_FRICTION = 0.95  # Ground friction
    MINIMUM_SPEED = 0.01  # Minimum speed before stopping
    FADE_IN_OUT_TIME = 0.2  # Fade time

    def update(self, dt: float, bounds: tuple[float, float, float]) -> None:
        """Update sphere physics.

        Args:
            dt: Time delta in seconds
            bounds: (width, height, length) of the display
        """
        # Apply gravity
        self.vy -= self.GRAVITY * dt

        # Apply air resistance
        self.vx *= self.AIR_DAMPING
        self.vy *= self.AIR_DAMPING
        self.vz *= self.AIR_DAMPING

        # Apply ground friction
        if self.y - self.radius <= 0:
            self.vx *= self.GROUND_FRICTION
            self.vz *= self.GROUND_FRICTION

        # Stop very slow movement
        speed = math.sqrt(self.vx * self.vx + self.vy * self.vy + self.vz * self.vz)
        if speed < self.MINIMUM_SPEED:
            self.vx = self.vy = self.vz = 0

        # Update position
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt

        # Bounce off walls
        width, height, length = bounds
        if self.x - self.radius < 0:
            self.x = self.radius
            self.vx = abs(self.vx) * self.ELASTICITY
        elif self.x + self.radius > width - 1:
            self.x = width - 1 - self.radius
            self.vx = -abs(self.vx) * self.ELASTICITY

        if self.y - self.radius < 0:
            self.y = self.radius
            self.vy = abs(self.vy) * self.ELASTICITY
        elif self.y + self.radius > height - 1:
            self.y = height - 1 - self.radius
            self.vy = -abs(self.vy) * self.ELASTICITY

        if self.z - self.radius < 0:
            self.z = self.radius
            self.vz = abs(self.vz) * self.ELASTICITY
        elif self.z + self.radius > length - 1:
            self.z = length - 1 - self.radius
            self.vz = -abs(self.vz) * self.ELASTICITY

    def collide_with(self, other: "Sphere") -> None:
        """Handle elastic collision with another sphere."""
        # Calculate distance between centers
        dx = other.x - self.x
        dy = other.y - self.y
        dz = other.z - self.z
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)

        # Check for collision
        if distance < self.radius + other.radius and distance > 0:
            # Normal vector of collision
            nx = dx / distance
            ny = dy / distance
            nz = dz / distance

            # Relative velocity
            rvx = other.vx - self.vx
            rvy = other.vy - self.vy
            rvz = other.vz - self.vz

            # Normal velocity
            normal_vel = rvx * nx + rvy * ny + rvz * nz

            # Only collide if moving toward each other
            if normal_vel < 0:
                # Calculate impulse
                impulse = -(1 + self.ELASTICITY) * normal_vel / (1 / self.mass + 1 / other.mass)

                # Apply impulse
                self.vx -= (impulse / self.mass) * nx
                self.vy -= (impulse / self.mass) * ny
                self.vz -= (impulse / self.mass) * nz
                other.vx += (impulse / other.mass) * nx
                other.vy += (impulse / other.mass) * ny
                other.vz += (impulse / other.mass) * nz

                # Separate spheres
                overlap = (self.radius + other.radius - distance) / 2
                self.x -= nx * overlap
                self.y -= ny * overlap
                self.z -= nz * overlap
                other.x += nx * overlap
                other.y += ny * overlap
                other.z += nz * overlap

    def is_expired(self, current_time: float) -> bool:
        """Check if sphere has expired."""
        return current_time - self.birth_time > self.lifetime

    def get_current_radius(self, current_time: float) -> float:
        """Get current radius with fade effects."""
        age = current_time - self.birth_time
        if age < self.FADE_IN_OUT_TIME:
            return self.radius * age / self.FADE_IN_OUT_TIME
        elif age > self.lifetime - self.FADE_IN_OUT_TIME:
            return self.radius * (self.lifetime - age) / self.FADE_IN_OUT_TIME
        return self.radius


class SphereAnimation(MenuAnimation):
    """A bouncing spheres animation for the menu screen."""

    RENDER_FADE_MARGIN = 0.2  # Fade margin for sphere edges

    def __init__(self, width: int, height: int, length: int):
        super().__init__(width, height, length)
        self.spheres: List[Sphere] = []
        self.next_spawn = 0.0
        self.spawn_interval = 2.0  # Base spawn interval
        self.bounds = (width, height, length)

    def spawn_sphere(self, current_time: float) -> Sphere:
        """Create a new sphere."""
        width, height, length = self.bounds
        radius = random.uniform(1.5, 3.0)

        # Random position
        x = random.uniform(radius, width - radius)
        y = random.uniform(height / 2, height - radius)  # Start in upper half
        z = random.uniform(radius, length - radius)

        # Random velocity
        speed = random.uniform(8.0, 32.0)
        angle = random.uniform(0, 2 * math.pi)
        vx = speed * math.cos(angle)
        vy = random.uniform(8, 32.0)
        vz = speed * math.sin(angle)

        # Random color and mass
        color = RGB.from_hsv(HSV(random.randint(0, 255), 255, 255))
        mass = radius**3

        return Sphere(
            x=x,
            y=y,
            z=z,
            vx=vx,
            vy=vy,
            vz=vz,
            radius=radius,
            birth_time=current_time,
            lifetime=random.uniform(5.0, 30.0),
            color=color,
            mass=mass,
        )

    def render(self, raster: Raster):
        """Render the bouncing spheres animation."""
        # Clear the raster
        raster.data.fill(0)

        current_time = self.last_update_time
        dt = 1.0 / 60.0  # Fixed timestep for stable physics

        # Spawn new spheres
        spawn_interval = self.spawn_interval / (1 + self.state.input_intensity)
        if current_time >= self.next_spawn:
            self.spheres.append(self.spawn_sphere(current_time))
            self.next_spawn = current_time + spawn_interval

        # Update physics
        for sphere in self.spheres:
            sphere.update(dt, self.bounds)

        # Handle collisions
        for i, sphere1 in enumerate(self.spheres):
            for sphere2 in self.spheres[i + 1 :]:
                sphere1.collide_with(sphere2)

        # Remove expired spheres
        self.spheres = [s for s in self.spheres if not s.is_expired(current_time)]

        # Render spheres
        for sphere in self.spheres:
            current_radius = sphere.get_current_radius(current_time)
            if current_radius <= 0.1:
                continue

            # Determine bounding box
            x_min = max(0, int(sphere.x - current_radius))
            x_max = min(self.width - 1, int(sphere.x + current_radius))
            y_min = max(0, int(sphere.y - current_radius))
            y_max = min(self.height - 1, int(sphere.y + current_radius))
            z_min = max(0, int(sphere.z - current_radius))
            z_max = min(self.length - 1, int(sphere.z + current_radius))

            # Render sphere
            for z in range(z_min, z_max + 1):
                for y in range(y_min, y_max + 1):
                    for x in range(x_min, x_max + 1):
                        # Calculate distance to sphere center
                        dx = x - sphere.x
                        dy = y - sphere.y
                        dz = z - sphere.z
                        distance = math.sqrt(dx * dx + dy * dy + dz * dz)

                        # Check if voxel is inside sphere
                        if distance <= current_radius:
                            # Calculate intensity with edge fade
                            intensity = 1.0
                            fade_start = current_radius * (1.0 - self.RENDER_FADE_MARGIN)
                            if distance > fade_start:
                                intensity = 1.0 - (distance - fade_start) / (
                                    current_radius * self.RENDER_FADE_MARGIN
                                )

                            # Adjust intensity based on voting state
                            if self.state.active_players:
                                vote_percentage = len(self.state.voted_players) / len(
                                    self.state.active_players
                                )
                                intensity *= 0.5 + vote_percentage * 0.5

                            # Apply color with intensity
                            new_r = int(sphere.color.red * intensity)
                            new_g = int(sphere.color.green * intensity)
                            new_b = int(sphere.color.blue * intensity)

                            # Use maximum blending
                            raster.data[z, y, x, 0] = max(raster.data[z, y, x, 0], new_r)
                            raster.data[z, y, x, 1] = max(raster.data[z, y, x, 1], new_g)
                            raster.data[z, y, x, 2] = max(raster.data[z, y, x, 2], new_b)
