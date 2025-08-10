import math
import random
from dataclasses import dataclass
from typing import List

from artnet import HSV, RGB, Raster, Scene


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

    # Physics constants
    GRAVITY = 20.0  # Gravity acceleration
    ELASTICITY = 0.95  # Bounce elasticity (1.0 = perfect bounce)
    AIR_DAMPING = 0.999  # Air resistance (velocity multiplier per update)
    GROUND_FRICTION = 0.95  # Additional friction when touching ground
    MINIMUM_SPEED = 0.01  # Speed below which we stop movement
    FADE_IN_OUT_TIME = 0.2  # Time to fade in and out

    def update(self, dt: float, bounds: tuple[float, float, float]):
        # Apply gravity
        self.vy -= self.GRAVITY * dt

        # Apply air resistance
        self.vx *= self.AIR_DAMPING
        self.vy *= self.AIR_DAMPING
        self.vz *= self.AIR_DAMPING

        # Apply additional ground friction when touching bottom
        if self.y - self.radius <= 0:
            self.vx *= self.GROUND_FRICTION
            self.vz *= self.GROUND_FRICTION

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
        return current_time - self.birth_time > self.lifetime

    def get_current_radius(self, current_time: float) -> float:
        age = current_time - self.birth_time
        if age < self.FADE_IN_OUT_TIME:
            # Growing at the beginning
            return self.radius * age / self.FADE_IN_OUT_TIME
        elif age > self.lifetime - self.FADE_IN_OUT_TIME:
            # Shrinking at the end
            return self.radius * (self.lifetime - age) / self.FADE_IN_OUT_TIME
        return self.radius


class BouncingSphereScene(Scene):

    RENDER_FADE_MARGIN = 0.2  # Fade out near the edge

    def __init__(self, **kwargs):
        self.spheres: List[Sphere] = []
        self.next_spawn = 0.0
        self.spawn_interval = 2.0  # New sphere every 2 seconds

    def spawn_sphere(self, time: float, bounds: tuple[float, float, float]) -> Sphere:
        width, height, length = bounds
        radius = random.uniform(1.5, 3.0)

        # Random position (keeping sphere inside bounds)
        x = random.uniform(radius, width - radius)
        y = random.uniform(height / 2, height - radius)  # Start in upper half
        z = random.uniform(radius, length - radius)

        # Random initial velocity
        speed = random.uniform(8.0, 32.0)
        angle = random.uniform(0, 2 * math.pi)

        vx = speed * math.cos(angle)
        vy = random.uniform(8, 32.0)  # Initial upward velocity
        vz = speed * math.sin(angle)

        # Random color
        color = RGB.from_hsv(HSV(random.randint(0, 255), 255, 255))

        mass = radius**3  # Mass scales by the radius cubed

        return Sphere(
            x=x,
            y=y,
            z=z,
            vx=vx,
            vy=vy,
            vz=vz,
            radius=radius,
            birth_time=time,
            lifetime=random.uniform(5.0, 30.0),
            color=color,
            mass=mass,
        )

    def render(self, raster: Raster, time: float):
        # Clear the raster
        for i in range(len(raster.data)):
            raster.data[i] = RGB(0, 0, 0)

        # Spawn new spheres
        if time >= self.next_spawn:
            self.spheres.append(
                self.spawn_sphere(time, (raster.width, raster.height, raster.length))
            )
            self.next_spawn = time + self.spawn_interval

        # Update and render spheres
        bounds = (raster.width, raster.height, raster.length)
        new_spheres = []
        dt = 0.01  # Smaller timestep for better physics

        # Update physics
        for sphere in self.spheres:
            if not sphere.is_expired(time):
                sphere.update(dt, bounds)

                # Check collisions with other spheres
                for other in self.spheres:
                    if sphere != other and not other.is_expired(time):
                        sphere.collide_with(other)

                        # Get current radius (for growing and shrinking effect)
                        current_radius = sphere.get_current_radius(time)

                        # Render sphere
                        for x in range(
                            max(0, int(sphere.x - current_radius)),
                            min(raster.width, int(sphere.x + current_radius + 1)),
                        ):
                            for y in range(
                                max(0, int(sphere.y - current_radius)),
                                min(raster.height, int(sphere.y + current_radius + 1)),
                            ):
                                for z in range(
                                    max(0, int(sphere.z - current_radius)),
                                    min(
                                        raster.length,
                                        int(sphere.z + current_radius + 1),
                                    ),
                                ):
                                    # Calculate distance from sphere center
                                    dx = x - sphere.x
                                    dy = y - sphere.y
                                    dz = z - sphere.z
                                    distance = math.sqrt(dx * dx + dy * dy + dz * dz)

                                    if distance <= current_radius:
                                        # Calculate intensity based on distance from center

                                        if distance < current_radius * (
                                            1 - self.RENDER_FADE_MARGIN
                                        ):
                                            intensity = 1.0
                                        else:
                                            intensity = 1.0 - (
                                                distance
                                                - current_radius * (1 - self.RENDER_FADE_MARGIN)
                                            ) / (current_radius * self.RENDER_FADE_MARGIN)
                                        idx = (
                                            y * raster.width + x + z * raster.width * raster.height
                                        )

                                        # Blend with existing color
                                        existing = raster.data[idx]
                                        raster.data[idx] = RGB(
                                            max(
                                                existing.red,
                                                int(sphere.color.red * intensity),
                                            ),
                                            max(
                                                existing.green,
                                                int(sphere.color.green * intensity),
                                            ),
                                            max(
                                                existing.blue,
                                                int(sphere.color.blue * intensity),
                                            ),
                                        )

                new_spheres.append(sphere)

        self.spheres = new_spheres
