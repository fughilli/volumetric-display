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
        if distance < self.radius + other.radius and distance > 0:
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

    RENDER_FADE_MARGIN = 0.2

    def __init__(self, **kwargs):
        properties = kwargs.get("properties")
        if not properties:
            raise ValueError("BouncingSphereScene requires a 'properties' object.")

        self.spheres: List[Sphere] = []
        self.next_spawn = 0.0
        self.spawn_interval = 2.0
        self.width = properties.width
        self.height = properties.height
        self.length = properties.length
        self.bounds = (self.width, self.height, self.length)

        # For debug tracking
        self.last_debug_time = 0

        # Debug: Print initialization info
        print(
            f"BouncingSphereScene initialized with bounds: {self.width}x{self.height}x{self.length}"
        )

    def spawn_sphere(self, time: float) -> Sphere:
        """Spawns a new sphere within the scene's total bounds."""
        width, height, length = self.bounds
        radius = random.uniform(1.5, 3.0)

        # Random position (keeping sphere inside bounds)
        x = random.uniform(radius, width - radius)
        y = random.uniform(height / 2, height - radius)  # Start in upper half
        z = random.uniform(radius, length - radius)

        # Random initial velocity
        speed = random.uniform(8.0, 32.0)
        angle = random.uniform(0, 2 * math.pi)
        vx = speed * math.cos(angle)
        vy = random.uniform(8, 32.0)
        vz = speed * math.sin(angle)

        # Random color and mass
        color = RGB.from_hsv(HSV(random.randint(0, 255), 255, 255))
        mass = radius**3

        # Debug: Print spawn info with color
        # print(f"Spawning sphere at ({x:.1f}, {y:.1f}, {z:.1f})
        # radius={radius:.1f} color=RGB({color.red},{color.green},{color.blue})")

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
        """
        Updates physics and renders spheres using high-performance NumPy operations.
        """
        dt = 1.0 / 60.0  # Use a fixed timestep for stable physics

        # Debug raster info on first frame
        # if time < dt:
        #    print(f"Raster shape: {raster.data.shape}")
        #    print(f"Raster dtype: {raster.data.dtype}")
        #    print(f"Raster dimensions: width={raster.width}, height={raster.height}, length={raster.length}")

        # --- Fast Raster Clearing ---
        raster.data.fill(0)

        # --- Spawning & Physics ---
        if time >= self.next_spawn:
            self.spheres.append(self.spawn_sphere(time))
            self.next_spawn = time + self.spawn_interval

        for sphere in self.spheres:
            sphere.update(dt, self.bounds)

        for i, sphere1 in enumerate(self.spheres):
            for sphere2 in self.spheres[i + 1 :]:
                sphere1.collide_with(sphere2)

        self.spheres = [s for s in self.spheres if not s.is_expired(time)]

        # Track total lit voxels for debugging
        total_lit_voxels = 0
        spheres_rendered = 0

        # --- Simple per-voxel rendering (avoiding complex numpy indexing) ---
        for sphere in self.spheres:
            current_radius = sphere.get_current_radius(time)
            if current_radius <= 0.1:
                continue

            spheres_rendered += 1

            # Determine bounding box for this sphere
            x_min = max(0, int(sphere.x - current_radius))
            x_max = min(raster.width - 1, int(sphere.x + current_radius))
            y_min = max(0, int(sphere.y - current_radius))
            y_max = min(raster.height - 1, int(sphere.y + current_radius))
            z_min = max(0, int(sphere.z - current_radius))
            z_max = min(raster.length - 1, int(sphere.z + current_radius))

            # Debug first sphere's bounds
            # if spheres_rendered == 1 and time - self.last_debug_time > 2.0:
            #    print(f"  Sphere bounds: x[{x_min},{x_max}] y[{y_min},{y_max}] z[{z_min},{z_max}]")

            # Iterate through the bounding box
            for z in range(z_min, z_max + 1):
                for y in range(y_min, y_max + 1):
                    for x in range(x_min, x_max + 1):
                        # Calculate distance from voxel to sphere center
                        dx = x - sphere.x
                        dy = y - sphere.y
                        dz = z - sphere.z
                        distance = math.sqrt(dx * dx + dy * dy + dz * dz)

                        # Check if voxel is inside sphere
                        if distance <= current_radius:
                            # Calculate intensity (fade at edges)
                            intensity = 1.0
                            fade_start = current_radius * (1.0 - self.RENDER_FADE_MARGIN)
                            if distance > fade_start:
                                intensity = 1.0 - (distance - fade_start) / (
                                    current_radius * self.RENDER_FADE_MARGIN
                                )

                            # Apply color with intensity
                            new_r = int(sphere.color.red * intensity)
                            new_g = int(sphere.color.green * intensity)
                            new_b = int(sphere.color.blue * intensity)

                            # Use maximum blending
                            raster.data[z, y, x, 0] = max(raster.data[z, y, x, 0], new_r)
                            raster.data[z, y, x, 1] = max(raster.data[z, y, x, 1], new_g)
                            raster.data[z, y, x, 2] = max(raster.data[z, y, x, 2], new_b)

                            total_lit_voxels += 1

        # Debug output every 2 seconds
        """
        if time - self.last_debug_time > 2.0:
            # Check actual raster data
            non_zero_voxels = np.sum(np.any(raster.data > 0, axis=-1))
            max_brightness = np.max(raster.data)
            mean_brightness = np.mean(raster.data[raster.data > 0]) if non_zero_voxels > 0 else 0

            print(f"[{time:.1f}s] Spheres: {len(self.spheres)} | Rendered: {spheres_rendered} | "
                  f"Lit voxels: {non_zero_voxels}/{raster.width*raster.height*raster.length} | "
                  f"Max brightness: {max_brightness} | Mean brightness: {mean_brightness:.1f}")

            # Sample a few lit voxels to see actual values
            if non_zero_voxels > 0:
                lit_positions = np.argwhere(np.any(raster.data > 0, axis=-1))
                if len(lit_positions) > 0:
                    sample_idx = min(3, len(lit_positions))
                    for i in range(sample_idx):
                        z, y, x = lit_positions[i]
                        color = raster.data[z, y, x]
                        print(f"  Sample voxel at ({x},{y},{z}): RGB({color[0]},{color[1]},{color[2]})")

            self.last_debug_time = time
        """
