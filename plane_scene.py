from artnet import Scene, RGB, HSV
import random
import math
import numpy


class Plane:

    def __init__(self, dimensions):
        # Compute the diagonal of the raster
        raster_size = math.sqrt(sum(d**2 for d in dimensions))
        self.position = -(raster_size / 2 + 1)
        self.end_position = -self.position
        self.normal = [random.uniform(-1, 1) for _ in range(3)]
        norm = math.sqrt(sum(n**2 for n in self.normal))
        self.normal = [n / norm
                       for n in self.normal]  # Normalize the normal vector
        self.velocity = random.uniform(0.05, 0.2) * 10
        self.color = RGB.from_hsv(HSV(random.randint(0, 255), 255, 255))

    def update(self, delta_time):
        self.position += self.velocity * delta_time

    def is_alive(self):
        return self.position < self.end_position

    def distance_to_point(self, point):
        plane_point = [self.position * n for n in self.normal]
        return distance_to_plane(plane_point, self.normal, point)


def distance_to_plane(plane_point, plane_normal, point):
    return abs(
        numpy.dot(plane_normal, [point[i] - plane_point[i] for i in range(3)]))


class PlaneScene(Scene):

    def __init__(self, **kwargs):
        super().__init__()
        self.dimensions = (0, 0, 0)
        self.planes = []

    def spawn_plane(self):
        self.planes.append(Plane(self.dimensions))

    def update_planes(self, delta_time):
        for plane in self.planes:
            plane.update(delta_time)
        self.planes = [plane for plane in self.planes if plane.is_alive()]

    def render(self, raster, time):
        self.dimensions = (raster.width, raster.height, raster.length)
        if random.random() < 0.1 and len(self.planes) < 3:  # Randomly spawn new planes
            self.spawn_plane()
        self.update_planes(
            1)  # Update planes with a fixed delta time of 1 for simplicity

        for x in range(raster.width):
            for y in range(raster.height):
                for z in range(raster.length):
                    point = [
                        x - raster.width / 2, y - raster.height / 2,
                        z - raster.length / 2
                    ]
                    colors = []
                    distances = []
                    for plane in self.planes:
                        distance = plane.distance_to_point(point)
                        if distance < 0.5:  # Consider planes that are close enough
                            colors.append(plane.color)
                            distances.append(distance)
                    if colors:
                        total_distance = sum(1 / d for d in distances)
                        interpolated_color = RGB(0, 0, 0)
                        for color, distance in zip(colors, distances):
                            weight = (1 / distance) / total_distance
                            interpolated_color.red += color.red * weight
                            interpolated_color.green += color.green * weight
                            interpolated_color.blue += color.blue * weight
                        raster.set_pix(x, y, z, interpolated_color)
                    else:
                        raster.set_pix(x, y, z, RGB(0, 0, 0))
