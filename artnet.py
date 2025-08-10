import dataclasses
import importlib.util
import os.path
import socket
import struct
import sys
from abc import ABC, abstractmethod


@dataclasses.dataclass
class RGB:
    """
    Simple RGB color data class.
    """

    red: int
    green: int
    blue: int

    def from_hsv(hsv):
        """
        Convert an HSV color to RGB.
        """
        h = hsv.hue / (256 / 6)
        s = hsv.saturation / 255
        v = hsv.value / 255

        c = v * s
        x = c * (1 - abs(h % 2 - 1))
        m = v - c

        if h < 1:
            r, g, b = c, x, 0
        elif h < 2:
            r, g, b = x, c, 0
        elif h < 3:
            r, g, b = 0, c, x
        elif h < 4:
            r, g, b = 0, x, c
        elif h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        return RGB(
            saturate_u8((r + m) * 255),
            saturate_u8((g + m) * 255),
            saturate_u8((b + m) * 255),
        )


@dataclasses.dataclass
class HSV:
    """
    Simple HSV color data class.
    """

    hue: int
    saturation: int
    value: int


@dataclasses.dataclass
class Raster:
    """
    Simple raster data class.
    """

    width: int
    height: int
    length: int
    brightness: float
    data: list[RGB]
    orientation: list[str]

    def __init__(self, width, height, length, orientation=None):
        self.width = width
        self.height = height
        self.length = length
        self.brightness = 1.0
        self.data = [RGB(0, 0, 0) for _ in range((width * height * length))]
        self.orientation = orientation or ["X", "Y", "Z"]
        self._compute_transform()

    def _compute_transform(self):
        """Compute the transformation matrix for coordinate mapping."""
        self.transform = []
        for coord in self.orientation:
            axis = coord[-1]  # Get the axis (X, Y, or Z)
            sign = -1 if coord.startswith("-") else 1
            if axis == "X":
                self.transform.append((0, sign))
            elif axis == "Y":
                self.transform.append((1, sign))
            else:  # Z
                self.transform.append((2, sign))

    def _transform_coords(self, x, y, z):
        """Transform coordinates according to the orientation configuration."""
        coords = [x, y, z]
        result = [0, 0, 0]
        for i, (axis, sign) in enumerate(self.transform):
            if sign == 1:
                result[i] = coords[axis]
            else:  # sign == -1
                # For negative axes, subtract from the maximum value
                if axis == 0:  # X
                    result[i] = self.width - 1 - coords[axis]
                elif axis == 1:  # Y
                    result[i] = self.height - 1 - coords[axis]
                else:  # Z
                    result[i] = self.length - 1 - coords[axis]
        return tuple(result)

    def set_pix(self, x, y, z, color):
        """
        Set a pixel color with coordinate transformation.

        Args:
            x, y, z: Original coordinates
            color: RGB color to set
        """
        assert x >= 0 and x < self.width, f"x: {x} width: {self.width}"
        assert y >= 0 and y < self.height, f"y: {y} height: {self.height}"
        assert z >= 0 and z < self.length, f"z: {z} length: {self.length}"

        # Transform coordinates
        tx, ty, tz = self._transform_coords(x, y, z)
        # Calculate index in the data array
        idx = ty * self.width + tx + tz * self.width * self.height
        self.data[idx] = color

    def clear(self):
        """
        Clear the raster.
        """
        self.data = [RGB(0, 0, 0) for _ in range((self.width * self.height * self.length))]


def saturate_u8(value):
    """
    Saturate a value to the range [0, 255].
    """
    return int(max(0, min(value, 255)))


class Scene(ABC):
    """Base class for scene plugins"""

    @abstractmethod
    def render(self, raster: Raster, time: float) -> None:
        """
        Update the raster for the current frame

        Args:
            raster: The Raster object to update
            time: Current time in seconds
        """
        pass


def load_scene(path: str, config=None) -> Scene:
    """
    Load a scene plugin from a Python file

    Args:
        path: Path to the Python file containing the scene
        config: Optional configuration to pass to the scene constructor

    Returns:
        An instance of the scene class

    Raises:
        ImportError: If the scene cannot be loaded
        ValueError: If the scene doesn't contain exactly one Scene subclass
    """
    # Get absolute path
    path = os.path.abspath(path)
    scene_dir = os.path.dirname(path)

    # Store original sys.path and add scene directory
    original_sys_path = list(sys.path)
    if scene_dir not in sys.path:
        sys.path.insert(0, scene_dir)

    try:
        # Load module
        spec = importlib.util.spec_from_file_location("scene_module", path)
        if not spec or not spec.loader:
            raise ImportError(f"Could not load scene from {path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find Scene subclass
        scene_classes = [
            cls
            for cls in module.__dict__.values()
            if isinstance(cls, type) and issubclass(cls, Scene) and cls != Scene
        ]

        if not scene_classes:
            raise ValueError(f"No Scene subclass found in {path}")
        if len(scene_classes) > 1:
            raise ValueError(f"Multiple Scene subclasses found in {path}")

        # Create and return instance
        return scene_classes[0](config=config)

    finally:
        # Restore original sys.path
        sys.path = original_sys_path


try:
    from src.artnet.artnet_rs import ArtNetController

    print("Loaded Rust-based ArtNetController")
except ImportError:
    print("Falling back to Python-based ArtNetController")

    class ArtNetController:

        def __init__(self, ip, port):
            self.ip = ip
            self.port = port
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        def __del__(self):
            self.sock.close()

        def create_dmx_packet(self, universe, data):
            """
            Manually construct an ArtNet DMX packet.
            """
            packet = bytearray()

            # ArtNet Header
            packet.extend(b"Art-Net\x00")  # Protocol header
            packet.extend(struct.pack("<H", 0x5000))  # OpCode for DMX (0x5000)
            packet.extend(struct.pack("!H", 14))  # Protocol version (0x000E, big-endian)
            packet.extend(struct.pack("B", 0))  # Sequence (0 = disabled)
            packet.extend(struct.pack("B", 0))  # Physical port (0 = ignored)
            packet.extend(struct.pack("<H", universe))  # Universe (little endian)
            packet.extend(struct.pack("!H", len(data)))  # Length of DMX data (big endian)

            # DMX Data
            packet.extend(data)  # Append DMX data

            return packet

        def create_sync_packet(self):
            """
            Manually construct an ArtNet Sync packet.
            """
            packet = bytearray()

            # ArtNet Header
            packet.extend(b"Art-Net\x00")  # Protocol header
            packet.extend(struct.pack("<H", 0x5200))  # OpCode for Sync (0x5200)
            packet.extend(struct.pack("!H", 14))  # Protocol version (0x000E, big-endian)
            packet.extend(struct.pack("B", 0))  # Sequence (ignored)
            packet.extend(struct.pack("B", 0))  # Physical port (ignored)

            return packet

        def send_dmx(
            self,
            base_universe,
            raster,
            channels_per_universe=510,
            universes_per_layer=3,
            channel_span=1,
            z_indices=None,
        ):
            """
            Send the ArtNet DMX packet via UDP.
            """
            # Send DMX Data Packet
            data = bytearray()
            if z_indices is None:
                z_indices = range(0, raster.length, channel_span)

            for out_z, z in enumerate(z_indices):
                universe = (out_z // channel_span) * universes_per_layer + base_universe
                layer = raster.data[
                    z * raster.width * raster.height : (z + 1) * raster.width * raster.height
                ]

                for rgb in layer:
                    data.extend(struct.pack("B", saturate_u8(rgb.red * raster.brightness)))
                    data.extend(struct.pack("B", saturate_u8(rgb.green * raster.brightness)))
                    data.extend(struct.pack("B", saturate_u8(rgb.blue * raster.brightness)))

                while len(data) > 0:
                    dmx_packet = self.create_dmx_packet(universe, data[:channels_per_universe])
                    self.sock.sendto(dmx_packet, (self.ip, self.port))
                    data = data[channels_per_universe:]
                    universe += 1

                # Send Sync Packet
                sync_packet = self.create_sync_packet()
                self.sock.sendto(sync_packet, (self.ip, self.port))
