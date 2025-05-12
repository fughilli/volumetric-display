from abc import ABC, abstractmethod
import dataclasses
import importlib.util
import os.path
import socket
import struct
import time
import sys


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

        return RGB(saturate_u8((r + m) * 255), saturate_u8((g + m) * 255),
                   saturate_u8((b + m) * 255))


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

    def __init__(self, width, height, length):
        self.width = width
        self.height = height
        self.length = length
        self.brightness = 1.0
        self.data = [RGB(0, 0, 0) for _ in range((width * height * length))]


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


def load_scene(path: str) -> Scene:
    """
    Load a scene plugin from a Python file

    Args:
        path: Path to the Python file containing the scene

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

    # Load module
    spec = importlib.util.spec_from_file_location("scene_module", path)
    if not spec or not spec.loader:
        # Restore sys.path before raising error
        sys.path = original_sys_path
        raise ImportError(f"Could not load scene from {path}")

    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
    finally:
        # Ensure original sys.path is restored
        sys.path = original_sys_path

    # Find Scene subclass
    scene_classes = [
        cls for cls in module.__dict__.values()
        if isinstance(cls, type) and issubclass(cls, Scene) and cls != Scene
    ]

    if not scene_classes:
        raise ValueError(f"No Scene subclass found in {path}")
    if len(scene_classes) > 1:
        raise ValueError(f"Multiple Scene subclasses found in {path}")

    # Create and return instance
    return scene_classes[0]()


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
        packet.extend(b'Art-Net\x00')  # Protocol header
        packet.extend(struct.pack('<H', 0x5000))  # OpCode for DMX (0x5000)
        packet.extend(struct.pack('!H',
                                  14))  # Protocol version (0x000E, big-endian)
        packet.extend(struct.pack('B', 0))  # Sequence (0 = disabled)
        packet.extend(struct.pack('B', 0))  # Physical port (0 = ignored)
        packet.extend(struct.pack('<H', universe))  # Universe (little endian)
        packet.extend(struct.pack(
            '!H', len(data)))  # Length of DMX data (big endian)

        # DMX Data
        packet.extend(data)  # Append DMX data

        return packet

    def create_sync_packet(self):
        """
        Manually construct an ArtNet Sync packet.
        """
        packet = bytearray()

        # ArtNet Header
        packet.extend(b'Art-Net\x00')  # Protocol header
        packet.extend(struct.pack('<H', 0x5200))  # OpCode for Sync (0x5200)
        packet.extend(struct.pack('!H',
                                  14))  # Protocol version (0x000E, big-endian)
        packet.extend(struct.pack('B', 0))  # Sequence (ignored)
        packet.extend(struct.pack('B', 0))  # Physical port (ignored)

        return packet

    def send_dmx(self,
                 base_universe,
                 raster,
                 channels_per_universe=510,
                 universes_per_layer=3,
                 channel_span=1,
                 z_indices=None):
        """
        Send the ArtNet DMX packet via UDP.
        """
        # Send DMX Data Packet
        data = bytearray()
        if z_indices is None:
            z_indices = range(0, raster.length, channel_span)

        for out_z, z in enumerate(z_indices):
            universe = (out_z //
                        channel_span) * universes_per_layer + base_universe
            layer = raster.data[z * raster.width * raster.height:(z + 1) *
                                raster.width * raster.height]

            for rgb in layer:
                data.extend(
                    struct.pack('B', saturate_u8(rgb.red * raster.brightness)))
                data.extend(
                    struct.pack('B',
                                saturate_u8(rgb.green * raster.brightness)))
                data.extend(
                    struct.pack('B',
                                saturate_u8(rgb.blue * raster.brightness)))

            while len(data) > 0:
                dmx_packet = self.create_dmx_packet(
                    universe, data[:channels_per_universe])
                self.sock.sendto(dmx_packet, (self.ip, self.port))
                data = data[channels_per_universe:]
                universe += 1

        # Send Sync Packet
        sync_packet = self.create_sync_packet()
        self.sock.sendto(sync_packet, (self.ip, self.port))
