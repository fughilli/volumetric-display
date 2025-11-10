import dataclasses
import importlib.util
import os.path
import socket
import struct
import sys
from abc import ABC, abstractmethod
from dataclasses import field

import numpy as np


@dataclasses.dataclass
class DisplayProperties:
    """A structured way to hold the global properties of the display."""

    width: int
    height: int
    length: int


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
    A NumPy-backed raster for high-performance operations.
    Data is stored in (L, H, W, C) -> (z, y, x, rgb) order.
    """

    width: int
    height: int
    length: int
    brightness: float = 1.0
    orientation: list[str] = field(default_factory=lambda: ["X", "Y", "Z"])
    data: np.ndarray = field(init=False, repr=False)
    transform: list = field(init=False, repr=False)

    def __post_init__(self):
        """This special method is called automatically after the object is created."""
        # Initialize the data buffer now that we have width, height, and length
        self.data = np.zeros((self.length, self.height, self.width, 3), dtype=np.uint8)
        # Run the setup for coordinate transformation
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

    def get_pix(self, x, y, z):
        """
        Get a pixel color with coordinate transformation.

        Args:
            x, y, z: Original coordinates

        Returns:
            The RGB color at the specified coordinates.
        """
        assert x >= 0 and x < self.width, f"x: {x} width: {self.width}"
        assert y >= 0 and y < self.height, f"y: {y} height: {self.height}"
        assert z >= 0 and z < self.length, f"z: {z} length: {self.length}"

        # Transform coordinates to match how they are set
        tx, ty, tz = self._transform_coords(x, y, z)

        # Calculate index in the data array
        r, g, b = self.data[tz, ty, tx]
        return RGB(r, g, b)

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
        self.data[tz, ty, tx] = [color.red, color.green, color.blue]

    def clear(self):
        """
        Clear the raster.
        """
        self.data.fill(0)


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


def load_scene(path: str, **kwargs) -> Scene:
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
        return scene_classes[0](**kwargs)

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
            universes_per_layer=3,  # No longer used, but kept for compatibility
            channel_span=1,  # No longer used, but kept for compatibility
            z_indices=None,
        ):
            """Sends the raster data via ArtNet using NumPy for high performance."""
            if z_indices is None:
                z_indices = range(raster.length)

            current_universe = base_universe
            # Process one Z-layer at a time
            for z in z_indices:
                # Get the Z-layer directly from the NumPy array
                # The data is already in [y, x, (r,g,b)] order for a given Z
                layer_data = raster.data[z]

                # Apply brightness and flatten the (height, width, 3) layer into a 1D byte array
                pixel_bytes = (
                    (layer_data * raster.brightness).clip(0, 255).astype(np.uint8).tobytes()
                )

                # Send the layer data in chunks of 510 bytes (170 pixels)
                for i in range(0, len(pixel_bytes), channels_per_universe):
                    chunk = pixel_bytes[i : i + channels_per_universe]
                    if not chunk:  # Don't send empty packets
                        continue
                    dmx_packet = self.create_dmx_packet(current_universe, chunk)
                    self.sock.sendto(dmx_packet, (self.ip, self.port))
                    current_universe += 1

            # Send a sync packet after all data for this controller is sent
            sync_packet = self.create_sync_packet()
            self.sock.sendto(sync_packet, (self.ip, self.port))

        def send_dmx_scatter_gather(
            self,
            world_raster,
            universe,
            coords,
            channels_per_universe=510,
        ):
            """
            Send ArtNet DMX data using scatter gather mode.
            Samples from world_raster at the specified coordinates.

            Args:
                world_raster: The world raster to sample from
                universe: Starting universe number for this strand
                coords: List of [x, y, z] coordinate tuples to sample from
                channels_per_universe: Number of DMX channels per universe (default 510)
            """
            import numpy as np

            # Sample pixels from world_raster at the specified coordinates
            data_bytes = bytearray()
            for coord in coords:
                x, y, z = coord
                # Ensure coordinates are within bounds
                if (
                    0 <= x < world_raster.width
                    and 0 <= y < world_raster.height
                    and 0 <= z < world_raster.length
                ):
                    # Sample from world_raster (NumPy array format: [z, y, x, 3])
                    pixel = world_raster.data[z, y, x]
                    # Apply brightness and convert to bytes
                    r = int(np.clip(pixel[0] * world_raster.brightness, 0, 255))
                    g = int(np.clip(pixel[1] * world_raster.brightness, 0, 255))
                    b = int(np.clip(pixel[2] * world_raster.brightness, 0, 255))
                    data_bytes.extend([r, g, b])
                else:
                    # Out of bounds - send black
                    data_bytes.extend([0, 0, 0])

            # Send data in chunks of channels_per_universe bytes
            current_universe = universe
            for i in range(0, len(data_bytes), channels_per_universe):
                chunk = bytes(data_bytes[i : i + channels_per_universe])
                if not chunk:  # Don't send empty packets
                    continue
                dmx_packet = self.create_dmx_packet(current_universe, chunk)
                self.sock.sendto(dmx_packet, (self.ip, self.port))
                current_universe += 1

            # Send a sync packet after all data for this strand is sent
            sync_packet = self.create_sync_packet()
            self.sock.sendto(sync_packet, (self.ip, self.port))
