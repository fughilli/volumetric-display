import abc
import time
from dataclasses import dataclass
from typing import Set

from artnet import Raster


@dataclass
class AnimationState:
    """State information for menu animations."""

    active_players: Set[int]  # Set of active player IDs
    voted_players: Set[int]  # Set of players who have voted
    last_input_time: float  # Time of last input event
    input_intensity: float  # Current input intensity (decays over time)

    @classmethod
    def create_empty(cls) -> "AnimationState":
        """Create a new empty animation state."""
        return cls(
            active_players=set(),
            voted_players=set(),
            last_input_time=0.0,
            input_intensity=0.0,
        )


class MenuAnimation(abc.ABC):
    """Base class for menu animations."""

    def __init__(self, width: int, height: int, length: int):
        """Initialize the animation.

        Args:
            width: Display width in pixels
            height: Display height in pixels
            length: Display length in pixels
        """
        self.width = width
        self.height = height
        self.length = length
        self.state = AnimationState.create_empty()
        self.last_update_time = time.monotonic()

    def update_state(self, active_players: Set[int], voted_players: Set[int]):
        """Update the animation state with current player information.

        Args:
            active_players: Set of currently active player IDs
            voted_players: Set of player IDs that have voted
        """
        current_time = time.monotonic()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time

        # Update player sets
        self.state.active_players = active_players
        self.state.voted_players = voted_players

        # Decay input intensity over time
        decay_rate = 2.0  # Input effect decays in 0.5 seconds
        self.state.input_intensity = max(0.0, self.state.input_intensity - decay_rate * dt)

    def handle_input_event(self):
        """Handle an input event from any player."""
        self.state.last_input_time = time.monotonic()
        self.state.input_intensity = 1.0  # Full intensity on input

    @abc.abstractmethod
    def render(self, raster: Raster):
        """Render the current animation frame.

        Args:
            raster: The volumetric raster to render to
        """
        pass
