import random
import time
from typing import List, Optional, Type

from artnet import Raster
from games.util.menu_animations.base import MenuAnimation


class MenuAnimationManager:
    """Manages menu animations with automatic transitions."""

    def __init__(self, width: int, height: int, length: int):
        """Initialize the animation manager.

        Args:
            width: Display width in pixels
            height: Display height in pixels
            length: Display length in pixels
        """
        self.width = width
        self.height = height
        self.length = length

        # Animation state
        self.available_animations: List[Type[MenuAnimation]] = []
        self.current_animation: Optional[MenuAnimation] = None
        self.last_transition_time = time.monotonic()
        self.transition_interval = 300  # 5 minutes in seconds

    def register_animation(self, animation_class: Type[MenuAnimation]):
        """Register a new animation type.

        Args:
            animation_class: The animation class to register
        """
        self.available_animations.append(animation_class)

        # If this is our first animation, use it
        if not self.current_animation:
            self.current_animation = animation_class(self.width, self.height, self.length)

    def select_random_animation(self) -> None:
        """Select a random animation from available ones, different from current."""
        if not self.available_animations:
            return

        # Get all animations except current one
        available = [
            a
            for a in self.available_animations
            if not self.current_animation or not isinstance(self.current_animation, a)
        ]

        # If no other animations, use current type
        if not available:
            available = self.available_animations

        # Select random animation
        animation_class = random.choice(available)
        self.current_animation = animation_class(self.width, self.height, self.length)
        self.last_transition_time = time.monotonic()

    def check_transition(self) -> None:
        """Check if it's time to transition to a new animation."""
        current_time = time.monotonic()
        if current_time - self.last_transition_time >= self.transition_interval:
            self.select_random_animation()

    def update_state(self, active_players: set[int], voted_players: set[int]) -> None:
        """Update the current animation state.

        Args:
            active_players: Set of currently active player IDs
            voted_players: Set of player IDs that have voted
        """
        if self.current_animation:
            self.current_animation.update_state(active_players, voted_players)

    def handle_input_event(self) -> None:
        """Handle an input event."""
        if self.current_animation:
            self.current_animation.handle_input_event()

    def render(self, raster: Raster) -> None:
        """Render the current animation.

        Args:
            raster: The volumetric raster to render to
        """
        self.check_transition()
        if self.current_animation:
            self.current_animation.render(raster)
