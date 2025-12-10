"""
Sound Manager for Game Audio
Plays pre-rendered WAV sound files using pygame.mixer
"""

import os
import threading
from enum import Enum
from pathlib import Path
from typing import Optional

try:
    import pygame.mixer

    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: pygame not available. Sound will be disabled.")


class SoundEffect(Enum):
    """Enum for different sound effects"""

    CLICK = "click"
    POP = "pop"
    BEEP = "beep"
    CRUNCH = "crunch"
    WARBLE = "warble"
    WOOSH = "woosh"
    DING = "ding"
    THUD = "thud"


class SoundManager:
    """Manages sound effects for games and menus using pre-rendered MP3 files"""

    def __init__(self, enabled: bool = True):
        """
        Initialize the sound manager

        Args:
            enabled: Whether sound is enabled (default: True)
        """
        self.enabled = enabled and PYGAME_AVAILABLE
        self._lock = threading.Lock()
        self._sound_files: dict[SoundEffect, Optional[pygame.mixer.Sound]] = {}
        self._mixer_initialized = False

        if self.enabled:
            try:
                # Initialize pygame mixer if not already initialized
                if not pygame.mixer.get_init():
                    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
                    self._mixer_initialized = True
                else:
                    self._mixer_initialized = False  # Already initialized elsewhere

                # Load sound files
                self._load_sound_files()
                print("SoundManager: Initialized with WAV sound files")
            except Exception as e:
                print(f"SoundManager: Failed to initialize sound system: {e}")
                self.enabled = False

    def _load_sound_files(self):
        """Load all sound effect MP3 files from runfiles"""
        try:
            from rules_python.python.runfiles import runfiles

            r = runfiles.Create()
        except ImportError:
            # Fallback: try to find sounds directory relative to this file
            r = None

        for sound_effect in SoundEffect:
            sound_path = None

            # Try to find WAV file in runfiles
            if r:
                sound_path = r.Rlocation(f"sounds/{sound_effect.value}.wav")

            # Fallback: look relative to current file
            if not sound_path or not os.path.exists(sound_path):
                # Try to find sounds directory
                current_file = Path(__file__)
                # Go up from games/util/ to project root, then to sounds/
                possible_paths = [
                    current_file.parent.parent.parent / "sounds" / f"{sound_effect.value}.wav",
                    current_file.parent.parent.parent.parent
                    / "sounds"
                    / f"{sound_effect.value}.wav",
                ]
                for path in possible_paths:
                    if path.exists():
                        sound_path = str(path)
                        break

            if sound_path and os.path.exists(sound_path):
                try:
                    self._sound_files[sound_effect] = pygame.mixer.Sound(sound_path)
                except Exception as e:
                    print(f"SoundManager: Failed to load {sound_effect.value}.wav: {e}")
                    self._sound_files[sound_effect] = None
            else:
                print(f"SoundManager: Sound file not found: {sound_effect.value}.wav")
                self._sound_files[sound_effect] = None

    def play_sound(self, sound_effect: SoundEffect) -> None:
        """
        Play a sound effect

        Args:
            sound_effect: The sound effect to play
        """
        if not self.enabled:
            return

        sound = self._sound_files.get(sound_effect)
        if sound is None:
            return

        try:
            with self._lock:
                channel = sound.play()
                # Wait a tiny bit to ensure playback starts
                import time

                time.sleep(0.01)
                # Keep a reference to prevent garbage collection
                if channel:
                    # Store channel reference to prevent it from being garbage collected
                    if not hasattr(self, "_active_channels"):
                        self._active_channels = []
                    self._active_channels.append(channel)
                    # Clean up old finished channels
                    self._active_channels = [c for c in self._active_channels if c.get_busy()]
        except Exception as e:
            print(f"SoundManager: Failed to play sound {sound_effect.value}: {e}")

    def play_click(self) -> None:
        """Play a click sound for UI interactions"""
        self.play_sound(SoundEffect.CLICK)

    def play_pop(self) -> None:
        """Play a pop sound for game events"""
        self.play_sound(SoundEffect.POP)

    def play_beep(self) -> None:
        """Play a beep sound for notifications"""
        self.play_sound(SoundEffect.BEEP)

    def play_crunch(self) -> None:
        """Play a crunch sound for collision events"""
        self.play_sound(SoundEffect.CRUNCH)

    def play_warble(self) -> None:
        """Play a warble sound for special events"""
        self.play_sound(SoundEffect.WARBLE)

    def play_woosh(self) -> None:
        """Play a woosh sound for movement"""
        self.play_sound(SoundEffect.WOOSH)

    def play_ding(self) -> None:
        """Play a ding sound for scoring"""
        self.play_sound(SoundEffect.DING)

    def play_thud(self) -> None:
        """Play a thud sound for impacts"""
        self.play_sound(SoundEffect.THUD)

    def set_enabled(self, enabled: bool) -> None:
        """
        Enable or disable sound

        Args:
            enabled: Whether sound should be enabled
        """
        self.enabled = enabled and PYGAME_AVAILABLE

    def is_enabled(self) -> bool:
        """Check if sound is enabled"""
        return self.enabled

    def cleanup(self) -> None:
        """Clean up resources"""
        if self._mixer_initialized:
            try:
                pygame.mixer.quit()
            except Exception:
                pass
        self._sound_files.clear()
        print("SoundManager: Cleaned up")


# Global sound manager instance
_global_sound_manager: Optional[SoundManager] = None


def get_sound_manager() -> SoundManager:
    """Get the global sound manager instance"""
    global _global_sound_manager
    if _global_sound_manager is None:
        _global_sound_manager = SoundManager()
    return _global_sound_manager


def set_global_sound_manager(sound_manager: SoundManager) -> None:
    """Set the global sound manager instance"""
    global _global_sound_manager
    _global_sound_manager = sound_manager


def cleanup_global_sound_manager() -> None:
    """Clean up the global sound manager"""
    global _global_sound_manager
    if _global_sound_manager:
        _global_sound_manager.cleanup()
        _global_sound_manager = None
