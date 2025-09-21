"""
Sound Manager for Game Audio
Handles communication with Chuck sound server via OSC
"""

import threading
from enum import Enum
from typing import Optional

try:
    from pythonosc import udp_client

    OSC_AVAILABLE = True
except ImportError:
    OSC_AVAILABLE = False
    print("Warning: pythonosc not available. Sound will be disabled.")


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
    """Manages sound effects for games and menus"""

    def __init__(self, host: str = "localhost", port: int = 6449, enabled: bool = True):
        """
        Initialize the sound manager

        Args:
            host: OSC server host (default: localhost)
            port: OSC server port (default: 6449)
            enabled: Whether sound is enabled (default: True)
        """
        self.host = host
        self.port = port
        self.enabled = enabled and OSC_AVAILABLE
        self.client: Optional[udp_client.SimpleUDPClient] = None
        self._lock = threading.Lock()

        if self.enabled:
            try:
                self.client = udp_client.SimpleUDPClient(host, port)
                print(f"SoundManager: Connected to Chuck sound server at {host}:{port}")
            except Exception as e:
                print(f"SoundManager: Failed to connect to sound server: {e}")
                self.enabled = False

    def play_sound(self, sound_effect: SoundEffect) -> None:
        """
        Play a sound effect

        Args:
            sound_effect: The sound effect to play
        """
        if not self.enabled or not self.client:
            return

        try:
            with self._lock:
                osc_address = f"/sound/{sound_effect.value}"
                self.client.send_message(osc_address, [])
                print(f"SoundManager: Playing {sound_effect.value}")
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
        self.enabled = enabled and OSC_AVAILABLE
        if self.enabled and not self.client:
            try:
                self.client = udp_client.SimpleUDPClient(self.host, self.port)
                print(f"SoundManager: Reconnected to Chuck sound server at {self.host}:{self.port}")
            except Exception as e:
                print(f"SoundManager: Failed to reconnect to sound server: {e}")
                self.enabled = False

    def is_enabled(self) -> bool:
        """Check if sound is enabled"""
        return self.enabled

    def cleanup(self) -> None:
        """Clean up resources"""
        if self.client:
            self.client = None
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
