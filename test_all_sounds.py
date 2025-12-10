"""
Test script that plays all available sounds in a loop with 1 second delay between each.
Uses pre-rendered MP3 sound files via pygame.mixer.
"""

import sys
import time

try:
    from games.util.sound_manager import SoundEffect, get_sound_manager

    SOUND_AVAILABLE = True
except ImportError:
    SOUND_AVAILABLE = False
    print("Error: Could not import sound_manager")


def main():
    if not SOUND_AVAILABLE:
        print("❌ Sound system not available")
        print("   Make sure games.util.sound_manager can be imported")
        return 1

    print("🔊 Initializing sound manager...")
    print("   Loading pre-rendered WAV sound files...")

    sound_manager = get_sound_manager()

    if not sound_manager.is_enabled():
        print("⚠️  Sound manager is not enabled!")
        print("\nPossible issues:")
        print("   1. pygame is not installed")
        print("   2. MP3 sound files not found in runfiles")
        print("   3. Audio system not available")
        return 1

    print("✅ Sound manager initialized")
    print("   Using pre-rendered WAV files from Bazel runfiles")

    # Get all available sound effects
    all_sounds = list(SoundEffect)
    print(f"\nFound {len(all_sounds)} sound effects: {[s.value for s in all_sounds]}")
    print("\n🎵 Playing all sounds in a loop (Ctrl+C to stop)...\n")

    try:
        loop_count = 0
        while True:
            loop_count += 1
            print(f"=== Loop {loop_count} ===")
            for sound_effect in all_sounds:
                print(f"Playing: {sound_effect.value}", end="", flush=True)
                try:
                    sound_manager.play_sound(sound_effect)
                    print(" ✓")
                except Exception as e:
                    print(f" ✗ Error: {e}")
                time.sleep(1.0)  # 1 second delay between sounds

            print("\n--- Loop complete, starting again ---\n")
            time.sleep(0.5)  # Brief pause before next loop

    except KeyboardInterrupt:
        print("\n\n🛑 Stopped by user")
    finally:
        # Clean up
        sound_manager.cleanup()
        print("✅ Sound manager cleaned up")

    return 0


if __name__ == "__main__":
    sys.exit(main())
