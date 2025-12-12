"""
Test script that plays all available sounds in a loop with 1 second delay between each.
Uses pre-rendered WAV sound files via pygame.mixer.
Can also play a specific sound by name if provided as an argument.
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
        print("   2. WAV sound files not found in runfiles")
        print("   3. Audio system not available")
        return 1

    print("✅ Sound manager initialized")
    print("   Using pre-rendered WAV files from Bazel runfiles")

    # Get all available sound effects
    all_sounds = list(SoundEffect)
    sound_dict = {s.value: s for s in all_sounds}

    # Check if a specific sound was requested
    if len(sys.argv) > 1:
        sound_name = sys.argv[1].lower()
        if sound_name in sound_dict:
            sound_effect = sound_dict[sound_name]
            print(f"\n🎵 Playing sound: {sound_name}")
            print("   (Press Ctrl+C to stop)\n")
            try:
                while True:
                    print(f"Playing: {sound_name}", end="", flush=True)
                    try:
                        sound_manager.play_sound(sound_effect)
                        print(" ✓")
                    except Exception as e:
                        print(f" ✗ Error: {e}")
                    time.sleep(1.0)
            except KeyboardInterrupt:
                print("\n\n🛑 Stopped by user")
            finally:
                sound_manager.cleanup()
                print("✅ Sound manager cleaned up")
            return 0
        else:
            print(f"\n❌ Sound '{sound_name}' not found!")
            print("\nAvailable sounds:")
            for sound_value in sorted(sound_dict.keys()):
                print(f"  - {sound_value}")
            return 1

    print(f"\nFound {len(all_sounds)} sound effects: {[s.value for s in all_sounds]}")
    print("\n🎵 Playing all sounds in a loop (Ctrl+C to stop)...")
    print("   Usage: test_all_sounds [sound_name] to play a specific sound\n")

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
