#!/usr/bin/env python3
"""
Test script for the sound system
"""

import sys
import time


def test_sound_system():
    """Test the sound system by playing all available sounds"""

    print("Testing Sound System")
    print("=" * 30)

    # Try to import the sound manager
    try:
        from games.util.sound_manager import get_sound_manager

        print("✅ Sound manager imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import sound manager: {e}")
        return False

    # Get the sound manager
    sound_manager = get_sound_manager()
    print(f"✅ Sound manager created (enabled: {sound_manager.is_enabled()})")

    if not sound_manager.is_enabled():
        print("⚠️  Sound is disabled - make sure pygame is installed")
        return False

    # Test all sound effects
    sounds_to_test = [
        ("Click", sound_manager.play_click),
        ("Pop", sound_manager.play_pop),
        ("Beep", sound_manager.play_beep),
        ("Crunch", sound_manager.play_crunch),
        ("Warble", sound_manager.play_warble),
        ("Woosh", sound_manager.play_woosh),
        ("Ding", sound_manager.play_ding),
        ("Thud", sound_manager.play_thud),
    ]

    print("\nPlaying sound effects:")
    for name, sound_func in sounds_to_test:
        print(f"  Playing {name}...")
        sound_func()
        time.sleep(0.5)  # Wait between sounds

    print("\n✅ Sound system test completed!")
    return True


def main():
    """Main test function"""
    print("Sound System Test")
    print("=" * 50)
    print("Testing pre-rendered MP3 sound files")
    print()

    # Test sound system
    if not test_sound_system():
        print("\n❌ Sound system test failed")
        return 1

    print("\n🎉 All tests passed! Sound system is working correctly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
