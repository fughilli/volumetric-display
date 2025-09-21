#!/usr/bin/env python3
"""
Test script for the sound system
"""

import subprocess
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
        print("⚠️  Sound is disabled - make sure Chuck is installed and sound server is running")
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


def test_chuck_installation():
    """Test if Chuck is installed"""
    print("Testing Chuck Installation")
    print("=" * 30)

    try:
        result = subprocess.run(["chuck", "--version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✅ Chuck is installed: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ Chuck command failed: {result.stderr}")
            return False
    except FileNotFoundError:
        print("❌ Chuck not found in PATH")
        print("Please install Chuck from https://chuck.cs.princeton.edu/")
        print("On macOS: brew install chuck")
        print("On Ubuntu: sudo apt-get install chuck")
        return False
    except subprocess.TimeoutExpired:
        print("❌ Chuck command timed out")
        return False


def main():
    """Main test function"""
    print("Sound System Test")
    print("=" * 50)

    # Test Chuck installation first
    if not test_chuck_installation():
        print("\n❌ Chuck installation test failed")
        return 1

    print()

    # Test sound system
    if not test_sound_system():
        print("\n❌ Sound system test failed")
        return 1

    print("\n🎉 All tests passed! Sound system is working correctly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
