#!/usr/bin/env python3
"""
Test script for the Rust-based control port implementation.

This script demonstrates the usage and validates the functionality of the new
Rust-based control port system.
"""

import asyncio
import json
import time

try:
    from control_port_rust import create_control_port_from_config
    from games.util.game_util_rust import ControllerInputHandlerRust

    RUST_AVAILABLE = True
except ImportError as e:
    print(f"Rust implementation not available: {e}")
    RUST_AVAILABLE = False


async def test_control_port_basic():
    """Test basic control port functionality."""
    print("=== Testing Basic Control Port ===")

    if not RUST_AVAILABLE:
        print("Skipping Rust tests - not available")
        return False

    try:
        # Create control port from config
        cp = create_control_port_from_config("config.json", web_monitor_port=8081)

        # Enumerate controllers
        controllers = await cp.enumerate()
        print(f"Found {len(controllers)} controllers: {list(controllers.keys())}")

        # Test basic display functionality
        for dip, controller in controllers.items():
            print(f"Testing controller {dip}...")

            # Clear and write to display
            controller.clear()
            controller.write_lcd(0, 0, "Rust Test")
            controller.write_lcd(0, 1, f"Controller {dip}")
            controller.write_lcd(0, 2, "Line 3")
            await controller.commit()

            print(f"  Display updated for controller {dip}")

            # Test LED colors (rainbow pattern)
            rainbow_colors = [
                (255, 0, 0),  # Red
                (255, 127, 0),  # Orange
                (255, 255, 0),  # Yellow
                (0, 255, 0),  # Green
                (0, 0, 255),  # Blue
                (75, 0, 130),  # Indigo
                (148, 0, 211),  # Violet
            ]
            await controller.set_leds(rainbow_colors)
            print(f"  LEDs set for controller {dip}")

            # Test backlight
            await controller.set_backlights([True, False, True, False])
            print(f"  Backlights set for controller {dip}")

        # Get statistics
        stats = cp.get_stats()
        print(f"Controller stats: {len(stats)} entries")
        for stat in stats:
            print(
                f"  Controller {stat.get('dip', 'unknown')}: "
                f"connected={stat.get('connected', False)}, "
                f"messages_sent={stat.get('messages_sent', 0)}"
            )

        print("Basic control port test completed successfully!")

        # Clean up
        cp.shutdown()
        return True

    except Exception as e:
        print(f"Basic control port test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_input_handler():
    """Test the controller input handler."""
    print("\n=== Testing Controller Input Handler ===")

    if not RUST_AVAILABLE:
        print("Skipping Rust input handler tests - not available")
        return False

    try:
        # Create controller mapping
        controller_mapping = {
            "0": "player1",
            "1": "player2",
            "2": "player3",
            "3": "player4",
        }

        # Create input handler
        handler = ControllerInputHandlerRust(
            controller_mapping=controller_mapping, config_path="config.json", web_monitor_port=8082
        )

        # Set up button callbacks
        def button_callback(player_id, button, button_state):
            print(f"Button event: {player_id} pressed {button.name} ({button_state.name})")

        # Initialize
        if handler.start_initialization():
            print(
                f"Input handler initialized successfully with {len(handler.controllers)} controllers"
            )
            print(f"Web monitor available at: {handler.get_web_monitor_url()}")

            # Register callbacks
            for controller_id in handler.controllers.keys():
                handler.register_button_callback(controller_id, button_callback)
                print(f"Registered button callback for controller {controller_id}")

            # Let it run for a bit to collect some events
            print("Listening for button events for 5 seconds...")
            time.sleep(5)

            # Check for any queued events
            event_count = 0
            while True:
                event = handler.get_direction_key()
                if event is None:
                    break
                event_count += 1
                print(f"Queued event {event_count}: {event}")

            print(f"Found {event_count} queued events")

            # Get statistics
            stats = handler.get_stats()
            print(f"Handler statistics: {len(stats)} controllers")

            print("Input handler test completed successfully!")

            # Clean up
            handler.stop()
            return True
        else:
            print("Failed to initialize input handler")
            return False

    except Exception as e:
        print(f"Input handler test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def create_test_config():
    """Create a test configuration file if it doesn't exist."""
    try:
        with open("config.json", "r") as f:
            json.load(f)
            print("Using existing config.json")
            return True
    except FileNotFoundError:
        print("Creating test config.json...")
        test_config = {
            "geometry": "20x20x20",
            "controller_addresses": {
                "0": {"ip": "127.0.0.1", "port": 51330},
                "1": {"ip": "127.0.0.1", "port": 51331},
                "2": {"ip": "127.0.0.1", "port": 51332},
                "3": {"ip": "127.0.0.1", "port": 51333},
            },
        }

        try:
            with open("config.json", "w") as f:
                json.dump(test_config, f, indent=2)
            print("Test config.json created")
            return True
        except Exception as e:
            print(f"Failed to create test config: {e}")
            return False


async def main():
    """Main test function."""
    print("Rust Control Port Test Suite")
    print("=" * 40)

    # Check if config exirts
    if not create_test_config():
        print("Cannot proceed without config file")
        return

    # Run tests
    results = []

    # Test basic control port
    results.append(await test_control_port_basic())

    # Test input handler
    results.append(test_input_handler())

    # Summary
    print("\n" + "=" * 40)
    print("Test Results Summary:")
    print(f"Basic Control Port: {'PASS' if results[0] else 'FAIL'}")
    print(f"Input Handler: {'PASS' if results[1] else 'FAIL'}")

    if all(results):
        print("\nAll tests passed! 🎉")
        print("\nTo monitor the system:")
        print("1. Run your application with the Rust control port")
        print("2. Open http://localhost:8080 in your browser")
        print("3. View real-time controller status and communication logs")
    else:
        print("\nSome tests failed. Check the output above for details.")

    print(f"\nRust implementation available: {RUST_AVAILABLE}")


if __name__ == "__main__":
    asyncio.run(main())
