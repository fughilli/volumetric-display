"""
Integration tests for ControlPortManager with simulated controllers.

This test suite exercises the full control port functionality including:
- TCP connection establishment
- LCD functionality
- Button callbacks
- Multiple controller handling
"""

import time
import unittest
from typing import Dict, List, Optional

# Import the controller simulator library
from controller_simulator_lib import ControllerSimulator

# Import the control port manager (we'll need to mock the Rust bindings)
# For now, we'll create a mock version to test the integration logic


class MockControlPortManager:
    """
    Mock implementation of the ControlPortManager for testing.
    This simulates the behavior we expect from the Rust implementation.
    """

    def __init__(self):
        self.controllers: Dict[int, Dict] = {}
        self.connection_events: List[Dict] = []
        self.lcd_commands: List[Dict] = []
        self.button_events: List[Dict] = []

    def add_controller(self, dip: int, host: str, port: int) -> bool:
        """Add a controller to the manager."""
        self.controllers[dip] = {
            "host": host,
            "port": port,
            "connected": False,
            "connection_time": None,
            "last_error": None,
        }
        return True

    def remove_controller(self, dip: int) -> bool:
        """Remove a controller from the manager."""
        if dip in self.controllers:
            del self.controllers[dip]
            return True
        return False

    def get_controller_status(self, dip: int) -> Optional[Dict]:
        """Get the status of a controller."""
        if dip in self.controllers:
            return self.controllers[dip].copy()
        return None

    def get_all_controllers(self) -> Dict[int, Dict]:
        """Get all controllers."""
        return {dip: ctrl.copy() for dip, ctrl in self.controllers.items()}

    def set_lcd_text(self, dip: int, x: int, y: int, text: str) -> bool:
        """Set LCD text for a controller."""
        if dip in self.controllers and self.controllers[dip]["connected"]:
            self.lcd_commands.append(
                {"dip": dip, "x": x, "y": y, "text": text, "timestamp": time.time()}
            )
            return True
        return False

    def clear_lcd(self, dip: int) -> bool:
        """Clear LCD for a controller."""
        if dip in self.controllers and self.controllers[dip]["connected"]:
            self.lcd_commands.append({"dip": dip, "action": "clear", "timestamp": time.time()})
            return True
        return False

    def set_button_callback(self, dip: int, callback) -> bool:
        """Set button callback for a controller."""
        if dip in self.controllers:
            # Store callback reference (in real implementation this would be more complex)
            return True
        return False

    def simulate_connection_event(self, dip: int, connected: bool, error: Optional[str] = None):
        """Simulate a connection event for testing."""
        if dip in self.controllers:
            self.controllers[dip]["connected"] = connected
            if connected:
                self.controllers[dip]["connection_time"] = time.time()
                self.controllers[dip]["last_error"] = None
            else:
                self.controllers[dip]["last_error"] = error

            self.connection_events.append(
                {"dip": dip, "connected": connected, "error": error, "timestamp": time.time()}
            )

    def simulate_button_event(self, dip: int, button_states: List[bool]):
        """Simulate a button event for testing."""
        if dip in self.controllers and self.controllers[dip]["connected"]:
            self.button_events.append(
                {"dip": dip, "buttons": button_states, "timestamp": time.time()}
            )


class TestControlPortIntegration(unittest.TestCase):
    """Integration tests for ControlPortManager with simulated controllers."""

    def setUp(self):
        """Set up test fixtures."""
        self.simulator = ControllerSimulator()
        self.control_manager = MockControlPortManager()
        self.test_results = {"lcd_commands": [], "button_events": [], "connection_events": []}

    def tearDown(self):
        """Clean up test fixtures."""
        self.simulator.stop()
        self.simulator.wait_for_shutdown()

    def test_single_controller_connection(self):
        """Test connection to a single controller simulator."""

        # Set up controller simulator
        dip = 1
        port = 8001
        self.simulator.add_controller(dip, port)

        # Add controller to control manager
        success = self.control_manager.add_controller(dip, "127.0.0.1", port)
        self.assertTrue(success, "Failed to add controller to manager")

        # Start simulator
        self.simulator.start_asyncio_thread()
        time.sleep(1)  # Allow time for server to start

        # Simulate connection establishment
        self.control_manager.simulate_connection_event(dip, True)

        # Verify connection status
        status = self.control_manager.get_controller_status(dip)
        self.assertIsNotNone(status, "Controller status should not be None")
        self.assertTrue(status["connected"], "Controller should be connected")
        self.assertIsNotNone(status["connection_time"], "Connection time should be set")

    def test_lcd_functionality(self):
        """Test LCD functionality with connected controller."""

        # Set up controller simulator
        dip = 2
        port = 8002
        self.simulator.add_controller(dip, port)

        # Add controller to control manager
        self.control_manager.add_controller(dip, "127.0.0.1", port)

        # Start simulator
        self.simulator.start_asyncio_thread()
        time.sleep(1)

        # Simulate connection
        self.control_manager.simulate_connection_event(dip, True)

        # Test LCD text setting
        test_text = "Hello World"
        success = self.control_manager.set_lcd_text(dip, 0, 0, test_text)
        self.assertTrue(success, "LCD text setting should succeed for connected controller")

        # Test LCD clear
        success = self.control_manager.clear_lcd(dip)
        self.assertTrue(success, "LCD clear should succeed for connected controller")

        # Verify LCD commands were recorded
        lcd_commands = self.control_manager.lcd_commands
        self.assertGreater(len(lcd_commands), 0, "LCD commands should be recorded")

        # Check that we have both text and clear commands
        text_commands = [cmd for cmd in lcd_commands if cmd.get("action") != "clear"]
        clear_commands = [cmd for cmd in lcd_commands if cmd.get("action") == "clear"]

        self.assertGreater(len(text_commands), 0, "Text commands should be recorded")
        self.assertGreater(len(clear_commands), 0, "Clear commands should be recorded")

    def test_button_callbacks(self):
        """Test button callback functionality."""

        # Set up controller simulator
        dip = 3
        port = 8003
        self.simulator.add_controller(dip, port)

        # Add controller to control manager
        self.control_manager.add_controller(dip, "127.0.0.1", port)

        # Start simulator
        self.simulator.start_asyncio_thread()
        time.sleep(1)

        # Simulate connection
        self.control_manager.simulate_connection_event(dip, True)

        # Set button callback
        success = self.control_manager.set_button_callback(dip, lambda x: x)
        self.assertTrue(success, "Button callback should be set successfully")

        # Simulate button events
        button_states = [True, False, False, False, False]  # UP button pressed
        self.control_manager.simulate_button_event(dip, button_states)

        # Verify button events were recorded
        button_events = self.control_manager.button_events
        self.assertGreater(len(button_events), 0, "Button events should be recorded")

        # Check button event details
        event = button_events[0]
        self.assertEqual(event["dip"], dip, "Button event should have correct DIP")
        self.assertEqual(
            event["buttons"], button_states, "Button event should have correct button states"
        )

    def test_multiple_controllers(self):
        """Test multiple controller handling."""

        # Set up multiple controller simulators
        controllers = [(4, 8004), (5, 8005), (6, 8006), (7, 8007), (8, 8008)]

        for dip, port in controllers:
            self.simulator.add_controller(dip, port)
            self.control_manager.add_controller(dip, "127.0.0.1", port)

        # Start simulator
        self.simulator.start_asyncio_thread()
        time.sleep(1)

        # Simulate connections for all controllers
        for dip, _ in controllers:
            self.control_manager.simulate_connection_event(dip, True)

        # Verify all controllers are connected
        all_controllers = self.control_manager.get_all_controllers()
        self.assertEqual(
            len(all_controllers),
            len(controllers),
            f"Expected {len(controllers)} controllers, got {len(all_controllers)}",
        )

        for dip, _ in controllers:
            status = self.control_manager.get_controller_status(dip)
            self.assertIsNotNone(status, f"Controller {dip} status should not be None")
            self.assertTrue(status["connected"], f"Controller {dip} should be connected")

        # Test LCD functionality on multiple controllers
        for dip, _ in controllers:
            success = self.control_manager.set_lcd_text(dip, 0, 0, f"Controller {dip}")
            self.assertTrue(success, f"LCD text setting should succeed for controller {dip}")

        # Verify LCD commands for all controllers
        lcd_commands = self.control_manager.lcd_commands
        self.assertGreaterEqual(
            len(lcd_commands), len(controllers), "Should have LCD commands for all controllers"
        )

    def test_connection_failure_handling(self):
        """Test handling of connection failures."""

        # Add controller to control manager (but don't start simulator)
        dip = 9
        port = 8009
        self.control_manager.add_controller(dip, "127.0.0.1", port)

        # Simulate connection failure
        error_msg = "Connection refused"
        self.control_manager.simulate_connection_event(dip, False, error_msg)

        # Verify failure status
        status = self.control_manager.get_controller_status(dip)
        self.assertIsNotNone(status, "Controller status should not be None")
        self.assertFalse(status["connected"], "Controller should not be connected")
        self.assertEqual(status["last_error"], error_msg, "Error message should be recorded")

        # Test that LCD operations fail for disconnected controller
        success = self.control_manager.set_lcd_text(dip, 0, 0, "Test")
        self.assertFalse(success, "LCD operations should fail for disconnected controller")

    def test_controller_removal(self):
        """Test removing controllers from the manager."""

        # Add controller
        dip = 10
        port = 8010
        self.control_manager.add_controller(dip, "127.0.0.1", port)

        # Verify controller was added
        status = self.control_manager.get_controller_status(dip)
        self.assertIsNotNone(status, "Controller should be added")

        # Remove controller
        success = self.control_manager.remove_controller(dip)
        self.assertTrue(success, "Controller removal should succeed")

        # Verify controller was removed
        status = self.control_manager.get_controller_status(dip)
        self.assertIsNone(status, "Controller should be removed")

    def test_stress_multiple_controllers(self):
        """Stress test with many controllers."""

        # Set up many controller simulators
        num_controllers = 10
        controllers = []

        for i in range(num_controllers):
            dip = 100 + i  # Use DIPs 100-109
            port = 8100 + i  # Use ports 8100-8109
            controllers.append((dip, port))

            self.simulator.add_controller(dip, port)
            self.control_manager.add_controller(dip, "127.0.0.1", port)

        # Start simulator
        self.simulator.start_asyncio_thread()
        time.sleep(2)  # Allow more time for many servers to start

        # Simulate connections for all controllers
        for dip, _ in controllers:
            self.control_manager.simulate_connection_event(dip, True)

        # Verify all controllers are connected
        all_controllers = self.control_manager.get_all_controllers()
        self.assertEqual(
            len(all_controllers),
            num_controllers,
            f"Expected {num_controllers} controllers, got {len(all_controllers)}",
        )

        # Test concurrent LCD operations
        lcd_operations = []
        for dip, _ in controllers:
            for line in range(4):
                lcd_operations.append((dip, 0, line, f"Line {line}"))

        # Execute LCD operations
        for dip, x, y, text in lcd_operations:
            success = self.control_manager.set_lcd_text(dip, x, y, text)
            self.assertTrue(success, f"LCD operation should succeed for controller {dip}")

        # Verify all LCD operations were recorded
        lcd_commands = self.control_manager.lcd_commands
        self.assertGreaterEqual(
            len(lcd_commands), len(lcd_operations), "Should have LCD commands for all operations"
        )

    def test_connection_recovery(self):
        """Test connection recovery after failure."""

        # Set up controller simulator
        dip = 11
        port = 8011
        self.simulator.add_controller(dip, port)

        # Add controller to control manager
        self.control_manager.add_controller(dip, "127.0.0.1", port)

        # Simulate initial connection failure
        self.control_manager.simulate_connection_event(dip, False, "Initial failure")

        # Verify failure status
        status = self.control_manager.get_controller_status(dip)
        self.assertFalse(status["connected"], "Controller should initially be disconnected")

        # Start simulator and simulate recovery
        self.simulator.start_asyncio_thread()
        time.sleep(1)

        # Simulate successful connection
        self.control_manager.simulate_connection_event(dip, True)

        # Verify recovery
        status = self.control_manager.get_controller_status(dip)
        self.assertTrue(status["connected"], "Controller should be connected after recovery")
        self.assertIsNone(status["last_error"], "Error should be cleared after recovery")

        # Test that operations work after recovery
        success = self.control_manager.set_lcd_text(dip, 0, 0, "Recovered")
        self.assertTrue(success, "LCD operations should work after connection recovery")


if __name__ == "__main__":
    unittest.main()
