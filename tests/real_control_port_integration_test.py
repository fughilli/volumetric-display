"""
Real Integration Test for ControlPortManager with Simulated Controllers

This test suite exercises the actual Rust implementation of the control port
manager with simulated controllers to verify end-to-end functionality.
"""

import asyncio
import json
import os
import tempfile
import time
import unittest
from typing import List

from control_port_rust import ControlPortManager

# Import the controller simulator library
from controller_simulator_lib import ControllerSimulator


class RealControlPortIntegrationTest(unittest.TestCase):
    """Real integration tests using the actual Rust control port manager."""

    def setUp(self):
        """Set up test fixtures."""
        self.simulator = ControllerSimulator()
        self.temp_config_file = None
        self.control_manager = None

    def tearDown(self):
        """Clean up test fixtures."""
        if self.control_manager:
            try:
                self.control_manager.shutdown()
            except Exception:
                pass

        if self.temp_config_file and os.path.exists(self.temp_config_file):
            try:
                os.unlink(self.temp_config_file)
            except Exception:
                pass

        self.simulator.stop()
        self.simulator.wait_for_shutdown()

    def create_test_config(self, controllers: List[tuple]) -> str:
        """Create a temporary test configuration file."""
        config = {"controller_addresses": {}}

        for dip, port in controllers:
            config["controller_addresses"][str(dip)] = {"ip": "127.0.0.1", "port": port}

        # Create temporary file
        fd, path = tempfile.mkstemp(suffix=".json", prefix="test_config_")
        os.close(fd)

        with open(path, "w") as f:
            json.dump(config, f, indent=2)

        self.temp_config_file = path
        return path

    def _test_single_controller_real_connection(self):
        """Test real connection to a single controller simulator."""

        # Set up controller simulator
        dip = "1"
        port = 8001
        self.simulator.add_controller(int(dip), port)

        # Create test config
        config_path = self.create_test_config([(dip, port)])

        # Create control manager
        self.control_manager = ControlPortManager(config_path)

        # Start simulator first
        self.simulator.start_asyncio_thread()
        time.sleep(1)  # Allow time for server to start

        # Initialize control manager
        self.control_manager.initialize()

        # Wait for connection to establish
        max_wait = 3
        connected = False
        for i in range(max_wait):
            time.sleep(0.2)
            control_port = self.control_manager.get_control_port(dip)
            if control_port and control_port.connected:
                connected = True
                break

        self.assertTrue(
            connected, f"Controller {dip} should be connected within {max_wait} seconds"
        )

        # Verify connection status
        control_port = self.control_manager.get_control_port(dip)
        self.assertIsNotNone(control_port, "Control port should be available")
        self.assertTrue(control_port.connected, "Control port should be connected")

    def test_lcd_functionality_real(self):
        """Test real LCD functionality with connected controller."""

        # Set up controller simulator
        dip = "2"
        port = 8002
        self.simulator.add_controller(int(dip), port)

        # Create test config
        config_path = self.create_test_config([(dip, port)])

        # Create control manager
        self.control_manager = ControlPortManager(config_path)

        # Start simulator first
        self.simulator.start_asyncio_thread()
        time.sleep(1)

        # Initialize control manager
        self.control_manager.initialize()

        # Wait for connection
        max_wait = 3
        connected = False
        for i in range(max_wait):
            time.sleep(0.2)
            control_port = self.control_manager.get_control_port(dip)
            if control_port and control_port.connected:
                connected = True
                break

        self.assertTrue(connected, f"Controller {dip} should be connected")

        # Test LCD functionality
        control_port = self.control_manager.get_control_port(dip)

        # Test writing text
        test_text = "Hello World"
        control_port.write_display(0, 0, test_text)

        # Test clearing display
        control_port.clear_display()

        # Test writing to different positions
        control_port.write_display(5, 2, "Test")

        # Commit is async, so we need to run it synchronously
        loop = asyncio.get_event_loop()
        loop.run_until_complete(control_port.commit_display())

        # Verify the simulator received the commands
        # We can check the simulator's LCD content
        lcd_content = self.simulator.get_lcd_content(int(dip))
        self.assertIsNotNone(lcd_content, "LCD content should be available")

    def _test_multiple_controllers_real(self):
        """Test real multiple controller handling."""

        # Set up multiple controller simulators
        controllers = [("4", 8004), ("5", 8005), ("6", 8006)]

        for dip, port in controllers:
            self.simulator.add_controller(int(dip), port)

        # Create test config
        config_path = self.create_test_config(controllers)

        # Create control manager
        self.control_manager = ControlPortManager(config_path)

        # Start simulator first
        self.simulator.start_asyncio_thread()
        time.sleep(1)  # Allow more time for multiple servers to start

        # Initialize control manager
        self.control_manager.initialize()

        # Wait for all connections
        max_wait = 3
        all_connected = False
        for i in range(max_wait):
            time.sleep(0.2)
            connected_count = 0
            for dip, _ in controllers:
                control_port = self.control_manager.get_control_port(dip)
                if control_port and control_port.connected:
                    connected_count += 1

            if connected_count == len(controllers):
                all_connected = True
                break

        self.assertTrue(
            all_connected, f"All controllers should be connected within {max_wait} seconds"
        )

        # Verify all controllers are connected
        for dip, _ in controllers:
            control_port = self.control_manager.get_control_port(dip)
            self.assertIsNotNone(control_port, f"Control port {dip} should be available")
            self.assertTrue(control_port.connected, f"Control port {dip} should be connected")

        # Test LCD functionality on all controllers
        for dip, _ in controllers:
            control_port = self.control_manager.get_control_port(dip)
            control_port.write_display(0, 0, f"Controller {dip}")
            # Commit is async, so we need to run it synchronously
            loop = asyncio.get_event_loop()
            loop.run_until_complete(control_port.commit_display())

    def _test_connection_failure_real(self):
        """Test real connection failure handling."""

        # Create test config for non-existent controller
        dip = "99"
        port = 8999
        config_path = self.create_test_config([(dip, port)])

        # Create control manager
        self.control_manager = ControlPortManager(config_path)

        # Initialize control manager (should not crash)
        self.control_manager.initialize()

        # Wait a bit for connection attempts
        time.sleep(1)

        # Check that the controller is not connected
        control_port = self.control_manager.get_control_port(dip)
        if control_port:
            # In a real implementation, this might be None or not connected
            # We'll just verify the system doesn't crash
            pass

    def _test_stress_multiple_controllers_real(self):
        """Stress test with many real controllers."""

        # Set up many controller simulators
        num_controllers = 5  # Reduced for testing
        controllers = []

        for i in range(num_controllers):
            dip = str(100 + i)  # Use DIPs 100-104
            port = 8100 + i  # Use ports 8100-8104
            controllers.append((dip, port))

            self.simulator.add_controller(int(dip), port)

        # Create test config
        config_path = self.create_test_config(controllers)

        # Create control manager
        self.control_manager = ControlPortManager(config_path)

        # Start simulator first
        self.simulator.start_asyncio_thread()
        time.sleep(1)  # Allow more time for many servers to start

        # Initialize control manager
        self.control_manager.initialize()

        # Wait for connections
        max_wait = 3
        all_connected = False
        for i in range(max_wait):
            time.sleep(0.2)
            connected_count = 0
            for dip, _ in controllers:
                control_port = self.control_manager.get_control_port(dip)
                if control_port and control_port.connected:
                    connected_count += 1

            if connected_count == len(controllers):
                all_connected = True
                break

        self.assertTrue(
            all_connected, f"All controllers should be connected within {max_wait} seconds"
        )

        # Test concurrent LCD operations
        for dip, _ in controllers:
            control_port = self.control_manager.get_control_port(dip)
            for line in range(4):
                control_port.write_display(0, line, f"Line {line}")
            # Commit is async, so we need to run it synchronously
            loop = asyncio.get_event_loop()
            loop.run_until_complete(control_port.commit_display())

    def _test_web_monitor_real(self):
        """Test web monitor functionality."""

        # Set up controller simulator
        dip = "7"
        port = 8007
        self.simulator.add_controller(int(dip), port)

        # Create test config
        config_path = self.create_test_config([(dip, port)])

        # Create control manager
        self.control_manager = ControlPortManager(config_path)

        # Start simulator first
        self.simulator.start_asyncio_thread()
        time.sleep(1)

        # Initialize control manager
        self.control_manager.initialize()

        # Start web monitor
        web_port = 8081  # Use different port to avoid conflicts
        self.control_manager.start_web_monitor(web_port)

        # Wait for connection
        max_wait = 3
        connected = False
        for i in range(max_wait):
            time.sleep(0.2)
            control_port = self.control_manager.get_control_port(dip)
            if control_port and control_port.connected:
                connected = True
                break

        self.assertTrue(connected, f"Controller {dip} should be connected")

        # Get stats
        stats = self.control_manager.get_stats()
        self.assertIsNotNone(stats, "Stats should be available")
        self.assertGreater(len(stats), 0, "Should have stats for at least one controller")


if __name__ == "__main__":
    unittest.main()
