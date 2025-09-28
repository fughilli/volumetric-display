import json
import time
import unittest

try:
    from sender_monitor_rust import (
        create_sender_monitor,
        create_sender_monitor_with_web_interface,
    )

    SENDER_MONITOR_AVAILABLE = True
except ImportError:
    SENDER_MONITOR_AVAILABLE = False


class TestSenderMonitorBasic(unittest.TestCase):
    """Test basic sender monitor functionality."""

    def setUp(self):
        """Set up test fixtures."""
        if not SENDER_MONITOR_AVAILABLE:
            self.skipTest("Rust sender monitor implementation not available")

    def test_sender_monitor_creation(self):
        """Test that sender monitor can be created."""
        monitor = create_sender_monitor()
        self.assertIsNotNone(monitor, "Failed to create sender monitor")

    def test_controller_registration(self):
        """Test controller registration and counting."""
        monitor = create_sender_monitor()
        self.assertIsNotNone(monitor)

        test_controllers = [
            ("192.168.1.100", 51330),
            ("192.168.1.101", 51331),
            ("192.168.1.102", 51332),
            ("192.168.1.103", 51333),
        ]

        # Register controllers
        for ip, port in test_controllers:
            monitor.register_controller(ip, port)

        # Check controller count
        controller_count = monitor.get_controller_count()
        self.assertEqual(
            controller_count,
            len(test_controllers),
            f"Expected {len(test_controllers)} controllers, got {controller_count}",
        )

        # Check routable controller count (should be same initially)
        routable_count = monitor.get_routable_controller_count()
        self.assertEqual(
            routable_count,
            len(test_controllers),
            f"Expected {len(test_controllers)} routable controllers, got {routable_count}",
        )

    def test_success_failure_reporting(self):
        """Test success and failure reporting."""
        monitor = create_sender_monitor()
        self.assertIsNotNone(monitor)

        test_controllers = [
            ("192.168.1.100", 51330),
            ("192.168.1.101", 51331),
        ]

        # Register controllers
        for ip, port in test_controllers:
            monitor.register_controller(ip, port)

        # Test success reporting
        for ip, port in test_controllers:
            monitor.report_controller_success(ip, port)

        # Test failure reporting
        monitor.report_controller_failure("192.168.1.100", 51330, "Connection timeout")

    def test_frame_reporting(self):
        """Test frame reporting functionality."""
        monitor = create_sender_monitor()
        self.assertIsNotNone(monitor)

        # Test frame reporting
        for _ in range(10):
            monitor.report_frame()

    def test_cooldown_duration_setting(self):
        """Test cooldown duration setting."""
        monitor = create_sender_monitor()
        self.assertIsNotNone(monitor)

        # Test cooldown duration setting
        monitor.set_cooldown_duration(5)  # 5 seconds

    def test_same_ip_different_ports(self):
        """Test that controllers with same IP but different ports are treated separately."""
        monitor = create_sender_monitor()
        self.assertIsNotNone(monitor)

        # Register controllers with same IP but different ports
        same_ip_controllers = [
            ("192.168.1.100", 51330),
            ("192.168.1.100", 51331),
            ("192.168.1.100", 51332),
        ]

        for ip, port in same_ip_controllers:
            monitor.register_controller(ip, port)

        # Check that all controllers are registered separately
        controller_count = monitor.get_controller_count()
        self.assertEqual(
            controller_count,
            len(same_ip_controllers),
            f"Expected {len(same_ip_controllers)} controllers, got {controller_count}",
        )

        # Test success reporting for each controller individually
        for ip, port in same_ip_controllers:
            monitor.report_controller_success(ip, port)

        # Test failure reporting for one specific controller
        monitor.report_controller_failure("192.168.1.100", 51331, "Port-specific failure")

        # Wait a moment for the async failure reporting to complete
        time.sleep(0.1)

        # All controllers should still be routable except the one that failed
        routable_count = monitor.get_routable_controller_count()
        self.assertEqual(
            routable_count,
            len(same_ip_controllers) - 1,
            f"Expected {len(same_ip_controllers) - 1} routable controllers, got {routable_count}",
        )


class TestSenderMonitorWebInterface(unittest.TestCase):
    """Test web interface functionality."""

    def setUp(self):
        """Set up test fixtures."""
        if not SENDER_MONITOR_AVAILABLE:
            self.skipTest("Rust sender monitor implementation not available")

    def test_web_interface_creation(self):
        """Test that sender monitor with web interface can be created."""
        monitor = create_sender_monitor_with_web_interface(port=8083, cooldown_seconds=10)
        self.assertIsNotNone(monitor, "Failed to create sender monitor with web interface")

    def test_web_interface_with_controllers(self):
        """Test web interface with registered controllers."""
        monitor = create_sender_monitor_with_web_interface(port=8084, cooldown_seconds=10)
        self.assertIsNotNone(monitor)

        # Register some test controllers
        test_controllers = [
            ("127.0.0.1", 51330),
            ("127.0.0.2", 51331),
        ]

        for ip, port in test_controllers:
            monitor.register_controller(ip, port)

        # Simulate some activity
        for _ in range(5):
            for ip, port in test_controllers:
                monitor.report_controller_success(ip, port)
            monitor.report_frame()
            time.sleep(0.1)

        # Check that controllers are registered
        controller_count = monitor.get_controller_count()
        self.assertEqual(
            controller_count,
            len(test_controllers),
            f"Expected {len(test_controllers)} controllers, got {controller_count}",
        )


class TestSenderMonitorScenarios(unittest.TestCase):
    """Test various monitoring scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        if not SENDER_MONITOR_AVAILABLE:
            self.skipTest("Rust sender monitor implementation not available")

    def test_all_controllers_working(self):
        """Test scenario where all controllers are working."""
        monitor = create_sender_monitor()
        self.assertIsNotNone(monitor)

        # Set a short cooldown for testing
        monitor.set_cooldown_duration(2)  # 2 seconds

        # Register controllers
        controllers = [
            ("10.0.0.1", 51330),
            ("10.0.0.2", 51331),
            ("10.0.0.3", 51332),
        ]

        for ip, port in controllers:
            monitor.register_controller(ip, port)

        # All controllers working
        for ip, port in controllers:
            monitor.report_controller_success(ip, port)
            monitor.report_frame()

        # Check that all controllers are routable
        routable_count = monitor.get_routable_controller_count()
        self.assertEqual(
            routable_count,
            len(controllers),
            f"Expected {len(controllers)} routable controllers, got {routable_count}",
        )

    def test_controller_failure_and_recovery(self):
        """Test controller failure and recovery scenario."""
        monitor = create_sender_monitor()
        self.assertIsNotNone(monitor)

        # Set a short cooldown for testing
        monitor.set_cooldown_duration(1)  # 1 second

        # Register controllers
        controllers = [
            ("10.0.0.1", 51330),
            ("10.0.0.2", 51331),
        ]

        for ip, port in controllers:
            monitor.register_controller(ip, port)

        # Report failure for one controller
        monitor.report_controller_failure("10.0.0.1", 51330, "Network unreachable")

        # Continue reporting success for others
        for ip, port in controllers[1:]:
            monitor.report_controller_success(ip, port)
            monitor.report_frame()

        # Wait for cooldown and test recovery
        time.sleep(2)  # Wait longer than cooldown

        # Report success for the previously failed controller
        monitor.report_controller_success("10.0.0.1", 51330)

    def test_high_frame_rate(self):
        """Test high frame rate reporting."""
        monitor = create_sender_monitor()
        self.assertIsNotNone(monitor)

        # Register controllers
        controllers = [
            ("10.0.0.1", 51330),
            ("10.0.0.2", 51331),
        ]

        for ip, port in controllers:
            monitor.register_controller(ip, port)

        # High frame rate testing
        for i in range(100):
            monitor.report_frame()
            if i % 20 == 0:
                for ip, port in controllers:
                    monitor.report_controller_success(ip, port)


class TestSenderMonitorErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""

    def setUp(self):
        """Set up test fixtures."""
        if not SENDER_MONITOR_AVAILABLE:
            self.skipTest("Rust sender monitor implementation not available")

    def test_unregistered_controller_operations(self):
        """Test operations on unregistered controllers."""
        monitor = create_sender_monitor()
        self.assertIsNotNone(monitor)

        # Test reporting success for unregistered controller
        monitor.report_controller_success("192.168.99.99", 51330)

        # Test reporting failure for unregistered controller
        monitor.report_controller_failure("192.168.99.99", 51330, "Test error")

    def test_extreme_cooldown_values(self):
        """Test extreme cooldown duration values."""
        monitor = create_sender_monitor()
        self.assertIsNotNone(monitor)

        # Test extreme cooldown values
        monitor.set_cooldown_duration(0)
        monitor.set_cooldown_duration(3600)  # 1 hour

    def test_rapid_frame_reporting(self):
        """Test rapid frame reporting."""
        monitor = create_sender_monitor()
        self.assertIsNotNone(monitor)

        # Test rapid frame reporting
        for _ in range(1000):
            monitor.report_frame()


def create_test_config():
    """Create a test configuration file if it doesn't exist."""
    try:
        with open("config.json", "r") as f:
            json.load(f)
            return True
    except FileNotFoundError:
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
            return True
        except Exception:
            return False


if __name__ == "__main__":
    # Create test config if it doesn't exist
    create_test_config()

    # Run the test suite
    unittest.main(verbosity=2)
