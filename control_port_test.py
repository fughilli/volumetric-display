import asyncio
import unittest
from unittest.mock import AsyncMock, call

from control_port_rust import ControllerState


class TestControlPort(unittest.TestCase):
    def test_lcd_commit_on_empty_buffer_causes_clear_command(self):
        controller_state = ControllerState(ip="127.0.0.1", dip=1, port=1234)
        controller_state._connected = True
        controller_state._send = AsyncMock()

        controller_state.clear()
        # Commit is async, so we need to run it synchronously
        loop = asyncio.get_event_loop()
        loop.run_until_complete(controller_state.commit())

        # Assert that the clear command was sent
        controller_state._send.assert_called_once_with(b"lcd:clear\n")

    def test_lcd_commit_with_changes_causes_lcd_command_to_be_sent(self):
        controller_state = ControllerState(ip="127.0.0.1", dip=1, port=1234)
        controller_state._connected = True
        controller_state._send = AsyncMock()

        controller_state.write_lcd(0, 0, "Hello, world!")

        loop = asyncio.get_event_loop()
        loop.run_until_complete(controller_state.commit())

        # Assert that the lcd command was sent
        controller_state._send.assert_called_once_with(b"lcd:0:0:Hello, world!\n")

    def test_lcd_commit_with_minimal_change_causes_correct_command_sequence(self):
        controller_state = ControllerState(ip="127.0.0.1", dip=1, port=1234)
        controller_state._connected = True
        controller_state._send = AsyncMock()

        loop = asyncio.get_event_loop()

        controller_state.write_lcd(0, 0, "Hello, world!")
        loop.run_until_complete(controller_state.commit())

        controller_state.write_lcd(0, 0, "Hello, there!")
        loop.run_until_complete(controller_state.commit())

        # Assert that the lcd command was sent
        controller_state._send.assert_has_calls(
            [
                call(b"lcd:0:0:Hello, world!\n"),
                call(b"lcd:7:0:there\n"),
            ]
        )

    def test_lcd_commit_with_multiple_changes_causes_correct_command_sequence(self):
        controller_state = ControllerState(ip="127.0.0.1", dip=1, port=1234)
        controller_state._connected = True
        controller_state._send = AsyncMock()

        loop = asyncio.get_event_loop()

        controller_state.write_lcd(0, 0, "ABCDEFGH")
        controller_state.write_lcd(0, 1, "IJKLMNOP")
        loop.run_until_complete(controller_state.commit())

        controller_state.write_lcd(0, 0, "ABCDEFGG")
        controller_state.write_lcd(0, 1, "JJKLMNOP")
        loop.run_until_complete(controller_state.commit())

        # Assert that the lcd command was sent
        controller_state._send.assert_has_calls(
            [
                call(b"lcd:0:0:ABCDEFGH\n"),
                call(b"lcd:0:1:IJKLMNOP\n"),
                call(b"lcd:7:0:G\n"),
                call(b"lcd:0:1:J\n"),
            ]
        )


if __name__ == "__main__":
    unittest.main()
