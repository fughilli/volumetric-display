#!/usr/bin/env python3
"""
Chuck Sound Server Launcher
Launches the Chuck sound server using Bazel runfiles
"""

import os
import subprocess
import sys
from pathlib import Path


def find_chuck_sound_server():
    """Find the Chuck sound server file in Bazel runfiles"""
    # Try to find the sound server in runfiles
    runfiles_dir = os.environ.get("RUNFILES_DIR")
    if runfiles_dir:
        # Look for the sound server in the runfiles
        sound_server_path = os.path.join(runfiles_dir, "sounds/sound_server.ck")
        if os.path.exists(sound_server_path):
            return sound_server_path

    # Fallback: look in current directory
    current_dir = Path(__file__).parent
    sound_server_path = current_dir / "sound_server.ck"
    if sound_server_path.exists():
        return str(sound_server_path)

    # Fallback: look in workspace
    workspace_root = Path(__file__).parent.parent
    sound_server_path = workspace_root / "sounds" / "sound_server.ck"
    if sound_server_path.exists():
        return str(sound_server_path)

    return None


def check_chuck_installed():
    """Check if Chuck is installed and available"""
    try:
        result = subprocess.run(["chuck", "--version"], capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def main():
    """Main function to launch the Chuck sound server"""
    print("Chuck Sound Server Launcher")
    print("=" * 30)

    # Check if Chuck is installed
    if not check_chuck_installed():
        print("Error: Chuck is not installed or not in PATH")
        print("Please install Chuck from https://chuck.cs.princeton.edu/")
        print("On macOS: brew install chuck")
        print("On Ubuntu: sudo apt-get install chuck")
        sys.exit(1)

    # Find the sound server file
    sound_server_path = find_chuck_sound_server()
    if not sound_server_path:
        print("Error: Could not find sound_server.ck")
        print("Make sure the file exists in the sounds directory")
        sys.exit(1)

    print(f"Found sound server at: {sound_server_path}")
    print("Starting Chuck sound server on port 6449...")
    print("Press Ctrl+C to stop the server")
    print("-" * 30)

    try:
        # Start the Chuck sound server
        process = subprocess.Popen(["chuck", sound_server_path])

        # Wait for the process to complete
        process.wait()

    except KeyboardInterrupt:
        print("\nStopping sound server...")
        if process:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        print("Sound server stopped.")
    except Exception as e:
        print(f"Error running sound server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
