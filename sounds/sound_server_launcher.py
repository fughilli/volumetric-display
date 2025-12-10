#!/usr/bin/env python3
"""
Chuck Sound Server Launcher
Launches the Chuck sound server using Bazel runfiles
"""

import subprocess
import sys
from pathlib import Path

try:
    from rules_python.python.runfiles import runfiles

    RUNFILES_AVAILABLE = True
except ImportError:
    RUNFILES_AVAILABLE = False


def find_chuck_binary():
    """Find the Chuck binary in Bazel runfiles"""
    if RUNFILES_AVAILABLE:
        r = runfiles.Create()
        # Try the runfile path for @chuck//:bin/chuck
        chuck_path = r.Rlocation("chuck/bin/chuck")
        if chuck_path:
            return chuck_path

    # Fallback: try system chuck
    try:
        result = subprocess.run(["chuck", "--version"], capture_output=True, text=True, timeout=1)
        if result.returncode == 0:
            return "chuck"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return None


def find_chuck_sound_server():
    """Find the Chuck sound server file in Bazel runfiles"""
    if RUNFILES_AVAILABLE:
        r = runfiles.Create()
        # Try the runfile path for the sound server
        sound_server_path = r.Rlocation("sounds/sound_server.ck")
        if sound_server_path:
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


def main():
    """Main function to launch the Chuck sound server"""
    print("Chuck Sound Server Launcher")
    print("=" * 30)

    # Find Chuck binary from runfiles or system
    chuck_binary = find_chuck_binary()
    if not chuck_binary:
        print("Error: Chuck binary not found")
        print("Chuck should be available via Nix package in Bazel runfiles")
        sys.exit(1)

    print(f"Found Chuck binary at: {chuck_binary}")

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
        # Start the Chuck sound server using the binary from runfiles
        process = subprocess.Popen([chuck_binary, sound_server_path])

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
