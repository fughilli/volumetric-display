#!/bin/bash

# Start Chuck Sound Server
# This script starts the Chuck sound server for game audio

echo "Starting Chuck Sound Server..."

# Check if Chuck is installed
if ! command -v chuck &> /dev/null; then
    echo "Error: Chuck is not installed or not in PATH"
    echo "Please install Chuck from https://chuck.cs.princeton.edu/"
    exit 1
fi

# Check if the sound server file exists
if [ ! -f "sounds/sound_server.ck" ]; then
    echo "Error: sounds/sound_server.ck not found"
    exit 1
fi

# Start the Chuck sound server
echo "Launching sound server on port 6449..."
chuck sounds/sound_server.ck

echo "Sound server stopped."
