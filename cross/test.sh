#!/bin/bash

set -euo pipefail

# Build the container
docker build -t test cross/

if [ "$1" == "shell" ]; then
  # If the first argument is "shell", start a shell in the container
  docker run -it --rm -v $(pwd):/app test /bin/bash
  exit 0
else
  # Run the container, mounting the current directory to /app
  docker run -v $(pwd):/app test
fi
