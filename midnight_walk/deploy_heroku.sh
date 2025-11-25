#!/usr/bin/env bash
set -euo pipefail

# Get the repo root (one level up from this script's directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

HEROKU_APP="${HEROKU_APP:-midnight-walk}"
IMAGE_NAME="${IMAGE_NAME:-midnight-walk-server}"
HEROKU_IMAGE="registry.heroku.com/${HEROKU_APP}/web"

# Ensure we match Heroku's x86_64 runtime
: "${DOCKER_DEFAULT_PLATFORM:=linux/amd64}"

echo "🔨 Building wheel and Docker image…"
./midnight_walk/build_server_image.sh

echo "🪝 Logging into Heroku Container Registry…"
heroku container:login

echo "🏷️  Tagging image ${IMAGE_NAME} -> ${HEROKU_IMAGE}"
docker tag "${IMAGE_NAME}" "${HEROKU_IMAGE}"

echo "📤 Pushing image to Heroku…"
docker push "${HEROKU_IMAGE}"

echo "🚀 Releasing container on Heroku app '${HEROKU_APP}'"
heroku container:release web -a "${HEROKU_APP}"

echo "✅ Deployment complete."
