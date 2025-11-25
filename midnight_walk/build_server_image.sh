#!/usr/bin/env bash
set -euo pipefail

# Get the repo root (one level up from this script's directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if ! command -v bazelisk >/dev/null 2>&1; then
  echo "bazelisk is required to build the wheel" >&2
  exit 1
fi

WHEEL_TARGET="//midnight_walk:midnight_walk_server_wheel"
ALEMBIC_TAR_TARGET="//midnight_walk:alembic_bundle_tar"

bazelisk build "${WHEEL_TARGET}" "${ALEMBIC_TAR_TARGET}"
WHEEL_PATH="$(bazelisk cquery --output=files ${WHEEL_TARGET})"
WHEEL_BASENAME="$(basename "${WHEEL_PATH}")"
ALEMBIC_TAR_PATH="$(bazelisk cquery --output=files ${ALEMBIC_TAR_TARGET})"
ALEMBIC_TAR_BASENAME="$(basename "${ALEMBIC_TAR_PATH}")"

# Create a temporary directory for the Docker build context
BUILD_DIR="$(mktemp -d -t midnight-docker-build-XXXXXX)"
trap 'rm -rf "${BUILD_DIR}"' EXIT

cp midnight_walk/docker/Dockerfile "${BUILD_DIR}/Dockerfile"
cp midnight_walk/docker/entrypoint.sh "${BUILD_DIR}/entrypoint.sh"
cp "${WHEEL_PATH}" "${BUILD_DIR}/${WHEEL_BASENAME}"
cp "${ALEMBIC_TAR_PATH}" "${BUILD_DIR}/${ALEMBIC_TAR_BASENAME}"

: "${DOCKER_DEFAULT_PLATFORM:=linux/amd64}"

pushd "${BUILD_DIR}" >/dev/null
if [[ -n "${DOCKER_DEFAULT_PLATFORM:-}" ]]; then
  docker build \
    --platform "${DOCKER_DEFAULT_PLATFORM}" \
    --build-arg "SERVER_WHEEL_FILE=${WHEEL_BASENAME}" \
    --build-arg "ALEMBIC_TAR_FILE=${ALEMBIC_TAR_BASENAME}" \
    -t midnight-walk-server \
    .
else
  docker build \
    --build-arg "SERVER_WHEEL_FILE=${WHEEL_BASENAME}" \
    --build-arg "ALEMBIC_TAR_FILE=${ALEMBIC_TAR_BASENAME}" \
    -t midnight-walk-server \
    .
fi
popd >/dev/null
