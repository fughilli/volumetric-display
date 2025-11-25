#!/usr/bin/env bash
set -euo pipefail

: "${MIDNIGHT_SERVER_HOST:=0.0.0.0}"

exec midnight-walk
