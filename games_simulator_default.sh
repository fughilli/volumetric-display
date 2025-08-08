#!/bin/bash

function run_display_sim() {
  bazel run :simulator -- \
    --geometry=20x20x20 \
    --alpha 0.8 \
    -v 0 \
    --ip 0.0.0.0 \
    --rotate_rate=0,10,0 \
    --universes_per_layer=3 &
}

function run_controller_sim() {
  bazel run :controller_simulator -- --config "$(pwd)/controller_config.json" &
}

function run_game_server() {
  bazel run :sender -- \
    --config="$(pwd)/sim_config.json" \
    --scene="$(pwd)/game_scene.py" \
    --brightness=1 \
    --layer-span=1 &
}

run_display_sim
GAME_SIM_PID=$!
run_controller_sim
CONTROLLER_SIM_PID=$!
sleep 5
run_game_server
GAME_SERVER_PID=$!

wait $CONTROLLER_SIM_PID

kill $GAME_SIM_PID
kill $GAME_SERVER_PID

wait $GAME_SIM_PID
wait $GAME_SERVER_PID
