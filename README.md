# Volumetric Display Project

[![Build Status](https://github.com/fughilli/volumetric-display/actions/workflows/test.yaml/badge.svg)](https://github.com/fughilli/volumetric-display/blob/main/.github/workflows/test.yaml)

## Prerequisites

Make sure you have the following tools installed:

- [Bazelisk](https://github.com/bazelbuild/bazelisk)
- [Nix](https://nixos.org/download/)
- [Python 3.x](https://www.python.org/)

## Building the Project

Navigate to the project directory and build the project using Bazelisk:

```sh
bazelisk build //:simulator
```

## Running the Simulator

To run the volumetric display simulator, use the following command:

```sh
bazelisk run //:simulator -- --geometry=20x20x20 --ip=127.0.0.1 --port=6454 --universes-per-layer=6
```

You can adjust the `--geometry`, `--ip`, `--port`, and `--universes-per-layer` arguments as needed.

## Running a Scene

To run a specific scene, you can use a Python script like `:sender`. Here is an example command:

```sh
bazelisk run //:sender -- --scene=rainbow_scene.py --geometry=20x20x20 --ip=127.0.0.1 --port=6454 --brightness=1.0
```

Replace `rainbow_scene.py` with the scene file name.

## Running with TouchDesigner

An example project which renders a rotating donut is provided in `touchdesigner/donut.toe`.

To run the project, open the `donut.toe` file in TouchDesigner and start the simulation using the following command:

```sh
open touchdesigner/donut.toe

bazelisk run //:simulator -- --geometry=20x20x20 --alpha 0.5 -v 0 --ip 0.0.0.0
```

The DMX channel mapping passed to the DMX out CHOP can be generated with:

```sh
bazelisk run //:gen_routing_table -- --output-file=$(pwd)/touchdesigner/donut_routing_table.tsv
```
