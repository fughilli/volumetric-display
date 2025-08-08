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

You can adjust the `--geometry`, `--ip`, `--port`, and `--universes-per-layer`
arguments as needed.

## Running a Scene

To run a specific scene, you can use a Python script like `:sender`. Here is an
example command:

```sh
bazelisk run //:sender -- --scene=rainbow_scene.py --geometry=20x20x20 --ip=127.0.0.1 --port=6454 --brightness=1.0
```

Replace `rainbow_scene.py` with the scene file name.

## Running with TouchDesigner

An example project which renders a rotating donut is provided in
`touchdesigner/donut.toe`.

To run the project, open the `donut.toe` file in TouchDesigner and start the
simulation using the following command:

```sh
open touchdesigner/donut.toe

bazelisk run //:simulator -- --geometry=20x20x20 --alpha 0.5 -v 0 --ip 0.0.0.0
```

The DMX channel mapping passed to the DMX out CHOP can be generated with:

```sh
bazelisk run //:gen_routing_table -- --output-file=$(pwd)/touchdesigner/donut_routing_table.tsv
```

## Running Games

The games implementation lives in the `games/` subdirectory. To run the games
under simulation, you can start an instance of the display simulator, game
server, and controller simulator in three separate terminals like so:

```sh
# Terminal 1 (Display Simulator)
bazel run :simulator -- --geometry=20x20x20 --alpha 0.8 -v 0 --ip 0.0.0.0 --rotate_rate=0,0,0 --universes_per_layer=3
```

```sh
# Terminal 2 (Controller Simulator)
bazel run :controller_simulator -- --config (pwd)/controller_config.json
```

```sh
# Terminal 3 (Game Server)
bazel run :sender -- --config=(pwd)/sim_config.json --scene=$(pwd)/game_scene.py --brightness=1 --layer-span=1
```

Once all three are running, you should see a rotating point cloud in the shape
of a cube within the display, and the four controller displays should show a
game selection menu. You can interact with each controller using "WASD"-style
key assignments at 4 locations on the keyboard. E.g. controller 1 uses keys `2`,
`Q`, `W`, `E`, `3` for UP, LEFT, DOWN, RIGHT, SELECT, respectively.

Instead of starting all 3 sequentially, you can also use the convenience run
script:

```sh
./games_simulator_default.sh
```

## Development Setup

### Pre-commit Hooks

This project uses pre-commit hooks to ensure code quality and consistency. The
hooks include:

- **Python**: Black (formatting), isort (import sorting), flake8 (linting)
- **Rust**: rustfmt (formatting), clippy (linting)
- **C/C++**: clang-format (formatting)
- **Shell**: shellcheck (linting)
- **Bazel**: buildifier (BUILD file formatting)
- **Nix**: nixpkgs-fmt (formatting)
- **Markdown**: prettier (formatting), markdownlint (linting)
- **JSON/YAML**: prettier (formatting)
- **General**: end-of-line fixing, trailing whitespace removal

#### Installation

##### Option 1: Manual Setup

1. Install pre-commit:

   ```sh
   pip install pre-commit
   ```

2. Install the git hooks:

   ```sh
   pre-commit install
   ```

##### Option 2: Automated Setup

Run the provided setup script:

```sh
./setup-precommit.sh
```

_Note: Pre-commit is not managed through Bazel's requirements system and should
be installed separately for development._

#### Usage

- The hooks will run automatically on every commit
- To run manually on all files:

  ```sh
  pre-commit run --all-files
  ```

- To run on specific files:

  ```sh
  pre-commit run --files path/to/file
  ```

- To skip hooks for a commit (not recommended):

  ```sh
  git commit --no-verify
  ```

#### Configuration Files

- `.pre-commit-config.yaml`: Main pre-commit configuration
- `.clang-format`: C/C++ formatting rules
- `pyproject.toml`: Python tool configurations
- `.markdownlint.json`: Markdown linting rules
- `.prettierrc`: Prettier formatting rules
