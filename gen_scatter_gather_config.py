#!/usr/bin/env python3
"""
Generate scatter-gather configuration from shaped array geometry.

This tool converts curtain configuration data into a scatter-gather LED coordinate mapping.
"""

import argparse
import json
from typing import Any, Dict, List


def generate_scatter_gather_config(
    curtain_sequence: List[str],
    curtain_configs: Dict[str, List[int]],
    controllers_per_group: int = 12,
    universes_per_curtain: int = 3,
    leds_per_universe: int = 170,
    base_ip: str = "127.0.0.1",
    base_port: int = 6454,
    controller_base_port: int = 51330,
) -> Dict[str, Any]:
    """
    Generate a scatter-gather configuration from curtain geometry.

    Args:
        curtain_sequence: List of curtain config names in order (e.g., ["A", "A", "B", ...])
        curtain_configs: Dict mapping config names to lists of strand lengths
        controllers_per_group: Number of curtains per artnet_mapping (default: 12)
        universes_per_curtain: Number of universes per curtain (default: 3)
        leds_per_universe: Maximum LEDs per universe (default: 170)
        base_ip: IP address for artnet (default: "127.0.0.1")
        base_port: Port for artnet (default: 6454)
        controller_base_port: Starting port for controller addresses (default: 51330)

    Returns:
        Complete configuration dictionary
    """

    # Calculate world geometry
    max_x = max(max(curtain_configs[config]) for config in curtain_configs)
    max_y = max(len(curtain_configs[config]) for config in curtain_configs)
    max_z = len(curtain_sequence)

    world_geometry = f"{max_x}x{max_y}x{max_z}"

    # Generate scatter-gather cubes
    scatter_gather_cubes = []

    # Group curtains by controllers
    num_groups = (len(curtain_sequence) + controllers_per_group - 1) // controllers_per_group

    for group_idx in range(num_groups):
        start_curtain = group_idx * controllers_per_group
        end_curtain = min(start_curtain + controllers_per_group, len(curtain_sequence))

        channel_samples = []

        # Process each curtain in this group
        for curtain_idx in range(start_curtain, end_curtain):
            config_name = curtain_sequence[curtain_idx]
            strand_lengths = curtain_configs[config_name]

            # Collect all coordinates for this curtain
            coords = []

            # Generate coordinates for each LED in each strand
            for strand_idx, strand_length in enumerate(strand_lengths):
                for led_idx in range(strand_length):
                    coords.append([led_idx, strand_idx, curtain_idx])

            # Calculate universe number with stride
            # Curtain 0 -> universe 0, curtain 1 -> universe 3, curtain 2 -> universe 6, etc.
            local_curtain_idx = curtain_idx - start_curtain
            universe_num = local_curtain_idx * universes_per_curtain

            channel_samples.append({"universe": universe_num, "coords": coords})

        artnet_mapping = {
            "ip": base_ip,
            "port": str(base_port + group_idx),
            "channel_samples": channel_samples,
        }

        scatter_gather_cubes.append({"artnet_mappings": [artnet_mapping]})

    # Generate controller addresses
    controller_addresses = {}
    for i in range(num_groups):
        controller_addresses[str(i)] = {"ip": base_ip, "port": controller_base_port + i}

    # Build complete configuration
    config = {
        "world_geometry": world_geometry,
        "scatter_gather_cubes": scatter_gather_cubes,
        "orientation": ["X", "Y", "Z"],
        "controller_addresses": controller_addresses,
        "scene": {"3d_snake": {"controller_mapping": {f"p{i+1}": i for i in range(num_groups)}}},
    }

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Generate scatter-gather configuration from shaped array geometry"
    )
    parser.add_argument(
        "--curtain-sequence",
        type=str,
        help="Path to JSON file containing curtain sequence array",
        required=True,
    )
    parser.add_argument(
        "--curtain-configs",
        type=str,
        help="Path to JSON file containing curtain configurations",
        required=True,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="scatter_gather_output.json",
        help="Output file path (default: scatter_gather_output.json)",
    )
    parser.add_argument(
        "--controllers-per-group",
        type=int,
        default=12,
        help="Number of curtains per artnet mapping (default: 12)",
    )
    parser.add_argument(
        "--universes-per-curtain",
        type=int,
        default=3,
        help="Number of universes per curtain (default: 3)",
    )
    parser.add_argument(
        "--leds-per-universe",
        type=int,
        default=170,
        help="Maximum LEDs per universe (default: 170)",
    )
    parser.add_argument(
        "--ip", type=str, default="127.0.0.1", help="IP address for artnet (default: 127.0.0.1)"
    )
    parser.add_argument("--port", type=int, default=6454, help="Port for artnet (default: 6454)")
    parser.add_argument(
        "--controller-base-port",
        type=int,
        default=51330,
        help="Starting port for controller addresses (default: 51330)",
    )

    args = parser.parse_args()

    # Load input files
    with open(args.curtain_sequence, "r") as f:
        curtain_sequence = json.load(f)

    with open(args.curtain_configs, "r") as f:
        curtain_configs = json.load(f)

    # Generate configuration
    config = generate_scatter_gather_config(
        curtain_sequence=curtain_sequence,
        curtain_configs=curtain_configs,
        controllers_per_group=args.controllers_per_group,
        universes_per_curtain=args.universes_per_curtain,
        leds_per_universe=args.leds_per_universe,
        base_ip=args.ip,
        base_port=args.port,
        controller_base_port=args.controller_base_port,
    )

    # Write output
    with open(args.output, "w") as f:
        json.dump(config, f, indent=2)

    print(f"✓ Generated scatter-gather configuration: {args.output}")
    print(f"  World geometry: {config['world_geometry']}")
    print(f"  Total curtains: {len(curtain_sequence)}")
    print(f"  Artnet groups: {len(config['scatter_gather_cubes'])}")
    print(f"  Total LEDs: {sum(sum(curtain_configs[name]) for name in curtain_sequence)}")


if __name__ == "__main__":
    main()
