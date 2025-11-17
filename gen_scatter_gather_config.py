#!/usr/bin/env python3
"""
Generate scatter-gather configuration from shaped array geometry.

This tool converts curtain configuration data into a scatter-gather LED coordinate mapping.
"""

import argparse
import json
from typing import Any, Dict, List


def generate_scatter_gather_config(
    curtain_sequence: Dict[str, Dict[str, Any]],
    curtain_configs: Dict[str, List[int]],
    universes_per_curtain: int = 3,
    leds_per_universe: int = 170,
    controller_base_port: int = 51330,
) -> Dict[str, Any]:
    """
    Generate a scatter-gather configuration from curtain geometry.

    Args:
        curtain_sequence: Dict mapping "ip:port" to dict with "configs" list and optional "start_z"
        curtain_configs: Dict mapping config names to lists of strand lengths
        universes_per_curtain: Number of universes per curtain (default: 3)
        leds_per_universe: Maximum LEDs per universe (default: 170)
        controller_base_port: Starting port for controller addresses (default: 51330)

    Returns:
        Complete configuration dictionary
    """

    # Calculate world geometry
    max_x = max(max(curtain_configs[config]) for config in curtain_configs)
    max_y = max(len(curtain_configs[config]) for config in curtain_configs)

    # Calculate total number of curtains across all controllers
    total_curtains = sum(len(entry["configs"]) for entry in curtain_sequence.values())
    max_z = total_curtains

    world_geometry = f"{max_x}x{max_y}x{max_z}"

    # Generate scatter-gather cubes
    scatter_gather_cubes = []

    # Track global curtain index (z coordinate)
    global_curtain_idx = 0

    # Process each controller (each IP:port entry)
    for controller_idx, (ip_port, entry) in enumerate(curtain_sequence.items()):
        # Parse IP and port from the key
        ip, port = ip_port.split(":")

        # Extract configs list and optional start_z
        curtains = entry["configs"]
        start_z = entry.get("start_z")

        # Reset global curtain index if start_z is specified
        if start_z is not None:
            global_curtain_idx = start_z

        channel_samples = []

        # Process each curtain on this controller
        for local_curtain_idx, config_name in enumerate(curtains):
            strand_lengths = curtain_configs[config_name]

            # Collect all coordinates for this curtain
            coords = []

            # Generate coordinates for each LED in each strand
            for strand_idx, strand_length in enumerate(strand_lengths):
                for led_idx in range(strand_length):
                    coords.append([led_idx, strand_idx, global_curtain_idx])

            # Calculate universe number with stride
            # Curtain 0 -> universe 0, curtain 1 -> universe 3, curtain 2 -> universe 6, etc.
            universe_num = local_curtain_idx * universes_per_curtain

            channel_samples.append({"universe": universe_num, "coords": coords})

            global_curtain_idx += 1

        artnet_mapping = {
            "ip": ip,
            "port": port,
            "channel_samples": channel_samples,
        }

        scatter_gather_cubes.append({"artnet_mappings": [artnet_mapping]})

    # Generate controller addresses
    controller_addresses = {}
    for i, ip_port in enumerate(sorted(curtain_sequence.keys())):
        ip, _ = ip_port.split(":")
        controller_addresses[str(i)] = {"ip": ip, "port": controller_base_port + i}

    # Build complete configuration
    num_controllers = len(curtain_sequence)
    config = {
        "world_geometry": world_geometry,
        "scatter_gather_cubes": scatter_gather_cubes,
        "orientation": ["X", "Y", "Z"],
        "controller_addresses": controller_addresses,
        "scene": {
            "3d_snake": {"controller_mapping": {f"p{i+1}": i for i in range(num_controllers)}}
        },
    }

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Generate scatter-gather configuration from shaped array geometry"
    )
    parser.add_argument(
        "--curtain-sequence",
        type=str,
        help="Path to JSON file containing curtain sequence dict mapping IP:port to curtain list",
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
        universes_per_curtain=args.universes_per_curtain,
        leds_per_universe=args.leds_per_universe,
        controller_base_port=args.controller_base_port,
    )

    # Write output
    with open(args.output, "w") as f:
        json.dump(config, f, indent=2)

    # Calculate summary statistics
    total_curtains = sum(len(entry["configs"]) for entry in curtain_sequence.values())
    total_leds = sum(
        sum(curtain_configs[name])
        for entry in curtain_sequence.values()
        for name in entry["configs"]
    )

    print(f"✓ Generated scatter-gather configuration: {args.output}")
    print(f"  World geometry: {config['world_geometry']}")
    print(f"  Total controllers: {len(curtain_sequence)}")
    print(f"  Total curtains: {total_curtains}")
    print(f"  Total LEDs: {total_leds}")


if __name__ == "__main__":
    main()
