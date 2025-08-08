import argparse
import csv


def export_routing_table(output_file, channels, nets, subnets, universes):
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t", lineterminator="\n")
        writer.writerow(["channel", "net", "subnet", "universe"])
        for row in zip(channels, nets, subnets, universes):
            writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate routing table")
    parser.add_argument("--output-file", help="Output file")
    parser.add_argument("--num-layers", type=int, default=20, help="Raster layers")
    parser.add_argument("--universes-per-layer", type=int, default=6, help="Universes per layer")
    args = parser.parse_args()

    channels = []
    nets = []
    subnets = []
    universes = []

    for layer in range(args.num_layers):
        for i in range(args.universes_per_layer):
            universe_base = layer * args.universes_per_layer + i

            universe = universe_base % 16
            subnet = (universe_base // 16) % 16
            net = universe_base // 256
            channel_index = 0
            channels.append("ch{}_{}".format(layer, i))
            nets.append(net)
            subnets.append(subnet)
            universes.append(universe)

    export_routing_table(args.output_file, channels, nets, subnets, universes)
