import matplotlib.pyplot as plt

labels = [
    "A",
    "A",
    "A",
    "A",
    "A",
    "B",
    "C",
    "D",
    "E",
    "E",
    "E",
    "E",
    "E",
    "D",
    "C1",
    "B1",
    "A1",
    "A1",
    "A1",
    "A1",
    "A1",
    "B1",
    "C1",
    "D",
    "E",
    "E",
    "E",
    "E",
    "E",
    "D",
    "C",
    "B",
    "A",
    "A",
    "A",
    "A",
    "A",
    "B",
    "C1",
    "D1",
    "E1",
    "E1",
    "E1",
    "E1",
    "E1",
    "D1",
    "C1",
    "B",
    "A",
    "A",
    "A",
    "A",
    "A",
    "B",
    "C",
    "D",
    "E",
    "E",
    "E",
    "E",
    "E",
    "D",
    "C1",
    "B1",
    "A1",
    "A1",
    "A1",
    "A1",
    "A1",
    "B1",
    "C1",
    "D",
    "E",
    "E",
    "E",
    "E",
    "E",
    "D",
    "C",
    "B",
    "A",
    "A",
    "A",
    "A",
    "A",
]


def plot_vertical_lines(labels):
    fig, ax = plt.subplots(figsize=(len(labels) * 0.15, 6))

    x_positions = range(len(labels))
    center = len(labels) // 2
    last = len(labels) - 1

    # Draw the lines with color coding
    for x in x_positions:
        # Determine color and linewidth based on position
        if x == center:
            color = "red"
            linewidth = 2.5
        elif x == 0 or x == last:
            color = "black"
            linewidth = 2.5
        elif abs(x - center) % 5 == 0:
            color = "blue"
            linewidth = 1.5
        else:
            color = "black"
            linewidth = 1

        ax.plot([x, x], [0, 1], color=color, linewidth=linewidth)

    # Add labels with color coding
    ax.set_xticks(x_positions)
    label_colors = []
    label_weights = []
    for x in x_positions:
        if x == center:
            label_colors.append("red")
            label_weights.append("bold")
        elif x == 0 or x == last:
            label_colors.append("black")
            label_weights.append("bold")
        elif abs(x - center) % 5 == 0:
            label_colors.append("blue")
            label_weights.append("normal")
        else:
            label_colors.append("black")
            label_weights.append("normal")

    ax.set_xticklabels(labels, rotation=90)

    # Color and style each label individually
    for ticklabel, color, weight in zip(ax.get_xticklabels(), label_colors, label_weights):
        ticklabel.set_color(color)
        ticklabel.set_fontweight(weight)

    # Add numbers for 1, 5, 10, 15, ..., 43 (center), ..., 85 along the top
    for x in x_positions:
        if x == 0 or (x + 1) % 5 == 0 or x == center:
            ax.text(x, 1.05, str(x + 1), ha="center", va="bottom", fontsize=8, color="gray")

    # Clean up the plot
    ax.set_yticks([])
    ax.set_xlim(-1, len(labels))
    ax.set_ylim(0, 1.15)  # Extended to accommodate top numbers
    ax.set_title("Curtain Label Diagram")

    # Remove the outer bounding box
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.tight_layout()
    plt.savefig("curtain_diagram.pdf", format="pdf", bbox_inches="tight")
    print("Diagram saved to curtain_diagram.pdf")
    plt.show()


plot_vertical_lines(labels)
