import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 2. Plot the network
def plot_network(nodes, elements):
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot each node
    for node, (x, y) in nodes.items():
        ax.scatter(x, y, color="blue")
        ax.text(x + 0.1, y + 0.1, str(node), fontsize=12)

    # Plot each element (line between connected nodes)
    for start, end in elements:
        x_start, y_start = nodes[abs(start)]
        x_end, y_end = nodes[abs(end)]

        # Reverse the direction if start is negative
        if start < 0:
            ax.plot([x_end, x_start], [y_end, y_start], color="black")
        else:
            ax.plot([x_start, x_end], [y_start, y_end], color="black")

    # Set equal scaling and grid
    ax.set_aspect("equal")
    ax.grid(True)
    plt.show()


def plot_network3D(nodes, elements):
    # Create a 3D figure and axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot each node
    for node, (x, y, z) in nodes.items():
        ax.scatter(x, y, z, color="blue", s=50)  # Plot nodes as blue points
        ax.text(x + 0.1, y + 0.1, z, str(node), fontsize=12)  # Add node labels

    # Plot each element (line between connected nodes)
    for start, end in elements:
        x_start, y_start, z_start = nodes[abs(start)]
        x_end, y_end, z_end = nodes[abs(end)]

        # Reverse the direction if start is negative
        if start < 0:
            ax.plot(
                [x_end, x_start],
                [y_end, y_start],
                [z_end, z_start],
                color="black",
            )
        else:
            ax.plot(
                [x_start, x_end],
                [y_start, y_end],
                [z_start, z_end],
                color="black",
            )

    # Set labels for axes
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Display the plot
    plt.show()


def plot_network3D_2(nodes, elements, fixed_nodes, external_loads):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot elements
    for n1, n2 in elements:
        x_vals = [nodes[n1][0], nodes[n2][0]]
        y_vals = [nodes[n1][1], nodes[n2][1]]
        z_vals = [nodes[n1][2], nodes[n2][2]]
        ax.plot(x_vals, y_vals, z_vals, "k-", alpha=0.5)

    # Plot nodes
    for node_id, (x, y, z) in nodes.items():
        if node_id in fixed_nodes:
            ax.scatter(
                x,
                y,
                z,
                color="blue",
                s=50,
                label="Fixed" if node_id == fixed_nodes[0] else "",
            )
        elif any(val != 0.0 for val in external_loads[node_id]):
            ax.scatter(
                x,
                y,
                z,
                color="red",
                s=50,
                label="Loaded"
                if node_id == list(external_loads.keys())[0]
                else "",
            )
        else:
            ax.scatter(
                x,
                y,
                z,
                color="black",
                s=25,
                label="Normal" if node_id == 1 else "",
            )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.legend()
    plt.show()
