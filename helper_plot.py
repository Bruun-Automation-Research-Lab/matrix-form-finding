import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_network3D(nodes, elements, nodes_loads, nodes_fixed):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot elements
    for n1, n2 in elements - 1:  # Convert 1-based to 0-based indexing
        x_vals = [nodes[n1, 0], nodes[n2, 0]]
        y_vals = [nodes[n1, 1], nodes[n2, 1]]
        z_vals = [nodes[n1, 2], nodes[n2, 2]]
        ax.plot(x_vals, y_vals, z_vals, "k-", alpha=0.5)

    # Plot nodes
    for i, (x, y, z) in enumerate(nodes):
        if nodes_fixed[i]:  # Fixed node
            ax.scatter(
                x,
                y,
                z,
                color="blue",
                s=50,
                label=(
                    "Fixed"
                    if i == np.where(nodes_fixed[:, 0] == 1)[0][0]
                    else ""
                ),
            )
        elif np.any(nodes_loads[i] != 0):  # Loaded node
            ax.scatter(
                x,
                y,
                z,
                color="red",
                s=50,
                label=(
                    "Loaded"
                    if i == np.where(np.any(nodes_loads != 0, axis=1))[0][0]
                    else ""
                ),
            )
        else:  # Normal node
            ax.scatter(
                x, y, z, color="black", s=5, label="Normal" if i == 0 else ""
            )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.legend()
    plt.show()


def plot_network3D_old(nodes, elements, fixed_nodes, external_loads):
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
                label=(
                    "Loaded"
                    if node_id == list(external_loads.keys())[0]
                    else ""
                ),
            )
        else:
            ax.scatter(
                x,
                y,
                z,
                color="black",
                s=5,
                label="Normal" if node_id == 1 else "",
            )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.legend()
    plt.show()


# Function to plot the network at each step of the iteration
def plot_network_animated(ax, nodes, elements, fixed_nodes):
    ax.cla()  # Clear the axes
    x_vals = [nodes[node][0] for node in nodes]
    y_vals = [nodes[node][1] for node in nodes]
    z_vals = [nodes[node][2] for node in nodes]

    # Plot nodes
    ax.scatter(x_vals, y_vals, z_vals, c="b", marker=".", label="Nodes")

    # Highlight fixed nodes in red
    x_fixed = [nodes[node][0] for node in fixed_nodes]
    y_fixed = [nodes[node][1] for node in fixed_nodes]
    z_fixed = [nodes[node][2] for node in fixed_nodes]
    ax.scatter(
        x_fixed, y_fixed, z_fixed, c="r", marker="x", label="Fixed Nodes"
    )

    # Plot the elements (edges between nodes)
    for start, end in elements:
        x_start, y_start, z_start = nodes[abs(start)]
        x_end, y_end, z_end = nodes[abs(end)]
        ax.plot(
            [x_start, x_end],
            [y_start, y_end],
            [z_start, z_end],
            c="g",
            linestyle="-",
            linewidth=1,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
