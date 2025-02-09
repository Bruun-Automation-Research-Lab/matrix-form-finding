import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_network3D(nodes, elements, fixed_nodes, external_loads):
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
                s=25,
                label="Normal" if node_id == 1 else "",
            )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.legend()
    plt.show()
