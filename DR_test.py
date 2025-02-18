import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class DynamicRelaxation:
    def __init__(
        self,
        nodes,
        edges,
        fixed,
        mass=1.0,
        damping=0.30,
        dt=0.1,
        tol=1e-3,
        external_forces=None,
    ):
        self.nodes = np.array(nodes, dtype=float)  # Node coordinates
        self.edges = np.array(edges)  # Connectivity
        self.fixed = fixed  # Fixed nodes
        self.mass = mass  # Mass per node
        self.damping = damping  # Damping factor
        self.dt = dt  # Time step
        self.tol = tol  # Convergence tolerance
        self.velocities = np.zeros_like(self.nodes)  # Initial velocities
        self.external_forces = (
            np.array(external_forces)
            if external_forces is not None
            else np.zeros_like(self.nodes)
        )
        self.frames = []  # Store frames for animation

    def length(self, edge):
        """Compute length of an edge."""
        return np.linalg.norm(self.nodes[edge[1]] - self.nodes[edge[0]])

    def solve(self, target_length=None, save_every_n=20):
        """Iteratively solve for equilibrium."""
        converged = False
        iteration = 0
        forces = np.zeros_like(self.nodes)
        target_length = target_length or {
            tuple(e): self.length(e) for e in self.edges
        }

        while not converged:
            forces.fill(0)  # Reset forces

            # Compute internal forces
            for edge in self.edges:
                i, j = edge
                vec = self.nodes[j] - self.nodes[i]
                length = np.linalg.norm(vec)
                direction = vec / (length + 1e-9)
                force_magnitude = length - target_length[tuple(edge)]
                forces[i] += force_magnitude * direction
                forces[j] -= force_magnitude * direction

            # Apply external forces
            forces += self.external_forces

            # Apply boundary conditions
            forces[self.fixed] = 0

            # Update velocities and positions
            self.velocities += (forces / self.mass) * self.dt
            self.velocities *= self.damping
            self.nodes += self.velocities * self.dt

            # Store frames for animation
            if iteration % save_every_n == 0:
                self.frames.append(np.copy(self.nodes))

            # Check for convergence
            max_displacement = np.max(np.abs(self.velocities))

            # Print iteration info every 100 steps
            if iteration % 100 == 0:
                print(
                    f"Iter {iteration:6d} | Max Disp: {max_displacement:.3e}"
                )

            if max_displacement < self.tol:
                converged = True

            iteration += 1
            if iteration > 10000:
                print("Warning: Max iterations reached")
                break

        print(f"Converged in {iteration} iterations.")
        return self.nodes

    def plot_structure(
        self, nodes, edges, fixed, external_forces, original_nodes=None
    ):
        """Plot the structural configuration."""
        plt.figure(figsize=(8, 6))

        # Plot original structure if provided
        if original_nodes is not None:
            for edge in edges:
                plt.plot(
                    *zip(*np.array(original_nodes)[edge]),
                    color="gray",
                    linestyle="dashed",
                    linewidth=0.5,
                    alpha=0.5,
                )

        # Plot final structure
        for edge in edges:
            plt.plot(*zip(*nodes[edge]), color="black")

        # Identify nodes with applied loads
        loaded_nodes = np.any(external_forces != 0, axis=1)

        # Color nodes based on type
        nodes = np.array(nodes)
        plt.scatter(
            nodes[:, 0], nodes[:, 1], color="black", label="Free Nodes"
        )
        plt.scatter(
            nodes[fixed, 0], nodes[fixed, 1], color="red", label="Fixed Nodes"
        )
        plt.scatter(
            nodes[loaded_nodes, 0],
            nodes[loaded_nodes, 1],
            color="green",
            label="Loaded Nodes",
        )

        # Label nodes with their index, shifted slightly to avoid overlap
        for i, (x, y) in enumerate(nodes):
            plt.text(
                x + 0.15,
                y + 0.15,
                str(i),
                fontsize=12,
                ha="right",
                color="black",
            )

        plt.legend()
        plt.show()

    def animate_convergence(self, edges, fixed, external_forces):
        """Create an animation of the convergence process."""
        fig, ax = plt.subplots(figsize=(8, 6))

        def update(frame):
            ax.clear()
            nodes = self.frames[frame]
            for edge in edges:
                ax.plot(*zip(*nodes[edge]), color="black")

            loaded_nodes = np.any(external_forces != 0, axis=1)
            ax.scatter(nodes[:, 0], nodes[:, 1], color="black")
            ax.scatter(nodes[fixed, 0], nodes[fixed, 1], color="red")
            ax.scatter(
                nodes[loaded_nodes, 0], nodes[loaded_nodes, 1], color="green"
            )

            for i, (x, y) in enumerate(nodes):
                ax.text(
                    x + 0.15,
                    y + 0.15,
                    str(i),
                    fontsize=12,
                    ha="right",
                    color="black",
                )

            # Calculate max displacement as the maximum change in node position
            if frame > 0:
                prev_nodes = self.frames[frame - 1]
                displacement = np.linalg.norm(nodes - prev_nodes, axis=1)
                max_displacement = np.max(displacement)
            else:
                max_displacement = 0.0

            # Display iteration number and max displacement (in sci notation)
            ax.text(
                0.1,
                3.5,
                f"Iter: {frame * 20}\nMax Disp. (m): {max_displacement:.3e}",
                fontsize=12,
                color="blue",
            )
            ax.set_xlim(-1, 10)
            ax.set_ylim(-1, 4)

        _ = animation.FuncAnimation(
            fig, update, frames=len(self.frames), interval=100
        )
        plt.show()


def generate_grid(rows, cols, spacing=1.0):
    """Generate a 2D grid of nodes and edges."""
    nodes = []
    edges = []

    # Create nodes
    for i in range(rows):
        for j in range(cols):
            nodes.append([j * spacing, i * spacing])

    nodes = np.array(nodes)

    # Create edges (horizontal and vertical)
    for i in range(rows):
        for j in range(cols):
            index = i * cols + j
            # Connect to right neighbor
            if j < cols - 1:
                edges.append([index, index + 1])
            # Connect to upper neighbor
            if i < rows - 1:
                edges.append([index, index + cols])

    return nodes, np.array(edges)


# Example usage
def example():
    nodes = [
        [0, 0],
        [1, 2],
        [2, 1],
        [3, 3],
        [4, 0],
        [5, 2],
        [6, 1],
        [7, 3],
        [8, 0],
        [9, 2],
    ]  # Initial positions
    edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [8, 9],
        [9, 0],
        [1, 3],
        [3, 5],
        [5, 7],
        [7, 9],
        [9, 1],
    ]  # Connectivity
    fixed = [0, 4, 8]  # Fixed nodes
    external_forces = np.zeros((10, 2))
    external_forces[5] = [1.0, -1.5]  # Apply downward force to node 5

    dr = DynamicRelaxation(
        nodes, edges, fixed, external_forces=external_forces
    )
    relaxed_nodes = dr.solve()
    dr.plot_structure(
        relaxed_nodes, edges, fixed, external_forces, original_nodes=nodes
    )
    dr.animate_convergence(edges, fixed, external_forces)


def example2():
    rows, cols = 5, 6  # 5x6 grid
    nodes, edges = generate_grid(rows, cols)

    fixed = list(range(cols))  # Bottom row is fixed (first 'cols' nodes)

    external_forces = np.zeros((rows * cols, 2))
    mid_right = (rows // 2) * cols + (cols - 1)  # Middle node on right side
    external_forces[mid_right] = [1.0, 0.5]  # Rightward & upward force

    dr = DynamicRelaxation(
        nodes, edges, fixed, external_forces=external_forces
    )
    relaxed_nodes = dr.solve()
    dr.plot_structure(
        relaxed_nodes, edges, fixed, external_forces, original_nodes=nodes
    )
    dr.animate_convergence(edges, fixed, external_forces)


example2()
