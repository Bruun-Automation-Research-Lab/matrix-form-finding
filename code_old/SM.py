import numpy as np
import matplotlib.pyplot as plt


def truss_solver(
    nodes, elements, nodes_loads, nodes_fixed, E, A, tol=1e-5, max_iter=100
):
    num_nodes = len(nodes)
    dof = 2 * num_nodes  # Degrees of freedom (2 per node)
    K_global = np.zeros((dof, dof))  # Global stiffness matrix
    F_global = np.zeros(dof)  # Global force vector

    # Initial member lengths
    member_lengths = {}
    for n1, n2 in elements:
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        L = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        member_lengths[(n1, n2)] = L

    for iteration in range(max_iter):
        K_global.fill(0)  # Reset the global stiffness matrix
        # Assemble global stiffness matrix
        for n1, n2 in elements:
            x1, y1 = nodes[n1]
            x2, y2 = nodes[n2]
            L = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            c = (x2 - x1) / L
            s = (y2 - y1) / L

            k_local = (E * A / L) * np.array(
                [
                    [c**2, c * s, -(c**2), -c * s],
                    [c * s, s**2, -c * s, -(s**2)],
                    [-(c**2), -c * s, c**2, c * s],
                    [-c * s, -(s**2), c * s, s**2],
                ]
            )

            indices = [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1]
            for i in range(4):
                for j in range(4):
                    K_global[indices[i], indices[j]] += k_local[i, j]

        # Apply nodal loads
        F_global.fill(0)  # Reset force vector
        for node, (fx, fy) in nodes_loads.items():
            F_global[2 * node] = fx
            F_global[2 * node + 1] = fy

        # Apply boundary conditions
        free_dofs = []
        fixed_dofs = []
        for node, is_fixed in nodes_fixed.items():
            if is_fixed:
                fixed_dofs.extend([2 * node, 2 * node + 1])
            else:
                free_dofs.extend([2 * node, 2 * node + 1])

        # Partition system
        K_ff = K_global[np.ix_(free_dofs, free_dofs)]
        F_f = F_global[free_dofs]

        # Solve for displacements
        U_f = np.linalg.solve(K_ff, F_f)

        # Reconstruct full displacement vector
        U_global = np.zeros(dof)
        U_global[free_dofs] = U_f

        # Compute new member lengths and check for convergence
        new_member_lengths = {}
        for n1, n2 in elements:
            x1, y1 = nodes[n1] + U_global[2 * n1 : 2 * n1 + 2]
            x2, y2 = nodes[n2] + U_global[2 * n2 : 2 * n2 + 2]
            new_L = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            new_member_lengths[(n1, n2)] = new_L

        # Check if the lengths are converging
        max_change = max(
            abs(new_member_lengths[(n1, n2)] - member_lengths[(n1, n2)])
            for (n1, n2) in elements
        )
        if max_change < tol:
            print(f"Converged after {iteration + 1} iterations")
            member_lengths = new_member_lengths
            break

        member_lengths = (
            new_member_lengths  # Update member lengths for the next iteration
        )

    # Compute element forces
    element_forces = []
    for n1, n2 in elements:
        x1, y1 = nodes[n1] + U_global[2 * n1 : 2 * n1 + 2]
        x2, y2 = nodes[n2] + U_global[2 * n2 : 2 * n2 + 2]
        L = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        c = (x2 - x1) / L
        s = (y2 - y1) / L

        T = np.array([c, s, -c, -s])

        Ue = np.array(
            [
                U_global[2 * n1],
                U_global[2 * n1 + 1],
                U_global[2 * n2],
                U_global[2 * n2 + 1],
            ]
        )

        force = (E * A / L) * np.dot(T, Ue)
        element_forces.append(force)

    return U_global.reshape(-1, 2), element_forces


def plot_truss(nodes, elements, displacements, scale=1.0):
    plt.figure(figsize=(8, 6))

    # Plot original truss
    for n1, n2 in elements:
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        plt.plot([x1, x2], [y1, y2], "k--", alpha=0.5)

    # Plot deformed truss
    deformed_nodes = {
        i: (
            nodes[i][0] + scale * displacements[i, 0],
            nodes[i][1] + scale * displacements[i, 1],
        )
        for i in nodes
    }
    for n1, n2 in elements:
        x1, y1 = deformed_nodes[n1]
        x2, y2 = deformed_nodes[n2]
        plt.plot([x1, x2], [y1, y2], "r-")

    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Truss Deformation (Scaled)")
    plt.legend(["Original", "Deformed"], loc="best")
    plt.axis("equal")
    plt.grid(True)
    plt.show()


# Example Usage
nodes = {
    0: (0.0, 0.0),
    1: (2.0, 0.0),
    2: (1.5, 2.0),
}

elements = [
    (0, 1),
    (1, 2),
    (2, 0),
]

nodes_loads = {
    0: (0.0, 0.0),
    1: (0.0, 0.0),
    2: (1.0, 1.0),
}

nodes_fixed = {
    0: 1,
    1: 1,
    2: 0,
}

E = 1  # Modulus of elasticity
A = 1  # Cross-sectional area

displacements, element_forces = truss_solver(
    nodes, elements, nodes_loads, nodes_fixed, E, A
)
plot_truss(nodes, elements, displacements, scale=1)
