import numpy as np


def compute_CTKC(C_dim, K_dim):
    # C_dim should be (m, n) and K_dim should be (n, n) for valid  mult
    C = np.random.rand(*C_dim)  # C is a matrix of dimensions (m, n)
    K = np.random.rand(*K_dim)  # K is a matrix of dimensions (n, n)

    # Calculate C^T * K * C
    result = C.T @ K @ C
    return result


def direction_cosines(x1, y1, x2, y2):
    # Compute direction cosines for the element from (x1, y1) to (x2, y2)
    L = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)  # Length of the element
    cos_theta = (x2 - x1) / L  # Direction cosine in x-direction
    sin_theta = (y2 - y1) / L  # Direction cosine in y-direction
    return L, cos_theta, sin_theta


def local_stiffness(E, A, L, cos_theta, sin_theta):
    # Local stiffness matrix for a 2D truss element
    k_local = (E * A / L) * np.array(
        [
            [
                cos_theta**2,
                cos_theta * sin_theta,
                -(cos_theta**2),
                -cos_theta * sin_theta,
            ],
            [
                cos_theta * sin_theta,
                sin_theta**2,
                -cos_theta * sin_theta,
                -(sin_theta**2),
            ],
            [
                -(cos_theta**2),
                -cos_theta * sin_theta,
                cos_theta**2,
                cos_theta * sin_theta,
            ],
            [
                -cos_theta * sin_theta,
                -(sin_theta**2),
                cos_theta * sin_theta,
                sin_theta**2,
            ],
        ]
    )
    return k_local


def assemble_global_stiffness(nodes, E, A, elements):
    # Initialize a global stiffness matrix (6x6 for 3 nodes, each with 2 DOF)
    num_dofs = len(nodes) * 2  # 2 DOF per node (x and y displacements)
    K_global = np.zeros((num_dofs, num_dofs))

    for element in elements:
        node1, node2 = element  # Nodes connected by the element

        # Get the coordinates of the nodes
        x1, y1 = nodes[node1]
        x2, y2 = nodes[node2]

        # Get the direction cosines and length of the element
        L, cos_theta, sin_theta = direction_cosines(x1, y1, x2, y2)

        # Compute the local stiffness matrix for this element
        k_local = local_stiffness(E, A, L, cos_theta, sin_theta)

        # Global DOF indices for the nodes
        dof_indices = [
            [2 * node1, 2 * node1 + 1, 2 * node2, 2 * node2 + 1]
            for node1, node2 in elements
        ]

        # Add the local stiffness matrix to the global stiffness matrix
        for i in range(4):
            for j in range(4):
                K_global[
                    dof_indices[node1][i], dof_indices[node2][j]
                ] += k_local[i, j]

    return K_global


# Define the load vector f
def create_load_vector(nodes_loads):
    f = np.zeros(2 * len(nodes_loads))
    for node, load in nodes_loads.items():
        f[2 * (node - 1) : 2 * node] = np.array(load)
    return f


# Function to apply boundary conditions and solve for displacements
def apply_boundary_conditions(K_global, f, nodes_fixed):
    # Remove the fixed degrees of freedom
    for node, fixed in nodes_fixed.items():
        if fixed == 1:
            # Zero the corresponding row and column for fixed nodes
            dof = 2 * (node - 1)
            K_global[dof, :] = 0
            K_global[:, dof] = 0
            K_global[dof, dof] = 1  # To avoid singular matrix
            f[dof] = 0  # No displacement at fixed DOF
    return K_global, f


# Solve the system for displacements
def solve_displacements(K_global, f):
    # Solve for displacements
    u = np.linalg.solve(K_global, f)
    return u


# Example Usage
# Define nodes with (x, y) coordinates
nodes = {
    0: (0.0, 0.0),  # Node 1 at (0, 0)
    1: (2.0, 0.0),  # Node 2 at (1, 0)
    2: (1.5, 2.0),  # Node 3 at (0.5, 1)
}

# Define the elements (pairs of connected nodes)
elements = [
    (0, 1),  # Element 1 connects Node 1 and Node 2
    (1, 2),  # Element 2 connects Node 2 and Node 3
    (2, 0),  # Element 3 connects Node 3 and Node 1
]

nodes_loads = {
    1: (0.0, 0.0),
    2: (0.0, 0.0),
    3: (0.0, 0.1),
}

nodes_fixed = {
    1: 1,
    2: 1,
    3: 0,
}

# Material and geometry properties
E = 1  # Modulus of elasticity (Pa)
A = 1  # Cross-sectional area (m^2)

# Assemble the global stiffness matrix
K_global = assemble_global_stiffness(nodes, E, A, elements)
print("Global Stiffness Matrix:\n", K_global)

# Define the load vector
f = create_load_vector(nodes_loads)

# Apply boundary conditions
K_global, f = apply_boundary_conditions(K_global, f, nodes_fixed)

# Solve for displacements
displacements = solve_displacements(K_global, f)

# Output the displacement vector
print("\nDisplacement Vector (u):\n", displacements)
