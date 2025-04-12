import numpy as np
import matplotlib.pyplot as plt


def create_structure(nelx, nely, structure_type="cantilever"):
    """
    Creates a structural grid
    returns fixed nodes, load nodes, and node coordinates.
    """
    num_nodes_x = nelx + 1
    num_nodes_y = nely + 1

    # Create Node Coordinates (row-major order)
    node_coords = [
        (i, j) for j in range(num_nodes_y) for i in range(num_nodes_x)
    ]

    # Define Fixed Nodes
    if structure_type == "cantilever":
        nodes_fixed_x = [r * num_nodes_x for r in range(num_nodes_y)]
        nodes_fixed_y = [r * num_nodes_x for r in range(num_nodes_y)]

        fixed_dofs = np.sort(
            np.array(
                [2 * n for n in nodes_fixed_x]  # X DOFs
                + [2 * n + 1 for n in nodes_fixed_y]  # Y DOFs
            )
        )

    else:
        raise ValueError(f"Unknown structure_type: {structure_type}")

    # Define Loaded DOFs
    if structure_type == "cantilever":
        nodes_loaded_x = []  # e.g., load in X direction if needed
        nodes_loaded_y = [num_nodes_x * num_nodes_y - 1]  # Y DOFs

        loaded_dofs = np.sort(
            np.array(
                [2 * n for n in nodes_loaded_x]  # X DOFs
                + [2 * n + 1 for n in nodes_loaded_y]  # Y DOFs
            )
        )

    return fixed_dofs, loaded_dofs, node_coords


def plot_initial_structure(
    nelx,
    nely,
    fixed_dofs,
    loaded_dofs,
    show_node_numbers=False,
    show_element_numbers=True,
):
    """
    Plots the 2D grid structure
    showing fixed DOFs, loaded DOFs, and optionally node/element numbers.
    """
    num_nodes_x = nelx + 1
    num_nodes_y = nely + 1
    node_coords = np.array(
        [(i, j) for j in range(num_nodes_y) for i in range(num_nodes_x)]
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot Element Grid
    for j in range(nely + 1):
        for i in range(nelx + 1):
            node_id = j * num_nodes_x + i
            x, y = node_coords[node_id]
            if i < nelx:
                right_node = j * num_nodes_x + (i + 1)
                ax.plot(
                    [x, node_coords[right_node][0]],
                    [y, node_coords[right_node][1]],
                    "k-",
                    lw=0.5,
                )
            if j < nely:
                top_node = (j + 1) * num_nodes_x + i
                ax.plot(
                    [x, node_coords[top_node][0]],
                    [y, node_coords[top_node][1]],
                    "k-",
                    lw=0.5,
                )

            if show_node_numbers:
                ax.text(
                    x,
                    y + 0.05,
                    str(node_id),
                    fontsize=8,
                    ha="center",
                    color="gray",
                )

    # Draw element numbers in the center of each element
    if show_element_numbers:
        for ely in range(nely):
            for elx in range(nelx):
                n1 = ely * (nelx + 1) + elx
                n2 = ely * (nelx + 1) + elx + 1
                n3 = (ely + 1) * (nelx + 1) + elx + 1
                n4 = (ely + 1) * (nelx + 1) + elx
                x_coords = [node_coords[n][0] for n in [n1, n2, n3, n4]]
                y_coords = [node_coords[n][1] for n in [n1, n2, n3, n4]]
                center_x = np.mean(x_coords)
                center_y = np.mean(y_coords)
                el_id = ely * nelx + elx
                ax.text(
                    center_x,
                    center_y,
                    str(el_id),
                    fontsize=8,
                    ha="center",
                    color="blue",
                )

    # Determine Fixed DOFs per node
    fixed_x = {fd // 2 for fd in fixed_dofs if fd % 2 == 0}
    fixed_y = {fd // 2 for fd in fixed_dofs if fd % 2 == 1}

    for n in range(len(node_coords)):
        x, y = node_coords[n]
        in_x = n in fixed_x
        in_y = n in fixed_y
        if in_x and in_y:
            ax.plot(
                x, y, "k^", markersize=8, label="Fixed XY" if n == 0 else ""
            )
        elif in_x:
            ax.plot(
                x, y, "bo", markersize=6, label="Fixed X" if n == 0 else ""
            )
        elif in_y:
            ax.plot(
                x, y, "go", markersize=6, label="Fixed Y" if n == 0 else ""
            )

    # Plot Loaded DOFs as Arrows
    for dof in loaded_dofs:
        node = dof // 2
        direction = dof % 2  # 0: x, 1: y
        x, y = node_coords[node]

        if direction == 0:  # X-direction
            dx, dy = 0.3, 0
        else:  # Y-direction
            dx, dy = 0, 0.3

        ax.arrow(
            x,
            y,
            dx,
            dy,
            head_width=0.1,
            head_length=0.1,
            fc="red",
            ec="red",
            label="Load" if dof == loaded_dofs[0] else "",
        )

    # Final Plot Settings
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Initial Structure")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()


def compute_element_stiffness_closed_form(E, nu, t=1.0, a=1.0, b=1.0):
    """
    Closed-form stiffness matrix rectangular Q4 plane stress element.

    Assumes uniform, axis-aligned rectangular element
    with width a and height b.
    """
    D = (E / (1 - nu**2)) * np.array(
        [[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]]
    )

    # Constants for integration over unit square using Gauss integration
    # Compute at the 4 Gauss points (±1/√3, ±1/√3)
    gauss_pts = np.array(
        [
            [-1 / np.sqrt(3), -1 / np.sqrt(3)],
            [1 / np.sqrt(3), -1 / np.sqrt(3)],
            [1 / np.sqrt(3), 1 / np.sqrt(3)],
            [-1 / np.sqrt(3), 1 / np.sqrt(3)],
        ]
    )

    # Coordinates of the element in the reference square
    coords = np.array(
        [[-a / 2, -b / 2], [a / 2, -b / 2], [a / 2, b / 2], [-a / 2, b / 2]]
    )

    Ke = np.zeros((8, 8))

    for xi, eta in gauss_pts:
        # Shape function derivatives with respect to xi, eta
        dN_dxi = 0.25 * np.array(
            [
                [-(1 - eta), -(1 - xi)],
                [(1 - eta), -(1 + xi)],
                [(1 + eta), (1 + xi)],
                [-(1 + eta), (1 - xi)],
            ]
        )

        # Jacobian
        J = dN_dxi.T @ coords
        detJ = np.linalg.det(J)
        J_inv = np.linalg.inv(J)

        # Derivatives wrt x, y
        dN_dx_dy = J_inv @ dN_dxi.T

        B = np.zeros((3, 8))
        for i in range(4):
            B[0, 2 * i] = dN_dx_dy[0, i]
            B[1, 2 * i + 1] = dN_dx_dy[1, i]
            B[2, 2 * i] = dN_dx_dy[1, i]
            B[2, 2 * i + 1] = dN_dx_dy[0, i]

        Ke += B.T @ D @ B * detJ * t

    return Ke


# # ========== ELEMENT STIFFNESS MATRIX ==========
# def element_stiffness_matrix(x1, y1, x2, y2, x3, y3, x4, y4):
#     """
#     Compute the element stiffness matrix for a
#     quadrilateral element using the 4-node quadrilateral element in FEM.
#     """
#     # Shape functions (bilinear for quadrilateral element)
#     B = np.zeros(
#         (3, 8)
#     )  # Strain-displacement matrix (size 3x8 for 4-node quadrilateral)

#     # Define the element stiffness matrix for a
#     # simple linear-elastic plane stress model
#     # This would be done using numerical integration (Gauss Quadrature)
#     # Simplified stiffness matrix for demonstration
#     A = np.array([[x1, y1, 1], [x2, y2, 1], [x3, y3, 1], [x4, y4, 1]])

#     # We assume the stiffness matrix is dummy for now
#     # in real applications, this is calculated using integration.
#     ke = np.zeros((8, 8))  # Placeholder for element stiffness matrix
#     return ke


# ========== ASSEMBLING THE GLOBAL STIFFNESS MATRIX ==========
def assemble_stiffness_matrix(nelx, nely, node_coords, density):
    """
    Assembles the global stiffness matrix
    for the entire structure by adding element stiffness matrices.
    Density field affects the element stiffness matrix.
    """
    num_nodes = (nelx + 1) * (nely + 1)
    K_global = np.zeros(
        (2 * num_nodes, 2 * num_nodes)
    )  # Global stiffness matrix (2 DOFs per node)

    # Loop through each element and assemble the global stiffness matrix
    for elx in range(nelx):
        for ely in range(nely):
            # Element node indices
            [n1, n2, n3, n4] = get_element_node_indices(elx, ely, nelx, nely)

            # Element node coordinates
            x1, y1 = node_coords[n1]
            x2, y2 = node_coords[n2]
            x3, y3 = node_coords[n3]
            x4, y4 = node_coords[n4]

            # Modify stiffness matrix based on material density (ESO)
            density_factor = density[elx, ely]
            # ke = (
            #     element_stiffness_matrix(x1, y1, x2, y2, x3, y3, x4, y4)
            #     * density_factor
            # )
            ke = compute_element_stiffness_closed_form(E, nu) * density_factor

            # Map the local stiffness matrix to the global stiffness matrix
            dof_map = [
                2 * n1,
                2 * n1 + 1,
                2 * n2,
                2 * n2 + 1,
                2 * n3,
                2 * n3 + 1,
                2 * n4,
                2 * n4 + 1,
            ]
            for i in range(8):
                for j in range(8):
                    K_global[dof_map[i], dof_map[j]] += ke[i, j]

    return K_global


# ========== BOUNDARY CONDITIONS ==========
def apply_boundary_conditions(K_global, force_vector, fixed_dofs):
    """
    Apply boundary conditions (fixed nodes and loads).
    Fixed nodes are set to zero, global stiffness matrix and force vector.
    """
    for dof in fixed_dofs:
        K_global[dof, :] = 0
        K_global[:, dof] = 0
        K_global[dof, dof] = 1  # Keep diagonal entry non-zero (for stability)
        force_vector[dof] = 0  # No force at fixed DOFs
    return K_global, force_vector


# ========== SOLVE THE SYSTEM OF EQUATIONS ==========
def solve_fea(K_global, force_vector):
    """
    Solve the linear system K*u = F for the displacement vector u.
    """
    u = np.linalg.solve(K_global, force_vector)
    return u


# ========== COMPLIANCE FUNCTION ==========
def compute_compliance(K_global, displacement_vector):
    """
    Compute the structural compliance, which is the sum of the strain energy.
    """
    return np.dot(displacement_vector.T, np.dot(K_global, displacement_vector))


def get_element_dof_indices(nodes):
    """
    Given 4 global node indices, return the corresponding 8 DOF indices.
    Each node has 2 DOFs: [u, v]
    """
    dofs = []
    for n in nodes:
        dofs.extend([2 * n, 2 * n + 1])
    return dofs


def get_element_node_indices(elx, ely, nelx, nely):
    n1 = ely * (nelx + 1) + elx
    n2 = ely * (nelx + 1) + elx + 1
    n3 = (ely + 1) * (nelx + 1) + elx + 1
    n4 = (ely + 1) * (nelx + 1) + elx

    return [n1, n2, n3, n4]


# ========== ESO FUNCTION ==========
def run_eso(nelx, nely, load_dofs, fixed_dofs, volume_fraction, max_iter=20):
    """
    Perform Evolutionary Structural Optimization (ESO).
    """
    # Initial density field (start with full material everywhere)
    density = np.ones((nelx, nely))

    # Apply volume fraction constraint
    target_volume = volume_fraction * nelx * nely

    # Initialize force vector
    num_nodes = (nelx + 1) * (nely + 1)
    force_vector = np.zeros(2 * num_nodes)
    force_vector[load_dofs] = -1

    # Main ESO loop
    for iteration in range(max_iter):
        # Assemble the global stiffness matrix
        K_global = assemble_stiffness_matrix(nelx, nely, node_coords, density)

        # Apply boundary conditions
        K_global, force_vector = apply_boundary_conditions(
            K_global, force_vector, fixed_dofs
        )

        # Solve for displacements
        displacement_vector = solve_fea(K_global, force_vector)

        # Compute element-wise strain energy = uᵀ * K_e * u
        strain_energy = np.zeros((nelx, nely))
        for elx in range(nelx):
            for ely in range(nely):
                nodes = get_element_node_indices(
                    elx, ely, nelx, nely
                )  # User-defined
                dofs = get_element_dof_indices(nodes)  # User-defined
                Ke = (
                    compute_element_stiffness_closed_form(E, nu)
                    * density[elx, ely]
                )  # Assuming linear scaling
                u_el = displacement_vector[dofs]
                strain_energy[elx, ely] = u_el @ Ke @ u_el

        # Rank elements by efficiency (strain energy) and keep top ones
        flattened_energy = strain_energy.flatten()
        flattened_density = density.flatten()

        # Sort elements by strain energy
        sort_idx = np.argsort(flattened_energy)  # Least efficient first
        num_elements_to_keep = int(target_volume)

        new_density = np.zeros_like(flattened_density)
        new_density[sort_idx[-num_elements_to_keep:]] = (
            1.0  # Keep top elements
        )

        density = new_density.reshape((nelx, nely))

        # Print and plot
        compliance = np.sum(flattened_energy)

        # # Ensure the volume fraction constraint is respected
        # # Reduce material in ele where density greater than target
        # current_volume = np.sum(density)
        # if current_volume > target_volume:
        #     density[
        #         density > 0
        #     ] -= 0.05

        density = np.maximum(density, 1e-3)
        print(
            f"""
        Iteration {iteration + 1}:
        Compliance = {compliance:.4f}
        Volume     = {np.sum(density):.2f}
        """
        )

        if iteration % 10 == 0:
            plt.imshow(density, cmap="gray", vmin=0, vmax=1)
            plt.title(f"Iteration {iteration} - Topology")
            plt.colorbar()
            plt.show()

    return density


# ========== PARAMETERS ==========
nelx, nely = 100, 40  # Number of elements in X and Y
volume_fraction = 0.7  # Target volume fraction
Emin, E = 1e-9, 1.0  # Minimum and maximum Young's modulus
nu = 0.3  # Poisson's ratio
t = 1.0  # Thickness of the structure (unit thickness for simplicity)

# Generate structure
fixed_dofs, load_dofs, node_coords = create_structure(nelx, nely)

# Plot the initial structure with node numbers
plot_initial_structure(
    nelx, nely, fixed_dofs, load_dofs, show_node_numbers=True
)


# ========== MATERIAL PROPERTIES ==========
# Plane stress constitutive matrix for linear elasticity
c = E / (1 - nu**2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])


# Run the ESO optimization
optimized_density = run_eso(nelx, nely, load_dofs, fixed_dofs, volume_fraction)

# ========== POSTPROCESSING: Plot Optimized Density ==========
plt.imshow(optimized_density.T, cmap="gray", interpolation="nearest")
plt.title("Optimized Material Distribution")
plt.colorbar(label="Material Density")
plt.show()
