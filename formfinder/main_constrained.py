import numpy as np
import matplotlib.pyplot as plt

## THIS CODE IS NOT FIGURED OUT YET
## JUST A QUICK CHECK ON CONSTRAINED FORCE DENSITY


def build_full_connectivity(branches, n_nodes):
    """
    Build full branch-node connectivity matrix C_s.
    """
    m = len(branches)
    C_s = np.zeros((m, n_nodes), dtype=float)

    for i, (j, k) in enumerate(branches):
        C_s[i, j - 1] = 1.0
        C_s[i, k - 1] = -1.0

    return C_s


def partition_connectivity(C_s, free_nodes, fixed_nodes):
    """
    Reorder connectivity into [free|fixed] node order and partition into C|C_f.
    """
    node_order = list(free_nodes) + list(fixed_nodes)
    node_index = {node: i for i, node in enumerate(node_order)}

    m = C_s.shape[0]
    C_reordered = np.zeros((m, len(node_order)), dtype=float)

    for row in range(m):
        for node in range(C_s.shape[1]):
            val = C_s[row, node]
            if val != 0:
                old_node = node + 1
                new_col = node_index[old_node]
                C_reordered[row, new_col] = val

    n_free = len(free_nodes)
    C = C_reordered[:, :n_free]
    C_f = C_reordered[:, n_free:]

    return C, C_f, node_order


def reorder_coordinates(xyz_free, xyz_fixed):
    """
    Stack coordinates in reordered form: [free | fixed]
    """
    return np.vstack([xyz_free, xyz_fixed])


def restore_original_node_order(xyz_reordered, node_order):
    """
    Convert reordered coordinates [free|fixed] back to original node numbering.
    """
    n_nodes = len(node_order)
    ndim = xyz_reordered.shape[1]
    xyz_original = np.zeros((n_nodes, ndim), dtype=float)

    for i, node in enumerate(node_order):
        xyz_original[node - 1] = xyz_reordered[i]

    return xyz_original


def solve_fdm_coordinates(C, C_f, q, xyz_fixed, loads_free):
    """
    Solve standard FDM equilibrium for all coordinate directions.
    """
    Q = np.diag(q)

    D = C.T @ Q @ C
    D_f = C.T @ Q @ C_f

    rhs = loads_free - D_f @ xyz_fixed
    xyz_free = np.linalg.solve(D, rhs)

    return xyz_free


def member_vectors(C_s, xyz):
    """
    Compute member vectors from nodal coordinates in original node order.
    """
    return C_s @ xyz


def member_lengths(C_s, xyz):
    """
    Compute member lengths from nodal coordinates in original node order.
    """
    vec = member_vectors(C_s, xyz)
    return np.linalg.norm(vec, axis=1)


def plot_structure(
    xyz,
    branches,
    fixed_nodes=None,
    free_nodes=None,
    ax=None,
    title="Structure",
    show_node_labels=True,
):
    """
    Plot 2D structure geometry.
    """
    if xyz.shape[1] != 2:
        raise ValueError("plot_structure currently expects 2D coordinates.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    for i, j in branches:
        xi, yi = xyz[i - 1]
        xj, yj = xyz[j - 1]
        ax.plot([xi, xj], [yi, yj], "k-", lw=1.5)

    if free_nodes is not None and len(free_nodes) > 0:
        free_idx = np.array(free_nodes) - 1
        ax.plot(
            xyz[free_idx, 0],
            xyz[free_idx, 1],
            "o",
            ms=8,
            label="free nodes",
        )

    if fixed_nodes is not None and len(fixed_nodes) > 0:
        fixed_idx = np.array(fixed_nodes) - 1
        ax.plot(
            xyz[fixed_idx, 0],
            xyz[fixed_idx, 1],
            "s",
            ms=8,
            label="fixed nodes",
        )

    if show_node_labels:
        for n in range(xyz.shape[0]):
            ax.text(xyz[n, 0], xyz[n, 1], f"  {n + 1}", fontsize=10)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()

    return ax


def plot_before_after(
    xyz_before,
    xyz_after,
    branches,
    fixed_nodes=None,
    free_nodes=None,
):
    """
    Plot initial and final geometry side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    plot_structure(
        xyz_before,
        branches,
        fixed_nodes=fixed_nodes,
        free_nodes=free_nodes,
        ax=axes[0],
        title="Before",
    )

    plot_structure(
        xyz_after,
        branches,
        fixed_nodes=fixed_nodes,
        free_nodes=free_nodes,
        ax=axes[1],
        title="After",
    )

    plt.tight_layout()
    plt.show()


def constrained_force_density_method(
    branches,
    xyz_fixed,
    xyz_free_init,
    free_nodes,
    fixed_nodes,
    q_init,
    loads_free=None,
    force_constraints=None,
    max_iter=200,
    tol=1e-6,
    min_length=1e-10,
    verbose=True,
):
    """
    Iterative constrained force density method.

    Members listed in force_constraints have prescribed force F_i.
    Their force density is updated iteratively using:

        q_i = F_i / L_i
    """
    n_nodes = len(free_nodes) + len(fixed_nodes)
    ndim = xyz_fixed.shape[1]

    if loads_free is None:
        loads_free = np.zeros((len(free_nodes), ndim), dtype=float)

    if force_constraints is None:
        force_constraints = {}

    q = np.array(q_init, dtype=float).copy()

    C_s = build_full_connectivity(branches, n_nodes)
    C, C_f, node_order = partition_connectivity(C_s, free_nodes, fixed_nodes)

    constrained_ids = np.array(list(force_constraints.keys()), dtype=int)

    xyz_initial_reordered = reorder_coordinates(xyz_free_init, xyz_fixed)
    xyz_initial = restore_original_node_order(
        xyz_initial_reordered, node_order
    )

    xyz_free = solve_fdm_coordinates(C, C_f, q, xyz_fixed, loads_free)

    for iteration in range(1, max_iter + 1):
        xyz_reordered = reorder_coordinates(xyz_free, xyz_fixed)
        xyz_current = restore_original_node_order(xyz_reordered, node_order)

        L = member_lengths(C_s, xyz_current)
        F = q * L

        if len(constrained_ids) > 0:
            target_forces = np.array(
                [force_constraints[i] for i in constrained_ids],
                dtype=float,
            )
            residual = F[constrained_ids] - target_forces
            error = np.max(np.abs(residual))
        else:
            error = 0.0

        if verbose:
            print(f"iter {iteration:03d} | max force residual = {error:.6e}")

        if error < tol:
            break

        for member_id, target_force in force_constraints.items():
            if L[member_id] <= min_length:
                raise ValueError(
                    f"Member {member_id} has near-zero length. "
                    "This usually means the current geometry is degenerate "
                    "for the chosen force constraint."
                )
            q[member_id] = target_force / L[member_id]

        xyz_free = solve_fdm_coordinates(C, C_f, q, xyz_fixed, loads_free)

    else:
        print("Warning: maximum iterations reached without convergence.")

    xyz_final_reordered = reorder_coordinates(xyz_free, xyz_fixed)
    xyz_final = restore_original_node_order(xyz_final_reordered, node_order)

    L = member_lengths(C_s, xyz_final)
    F = q * L

    return {
        "xyz_initial": xyz_initial,
        "xyz_final": xyz_final,
        "xyz_free": xyz_free,
        "xyz_fixed": xyz_fixed,
        "q": q,
        "L": L,
        "F": F,
        "C": C,
        "C_f": C_f,
        "C_s": C_s,
        "node_order": node_order,
    }


# -----------------------------------
# Example: small 2D cable net
# -----------------------------------

branches = [
    (1, 3),
    (2, 3),
    (1, 4),
    (2, 4),
    (3, 4),
]

free_nodes = [3, 4]
fixed_nodes = [1, 2]

xyz_fixed = np.array(
    [
        [0.0, 0.0],  # node 1
        [4.0, 0.0],  # node 2
    ]
)

xyz_free_init = np.array(
    [
        [1.5, -0.5],  # node 3
        [2.5, -0.5],  # node 4
    ]
)

loads_free = np.array(
    [
        [0.0, -1.0],
        [0.0, -1.0],
    ]
)

q_init = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

# Constrain member 0 = branch (1, 3)
# Do NOT constrain member 4 = branch (3, 4) in this symmetric example,
# because nodes 3 and 4 collapse to the same point in the first solve.
force_constraints = {
    0: 2.5,
    1: 2.5,
}

result = constrained_force_density_method(
    branches=branches,
    xyz_fixed=xyz_fixed,
    xyz_free_init=xyz_free_init,
    free_nodes=free_nodes,
    fixed_nodes=fixed_nodes,
    q_init=q_init,
    loads_free=loads_free,
    force_constraints=force_constraints,
    max_iter=100,
    tol=1e-8,
    verbose=True,
)

print("\nFinal free node coordinates:")
print(result["xyz_free"])

print("\nFinal member lengths:")
print(result["L"])

print("\nFinal member forces:")
print(result["F"])

print("\nFinal member force-density:")
print(result["F"] / result["L"])

plot_before_after(
    result["xyz_initial"],
    result["xyz_final"],
    branches,
    fixed_nodes=fixed_nodes,
    free_nodes=free_nodes,
)
