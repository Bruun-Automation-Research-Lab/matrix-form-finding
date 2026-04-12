import logging as log
import numpy as np


def print_table_1col(
    c1,
    name1="Col 1",
    scale1=1,
    decimals=4,
    col_width=9,
):
    c1 = np.asarray(c1, dtype=float).reshape(-1)

    fmt = f"{{:>{col_width}.{decimals}f}}"

    lines = []
    lines.append(f"{'':>2} | {name1:>{col_width}}")
    lines.append(f"{'':>2} | {f'{scale1} ×':>{col_width}}")

    for i, v1 in enumerate(c1, start=1):
        lines.append(f"{i:02d} | {fmt.format(v1/scale1)}")

    return "\n".join(lines)


def print_table_2col(
    c1,
    c2,
    name1="Col 1",
    name2="Col 2",
    scale1=1,
    scale2=1,
    decimals=4,
    col_width=9,
):
    c1 = np.asarray(c1, dtype=float).reshape(-1)
    c2 = np.asarray(c2, dtype=float).reshape(-1)

    if len(c1) != len(c2):
        raise ValueError("Both columns must have the same length.")

    fmt = f"{{:>{col_width}.{decimals}f}}"

    lines = []
    lines.append(f"{'':>2} | {name1:>{col_width}} | {name2:>{col_width}}")
    lines.append(
        f"{'':>2} | {f'{scale1} ×':>{col_width}} | {f'{scale2} ×':>{col_width}}"
    )

    for i, (v1, v2) in enumerate(zip(c1, c2), start=1):
        lines.append(
            f"{i:02d} | {fmt.format(v1/scale1)} | {fmt.format(v2/scale2)}"
        )

    return "\n".join(lines)


def print_table_3col(
    c1,
    c2,
    c3,
    name1="Col 1",
    name2="Col 2",
    name3="Col 3",
    scale1=1,
    scale2=1,
    scale3=1,
    decimals=4,
    col_width=9,
):
    c1 = np.asarray(c1, dtype=float).reshape(-1)
    c2 = np.asarray(c2, dtype=float).reshape(-1)
    c3 = np.asarray(c3, dtype=float).reshape(-1)

    if not (len(c1) == len(c2) == len(c3)):
        raise ValueError("All three columns must have the same length.")

    fmt = f"{{:>{col_width}.{decimals}f}}"

    lines = []
    lines.append(
        f"{'':>3} | {name1:>{col_width}} | {name2:>{col_width}} | {name3:>{col_width}}"
    )
    lines.append(
        f"{'':>3} | {f'{scale1} ×':>{col_width}} | {f'{scale2} ×':>{col_width}} | {f'{scale3} ×':>{col_width}}"
    )

    for i, (v1, v2, v3) in enumerate(zip(c1, c2, c3), start=1):
        lines.append(
            f"{i:>3} | {fmt.format(v1/scale1)} | {fmt.format(v2/scale2)} | {fmt.format(v3/scale3)}"
        )

    return "\n".join(lines)


def print_table_4col_labeled(
    labels,
    c1,
    c2,
    c3,
    name0="",
    name1="Label",
    name2="Col 1",
    name3="Col 2",
    name4="Col 3",
    scale1=1,
    scale2=1,
    scale3=1,
    decimals=4,
    col_width=9,
    label_width=6,
    index_width=2,
):
    labels = np.asarray(labels).reshape(-1)
    c1 = np.asarray(c1, dtype=float).reshape(-1)
    c2 = np.asarray(c2, dtype=float).reshape(-1)
    c3 = np.asarray(c3, dtype=float).reshape(-1)

    if not (len(labels) == len(c1) == len(c2) == len(c3)):
        raise ValueError(
            "labels, c1, c2, and c3 must all have the same length."
        )

    fmt = f"{{:>{col_width}.{decimals}f}}"

    lines = []
    lines.append(
        f"{name0:>{index_width}}  | {name1:>{label_width}} | {name2:>{col_width}} | {name3:>{col_width}} | {name4:>{col_width}}"
    )
    lines.append(
        f"{'':>{index_width}}  | {'':>{label_width}} | {f'{scale1} ×':>{col_width}} | {f'{scale2} ×':>{col_width}} | {f'{scale3} ×':>{col_width}}"
    )

    for i, (label, v1, v2, v3) in enumerate(zip(labels, c1, c2, c3), start=1):
        lines.append(
            f"{i:>3} | {int(label):>{label_width}d} | {fmt.format(v1/scale1)} | {fmt.format(v2/scale2)} | {fmt.format(v3/scale3)}"
        )

    return "\n".join(lines)


def print_stiffness_mat(
    K,
    row_labels=None,
    header_labels=None,
    name="K",
    scale=1,
    decimals=2,
    col_width=7,
):
    K = np.asarray(K, dtype=float)
    n_rows, n_cols = K.shape
    label_width = 3

    if row_labels is None:
        row_labels = [f"{i:02d}" for i in range(1, n_rows + 1)]
    else:
        row_labels = np.asarray(row_labels).reshape(-1)
        if len(row_labels) != n_rows:
            raise ValueError(
                "row_labels must have the same length as the number of rows in K."
            )

    if header_labels is None:
        header_labels = [f"{j:02d}" for j in range(1, n_cols + 1)]
    else:
        header_labels = np.asarray(header_labels).reshape(-1)
        if len(header_labels) != n_cols:
            raise ValueError(
                "header_labels must have the same length as the number of columns in K."
            )

    fmt = f"{{:>{col_width}.{decimals}f}}"
    hdr = f"{{:>{col_width}}}"
    lab = f"{{:>{label_width}}}"

    lines = [f"{name} =", f"{scale} ×"]

    header_num = f"{'':>3} | {lab.format('')} | " + " ".join(
        hdr.format(f"{j:0d}") for j in range(1, n_cols + 1)
    )
    lines.append(header_num)

    header_lab = f"{'':>3} | {lab.format('')} | " + " ".join(
        hdr.format(str(label)) for label in header_labels
    )
    lines.append(header_lab)

    lines.append("----+-----+-" + "-" * (len(header_lab) - 11))

    for i, (label, row) in enumerate(zip(row_labels, K), start=1):
        row_scaled = row / scale
        row_str = " ".join(
            hdr.format(".") if np.isclose(val, 0.0) else fmt.format(val)
            for val in row_scaled
        )
        lines.append(f"{i:>3} | {lab.format(str(label))} | {row_str}")

    return "\n".join(lines)


def print_connectivity_mat(
    C,
    header_labels=None,
    name="C",
    col_width=3,
):
    C = np.asarray(C)

    if header_labels is None:
        header_labels = [f"{j:0d}" for j in range(1, C.shape[1] + 1)]
    else:
        header_labels = np.asarray(header_labels).reshape(-1)
        if len(header_labels) != C.shape[1]:
            raise ValueError(
                "header_labels must have the same length as the number of columns in C."
            )

    lines = [f"{name} ="]

    # First header row: regular numbering
    header_num = "    | " + " ".join(
        f"{j:>{col_width}}"
        for j in [f"{k:02d}" for k in range(1, C.shape[1] + 1)]
    )
    lines.append(header_num)

    # Border under numbering header
    lines.append("----+-" + "-" * (len(header_num) - 5))

    # Second header row: custom labels
    header_lab = "    | " + " ".join(
        f"{str(label):>{col_width}}" for label in header_labels
    )
    lines.append(header_lab)

    # Border under label header
    lines.append("----+-" + "-" * (len(header_lab) - 5))

    # Matrix rows
    for i, row in enumerate(C, start=1):
        row_str = " ".join(
            f"{('.' if val == 0 else int(val)):>{col_width}}" for val in row
        )
        lines.append(f"{i:>3} | {row_str}")

    return "\n".join(lines)


############################################


# Set up log
def setup_logging(debug=False, filename="debug_log.txt"):
    log.getLogger("matplotlib").setLevel(log.WARNING)

    if debug:
        log.basicConfig(
            filename=f"./debug/debug_{filename}.txt",
            level=log.DEBUG,
            format="%(message)s",
            filemode="w",
        )
    else:
        log.basicConfig(level=log.INFO)


def debug_struct_input(n, e, e_l, n_l, n_f):

    log.debug("############################################")
    log.debug("# INPUT STRUCTURE")
    log.debug("############################################")

    log.debug("")
    log.debug("#-------------------------------------------")
    log.debug("NODES")
    log.debug("[nx3]")
    log.debug(
        "%s",
        print_table_3col(
            n[:, 0],
            n[:, 1],
            n[:, 2],
            name1="x",
            name2="y",
            name3="z",
            decimals=2,
            col_width=6,
        ),
    )

    log.debug("")
    log.debug("#-------------------------------------------")
    log.debug("ELEMENTS")
    log.debug("[m]")
    log.debug(
        "%s",
        print_table_3col(
            e[:, 0],
            e[:, 1],
            e_l,
            name1="n_i",
            name2="n_j",
            name3="preload",
            decimals=0,
            col_width=4,
            scale3=1,
        ),
    )

    log.debug("")
    log.debug("#-------------------------------------------")
    log.debug("NODAL LOADS")
    log.debug("[nx3]")
    log.debug(
        "%s",
        print_table_3col(
            n_l[:, 0],
            n_l[:, 1],
            n_l[:, 2],
            name1="p_x",
            name2="p_y",
            name3="p_z",
            decimals=2,
            col_width=7,
        ),
    )

    log.debug("")
    log.debug("#-------------------------------------------")
    log.debug("NODAL FIXITY")
    log.debug("[nx1]")
    log.debug("1=fixed")
    log.debug(
        "%s",
        print_table_1col(
            n_f,
            name1="fix?",
            decimals=0,
            col_width=4,
        ),
    )


def debug_struct_matrices(C_total, C, C_f, p_x, p_y, p_z, n_f):
    log.debug("")
    log.debug("############################################")
    log.debug("# INPUT STRUCTURE MATRICES")
    log.debug("############################################")

    labels_free = np.where(np.asarray(n_f).reshape(-1) == 0)[0] + 1
    labels_fix = np.where(np.asarray(n_f).reshape(-1) == 1)[0] + 1

    log.debug("")
    log.debug("#-------------------------------------------")
    log.debug("NODAL LOADS (FREE)")
    log.debug("[n_free x 1]")
    log.debug(
        "%s",
        print_table_4col_labeled(
            labels_free,
            p_x,
            p_y,
            p_z,
            name1="Node",
            name2="p_x",
            name3="p_y",
            name4="p_z",
            decimals=2,
            col_width=6,
            label_width=4,
        ),
    )

    log.debug("")
    log.debug("#-------------------------------------------")
    log.debug("BRANCH-NODE MATRIX")
    log.debug("[m x n]")
    log.debug("%s", print_connectivity_mat(C_total, name="C_total"))

    log.debug("")
    log.debug("#-------------------------------------------")
    log.debug("BRANCH-NODE MATRIX (FREE)")
    log.debug("[m x n_free]")
    log.debug(
        "%s",
        print_connectivity_mat(C, name="C", header_labels=labels_free),
    )

    log.debug("")
    log.debug("#-------------------------------------------")
    log.debug("BRANCH-NODE MATRIX (FIXED)")
    log.debug("[m x n_fixed]")
    log.debug(
        "%s",
        print_connectivity_mat(C_f, name="C_f", header_labels=labels_fix),
    )


def debug_iteration(iteration, solver):
    log.debug("")
    log.debug("############################################")
    log.debug("# ITERATION: %s", iteration)
    log.debug("# SOLVER: %s", solver)
    log.debug("############################################")


def debug_F_L_Q(F, L, Q, scale=1):
    log.debug("")
    log.debug("#-------------------------------------------")
    log.debug("FORCE/LENGTH/FORCE DENSITY")
    log.debug("[m x 1]")

    log.debug(
        "%s",
        print_table_3col(
            np.diag(F),
            np.diag(L),
            np.diag(Q),
            name1="Force",
            name2="Length",
            name3="q",
            scale1=scale,
            scale2=scale,
            scale3=1,
        ),
    )


def debug_stiffness_FD(K, D, D_f, n_f, scale=1):
    log.debug("")
    log.debug("#-------------------------------------------")
    log.debug("STIFFNESS MATRICES")

    labels_free = np.where(np.asarray(n_f).reshape(-1) == 0)[0] + 1
    labels_fix = np.where(np.asarray(n_f).reshape(-1) == 1)[0] + 1

    log.debug("")
    log.debug("K (FREE)")
    log.debug("[n_free x n_free]")
    log.debug(
        "%s",
        print_stiffness_mat(
            K,
            row_labels=labels_free,
            header_labels=labels_free,
            name="K",
            scale=1,
            decimals=2,
            col_width=6,
        ),
    )

    log.debug("")
    log.debug("D (FREE)")
    log.debug("[n_free x n_free]")
    log.debug(
        "%s",
        print_stiffness_mat(
            D,
            row_labels=labels_free,
            header_labels=labels_free,
            name="D",
            scale=1,
            decimals=2,
            col_width=6,
        ),
    )

    log.debug("")
    log.debug("D (FIXED)")
    log.debug("[n_free x n_fixed]")
    log.debug(
        "%s",
        print_stiffness_mat(
            D_f,
            row_labels=labels_free,
            header_labels=labels_fix,
            name="D_f",
            scale=1,
            decimals=2,
            col_width=6,
        ),
    )


def debug_stiffness_SM(K_g, K_e, K, D, D_f, n_f, scale=1):
    log.debug("")
    log.debug("#-------------------------------------------")
    log.debug("STIFFNESS MATRICES")

    labels_free = np.where(np.asarray(n_f).reshape(-1) == 0)[0] + 1
    labels_fix = np.where(np.asarray(n_f).reshape(-1) == 1)[0] + 1

    n_ele = int(K_g.shape[0] / 3)
    labels_3 = np.array(
        [
            f"{label}{dof}"
            for label in range(1, n_ele + 1)
            for dof in ["x", "y", "z"]
        ]
    )
    labels_free_3 = np.array(
        [f"{label}{dof}" for label in labels_free for dof in ["x", "y", "z"]]
    )
    labels_fix_3 = np.array(
        [f"{label}{dof}" for label in labels_fix for dof in ["x", "y", "z"]]
    )

    log.debug("")
    log.debug("K GEOMETRIC")
    log.debug("[3m x 3m]")
    log.debug(
        "%s",
        print_stiffness_mat(
            K_g,
            row_labels=labels_3,
            header_labels=labels_3,
            name="K_g",
            scale=1,
            decimals=2,
            col_width=6,
        ),
    )

    log.debug("")
    log.debug("K ELASTIC")
    log.debug("[3m x 3m]")
    log.debug(
        "%s",
        print_stiffness_mat(
            K_e,
            row_labels=labels_3,
            header_labels=labels_3,
            name="K_e",
            scale=1,
            decimals=2,
            col_width=6,
        ),
    )

    log.debug("")
    log.debug("K (FREE)")
    log.debug("[3n_free x 3n_free]")
    log.debug(
        "%s",
        print_stiffness_mat(
            K,
            row_labels=labels_free_3,
            header_labels=labels_free_3,
            name="K",
            scale=1,
            decimals=2,
            col_width=6,
        ),
    )

    log.debug("")
    log.debug("D (FREE)")
    log.debug("[3n_free x 3n_free]")
    log.debug(
        "%s",
        print_stiffness_mat(
            D,
            row_labels=labels_free_3,
            header_labels=labels_free_3,
            name="D",
            scale=1,
            decimals=2,
            col_width=6,
        ),
    )

    log.debug("")
    log.debug("D (FIXED)")
    log.debug("[3n_free x 3n_fixed]")
    log.debug(
        "%s",
        print_stiffness_mat(
            D_f,
            row_labels=labels_free_3,
            header_labels=labels_fix_3,
            name="D_f",
            scale=1,
            decimals=2,
            col_width=6,
        ),
    )


def debug_stiffness_DR(K_g, K_e, K, D, D_f, n_f, scale=1):
    log.debug("")
    log.debug("#-------------------------------------------")
    log.debug("STIFFNESS MATRICES")

    labels_free = np.where(np.asarray(n_f).reshape(-1) == 0)[0] + 1
    labels_fix = np.where(np.asarray(n_f).reshape(-1) == 1)[0] + 1

    log.debug("")
    log.debug("K GEOMETRIC")
    log.debug("[m x m]")
    log.debug(
        "%s",
        print_stiffness_mat(K_g, name="K_g", scale=1, decimals=2, col_width=6),
    )

    log.debug("")
    log.debug("K ELASTIC")
    log.debug("[m x m]")
    log.debug(
        "%s",
        print_stiffness_mat(K_e, name="K_e", scale=1, decimals=2, col_width=6),
    )

    log.debug("")
    log.debug("K (FREE)")
    log.debug("[n_free x n_free]")
    log.debug(
        "%s",
        print_stiffness_mat(
            K,
            row_labels=labels_free,
            header_labels=labels_free,
            name="K",
            scale=1,
            decimals=2,
            col_width=6,
        ),
    )

    log.debug("")
    log.debug("D (FREE)")
    log.debug("[n_free x n_free]")
    log.debug(
        "%s",
        print_stiffness_mat(
            D,
            row_labels=labels_free,
            header_labels=labels_free,
            name="D",
            scale=1,
            decimals=2,
            col_width=6,
        ),
    )

    log.debug("")
    log.debug("D (FIXED)")
    log.debug("[n_free x n_fixed]")
    log.debug(
        "%s",
        print_stiffness_mat(
            D_f,
            row_labels=labels_free,
            header_labels=labels_fix,
            name="D_f",
            scale=1,
            decimals=2,
            col_width=6,
        ),
    )


def debug_deltas(d_x, d_y, d_z, n_f, scale=1):
    log.debug("")
    log.debug("#-------------------------------------------")
    log.debug("DELTA DISPLACEMENTS (FREE)")
    log.debug("[n_free x 1]")

    labels = np.where(np.asarray(n_f).reshape(-1) == 0)[0] + 1

    log.debug(
        "%s",
        print_table_4col_labeled(
            labels,
            d_x,
            d_y,
            d_z,
            name1="Node",
            name2="d_x",
            name3="d_y",
            name4="d_z",
            decimals=4,
            col_width=8,
            label_width=4,
        ),
    )


def debug_new_nodes(n):
    log.debug("")
    log.debug("#-------------------------------------------")
    log.debug("NODES (NEW)")
    log.debug("[n x 3]")

    log.debug(
        "%s",
        print_table_3col(
            n[:, 0],
            n[:, 1],
            n[:, 2],
            name1="x",
            name2="y",
            name3="z",
            decimals=2,
            col_width=6,
        ),
    )


def debug_velocity_kinetic_energy(v_x, v_y, v_z, n_f, KE):
    log.debug("")
    log.debug("#-------------------------------------------")
    log.debug("NODE VELOCITIES (FREE)")
    log.debug("[n_free x 3]")

    labels = np.where(np.asarray(n_f).reshape(-1) == 0)[0] + 1

    log.debug(
        "%s",
        print_table_4col_labeled(
            labels,
            v_x,
            v_y,
            v_z,
            name1="Node",
            name2="v_x",
            name3="v_y",
            name4="v_z",
            decimals=4,
            col_width=8,
            label_width=4,
        ),
    )

    log.debug("")
    log.debug("KINETIC ENERGY: %s", KE)


def debug_energy_peak(q):
    log.debug("")
    log.debug("\nKINETIC ENERGY PEAK REACHED. APPLYING DAMPING.")
    log.debug(f"Interpolated q = {q:.3f}")


def debug_error(L_total, error):
    log.debug("")
    log.debug("#-------------------------------------------")
    log.debug(f"Total Len = {L_total:.3f}")
    log.debug(f"Max error = {error:.3f}")


def debug_final(n, F, L, Q, scale=1):
    log.debug("")
    log.debug("############################################")
    log.debug("############################################")
    log.debug("# FINAL STRUCTURE")
    log.debug("############################################")
    log.debug("############################################")

    log.debug("")
    log.debug("#-------------------------------------------")
    log.debug("FINAL NODES")
    log.debug("[n x 3]")

    log.debug(
        "%s",
        print_table_3col(
            n[:, 0],
            n[:, 1],
            n[:, 2],
            name1="x",
            name2="y",
            name3="z",
            decimals=2,
            col_width=6,
        ),
    )

    log.debug("")
    log.debug("#-------------------------------------------")
    log.debug("FINAL FORCE/LENGTH/FORCE DENSITY")
    log.debug("[m x 1]")

    log.debug(
        "%s",
        print_table_3col(
            np.diag(F),
            np.diag(L),
            np.diag(Q),
            name1="Force",
            name2="Length",
            name3="q",
            scale1=scale,
            scale2=scale,
            scale3=1,
        ),
    )

    log.debug("")
    log.debug("#-------------------------------------------")
    log.debug("FINAL FORCE/FORCE_AVG RATIO")
    log.debug("[m x 1]")

    log.debug(
        "%s",
        print_table_1col(
            np.diag(F) / np.average(np.diag(F)),
            name1="F/F_avg",
            scale1=scale,
        ),
    )


def debug_table(r1):
    """
    This function generates a debug table

    Args:
    - y_star (float): The value of t* (time).
    - E_star (float): The energy at t* (E(t*)).
    - energy_values (list of floats): List containing the other energy values.
    """

    # Ensure energy_values has exactly 3 points
    if len(r1) != 3:
        raise ValueError("energy_values should contain exactly 3 points.")

    # Prepare the ASCII table with reordered columns
    table = f"""
    +----+------------+-----------+-----------+
    | t  | t - 3dt/2  | t - dt/2  | t + dt/2  |
    +----+------------+-----------+-----------+
    | E  | {r1[0]:.3e}  | {r1[1]:.3e} | {r1[2]:.3e} |
    +----+------------+-----------+-----------+
    """

    log.debug("\n%s", table)
