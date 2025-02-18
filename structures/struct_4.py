def generate_struct(rows=5, cols=6, spacing=1.0):
    nodes = {}
    elements = {}
    elements_preload = {}
    nodes_load = {}
    nodes_fixed = {}

    # Create nodes
    node_id = 1
    for i in range(rows):
        for j in range(cols):
            nodes[node_id] = (
                j * spacing,
                i * spacing,
                0.0,
            )  # 2D grid with zero z-coordinates
            nodes_load[node_id] = (0.0, 0.0, 0.0)  # No load initially
            nodes_fixed[node_id] = 1 if i == 0 else 0  # Fix bottom row
            node_id += 1

    nodes_load[17] = (1.0, 0.0, -1.0)
    nodes_fixed[25] = 1
    nodes_fixed[26] = 1
    nodes_fixed[27] = 1
    nodes_fixed[28] = 1
    nodes_fixed[29] = 1
    nodes_fixed[30] = 1
    nodes_fixed[7] = 1
    nodes_fixed[13] = 1
    nodes_fixed[19] = 1
    nodes_fixed[12] = 1
    nodes_fixed[18] = 1
    nodes_fixed[24] = 1

    # Create elements
    element_id = 1
    for i in range(rows):
        for j in range(cols):
            index = i * cols + j + 1
            # Connect to right neighbor
            if j < cols - 1:
                elements[element_id] = (index, index + 1)
                elements_preload[element_id] = 1
                element_id += 1
            # Connect to upper neighbor
            if i < rows - 1:
                elements[element_id] = (index, index + cols)
                elements_preload[element_id] = 1
                element_id += 1

    return nodes, elements, elements_preload, nodes_load, nodes_fixed
