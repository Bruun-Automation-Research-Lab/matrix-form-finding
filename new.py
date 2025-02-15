import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# from structures.struct_3 import generate_struct
from structures.struct_2 import generate_struct

from helper_matrix import (
    create_connectivity_matrix,
    create_length_matrix,
    partition_connectivity_matrix,
)
from helper_plot import plot_network3D, plot_network_animated
from helper_log import setup_logging


setup_logging(debug=True)

nodes, elements, external_loads, fixed_nodes = generate_struct(5)

# Plot initial network
plot_network3D(nodes, elements, fixed_nodes, external_loads)


# Generate connectivity matrix
connectivity_matrix = create_connectivity_matrix(nodes, elements)

C, C_f = partition_connectivity_matrix(connectivity_matrix, nodes, fixed_nodes)

# Length
l_vec, L = create_length_matrix(nodes, elements)

# Force


logging.debug("Nodes:\n %s", nodes)
# logging.debug("\nConnectivity Matrix:\n %s", connectivity_matrix)
logging.debug("\nC (free nodes):\n %s", C)
logging.debug("\nCf (fixed nodes):\n %s", C_f)
logging.debug("\nL (element lengths):\n %s", l_vec)
logging.debug("\np_x:\n %s", p_x)
logging.debug("\np_y:\n %s", p_y)
logging.debug("\np_z:\n %s", p_z)
