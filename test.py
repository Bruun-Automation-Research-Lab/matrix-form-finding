import matplotlib.pyplot as plt
import numpy as np

from solver import quadratic_interp

from helper_plot import plot_quadratic_interp

x = [0, 0.5, 1.0]
# y = [-1.179e-17, 7.129e-20, -7.660e-17]
y = [-1.179e-17, 7.129e-20, -2.179e-17]

KE = y[2]
KE_prev = y[1]
KE_prev2 = y[0]

q1 = (KE_prev - KE) / ((KE_prev - KE) - (KE_prev2 - KE_prev))
q2, KE_q, x_interp, y_interp = quadratic_interp([KE_prev2, KE_prev, KE])

print(q1)
print(q2, KE_q)


plot_quadratic_interp(x, y, x_interp, y_interp, q1, q2, KE_q)
