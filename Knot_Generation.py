import numpy as np 
import math

def generate_point(n_in_knots, n_edge_knots):
    out_knots = open("knots.txt", "w")
    j = 0
    while j < n_in_knots:
        x = np.random.uniform(0, 1)
        if (abs(x) < 1E-20) | (abs(x - 1.0) < 1E-20):
            continue
        y = np.random.uniform(-1, 1)
        if (abs(y - 1.0) < 1E-20) | (y < (0.5 * (math.tanh(2 - 10 * x) - math.tanh(2)))):
            continue

        out_knots.write('%10.9f  %10.9f\n' % (x, y))
        j = j + 1

    out_knots.write('%10.9f  %10.9f\n' % (0.0, 0.0))
    out_knots.write('%10.9f  %10.9f\n' % (0.0, 1.0))
    out_knots.write('%10.9f  %10.9f\n' % (1.0, 1.0))
    out_knots.write('%10.9f  %10.9f\n' % (1.0, 0.5 * (math.tanh(-8) - math.tanh(2))))

    delta = 1 / (n_edge_knots - 1)

    for i in range(1, n_edge_knots-1):
        y_t = 1.0
        x =  delta * i
        y_b = 0.5 * (math.tanh(2 - 10 * x) - math.tanh(2))
        out_knots.write('%10.9f  %10.9f\n' % (x, y_t))
        out_knots.write('%10.9f  %10.9f\n' % (x, y_b))

        x_l = 0.0
        y_l = delta * i
        out_knots.write('%10.9f  %10.9f\n' % (x_l, y_l))

    delta = (1 - 0.5 * (math.tanh(-8) - math.tanh(2))) / (2 * n_edge_knots - 1)
    for i in range(1, 2*n_edge_knots-1):
        x_r = 1.0
        y_r = 1 - delta * i
        out_knots.write('%10.9f  %10.9f\n' % (x_r, y_r))

    out_knots.close()

    return