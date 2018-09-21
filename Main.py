from DQ_RBF_Solver import DQ_RBF
from Knot_Generation import generate_point
import numpy as np
import math

def main():
    n_edge_knots = 50
    
    n_sup_knots = 15
    n_in_knots = 1000
    n_total_knots = n_in_knots + (n_edge_knots - 2) * 3 + (2 * n_edge_knots - 2) + 4
    n_deriatives = 4
    shape_para = 1.0
    iteration = 1000
    tolerance = 1E-8

    #generate_point(n_in_knots, n_edge_knots)

    adv_dif_equ = DQ_RBF(n_sup_knots, n_in_knots, n_total_knots, n_deriatives, shape_para, iteration, tolerance)
    adv_dif_equ.read_knots()
    adv_dif_equ.get_w_coe()
    adv_dif_equ.get_jacobi_matrix()
    adv_dif_equ.nonlinear_equations()
    adv_dif_equ.broyden_solver()
    adv_dif_equ.save_results()

    out_05x_05y = open("05x_05y.txt", "a")
    out_05x = open("05x.txt", "a")
    out_05y = open("05y.txt", "a")
    
    x0 = 0.5
    y0 = 0.5
    delta_x = 1.0 / (n_edge_knots - 1)
    delta_y = (1.0 - 0.5 * (math.tanh(-3) - math.tanh(2))) / (2 * n_edge_knots - 1)

    u = adv_dif_equ.calculate(x0, y0)
    out_05x_05y.write('%5i  %5i   %10.6f\n' % (x0, y0, u))
    out_05x_05y.write("\n")

    for i in range(n_edge_knots):
        x = i*delta_x
        u = adv_dif_equ.calculate(x, y0)
        out_05y.write('%5i  %5i   %10.6f\n' % (x, y0, u))
    out_05y.write("\n")

    for i in range(2*n_edge_knots):
        y = 0.5*(math.tanh(-3)-math.tanh(2))+i*delta_y
        u = adv_dif_equ.calculate(x0, y)
        out_05x.write('%5i  %5i   %10.6f\n' % (x0, y, u))
    out_05x.write("\n")

    out_05x_05y.close()
    out_05x.close()
    out_05y.close()

    return

if __name__ == "__main__":
    main()
    

