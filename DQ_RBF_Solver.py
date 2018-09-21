from numba import jit
import sys
import numpy as np
import math
from pandas import Series
from scipy.interpolate import Rbf

class DQ_RBF():
    def __init__(self, n1, n2, n3, n4, n5, n6, n7):
        self.n_sup_knots = int(n1)       #the number of knots in the supporting region
        self.n_in_knots = int(n2)        #the number of knots in the inner of the computational domain
        self.n_total_knots = int(n3)     #the number of knots in the whole computational domain
        self.n_derivatives = int(n4)     #the number of derivatives. 
                                    #for assignment A, it's 4, i.e., the first and second order derivative with respect to x and y, respectively
        self.shape_para = float(n5)        #shape parameter for the MQ radial basis function
        self.iteration = int(n6)
        self.TOL = float(n7)

        self.u = np.full(shape=(self.n_total_knots), fill_value=0.4, dtype=float, order='F')
        self.w_coe = np.ndarray(shape=(self.n_in_knots, self.n_sup_knots+1, self.n_derivatives), dtype=float, order="F")

        return

    def read_knots(self):
        self.pos_total_knots = np.ndarray(shape=(self.n_total_knots, 2), dtype=float, order='F')        #store the x and y positions of all knots
        
        knots_file_name = "knots.txt"
        try:
            knots_file = open(knots_file_name, "r")
        except IOError:
            print("Unable to open file \"%s\"" % knots_file_name)
            sys.exit()
        with knots_file:
            konts_pos = knots_file.readlines()
            for i in range(self.n_total_knots):
                temp = konts_pos[i].split()
                self.pos_total_knots[i][0] = float(temp[0])
                self.pos_total_knots[i][1] = float(temp[1])
        
        for i in range(self.n_in_knots, self.n_total_knots):
            if abs(self.pos_total_knots[i][0]) < 1E-20:
                self.u[i] = self.pos_total_knots[i][1] ** 2
            elif abs(self.pos_total_knots[i][0] - 1.0) < 1E-20:
                self.u[i] = 1.0 + self.pos_total_knots[i][1] ** 2
            elif abs(self.pos_total_knots[i][1] - 1.0) < 1E-20:
                self.u[i] = 1.0 + self.pos_total_knots[i][0] ** 2
            else:
                self.u[i] = self.pos_total_knots[i][0] ** 2 + \
                 (0.5 * (math.tanh(2 - 10 * self.pos_total_knots[i][0]) - math.tanh(2))) ** 2
            
        return
    
    @jit
    def get_w_coe(self):
        dis = np.ndarray(shape=(self.n_total_knots), dtype=float, order="F")
        refer_knot = np.ndarray(shape=(2), dtype=float, order="F")
        self.local_sn = np.ndarray(shape=(self.n_in_knots, self.n_sup_knots+1), dtype=int, order="F")
        local_knots = np.ndarray(shape=(self.n_sup_knots+1, 2), dtype=float, order="F")     
        coe_matrix = np.ndarray(shape=(self.n_sup_knots+1, self.n_sup_knots+1), dtype=float, order="F")
        der_vectors = np.ndarray(shape=(self.n_sup_knots+1, self.n_derivatives), dtype=float, order="F")        #derivative vectors of the basis functions

        for k in range(self.n_in_knots):
            refer_knot[0] = self.pos_total_knots[k][0]
            refer_knot[1] = self.pos_total_knots[k][1]

            for i in range(self.n_total_knots):
                dx = self.pos_total_knots[i][0] - refer_knot[0]
                dy = self.pos_total_knots[i][1] - refer_knot[1]
                dis[i] = math.sqrt(dx * dx + dy * dy)
    
            refer_dis = Series(dis)
            refer_dis = refer_dis.sort_values()         #ascending order by distance between the reference knot and other knots
            scaling = refer_dis.values[self.n_sup_knots] * 2.0       #the diameter of the minimal circle enclosing all knots in the supporting region

            self.local_sn[k][:self.n_sup_knots] = refer_dis.index[1:self.n_sup_knots+1]     #serial number of local knots: 
            self.local_sn[k][self.n_sup_knots] = refer_dis.index[0]     #0-n_sup_knots-1 are support knots, n_sup_knots is reference knot
            
            for i in range(self.n_sup_knots+1):
                local_knots[i][0] = self.pos_total_knots[self.local_sn[k][i]][0]       #store the positions of the supporting knots
                local_knots[i][1] = self.pos_total_knots[self.local_sn[k][i]][1]
    
            for i in range(self.n_sup_knots+1):
                coe_matrix[self.n_sup_knots][i] = 1.0
            
            for i in range(self.n_sup_knots):
                for j in range(self.n_sup_knots+1):
                    dx = (local_knots[j][0] - local_knots[i][0]) / scaling
                    dy = (local_knots[j][1] - local_knots[i][1]) / scaling
                    dxk = (local_knots[j][0] - refer_knot[0]) / scaling
                    dyk = (local_knots[j][1] - refer_knot[1]) / scaling
                    coe_matrix[i][j] = math.sqrt(dx * dx + dy * dy + self.shape_para) - math.sqrt(dxk * dxk + dyk * dyk + self.shape_para)
    
            for i in range(self.n_sup_knots):
                dx = (-local_knots[i][0] + refer_knot[0]) / scaling
                dy = (-local_knots[i][1] + refer_knot[1]) / scaling
                ffunc = math.sqrt(dx * dx + dy * dy + self.shape_para)
                der_vectors[i][0] = dx / ffunc
                der_vectors[i][1] = dy / ffunc
                der_vectors[i][2] = (dy * dy + self.shape_para) / (ffunc ** 3.) - 1.0 / math.sqrt(self.shape_para)
                der_vectors[i][3] = (dx * dx + self.shape_para) / (ffunc ** 3.) - 1.0 / math.sqrt(self.shape_para)
                
            der_vectors[self.n_sup_knots][0] = 0.0
            der_vectors[self.n_sup_knots][1] = 0.0
            der_vectors[self.n_sup_knots][2] = 0.0
            der_vectors[self.n_sup_knots][3] = 0.0
    
            solution = self.LU_solver(coe_matrix, der_vectors)
    
            for i in range(self.n_derivatives):
                for j in range(self.n_sup_knots+1):
                    if (i == 0) or (i == 1):
                        solution[j][i] = solution[j][i] / scaling
                    else:
                        solution[j][i] = solution[j][i] / scaling / scaling
            
            assert self.local_sn[k][self.n_sup_knots] < self.n_in_knots

            self.w_coe[k] = solution        

        return

    def LU_solver(self, coe_matrix, der_vectors):
        max_coe_col_pos = np.ndarray(shape=(self.n_sup_knots+1), dtype=int, order="F")

        for k in range(self.n_sup_knots+1):
            max_coe = 0.0
            for i in range(k, self.n_sup_knots+1):         #search the maximum value of the elements in coefficient matrix
                for j in range(k, self.n_sup_knots+1):
                    if abs(coe_matrix[i][j]) > max_coe:
                        max_coe = abs(coe_matrix[i][j])
                        max_coe_col_pos[k] = j          #position of the maximum value
                        max_coe_row_pos = i
            
            if max_coe < 1E-20:
                print("The linear problem cannot be solved by LU decomposition method!\n")
                sys.exit()

            for j in range(k, self.n_sup_knots+1):
                temp = coe_matrix[k][j]
                coe_matrix[k][j] = coe_matrix[max_coe_row_pos][j]
                coe_matrix[max_coe_row_pos][j] = temp

            for j in range(self.n_derivatives):
                temp = der_vectors[k][j]
                der_vectors[k][j] = der_vectors[max_coe_row_pos][j]
                der_vectors[max_coe_row_pos][j] = temp

            for i in range(self.n_sup_knots+1):
                temp = coe_matrix[i][k]
                coe_matrix[i][k] = coe_matrix[i][max_coe_col_pos[k]]
                coe_matrix[i][max_coe_col_pos[k]] = temp

            for j in range(k+1, self.n_sup_knots+1):
                coe_matrix[k][j] = coe_matrix[k][j] / coe_matrix[k][k]

            for j in range(self.n_derivatives):
                der_vectors[k][j] = der_vectors[k][j] / coe_matrix[k][k]

            for i in range(self.n_sup_knots+1):
                if i != k:
                    for j in range(k+1, self.n_sup_knots+1):
                        coe_matrix[i][j] = coe_matrix[i][j] - coe_matrix[i][k] * coe_matrix[k][j]
                    for j in range(self.n_derivatives):
                        der_vectors[i][j] = der_vectors[i][j] - coe_matrix[i][k] * der_vectors[k][j]

        for k in range(self.n_sup_knots, -1, -1):
            for j in range(self.n_derivatives):
                temp = der_vectors[k][j]
                der_vectors[k][j] = der_vectors[max_coe_col_pos[k]][j]
                der_vectors[max_coe_col_pos[k]][j] = temp
        
        #print("The linear problem is solved by LU decomposition method!\n")

        return der_vectors
    
    @jit
    def get_jacobi_matrix(self):
        self.jacobi_matrix = np.full(shape=(self.n_in_knots, self.n_in_knots), fill_value=0.0, dtype=float, order="F")
        for i in range(self.n_in_knots):
            for j in range(self.n_in_knots):
                if j in self.local_sn[i]:
                    assert self.local_sn[i][self.n_sup_knots] == i
                    k = np.where(self.local_sn[i] == j)
                    k = int(k[0])
                    if j == i:
                        assert k == self.n_sup_knots
                        temp = 0.0
                        for l in range(self.n_sup_knots+1):
                            temp = temp + (self.w_coe[i][l][0] + self.w_coe[i][l][1]) * self.u[self.local_sn[i][l]]

                        self.jacobi_matrix[i][j] = self.w_coe[i][k][3] + self.w_coe[i][k][2] \
                         + temp + (self.w_coe[i][k][1] + self.w_coe[i][k][0]) * self.u[i] \
                         - 2 * (self.pos_total_knots[i][0] + self.pos_total_knots[i][1])
                    else:
                        self.jacobi_matrix[i][j] = self.w_coe[i][k][3] + self.w_coe[i][k][2] \
                         + (self.w_coe[i][k][1] + self.w_coe[i][k][0]) * self.u[i]
     
        return

    @jit
    def get_inverse_matrix(self, matrix, dimension):
        temp_matrix = np.eye(dimension, dtype=float, order="F")
        for i in range(dimension):
            i1 = i + 1
            i2 = i
            if i != dimension:
                for j in range(i1, dimension):
                    if abs(matrix[j][i]) > abs(matrix[i2][i]):
                        i2 = j
                if abs(matrix[i2][i]) < 1E-20:
                    print("This matrix has no iverse!")
                    sys.exit()
                if i2 != i:
                    for j in range(dimension):
                        temp = matrix[i][j]
                        matrix[i][j] = matrix[i2][j]
                        matrix[i2][j] = temp
                        temp = temp_matrix[i][j]
                        temp_matrix[i][j] = temp_matrix[i2][j]
                        temp_matrix[i2][j] = temp
            if abs(matrix[i][i]) < 1E-20:
                print("This matrix has no iverse!")
                sys.exit()
            for j in range(dimension):
                if j != i:
                    temp = matrix[j][i] / matrix[i][i]
                    if abs(temp) < 1E-20:
                        temp = 0.0
                    for k in range(dimension):
                        matrix[j][k] = matrix[j][k] - temp * matrix[i][k]
                        temp_matrix[j][k] = temp_matrix[j][k] - temp * temp_matrix[i][k]
                        
        for i in range(dimension):
            temp = matrix[i][i]
            for j in range(dimension):
                matrix[i][j] = temp_matrix[i][j] / temp
        
        return matrix

    def nonlinear_equations(self):
        self.V = np.ndarray(shape=(self.n_in_knots), dtype=float, order="F")
        for i in range(self.n_in_knots):
            temp1 = 0.0
            temp2 = 0.0
            for l in range(self.n_sup_knots+1):
                temp1 = temp1 + (self.w_coe[i][l][3] + self.w_coe[i][l][2]) * self.u[self.local_sn[i][l]]
                temp2 = temp2 + self.u[i] * (self.w_coe[i][l][1] + self.w_coe[i][l][0]) * self.u[self.local_sn[i][l]]
            temp3 = 2 * (self.pos_total_knots[i][0] + self.pos_total_knots[i][1]) * self.u[i] + 4
            self.V[i] = temp1 + temp2 - temp3
        
        return

    @jit
    def broyden_solver(self):
        W = np.ndarray(shape=(self.n_in_knots), dtype=float, order="F")
        S = np.ndarray(shape=(self.n_in_knots), dtype=float, order="F")
        Y = np.ndarray(shape=(self.n_in_knots), dtype=float, order="F")
        U = np.ndarray(shape=(self.n_in_knots), dtype=float, order="F")
        Z = np.ndarray(shape=(self.n_in_knots), dtype=float, order="F")
        self.l2_err = []

        self.inv_jacobi = self.get_inverse_matrix(self.jacobi_matrix, self.n_in_knots)
        k = 1
        S, sol_l2_norm = self.get_s(self.inv_jacobi, self.V)
        self.l2_err.append(float(sol_l2_norm))
        #print('%5i   %15.10f\n' % (k, sol_l2_norm))

        for i in range(self.n_in_knots):
            self.u[i] = self.u[i] + S[i]

        k = 2
        while k < self.iteration:
            W = self.V
            self.nonlinear_equations()
            Y = self.V - W            
            Z, Zn = self.get_s(self.inv_jacobi, Y)

            p = 0.0
            for i in range(self.n_in_knots):
                p = p - S[i] * Z[i]
                U[i] = 0.0
                for j in range(self.n_in_knots):
                    U[i] = U[i] + S[j] * self.inv_jacobi[j][i]
            
            for i in range(self.n_in_knots):
                for j in range(self.n_in_knots):
                    self.inv_jacobi[i][j] = self.inv_jacobi[i][j] + (S[i] + Z[i]) * U[j] / p
            
            S, sol_l2_norm = self.get_s(self.inv_jacobi, self.V)
            self.l2_err.append(float(sol_l2_norm))

            for i in range(self.n_in_knots):
                self.u[i] = self.u[i] + S[i]

            #print('%5i   %15.10f\n' % (k, sol_l2_norm))

            if sol_l2_norm < self.TOL:
                print("procedure completed successfully!")
                print("terminal iteration is ", k)
                print("final value of L2 norm of error is ", sol_l2_norm)
                print("supporing knots number is ", self.n_sup_knots)
                print("internal knots number is ", self.n_in_knots)
                print("shape parameter is ", self.shape_para)
                return

            if sol_l2_norm > 1E6:
                print("L2 norm of error is too large!")
                print("terminal iteration is ", k)
                print("final value of L2 norm of error is ", sol_l2_norm)
                print("supporing knots number is ", self.n_sup_knots)
                print("internal knots number is ", self.n_in_knots)
                print("shape parameter is ", self.shape_para)
                sys.exit()
            
            k = k + 1
        
        print("maximum number of iterations exceeded!")
        sys.exit()

        return

    @jit
    def get_s(self, inv_jacobi, solution):
        sol_2_norm = 0.0
        Z = np.zeros(shape=(self.n_in_knots), dtype=float, order="F")
        for i in range(self.n_in_knots):
            for j in range(self.n_in_knots):
                Z[i] = Z[i] - inv_jacobi[i][j] * solution[j]
            sol_2_norm = sol_2_norm + Z[i] * Z[i]
        sol_2_norm = math.sqrt(sol_2_norm)

        return Z, sol_2_norm

    def save_results(self):
        out_u = open("u.plt", "w")
        out_u.write('VARIABLES = "x", "y", "u"\n')
        item = ('ZONE T = "MQ_RBF", I = %d'', F = POINT\n') % (self.n_total_knots)
        out_u.write(item)
        for i in range(self.n_total_knots):
            item = '%10.6f     %10.6f     %10.6f\n' % \
            (self.pos_total_knots[i][0], self.pos_total_knots[i][1], self.u[i])
            out_u.write(item)
        
        out_err = open("err.txt", "w")
        for i in range(len(self.l2_err)):
            item = '%10.9f\n' % (self.l2_err[i])
            out_err.write(item)
        
        out_u.close()
        out_err.close()

        self.u_err = np.ndarray(shape=(self.n_total_knots), dtype=float, order='F')
        self.u_ana_sol = self.pos_total_knots[:, 0] ** 2 + self.pos_total_knots[:, 1] ** 2
        self.u_err = self.u - self.u_ana_sol

        np.savetxt("u_err.txt", self.u_err)

        l2_norm_u_err = np.sum((self.u_err / (self.u_ana_sol + 1E-8)) ** 2.) / self.n_total_knots
        print("L2 norm of ralative error is ", l2_norm_u_err)
        np.savetxt("l2_norm_u_err.txt", np.array([l2_norm_u_err]))
        return

    def calculate(self, coordinate_x, coordinate_y):
        dist = np.ndarray(shape=(self.n_total_knots), dtype=float, order="F")

        for i in range(self.n_total_knots):            
            dx = self.pos_total_knots[i][0] - coordinate_x
            dy = self.pos_total_knots[i][1] - coordinate_y
            dist[i] = math.sqrt(dx * dx + dy * dy)
    
        cal_dist = Series(dist)
        cal_dist = cal_dist.sort_values()         #ascending order by distance between the reference knot and other knots
            
        inter_sn = np.ndarray(shape=(self.n_sup_knots), dtype=int, order="F")
        inter_sn = cal_dist.index[1:self.n_sup_knots+1]

        x = np.ndarray(shape=(self.n_sup_knots), dtype=float, order="F")
        y = np.ndarray(shape=(self.n_sup_knots), dtype=float, order="F")
        z = np.ndarray(shape=(self.n_sup_knots), dtype=float, order="F")
        
        for i in range(self.n_sup_knots):
            x[i] = self.pos_total_knots[inter_sn[i]][0]
            y[i] = self.pos_total_knots[inter_sn[i]][1]
            z[i] = self.u[inter_sn[i]]

        inter_fun = Rbf(x, y, z)
        inter_u = inter_fun(coordinate_x, coordinate_y)

        return inter_u