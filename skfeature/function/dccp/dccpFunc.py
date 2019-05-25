
from cvxpy import *
from cvxopt import matrix
import numpy as np
import matplotlib.pyplot as plt
import dccp

import math
import scipy.io
import sys


def identity(n):
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]

def dccpFunc(T,Rho):
    eps = 10 ** -2
    # eps = 10
    _teta = 1 / eps
    # rho_list = [10**-1,1]
    # rho_list = [10**-1, 10**-2, 10**-3, 1, 10, 100, 1000, 10000, 100000,1000000]
    # rho_list = [10**-1,10**-2, 1, 10, 100, 1000, 10000]

    t = 0
    solution_list = []
    for index, rho in enumerate(Rho):
        rhoeps = rho / log(1 + 1 / eps)
        # create identitiy matrix def

        # Loading accuricy diversity matrix(my matrix)

        A = T
        I = identity(150)
        G = A * np.transpose(A)
        Gtilda = np.zeros((150, 150))

        for i in range(150):
            for j in range(150):
                Gtilda[i][j] = 0.5 * (G[i][j] / (G[i][i] + eps) + G[i][j] / (G[j][j] + eps))
            Gtilda[i][i] = G[i][i] / 150
        # -----------------
        A = Gtilda

        x = Variable(150, 1)  # result matrix
        y = Variable(150, 1)

        x.value = np.random.rand(150, 1)
        y.value = np.random.rand(150, 1)
        # ---------------------
        x0 = np.random.rand(150, 1)
        tao = pos(-lambda_min(A))
        constr = []
        constr.append(x >= -y)
        constr.append(x <= y)
        constr.append(norm(x) == 1)
        A = matrix(A)  # convert to matrix form
        prob = Problem(Minimize(tao.value * square(norm(x)) - (quad_form(x, A + mul_elemwise(tao.value, np.asarray(I)).value).value) - mul_elemwise(rhoeps.value, sum_entries(log(1 + y / eps)))), constr)
        # prob = Problem(Minimize(tao.value * square(norm(x))+ mul_elemwise(rho.value, sum_entries(mul_elemwise(teta,y).value)).value -(quad_form(x, A+ mul_elemwise(tao.value,np.asarray(I)).value).value) +  mul_elemwise(rho.value, sum_entries(mul_elemwise(teta,y).value - (1-exp( -(mul_elemwise(_teta,y).value)).value))).value), constr)
        result = prob.solve(method='dccp', solver='SCS')
        solution = [x_value.value for x_value in x]
        recover = np.matrix(solution)
        list_sol = []
        i = 0;
        j = 0;
        for index, item in enumerate(solution):
            if item > 0.1:
                i = i + 1;
                list_sol.append(index)
            if item > 0.09:
                j = j + 1;
        solution_list.append(list_sol)
    return solution_list
