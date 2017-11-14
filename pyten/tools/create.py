__author__ = 'Song'
__copyright__ = "Copyright 2016, The Helios Project"

import numpy as np
import pyten.tenclass


def create(problem='basic', siz=None, r=2, miss=0, tp='CP', aux=None, timestep=5):
    """ A function to create a Tensor decomposition or completion problem.
    Input:
     problem: Tensor completion/decomposition problem (basic,auxiliary,dynamic)
     siz: size of Tensor
     r: rank of Tensor
     miss: missing percentage of data
     tp: type of expect solution. (Tucker or CP)
    Output:
     T: generated Tensor;
     Omega: missing data matrix (0: Miss; 1:Exist);
     sol: solution.
                                             """
    if siz is None:
        if problem == 'dynamic':
            siz = np.ones([timestep, 3]) * 20
        else:
            siz = [20, 20, 20]

    if problem == 'dynamic':
        dims = len(siz[0])
    else:
        dims = len(siz)

    if type(r) == int:
        r = np.zeros(dims) + r

    if problem == 'basic':
        if tp == 'Tucker':
            # Solution Decomposition Matrices
            u = [np.random.random([siz[n], r[n]]) for n in range(dims)]
            core = pyten.tenclass.Tensor(np.random.random(r))
            sol = pyten.tenclass.Ttensor(core, u)

        elif tp == 'CP':
            # Solution Decomposition Matrices
            u = [np.random.random([siz[n], r[n]]) for n in range(dims)]
            syn_lambda = np.ones(r[0])
            sol = pyten.tenclass.Ktensor(syn_lambda, u)
        else:
            raise ValueError('No Such Method.')

    elif problem == 'auxiliary':
        if tp == 'sim':
            if aux is None:
                aux = [np.diag(np.ones(siz[n] - 1), -1) + np.diag(np.ones(siz[n] - 1), 1) for n in range(dims)]
                epsilon = [np.random.random([r[n], 2]) for n in range(dims)]
                # Solution Decomposition Matrices
                tmp = []
                for n in range(dims):
                    tmp.append(np.array([range(1, siz[n] + 1), np.ones(siz[n])]).T)
                u = [np.dot(tmp[n], epsilon[n].T) for n in range(dims)]
            else:
                # Solution Decomposition Matrices
                u = [np.random.multivariate_normal(np.zeros(siz[n]), aux[n], r[n]).T for n in range(dims)]
            syn_lambda = np.ones(r[0])
            sol = pyten.tenclass.Ktensor(syn_lambda, u)
        elif tp == 'couple':
            u = [np.random.random([siz[n], r[n]]) for n in range(dims)]
            syn_lambda = np.ones(r[0])
            sol = pyten.tenclass.Ktensor(syn_lambda, u)
            if aux is None:
                aux = [np.dot(sol.Us[n], np.random.random([r[n], siz[n]])) for n in range(dims)]
        else:
            raise ValueError('Do Not Support Such Auxiliary Format.')

    elif problem == 'dynamic':
        ten = []
        omega = []
        sol = []
        for t in range(timestep):
            u = [np.random.random([siz[t, n], r[n]]) for n in range(dims)]
            syn_lambda = np.ones(r[1])
            temp_sol = pyten.tenclass.Ktensor(syn_lambda, u)
            temp_omega = (np.random.random(siz[t]) > miss) * 1
            temp_ten = temp_sol.totensor()
            temp_ten.data[temp_omega == 0] -= temp_ten.data[temp_omega == 0]
            omega.append(temp_omega)
            sol.append(temp_sol)
            ten.append(temp_ten)
        return ten, omega, sol, siz, timestep
    else:
        raise ValueError('No Such Scenario.')

    ten = sol.totensor()
    omega = (np.random.random(siz) > miss) * 1
    ten.data[omega == 0] -= ten.data[omega == 0]

    if problem == 'basic':
        return ten, omega, sol
    elif problem == 'auxiliary':
        return ten, omega, sol, aux
    else:
        return ten, omega, sol
