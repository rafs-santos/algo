import cvxpy as cp
from cvxpy.atoms.norm1 import norm1
from scipy.fftpack import dct, idct

import numpy as np
import matplotlib.pyplot as plt

#import threading
import time
import multiprocessing as mp

import scipy.io

import concurrent.futures

# %% Setup

def cvx_minimization(Theta, y, L, k):
    # construct the problem.
    x = cp.Variable(L)
    prob = cp.Problem(cp.Minimize(norm1(x)), [Theta @ x == y])

    # the optimal objective value is returned by `prob.solve()`.
    result = prob.solve()
    #q.put(x.value)
    return [x.value, k]

# %% Main Func

if __name__ == '__main__':

    path = "../../../Documentos/Mestrado_UTFPR/MATLAB/BD/BD_Sinais_New.mat"

    dataB = scipy.io.loadmat(path)
    fs = dataB['fs'][0][0]

    saudavel = dataB['saudavel']
    edema = dataB['saudavel']
    nodulo = dataB['saudavel']

    vowelData = np.concatenate((saudavel, edema, nodulo), axis=0)

    ''' ------------ Code config '''
    L = 2500                    # Windows width
    points = 600                # Number of random points for each Windows
    perm = np.random.permutation(L)
    perm = perm[:points]


    # DCT transform
    Psi = dct(np.identity(L))
    Theta = Psi[perm,:]
    ''' ------------ Load .mat data'''
    LL = 10


# %% Start Process  
    result = []
    t = time.time()
    n_sig = 3
    with concurrent.futures.ProcessPoolExecutor() as executor:
        count = 0
        for j in range(n_sig):
            for k in range(LL):
                x = vowelData[k*L:(k+1)*L, j]
                y = x[perm]
                result.append(executor.submit(cvx_minimization, Theta, y, L, count))
                count = count + 1
                print(time.time()-t)

    print(time.time()-t)

    aux = []
    for i in range(count):
        aux.append(result[i].result())

    print([item[1] for item in aux])
    aux.sort(key = lambda aux: aux[1])
    print([item[1] for item in aux])
    sol = [item[0] for item in aux]
    
    solF = np.zeros((L*LL,n_sig))
    xrecon = np.zeros((L*LL,n_sig))
    for j in range(n_sig):
        auxT = []
        auxD = []
        for k in range(j*LL,(j+1)*LL):
            re = idct(sol[k])
            auxD = np.append(auxD, re)
            auxT = np.append(auxT, sol[k])
        xrecon[:,j] = auxD
        solF[:,j] = auxT
    print(solF.shape)
    scipy.io.savemat("matlab_matrix.mat", {'dcT':solF, 'xrecon':xrecon})
    '''
    xrecon =[]
    for i in range(LL):
        re = idct(sol[i])
        xrecon = np.append(xrecon, re)
        print(xrecon.shape)
    plt.plot(saudavel[1:5000,1],'b')
    plt.plot(xrecon,'r--')

    plt.show()
    '''

    '''
    result[:].result().sort(key = lambda x: x[1])
    print(result[:].result()[1])
    for i in range(LL):
        re = idct(result[i].result())
        aux = np.append(aux, re)
        print(aux.shape)
    plt.plot(saudavel[1:5000,1],'b')
    plt.plot(aux,'r--')

    plt.show()

'''
'''
#Teste com spawn 
    q = []
    for k in range(LL):
        x = saudavel[k*L:(k+1)*L,1]
        y = x[perm]
        ctx = mp.get_context('spawn')
        q.append(ctx.Queue())
        p = ctx.Process(target=cvx_minimization, args=(Theta, y, L, q[k]))
        p.start()
        result.append(p)

    print(time.time()-t)
    aux = []
    for i in range(LL):
        result[i].join()
        res = idct(q[i].get())
        aux = np.append(aux, res)
        print(aux.shape)
        #print(q.get())
        #p.join()

    print(time.time()-t)
    plt.plot(saudavel[0:LL*L,1], 'b')
    plt.plot(aux, 'r--')
    plt.show()
'''
'''
'''
