import cvxpy as cp
from cvxpy.atoms.norm1 import norm1
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct


#import threading
import time
import multiprocessing as mp

import scipy.io

resu = np.zeros((10,2500))

def cvx_minimization(Theta, y, L, q):
    # construct the problem.
    x = cp.Variable(L)
    prob = cp.Problem(cp.Minimize(norm1(x)), [Theta @ x == y])

    # the optimal objective value is returned by `prob.solve()`.
    result = prob.solve()
    #resu[k,:] = x.value
    q.put(x.value)
    #return x.value


if __name__ == '__main__':
    #path = "../BD/BD.mat"
    ''' ------------ Load .mat data'''
    path = "../../../Documentos/Mestrado_UTFPR/MATLAB/BD/BD_Sinais_New.mat"

    test = scipy.io.loadmat(path)
    fs = test['fs'][0][0]
    #print(test)
    saudavel = test['saudavel']

    ''' ------------ Code config '''
    L = 2500                    # Windows width
    points = 600                # Number of random points for each Windows
    perm = np.random.permutation(L)
    perm = perm[:points]


    # DCT transform
    Psi = dct(np.identity(L))
    Theta = Psi[perm,:]
    ''' ------------ Load .mat data'''
    LL = 2
    Thre = []
    t = time.time()


    #ctx = mp.get_context('spawn')
    q = mp.Queue()
    y = np.zeros((10, points))

    Process = []
    for k in range(LL):
        x = saudavel[k*L:(k+1)*L,1]
        y = x[perm]
        p = mp.Process(target=cvx_minimization, args=(Theta, y, L, q))
        Process.append(p)
        p.start()

    for k in range(LL):
        Process[k].join()
    aux = q.get()
    print(aux.shape)
    '''
    for k in range(LL):
        x = saudavel[k*L:(k+1)*L,1]
        y = x[perm]
        p = multiprocessing.Process(target=cvx_minimization, args=(Theta, y, L, k))
        Thre.append(p)
        p.start()
        #Thre.append(threading.Thread(target=cvx_minimization, args=(Theta, y, L, k)))


    for proc in Thre:
        #Thre[i].join()
        proc.join()
        #print(que.get)
        #res.append(que.get)
    # do stuff

    elapsed = time.time() - t
    print("------------------------------------------------")
    print(elapsed)
    xrecon = []
    print(resu.shape)
    for p in range(LL):
        aux = idct(resu[p,:])
        #print(aux.shape)
        xrecon = np.append(xrecon, aux, axis=0)
        #print(xrecon.shape)
        #xrecon.append(aux[:])

    plt.plot(saudavel[:,1],'b')
    plt.plot(xrecon,'r--')

    plt.show()
'''
