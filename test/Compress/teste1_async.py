#asyncio_executor_thread.py
import asyncio
import concurrent.futures
import logging
import sys
import time

import cvxpy as cp
from cvxpy.atoms.norm1 import norm1
from scipy.fftpack import dct, idct

import numpy as np
import matplotlib.pyplot as plt

#import threading
import time
import multiprocessing as mp

import scipy.io

def cvx_minimization(Theta, y, L, k):
    # construct the problem.
    x = cp.Variable(L)
    prob = cp.Problem(cp.Minimize(norm1(x)), [Theta @ x == y])

    # the optimal objective value is returned by `prob.solve()`.
    result = prob.solve()
    #q.put(x.value)
    return [x.value, k]


async def run_blocking_tasks(executor):
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
    LL = 10
    Thre = []
    result = []
    #x=saudavel[:2500,1]
    #y = x[perm]
    t = time.time()
    loop = asyncio.get_event_loop()
    blocking_tasks =[]
    for k in range(LL):
        x = saudavel[k*L:(k+1)*L,1]
        y = x[perm]
        blocking_tasks.append(loop.run_in_executor(executor, cvx_minimization, Theta, y, L, k))
        print(time.time()-t)
    #log.info('waiting for executor tasks')
    completed, pending = await asyncio.wait(blocking_tasks)
    print(time.time()-t)
    results = [t.result() for t in completed]
    print(results)

if __name__ == '__main__':
    # Configure logging to show the name of the thread
    # where the log message originates.

    # Create a limited thread pool.
    executor = concurrent.futures.ProcessPoolExecutor()
    #executor = concurrent.futures.ThreadPoolExecutor()

    event_loop = asyncio.get_event_loop()
    try:
        event_loop.run_until_complete(
            run_blocking_tasks(executor)
        )
    finally:
        event_loop.close()
