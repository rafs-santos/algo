import cvxpy as cp
from cvxpy.atoms.norm1 import norm1
import numpy as np
import matplotlib.pyplot as plt


# Exemplo

n = 512
m = 64

k = np.arange(0,n)
t = np.arange(0,n)

F = np.exp(-1j*2*np.pi*np.outer(k,t)/n)/np.sqrt(n)



freq = np.random.permutation(n)
freq = freq[:m]

A = np.concatenate((np.real(F[freq,:]), np.imag(F[freq,:])), axis=0)

S = 28
support = np.random.permutation(n)
support =  support[:S]

x0 = np.zeros(n)
x0[support] = np.random.randn(S)

b = A @ x0


# Construct the problem.
x = cp.Variable(n)
prob = cp.Problem(cp.Minimize(norm1(x)), [A @ x == b])

# The optimal objective value is returned by `prob.solve()`.
result = prob.solve()

plt.subplot(2,1,1)
plt.plot(x0, 'b')
plt.plot(x.value, 'r--')
plt.subplot(2,1,2)
plt.plot(F @ x0, 'b')
plt.plot(F @ x.value, 'r--')
plt.show()

