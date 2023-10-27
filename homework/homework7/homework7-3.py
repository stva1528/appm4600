import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from numpy.linalg import inv

def driver():
    # given data
    f = lambda x: 1/(1+(10*x)**2)

    N = 50

    Neval = 1000

    xeval = np.linspace(-1,1,Neval+1)

    xint = np.zeros(N+1)
    for j in range(N+1):
        xint[N-j] = np.cos(((2*j)*np.pi)/(2*N))

    yint = f(xint)

    yeval = np.zeros(Neval+1)
    for i in range(Neval+1):
        yeval[i] = eval_barycentric(xint, yint, N, xeval[i])

    # plot
    fx = f(xeval)
    
    plt.figure()
    plt.plot(xint, f(xint), 'o')
    plt.plot(xeval, yeval, label = "barycentric")
    plt.plot(xeval, fx, label = "orginal function")
    plt.legend()
    plt.show()
        
    
def eval_barycentric(xint,yint,N,xval):
    phi = 1
    for i in range(N+1):
        phi *= (xval-xint[i])

    
    sum = 0
    for i in range(N+1):
        if (xval != xint[i]):
            denom = 1
            for j in range(N+1):
                if (i != j):
                    denom *= (xint[i]-xint[j])
            w = 1/denom
            sum += (w*yint[i])/(xval-xint[i])

    poly = phi*sum
        
    return poly
        

driver()
