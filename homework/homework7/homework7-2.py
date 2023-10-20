import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from numpy.linalg import inv

def driver():
    # given data
    f = lambda x: 1/(1+(10*x)**2)

    N = 5

    Neval = 1000

    xeval = np.linspace(-1,1,Neval+1)

    xint = np.zeros(N+1)
    h = 2/N
    for i in range(1,N+2):
        xint[i-1] = -1+h*(i-1)

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
        phi = phi*(xval-xint[i])

    denom = 1
    for i in range(N+1):
        if (xval != xint[i]):
            denom = denom*(xval-xint[i])

    w = 1/denom

    sum = 0
    for i in range(N+1):
        sum += yint[i]*(w/(xval-xint[i]))

    poly = phi*sum
    return poly
        

driver()
