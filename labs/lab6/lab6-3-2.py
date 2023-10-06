import numpy as np
import math
from numpy.linalg import inv 
from numpy.linalg import norm

def driver():
    # initial values
    x0 = np.array([1, 0])
    tol = 10e-10
    Nmax = 100

    [xstar, ier, its, nj] = SlackerNewton(x0, tol, Nmax)
    print('approximate root is:', xstar)
    print('number of iterations:', its)
    print('number of times an updated jacobian was computed:', nj)
    print('error message:', ier)
    


def evalF(x): 
# evaluate vector function F
    F = np.zeros(2)
    F[0] = 4*x[0]**2 + x[1]**2 - 4
    F[1] = x[0] + x[1] - math.sin(x[0]-x[1])
    return F
    
def evalJ(x): 
# evaluate the jacobian
    J = np.array([[8*x[0], 2*x[1]],
                 [1.0-math.cos(x[0]-x[1]), 1.0+math.cos(x[0]-x[1])]])
    return J

def SlackerNewton(x0, tol, Nmax):
    '''
    Slacker Newton: use the inverse of the Jacobian for initial guess, only
        update the jacobian if the difference between iterates is large
    Input:
        x0 - initial guess
        tol - tolerance
        Nmax - max number of iterations
    Output:
        xstar - approximate root
        ier - error message
        its - number of iterations
        nj - number of times new jacobian was computed
        
    '''
    J = evalJ(x0)
    Jinv = inv(J)
    nj = 0
    for its in range(Nmax):
        F = evalF(x0)
        x1 = x0 - Jinv.dot(F)

        if (norm(x1-x0) > 0.1):
            nj = nj+1
            J = evalJ(x0)
            Jinv = inv(J)
            F = evalF(x0)
            x1 = x0 - Jinv.dot(F)
            

        if (norm(x1-x0) < tol):
            xstar = x1
            ier = 0
            return[xstar, ier, its, nj]
        x0 = x1

    xstar = x1
    ier = 1
    return[xstar, ier, its, nj]

driver()
