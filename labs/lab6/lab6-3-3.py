import numpy as np
import math
from numpy.linalg import inv 
from numpy.linalg import norm

def driver():
    # initial values
    x0 = np.array([1, 0])
    tol = 10e-10
    Nmax = 100

    h = 0.1
    [xstar, ier, its] = NewtonApproxJ(x0, h, tol, Nmax)
    print('when h=0.1:')
    print('approximate root is:', xstar)
    print('number of iterations:', its)
    print('error message:', ier)

    h = 0.001
    [xstar, ier, its] = NewtonApproxJ(x0, h, tol, Nmax)
    print('when h=0.001:')
    print('approximate root is:', xstar)
    print('number of iterations:', its)
    print('error message:', ier)

    # the method has less iterations with a smaller h


def evalF(x): 
# evaluate vector function F
    F = np.zeros(2)
    F[0] = 4*x[0]**2 + x[1]**2 - 4
    F[1] = x[0] + x[1] - math.sin(x[0]-x[1])
    return F
    
def approxJ(x, F, h): 
# evaluate the approximate jacobian

    # add h to the first value of x vector, compute F
    x1 = np.array([x[0]+h, x[1]])
    fx1 = evalF(x1)
    
    # add h to the second value of x vector, compute F
    x2 = np.array([x[0], x[1]+h])
    fx2 = evalF(x2)

    # find approxixmate derivative values 
    val1 = (fx1[0]-F[0])/h
    val2 = (fx2[0]-F[0])/h
    val3 = (fx1[1]-F[1])/h
    val4 = (fx2[1]-F[1])/h

    # set as approximate jacobian
    J = np.array([[val1, val2],
                   [val3, val4]])
    return J

def NewtonApproxJ(x0, h, tol, Nmax):
    '''
    Input:
        x0 - initial guess
        h - value to approximate jacobian
        tol - tolerance
        Nmax - max number of iterations
    Output:
        xstar - approximate root
        ier - error message
        its - number of iterations
    '''

    for its in range(Nmax):
        F = evalF(x0)
        J = approxJ(x0, F, h)
        Jinv = inv(J)

        x1 = x0 - Jinv.dot(F)

        if (norm(x1-x0) < tol):
            xstar = x1
            ier = 0
            return[xstar, ier, its]

        x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar, ier, its]

driver()
