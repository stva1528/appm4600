import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm 

def driver():
    x = np.array([1, 1])
    tol = 1e-10
    Nmax = 100

    [xstar, ier, its] = kindaNewton(x, tol, Nmax)
    print('approximate root:', xstar)
    print('number of iterations:', its)
    print('error message:', ier)

    [xstar, ier, its] = Newton(x, tol, Nmax)
    print('approximate root:', xstar)
    print('number of iterations:', its)
    print('error message:', ier)

def evalF(x): 
    # evaluate matrix function
    F = np.zeros(2)
    
    F[0] = 3*x[0]**2 - x[1]**2
    F[1] = 3*x[0]*x[1]**2 - x[0]**3 - 1
    return F

def evalJ(x): 
    # evaluate the jacobian 
    
    J = np.array([[6*x[0], -2*x[1]],
                  [3*x[1]**2-3*x[0]**2, 6*x[0]*x[1]]])
    return J
    
def kindaNewton(x0, tol, Nmax):
    '''
    kinda Newton: newton iteration with a fixed jacobian
    Input:
        x0 - initial guess
        tol - tolerance
        Nmax - max number of iterations
    Output:
        xstar - approximate root
        ier - error message
        its - number of iterations
    '''
    J = np.array([[1/6, 1/18],
                  [0, 1/6]])
    
    for its in range(Nmax):
       F = evalF(x0)
       
       x1 = x0 - J.dot(F)
       
       if (norm(x1-x0) < tol):
           xstar = x1
           ier = 0
           return[xstar, ier, its]
           
       x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar, ier, its]

def Newton(x0,tol,Nmax):
    '''
    Input:
        x0 - initial guess
        tol - tolerance
        Nmax - max number of iterations
    Output:
        xstar - approximate root
        ier - error message
        its - number of iterations
    '''
    for its in range(Nmax):
       J = evalJ(x0)
       Jinv = inv(J)
       F = evalF(x0)
       
       x1 = x0 - Jinv.dot(F)
       
       if (norm(x1-x0) < tol):
           xstar = x1
           ier =0
           return[xstar, ier, its]
           
       x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar,ier,its]

driver()
