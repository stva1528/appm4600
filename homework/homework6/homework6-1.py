import numpy as np
import math
import time
from numpy.linalg import inv 
from numpy.linalg import norm

def driver():

    x0 = np.array([2, -1])

    Nmax = 100
    tol = 1e-10

    t = time.time()
    for i in range(20):
        [xstar, ier, its] = Newton(x0, tol, Nmax)
    elapsed = time.time()-t
    print('*** Newton ***')
    print('the error message reads:',ier)
    print('the approximate root is:', xstar)
    print('time:',elapsed/20)
    print('number of iterations:', its)

    
    t = time.time()
    for i in range(20):
      [xstar,ier,its] =  LazyNewton(x0,tol,Nmax)
    elapsed = time.time()-t
    print('*** Lazy Newton ***')
    print('the error message reads:',ier)
    print('the approximate root is:', xstar)
    print('time:',elapsed/20)
    print('number of iterations:', its)
     
    t = time.time()
    for i in range(20):
      [xstar,ier,its] = Broyden(x0, tol,Nmax)     
    elapsed = time.time()-t
    print('*** Broyden ***')
    print('the error message reads:',ier)
    print('the approximate root is:', xstar)
    print('time:',elapsed/20)
    print('number of iterations:', its)


def evalF(x): 

    F = np.zeros(2)
    
    F[0] = x[0]**2+x[1]**2-4
    F[1] = np.exp(x[0])+x[1]-1
    return F
    
def evalJ(x): 

    
    J = np.array([[2*x[0], 2*x[1]], 
        [math.exp(x[0]), 1.0]])
    return J

def Newton(x0,tol,Nmax):
    '''
    input:
        x0 - initial guess
        tol - tolerance
        Nmax - max iterations
    output:
        xstarc - approx root
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

def LazyNewton(x0,tol,Nmax):
    '''
    Lazy Newton = use only the inverse of the Jacobian for initial guess
    input:
        x0 - initial guess
        tol - tolerance
        Nmax - max iterations
    output:
        xstar - approx root
        ier - error message
        its - number of iterations
    '''

    J = evalJ(x0)
    Jinv = inv(J)
    for its in range(Nmax):

       F = evalF(x0)
       x1 = x0 - Jinv.dot(F)
       
       if (norm(x1-x0) < tol):
           xstar = x1
           ier =0
           return[xstar, ier,its]
           
       x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar,ier,its]


def Broyden(x0,tol,Nmax):
    '''
    input:
        x0 - initial guess
        tol - desired accuracy
        Nmax - max number of iterations
    output:
        alpha - approx root
        ier - error message
        its - number of iterations
    '''
    
    A0 = evalJ(x0)

    v = evalF(x0)
    A = np.linalg.inv(A0)

    s = -A.dot(v)
    xk = x0+s
    for  its in range(Nmax):
       '''(save v from previous step)'''
       w = v
       ''' create new v'''
       v = evalF(xk)
       '''y_k = F(xk)-F(xk-1)'''
       y = v-w;                   
       '''-A_{k-1}^{-1}y_k'''
       z = -A.dot(y)
       ''' p = s_k^tA_{k-1}^{-1}y_k'''
       p = -np.dot(s,z)                 
       u = np.dot(s,A) 
       ''' A = A_k^{-1} via Morrison formula'''
       tmp = s+z
       tmp2 = np.outer(tmp,u)
       A = A+1./p*tmp2
       ''' -A_k^{-1}F(x_k)'''
       s = -A.dot(v)
       xk = xk+s
       if (norm(s)<tol):
          alpha = xk
          ier = 0
          return[alpha,ier,its]
    alpha = xk
    ier = 1
    return[alpha,ier,its]

driver()
