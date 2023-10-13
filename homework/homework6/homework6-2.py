import matplotlib.pyplot as plt
import numpy as np
import math
import time
from numpy.linalg import inv 
from numpy.linalg import norm

def driver():
    x0 = np.array([0, 0, 0])

    Nmax = 100
    tol = 1e-6

    t = time.time()
    for j in range(20):
      [xstar,ier,its] =  Newton(x0,tol,Nmax)
    elapsed = time.time()-t
    print('*** Newton ***')
    print('the error message reads:',ier)
    print('the approx root is:', xstar) 
    print('time:',elapsed/20)
    print('number of iterations:',its)

    t = time.time()
    for j in range(20):
      [xstar,ier,its] =  SteepestDescent(x0,tol,Nmax)
    elapsed = time.time()-t
    print('*** Steepest Descent ***')
    print('the error message reads:',ier)
    print('the approx root is:', xstar) 
    print('time:',elapsed/20)
    print('number of iterations:',its)

    t = time.time()
    for j in range(20):
      [xstar,ier,its] =  SteepestDescent(x0,(5*10**(-2)),Nmax)
      [xstar,ier,its] = Newton(xstar,tol,Nmax)
    elapsed = time.time()-t
    print('*** Steepest Descent & Newton ***')
    print('the error message reads:',ier)
    print('the approx root is:', xstar) 
    print('time:',elapsed/20)
    print('number of iterations:',its)

def evalF(x):

    F = np.zeros(3)
    F[0] = x[0] + math.cos(x[0]*x[1]*x[2])-1
    F[1] = (1-x[0])**(1/4)+x[1]+0.05*x[2]**2-0.15*x[2]-1
    F[2] = -x[0]**2-0.1*x[1]**2+0.001*x[1]+x[2]-1
    return F

def evalJ(x): 

    J =np.array([[1-x[1]*x[2]*math.sin(x[0]*x[1]*x[2]), -x[0]*x[2]*math.sin(x[0]*x[1]*x[2]), -x[1]*x[1]*math.sin(x[0]*x[1]*x[2])],
          [-(1/4)*(1-x[0])**(-3/4), 1, 0.1*x[2]-0.15],
          [-2*x[0], -0.2*x[1]+0.001, 1]])
    return J

def evalg(x):

    F = evalF(x)
    g = F[0]**2 + F[1]**2 + F[2]**2
    return g

def eval_gradg(x):
    F = evalF(x)
    J = evalJ(x)
    
    gradg = np.transpose(J).dot(F)
    return gradg


# Newton code
def Newton(x0,tol,Nmax):

    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

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


# steepest descent code
def SteepestDescent(x,tol,Nmax):
    
    for its in range(Nmax):
        g1 = evalg(x)
        z = eval_gradg(x)
        z0 = norm(z)

        if z0 == 0:
            print("zero gradient")
        z = z/z0
        alpha1 = 0
        alpha3 = 1
        dif_vec = x - alpha3*z
        g3 = evalg(dif_vec)

        while g3>=g1:
            alpha3 = alpha3/2
            dif_vec = x - alpha3*z
            g3 = evalg(dif_vec)
            
        if alpha3<tol:
            print("no likely improvement")
            ier = 0
            return [x,g1,ier]
        
        alpha2 = alpha3/2
        dif_vec = x - alpha2*z
        g2 = evalg(dif_vec)

        h1 = (g2 - g1)/alpha2
        h2 = (g3-g2)/(alpha3-alpha2)
        h3 = (h2-h1)/alpha3

        alpha0 = 0.5*(alpha2 - h1/h3)
        dif_vec = x - alpha0*z
        g0 = evalg(dif_vec)

        if g0<=g3:
            alpha = alpha0
            gval = g0

        else:
            alpha = alpha3
            gval =g3

        x = x - alpha*z

        if abs(gval - g1)<tol:
            ier = 0
            return [x,gval,ier]

    print('max iterations exceeded')    
    ier = 1        
    return [x,g1,ier]

driver()
