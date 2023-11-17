import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv 
from numpy.linalg import norm
import scipy
from gauss_legendre import *

def driver():
    f = lambda x: np.sin(1/x)

    n = 5
    a = 0.1
    b = 2

    tol = 10e-3

    (I_trap, X_trap, n_trap) = adaptive_quad(a,b,f,tol,n,eval_composite_trap)
    (I_simp, X_simp, n_simp) = adaptive_quad(a,b,f,tol,n,eval_composite_simpsons)
    (I_guass, X_guass, n_guass) = adaptive_quad(a,b,f,tol,n,eval_gauss_quad)
    
    print('*** Trapezoidal ***')
    print('approx. integral:', I_trap)
    print('num interval splits:', n_trap)
    
    print('*** Simpsons ***')
    print('approx. integral:', I_simp)
    print('num interval splits:', n_simp)
    
    print('*** Gaussian ***')
    print('approx. integral:', I_guass)
    print('num interval splits:', n_guass)

    

def eval_composite_trap(n,a,b,f):
    h = (b-a)/n
    xnode = a+np.arange(0,n+1)*h
    
    I_trap = h*f(xnode[0])*1/2
    
    for j in range(1,n):
         I_trap = I_trap+h*f(xnode[j])
    I_trap= I_trap + 1/2*h*f(xnode[n])

    w = None
    
    return I_trap,xnode,w

def eval_composite_simpsons(n,a,b,f):
    h = (b-a)/n
    xnode = a+np.arange(0,n+1)*h
    I_simp = f(xnode[0])

    nhalf = n/2
    for j in range(1,int(nhalf)+1):
         # even part 
         I_simp = I_simp+2*f(xnode[2*j])
         # odd part
         I_simp = I_simp +4*f(xnode[2*j-1])
    I_simp= I_simp + f(xnode[n])
    
    I_simp = h/3*I_simp

    w = None
    
    return I_simp,xnode,w

def lgwt(N,a,b):
    N = N-1
    N1 = N+1
    N2 = N+2
    eps = np.finfo(float).eps  
    xu = np.linspace(-1,1,N1)
  
    # Initial guess
    y = np.cos((2*np.arange(0,N1)+1)*np.pi/(2*N+2))+(0.27/N1)*np.sin(np.pi*xu*N/N2)

    # Legendre-Gauss Vandermonde Matrix
    L = np.zeros((N1,N2))
  
  # Compute the zeros of the N+1 Legendre Polynomial
  # using the recursion relation and the Newton-Raphson method
    y0 = 2.
    one = np.ones((N1,))
    zero = np.zeros((N1,))

  # Iterate until new points are uniformly within epsilon of old points
    while np.max(np.abs(y-y0)) > eps:
        L[:,0] = one
        L[:,1] = y
        for k in range(2,N1+1):
            L[:,k] = ((2*k-1)*y*L[:,k-1]-(k-1)*L[:,k-2])/k

        lp = N2*(L[:,N1-1]-y*L[:,N2-1])/(1-y**2)
        y0 = y
        y = y0-L[:,N2-1]/lp
  
    # Linear map from[-1,1] to [a,b]
    x=(a*(1-y)+b*(1+y))/2
  
    # Compute the weights
    w=(b-a)/((1-y**2)*lp**2)*(N2/N1)**2
    return x,w

def eval_gauss_quad(n,a,b,f):
    x,w = lgwt(n,a,b)
    I_hat = np.sum(f(x)*w)
    return I_hat,x,w

def adaptive_quad(a,b,f,tol,n,method):
    """
  Adaptive numerical integrator for \int_a^b f(x)dx
  
  Input:
  a,b - interval [a,b]
  f - function to integrate
  tol - absolute accuracy goal
  n - number of quadrature nodes per bisected interval
  method - function handle for integrating on subinterval
         - eg) eval_gauss_quad, eval_composite_simpsons etc.
  
  Output: I - the approximate integral
          X - final adapted grid nodes
          nsplit - number of interval splits
  """

    maxit = 50
    left_p = np.zeros((maxit,))
    right_p = np.zeros((maxit,))
    s = np.zeros((maxit,1))
    left_p[0] = a; right_p[0] = b;
    
    # initial approx and grid
    s[0],x,_ = method(n,a,b,f);
    
    # save grid
    X = []
    X.append(x)
    j = 1;
    I = 0;
    nsplit = 1;
    
    while j < maxit:
        # get midpoint to split interval into left and right
        c = 0.5*(left_p[j-1]+right_p[j-1]);
        # compute integral on left and right spilt intervals
        s1,x,_ = method(n,left_p[j-1],c,f); X.append(x)
        s2,x,_ = method(n,c,right_p[j-1],f); X.append(x)
        
        if np.max(np.abs(s1+s2-s[j-1])) > tol:
            left_p[j] = left_p[j-1]
            right_p[j] = 0.5*(left_p[j-1]+right_p[j-1])
            s[j] = s1
            left_p[j-1] = 0.5*(left_p[j-1]+right_p[j-1])
            s[j-1] = s2
            j = j+1
            nsplit = nsplit+1
        else:
            I = I+s1+s2
            j = j-1
            
            if j == 0:
                j = maxit
    return I,np.unique(X),nsplit

driver()
