import numpy as np
import scipy
from scipy import special

def driver():
    # given initial values    
    Ti = 20
    Ts = -15
    alpha = 0.138*10**(-6)
    t = 5.184*10**6

    # temp function and derivative
    T = lambda x: (Ti-Ts)*special.erf(x/(2*(alpha*t)**(1/2)))+Ts
    Tp = lambda x: (2/(np.pi)**(1/2))*(Ti-Ts)*np.exp(-x**2/4*alpha*t)

    tol = 10**(-13)
    Nmax = 100
    p0 = 0.01

    (pstar, ier) = newton(T, Tp, p0, tol, Nmax)
    print('the approximate root is', pstar)
    print('the error message reads:', ier)


def newton(f, fp, p0, tol, Nmax):
    """
    Input:
        f,fp - function and derivative
        p0 - initial guess for root
        tol - iteration stops when p_n,p_{n+1} are within tol
        Nmax - max number of iterations
    Output:
        pstar - root approximation
        ier  - error message (1 = failure, 0 = success)
    """
    # begin iteration
    for i in range(Nmax):
        p1 = p0 - (f(p0)/fp(p0))
        if (abs(p1-p0) < tol):
            pstar = p1
            ier = 0
            return [pstar, ier]
        p0 = p1
        
    # if escaped loop, failure
    pstar = p1
    ier = 1
    return [pstar, ier]
        
driver()
