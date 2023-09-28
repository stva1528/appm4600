import numpy as np
import scipy
from scipy import special

def driver():
    # given initial values    
    Ti = 20
    Ts = -15
    alpha = 0.138*10**(-6)
    t = 5.184*10**6

    # temp function 
    T = lambda x: (Ti-Ts)*special.erf(x/(2*(alpha*t)**(1/2)))+Ts

    a = 0
    b = 10
    tol = 10**(-13)

    [astar,ier] = bisection(T,a,b,tol)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('T(astar) =', T(astar))


def bisection(f,a,b,tol):
    """
    Input:
        f - function
        a,b - start and end points
        tol  - bisection stops when interval length < tol

    Output:
        astar - approximation of root
        ier - error message (1 = failure, 0 = success)
    """

    # verify a root exists
    fa = f(a)
    fb = f(b);
    if (fa*fb>0):
        ier = 1
        astar = a
        return [astar, ier]

    # verify endpoints are not roots
    if (fa == 0):
        astar = a
        ier =0
        return [astar, ier]
    if (fb ==0):
        astar = b
        ier = 0
        return [astar, ier]

    # begin iteration
    count = 0
    d = 0.5*(a+b)
    while (abs(d-a)> tol):
        fd = f(d)
        if (fd ==0):
            astar = d
            ier = 0
            return [astar, ier]
        if (fa*fd<0):
            b = d
        else:
            a = d
            fa = fd
        d = 0.5*(a+b)

    # if escaped while loop, failure
    astar = d
    ier = 0
    return [astar, ier]
      
driver()               

