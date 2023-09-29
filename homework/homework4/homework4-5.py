import numpy as np
import matplotlib.pyplot as plt

def driver():

    # function and derivative
    f = lambda x: x**6 -x -1
    fp = lambda x: 6*x**5 -1

    alpha = 1.1347
    
    tol = 10**(-13)
    Nmax = 100
    p0 = 2

    (pstar, p, ier) = newton(f, fp, p0, tol, Nmax)
    print('the approximate root is', pstar)
    print('the error terms are:', abs(p-alpha))
    print('the error message reads:', ier)

    p0 = 2
    p1 = 1

    (pstar2, p2, ier2) = secant(p0, p1, f, tol, Nmax)
    print('the approximate root is', pstar2)
    print('the error terms are:', abs(p2-alpha))
    print('the error message reads:', ier2)

    '''
    xk = np.zeros(10)
    for i in range(10):
        xk[i] = p[i]

    xk1 = np.zeros(9)
    for i in range(9):
        xk1 = xk1[i] = p[i+1]
        

    plt.yscale('log')
    plt.xscale('log')
    plt.plot(abs(xk1-alpha),abs(xk-alpha))
    '''


def newton(f, fp, p0, tol, Nmax):
    """
    Input:
        f,fp - function and derivative
        p0 - initial guess for root
        tol - iteration stops when p_n,p_{n+1} are within tol
        Nmax - max number of iterations
    Output:
        pstar - root approximation
        p - an array of the iterates
        ier  - error message (1 = failure, 0 = success)
    """
    p = np.zeros(Nmax+1);
    p[0] = p0

    # begin iteration
    for it in range(Nmax):
        p1 = p0-f(p0)/fp(p0)
        p[it+1] = p1
        if (abs(p1-p0) < tol):
            pstar = p1
            ier = 0
            return [pstar, p, ier]
        p0 = p1
    pstar = p1
    ier = 1
    return [pstar, p, ier]

def secant(p0, p1, f, tol, Nmax):
    """
    Input:
        p0,p1 - two points
        f - function
        tol - iteration stops when error is within tol
        Nmax - max number of iterations
    Output:
        pstar - approximate root
        p - an array of the iterates 
        ier - error message (1 = failure, 0 = success)
    """

    fp0 = f(p0)
    fp1 = f(p1)

    p = np.zeros(Nmax+1)
    
    # initial checks
    if (fp0 == 0):
        pstar = p0
        p[0] = p0
        ier = 0
        return [pstar, ier]
    if (fp1 == 0):
        pstar = p1
        p[0] = p1
        ier = 0
        return [pstar, ier]

    # begin iteration
    for i in range(Nmax):
        if (abs(fp1-fp0) == 0):
            pstar = p1
            p[i] = p1
            ier = 1
            print('cannot divide by zero')
            return [pstar, p, ier]
        p2 = p1 - ((fp1*(p1-p0))/(fp1-fp0))
        p[i] = p2
        if (abs(p2-p1) < tol):
            pstar = p2
            ier = 0
            return [pstar, p, ier]
        p0 = p1
        fp0 = fp1
        p1 = p2
        fp1 = f(p2)

    # if the loop is exited, failure
    pstar = p2
    ier = 1
    return [pstar, p, ier]
        
        
driver()




