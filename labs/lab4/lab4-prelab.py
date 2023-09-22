import numpy as np

def driver():
# test functions
    f1 = lambda x: 1+0.5*np.sin(x)
# fixed point is alpha1 = 1.4987....
    f2 = lambda x: 3+2*np.sin(x)
# fixed point is alpha2 = 3.09...
    Nmax = 100
    tol = 1e-6
# test f1 
    x0 = 0.0
    [xstar,ier,pHat,count] = fixedpt(f1,x0,tol,Nmax)
    print('the approximate fixed point is:',xstar)
    print('f1(xstar):',f1(xstar))
    print('the list of approximations is:', pHat)
    print('the number of values in this list is:', count)
    print('Error message reads:',ier)
# test f2
    x0 = 0.0
    [xstar,ier,pHat,count] = fixedpt(f2,x0,tol,Nmax)
    print('the approximate fixed point is:',xstar)
    print('f2(xstar):',f2(xstar))
    print('the list of approximations is:', pHat)
    print('the number of values in this list is:', count)
    print('Error message reads:',ier)

def fixedpt(f,x0,tol,Nmax):

# Input: 
#   x0 - initial guess
#   Nmax - max number of iterations
#   tol - stopping tolerance
# Return:
#   xstar - estimated fixed point
#   ier - error (0 = success, 1 = failure)
#   pHat - vector of the approximations of the fixed points
#   count - number of values in pHat

    pHat = np.zeros((Nmax, 1))

    count = 0
    while (count < Nmax):
        x1 = f(x0)
        if (abs(x1-x0) < tol):
            xstar = x1
            ier = 0
            return [xstar, ier, pHat, count]
        pHat[count] = x1
        x0 = x1
        count = count + 1

    xstar = x1
    ier = 1
    return [xstar, ier, pHat, count]

driver()
