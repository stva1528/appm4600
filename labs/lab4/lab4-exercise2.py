import numpy as np

def driver():
    g = lambda x: (10/(x+4))**(1/2)
    # fixed point is p = 1.3652300134141...

    Nmax = 100
    tol = 1e-9
    x0 = 1.5
    
    [xstar,ier,pHat,count] = fixedpt(g,x0,tol,Nmax)


    error = np.zeros((count,1))
    for i in range(count):
        error[i] = abs(pHat[i]-xstar)
        
    print('the difference in between errors is:', error)

    # the rate of convergence is linear
            

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
