import numpy as np

def driver():
    # Exercise 3
# I am getting an overflow error when I try to run this ):
    
    fixed_point = 7**(1/5)
    x0 = 1
    tol = 10e-10
    Nmax = 100
    
    # part (a)
    print('*** Part (a) ***')
    f = lambda x: x*(1+((7-x**5)/x**2))**3
    # apply fixed point iteration
    [xstar,ier] = fixedpt(f, x0, tol, Nmax)
    print('Error message reads:',ier)
    print('the approximate fixed point is:',xstar)
    print('f(xstar):',f1(xstar))

    # part (b)
    print('*** Part (b) ***')
    f = lambda x: x - ((x**5 - 7)/(x**2))
    # apply fixed point iteration
    [xstar,ier] = fixedpt(f, x0, tol, Nmax)
    print('Error message reads:',ier)
    print('the approximate fixed point is:',xstar)
    print('f(xstar):',f1(xstar))

    # part (c)
    print('*** Part (c) ***')
    f = lambda x: x - (x**5 - 7)/(5*x**4)
    # apply fixed point iteration
    [xstar,ier] = fixedpt(f, x0, tol, Nmax)
    print('Error message reads:',ier)
    print('the approximate fixed point is:',xstar)
    print('f(xstar):',f1(xstar))

    # part (d)
    print('*** Part (d) ***')
    f = lambda x: x - (x**5 - 7)/(12)
    # apply fixed point iteration
    [xstar,ier] = fixedpt(f, x0, tol, Nmax)
    print('Error message reads:',ier)
    print('the approximate fixed point is:',xstar)
    print('f(xstar):',f1(xstar))


def fixedpt(f,x0,tol,Nmax):

# Input: 
#   x0 - initial guess
#   Nmax - max number of iterations
#   tol - stopping tolerance
# Return:
#   xstar - estimated fixed point
#   ier - error (0 = success, 1 = failure)

    count = 0
    while (count < Nmax):
       count = count + 1
       x1 = f(x0)
       if (abs(x1-x0) < tol):
          xstar = x1
          ier = 0
          return [xstar,ier]
       x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier]

driver()
