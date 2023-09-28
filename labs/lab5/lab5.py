import numpy as np

def driver():
    """
    Trying to implement newtons method with bisection
    to guarentee the basin of convergence is met

    f = lambda x: (x-2)(x-5)e^x
    fp = lambda x: e^x(x^2 - 5x + 3)
    a = 2
    b = 4
    
    """

    f = lambda x: (x-2)*(x-5)*np.exp(x)
    fp = lambda x: (x-2)*(x-5)*np.exp(x)+(2*x-7)*np.exp(x) 

    Nmax = 100
    tol = 1.e-14
    a = 1
    b = 4

    (pstar, it, ier) = bisection_newton(f, fp, a, b, tol)
    print('the approximate root is', '%16.16e' % pstar)
    print('number of iterations:', '%d' % it)
    print('the error message reads:', '%d' % ier)


def bisection_newton(f, fp, a, b, tol):
    """
    Bisection with Newton's Method

    Input:
        f - function
        fp - function derivative
        a,b - start and endpoint
        tol - function terminates when interval length < tol
    Output:
        astar - approximation of root (within Newton's basin)
        count - number of iterations
        ier - error (1 = failure, 0 = success)
    """
    
    fa = f(a)
    fb = f(b)
    count = 0
    
    # initial chcecks 
    if (fa*fb > 0):
        ier = 1
        astar = a
        return [astar, count, ier]
    if (fa == 0):
        astar = a
        ier = 0
        return [astar, count, ier]
    if (fb == 0):
        astar = b
        ier = 0
        return [astar, count, ier]
    
    c = 0.5*(a+b)
    while (abs(c-a)> tol):
        
        # newton check
        fc = f(c)
        fpc = fp(c)
        n = c - (fc/fpc)
        if (n>a and n<b):
            (pstar, it, ier) = newton(f, fp, c, tol, 100)
            return [pstar, it, ier]
        else:
            # bisection continues
            if (fc == 0):
                astar = c
                ier = 0
                return [astar, count, ier]
            if (fa*fc < 0):
                b = c
            else:
                a = c
                fa = fc
                c = 0.5*(a+b)
            count = count +1

    astar = c
    ier = 0
    return [astar, count, ier]

def newton(f,fp,p0,tol,Nmax):
  """
  Newton iteration.
  
  Inputs:
    f,fp - function and derivative
    p0   - initial guess for root
    tol  - iteration stops when p_n,p_{n+1} are within tol
    Nmax - max number of iterations
  Returns:
    pstar - the last iterate
    it - number of iterations
    ier  - success message
          - 0 if we met tol
          - 1 if we hit Nmax iterations (fail)
     
  """
  p = np.zeros(Nmax+1);
  for it in range(Nmax):
      p1 = p0-f(p0)/fp(p0)
      p[it+1] = p1
      if (abs(p1-p0) < tol):
          pstar = p1
          ier = 0
          return [pstar,it,ier]
      p0 = p1
  pstar = p1
  ier = 1
  return [pstar,it,ier]

driver()
