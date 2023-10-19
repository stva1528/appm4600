import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv

def driver():
    f = lambda x: 1/(1+(10*x)**2)
    a = -1
    b = 1

    Nint = 10
    Neval = 1000
    xeval = np.linspace(a, b, Neval)

    yeval = eval_lin_spline(xeval,Neval,a,b,f,Nint)

    plt.figure()
    plt.plot(xeval, f(xeval), label="Original Function")
    plt.plot(xeval, yeval, label="Splines Approximate")
    plt.legend()
    plt.savefig('splines-approx-linear.pdf')
    


def line_eval(x1,x2,fx1,fx2,xval):
    slope = (fx2-fx1)/(x2-x1)
    intercept = fx1-(slope*x1)

    y = slope*xval+intercept

    return y
    

def  eval_lin_spline(xeval,Neval,a,b,f,Nint):
    '''create the intervals for piecewise approximations'''
    xint = np.linspace(a,b,Nint+1)
   
    '''create vector to store the evaluation of the linear splines'''
    yeval = np.zeros(Neval) 
    
    for jint in range(Nint):
        '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
        '''let ind denote the indices in the intervals'''
        '''let n denote the length of ind'''
        
        ind = np.where((xeval>=xint[jint]) & (xeval<=xint[jint+1]))
        n = len(ind)
        
        '''temporarily store your info for creating a line in the interval of 
         interest'''
        a1 = xint[jint]
        fa1 = f(a1)
        b1 = xint[jint+1]
        fb1 = f(b1)
        
        for kk in range(n):
            '''
            use your line evaluator to evaluate the lines at each of the points
            in the interval

            yeval(ind(kk)) = call your line evaluator at xeval(ind(kk)) with
            the points (a1,fa1) and (b1,fb1)
            '''
            yeval[ind[kk]] = line_eval(a1,b1,fa1,fb1,xeval[ind[kk]])

    return yeval
            
           
driver()
           
