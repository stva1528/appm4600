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

    yeval = eval_cubic_spline(xeval,Neval,a,b,f,Nint)

    plt.figure()
    plt.plot(xeval, f(xeval), label="Original Function")
    plt.plot(xeval, yeval, label="Splines Approximate")
    plt.legend()
    plt.savefig('splines-approx-cubic.pdf')
    


def Mcoeff(x_int,f_xint):
    h = x_int[1]-x_int[0]
    n = len(f_xint)-1
    c = np.full(shape=n,fill_value=1/3)
    m = np.diag(c)
    for i in range(n-1):
        for j in range(n-1):
            if (abs(i-j)==1):
                m[i][j]=1/12

    y = np.zeros(n)
    for i in range(n-1):
        y[i] = (f_xint[i+2]-2*f_xint[i+1]+f_xint[i])/(2*h**2)

    m_inv = inv(m)

    M = m_inv.dot(y)
    M = np.append(0,M)
    M = np.append(M,0)
    return M
    
def eval_cubic(M1,M2,x1,x2,fx1,fx2,x):
    h = x2-x1
    C = (fx1/h)-M1*(h/6)
    D = (fx2/h)-M2*(h/6)

    s = (M1*(x2-x)**3)/(6*h) + (M2*(x-x1)**3)/(6*h) + C*(x2-x) + D*(x-x1)
    return s
    
    

def  eval_cubic_spline(xeval,Neval,a,b,f,Nint):
    '''create the intervals for piecewise approximations'''
    xint = np.linspace(a,b,Nint+1)
   
    '''create vector to store the evaluation of the linear splines'''
    yeval = np.zeros(Neval)

    '''find M coefficents'''
    f_xint = f(xint)
    M = Mcoeff(xint,f_xint)
    
    for jint in range(Nint):
        '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
        '''let ind denote the indices in the intervals'''
        '''let n denote the length of ind'''
        
        ind = np.where((xeval>=xint[jint]) & (xeval<=xint[jint+1]))
        n = len(ind)
        
        '''temporarily store your info for creating cubic in the interval of 
         interest'''
        a1 = xint[jint]
        fa1 = f(a1)
        b1 = xint[jint+1]
        fb1 = f(b1)
        M1 = M[jint]
        M2 = M[jint+1]
        
        
        for kk in range(n):
            yeval[ind[kk]] = eval_cubic(M1,M2,a1,b1,fa1,fb1,xeval[ind[kk]])

    return yeval
            
           
driver()
