import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import math
from numpy.linalg import inv 

def driver():

    f = lambda x: 1/(1+(10*x)**2)

    N = 10
    ''' interval '''
    a = -1
    b = 1
   
    ''' create interpolation nodes '''
    xint = np.zeros(N+1)
    for j in range(1, N+1):
        val = ((2*j-1)*math.pi)/(2*N)
        xint[j] = math.cos(val)
    
    ''' create interpolation data '''
    yint = f(xint)
    
    ''' create points for evaluating the Lagrange interpolating polynomial '''
    Neval = 1000
    xeval = np.linspace(a,b,Neval+1)
    yeval_l= np.zeros(Neval+1)
    yeval_dd = np.zeros(Neval+1)
  
    ''' initialize and populate the first columns of the 
     divided difference matrix, we will pass the x vector '''
    y = np.zeros( (N+1, N+1) )
     
    for j in range(N+1):
       y[j][0]  = yint[j]

    y = dividedDiffTable(xint, y, N+1)
    
    ''' evaluate lagrange poly '''
    for kk in range(Neval+1):
       yeval_l[kk] = eval_lagrange(xeval[kk],xint,yint,N)
       yeval_dd[kk] = evalDDpoly(xeval[kk],xint,y,N)


    ''' find coeff for polynomial '''
    acoeff = vandermonde(xint, yint, N)
    
    ''' evaluate polynomial '''
    yeval_m = np.zeros(Neval+1)
    for i in range(Neval+1):
        yeval_m[i] = eval_vander(xeval[i], acoeff, N)

    ''' create vector with exact values'''
    fex = f(xeval)
       

    plt.figure()    
    plt.plot(xeval,fex,'ro-',label='Original Function')
    plt.plot(xeval,yeval_l,'bs--',label='Lagrange') 
    plt.plot(xeval,yeval_dd,'c.--',label='Newton DD')
    plt.plot(xeval,yeval_m,'m.--',label='Monomial Expansion')
    plt.legend()
    plt.savefig('plot-newnodes.pdf')

    plt.figure() 
    err_l = abs(yeval_l-fex)
    err_dd = abs(yeval_dd-fex)
    err_m = abs(yeval_m-fex)
    plt.semilogy(xeval,err_l,'ro--',label='Lagrange')
    plt.semilogy(xeval,err_dd,'bs--',label='Newton DD')
    plt.semilogy(xeval,err_m,'m.--', label='Monomial Expansion')
    plt.legend()
    plt.savefig('errorplot-newnodes.pdf')

''' lagrange polynomial '''
def eval_lagrange(xeval,xint,yint,N):

    lj = np.ones(N+1)
    
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
              lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    yeval = 0.
    
    for jj in range(N+1):
       yeval = yeval + yint[jj]*lj[jj]
  
    return(yeval)

''' monomial expansion '''
def vandermonde(xint, yint, n):

    vander = np.zeros((n+1,n+1))
    for i in range(n+1):
        for j in range(n+1):
            vander[i][j] = xint[i]**j

    inv_vander = inv(vander)
    acoeff = np.dot(inv_vander, yint)
    return acoeff

def eval_vander(x, a, N):
    sum = 0
    for i in range(N+1):
        sum += a[i]*x**i
    return sum 

''' create divided difference matrix '''
def dividedDiffTable(x, y, n):
 
    for i in range(1, n):
        for j in range(n - i):
            y[j][i] = ((y[j][i - 1] - y[j + 1][i - 1]) /
                                     (x[j] - x[i + j]));
    return y;
    
def evalDDpoly(xval, xint,y,N):
    ''' evaluate the polynomial terms '''
    ptmp = np.zeros(N+1)
    
    ptmp[0] = 1.
    for j in range(N):
      ptmp[j+1] = ptmp[j]*(xval-xint[j])
     
    ''' evaluate the divided difference polynomial '''
    yeval = 0.
    for j in range(N+1):
       yeval = yeval + y[0][j]*ptmp[j]  

    return yeval

    
driver()        
