import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv 
from numpy.linalg import norm

def driver():
    f = lambda x: np.sin(10*x)
    fp = lambda x: 10*np.cos(10*x)

    N = 5
    a = 0
    b = 2*np.pi

    xint = np.linspace(a,b,N+1)

    yint = np.zeros(N+1)
    ypint = np.zeros(N+1)
    for i in range(N+1):
        yint[i] = f(xint[i])
        ypint[i] = fp(xint[i])

    Neval = 1000
    xeval = np.linspace(a,b,Neval+1)
            
    (M,C,D) = create_natural_spline(yint,xint,N)
    yeval_NC = eval_cubic_spline(xeval,Neval,xint,N,M,C,D)

    (M2,C2,D2) = create_clamped_spline(yint,ypint,xint,N)
    yeval_CC = eval_cubic_spline(xeval,Neval,xint,N,M2,C2,D2)

    fex = f(xeval)

    plt.figure()
    plt.plot(xeval,fex,label='Original Function')
    plt.plot(xeval,yeval_NC,label='Natural Cubic Spline')
    plt.plot(xeval,yeval_CC,label='Clamped Cubic Spline')
    plt.plot(xint,f(xint),'ro')
    plt.legend()
    plt.savefig('periodic.pdf')

    errNC = abs(yeval_NC-fex)
    errCC = abs(yeval_CC-fex)
    plt.figure()
    plt.semilogy(xeval,errNC,label='Natural Cubic Spline Error')
    plt.semilogy(xeval,errCC,label='Clamped Cubic Spline Error')
    plt.legend()
    plt.savefig('periodicError.pdf')


def create_clamped_spline(yint,ypint,xint,N):
    # create the right hand side for the linear system
    b = np.zeros(N+1)
    
    # vector values
    h = np.zeros(N+1)
    for i in range(1,N):
        hi = xint[i]-xint[i-1]
        hip = xint[i+1] - xint[i]
        b[i] = (yint[i+1]-yint[i])/hip - (yint[i]-yint[i-1])/hi
        h[i-1] = hi
        h[i] = hip
    b[0] = -ypint[0]+((yint[1]-yint[0])/h[0])
    b[N] = -ypint[N]+((yint[N]-yint[N-1])/h[N-1])

    # create matrix so you can solve for the M values
    # this is made by filling one row at a time 
    A = np.zeros((N+1,N+1))
    A[0][0] = h[0]/3
    A[0][1] = h[0]/6
    for j in range(1,N):
        A[j][j-1] = h[j-1]/6
        A[j][j] = (h[j]+h[j-1])/3 
        A[j][j+1] = h[j]/6
    A[N][N-1] = h[N-1]/6
    A[N][N] = h[N-1]/3

    Ainv = inv(A)
    M = Ainv.dot(b)

    # create the linear coefficients
    C = np.zeros(N)
    D = np.zeros(N)
    for j in range(N):
        C[j] = yint[j]/h[j]-h[j]*M[j]/6
        D[j] = yint[j+1]/h[j]-h[j]*M[j+1]/6
        
    return(M,C,D)


def create_natural_spline(yint,xint,N):
    # create the right hand side for the linear system
    b = np.zeros(N+1)
    
    # vector values
    h = np.zeros(N+1)  
    for i in range(1,N):
        hi = xint[i]-xint[i-1]
        hip = xint[i+1] - xint[i]
        b[i] = (yint[i+1]-yint[i])/hip - (yint[i]-yint[i-1])/hi
        h[i-1] = hi
        h[i] = hip

    # create matrix so you can solve for the M values
    # this is made by filling one row at a time 
    A = np.zeros((N+1,N+1))
    A[0][0] = 1.0
    for j in range(1,N):
        A[j][j-1] = h[j-1]/6
        A[j][j] = (h[j]+h[j-1])/3 
        A[j][j+1] = h[j]/6
    A[N][N] = 1

    Ainv = inv(A)
    M = Ainv.dot(b)

    # create the linear coefficients
    C = np.zeros(N)
    D = np.zeros(N)
    for j in range(N):
        C[j] = yint[j]/h[j]-h[j]*M[j]/6
        D[j] = yint[j+1]/h[j]-h[j]*M[j+1]/6
        
    return(M,C,D)

       
def eval_local_spline(xeval,xi,xip,Mi,Mip,C,D):
# evaluates the local spline as defined in class
    # xip = x_{i+1}; xi = x_i
    # Mip = M_{i+1}; Mi = M_i

    hi = xip-xi
    yeval = (Mi*(xip-xeval)**3 +(xeval-xi)**3*Mip)/(6*hi) \
            + C*(xip-xeval) + D*(xeval-xi)
    return yeval 

    
def  eval_cubic_spline(xeval,Neval,xint,Nint,M,C,D):
    yeval = np.zeros(Neval+1)

    # find indices of xeval in interval (xint(jint),xint(jint+1))
    for j in range(Nint):
        atmp = xint[j]
        btmp= xint[j+1]
        
        # find indices of values of xeval in the interval
        ind= np.where((xeval >= atmp) & (xeval <= btmp))
        xloc = xeval[ind]

        # evaluate the spline
        yloc = eval_local_spline(xloc,atmp,btmp,M[j],M[j+1],C[j],D[j])
        
        # copy into yeval
        yeval[ind] = yloc

    return(yeval)

    

driver()
