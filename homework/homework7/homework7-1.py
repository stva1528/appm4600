import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from numpy.linalg import inv

def driver():
    # given data
    f = lambda x: 1/(1+(10*x)**2)

    N = 18

    Neval = 1000

    xeval = np.linspace(-1,1,Neval+1)

    xint = np.zeros(N+1)
    h = 2/N
    for i in range(1,N+2):
        xint[i-1] = -1+h*(i-1)

    yint = f(xint)

    # evaluate coefficients
    c = vandermonde(xint, yint, N)

    # solve for polynomial 
    yeval = np.zeros(Neval+1)
    for j in range(Neval+1):
        yeval[j] = eval_poly(xeval[j], c, N)

    # plot
    fx = f(xeval)
    
    plt.figure()
    plt.plot(xint, f(xint), 'o')
    plt.plot(xeval, yeval, label = "polynomial")
    plt.plot(xeval, fx, label = "orginal function")
    plt.legend()
    plt.savefig('monomial-18nodes.pdf')
        
    

def vandermonde(xint, yint, N):
    '''
    vandermonde: Uses the vandermonde matrix to solve for coefficients
                    of the approximate polynomial 
    Input:
        xint - interpolation nodes
        yint - function evaluated at interpolation nodes
        N - number of interpolation nodes
    Output:
        c - vector of coeffecients 
    '''
    vander = np.zeros((N+1,N+1))
    for i in range(N+1):
        for j in range(N+1):
            vander[i][j] = xint[i]**j

    inv_vander = inv(vander)
    c = inv_vander.dot(yint)
    return c

def eval_poly(x, c, N):
    '''
    eval_poly: evaluates the polynomial at a specific value of x
    Input:
        x - value of x to evaluate the polynomial 
        c - coefficients 
        N - length of coefficient vector
    Output:
        sum - value of the polynomial at x
    '''
    sum = 0
    for i in range(N+1):
        sum += c[i]*x**i
    return sum 

driver()
