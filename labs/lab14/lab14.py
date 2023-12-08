import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy.linalg as scila
import time


def driver():

     ''' create  matrix for testing different ways of solving a square 
     linear system'''

     '''' N = size of system'''
     N = 5000
 
     ''' Right hand side'''
     b = np.random.rand(N,1)
     A = np.random.rand(N,N)

     t = time.time()
     x1 = scila.solve(A,b)
     t1 = time.time()-t

     t = time.time()
     lu, p = scila.lu_factor(A)
     t_factor = time.time()-t

     t = time.time()
     x = scila.lu_solve((lu,p),b)
     t_solve = time.time()-t
     
     total_time = t_factor + t_solve

     print('time 1:', t1)
     #print('sol:', x1)
     print('time 2:', total_time)
     print('factor time:', t_factor)
     print('solve time:', t_solve)
     #print('sol:', x2)
     
     
     #test = np.matmul(A,x)
     #r = la.norm(test-b)
     #print(r)

     ''' Create an ill-conditioned rectangular matrix '''
     N = 10
     M = 5
     A = create_rect(N,M)     
     b = np.random.rand(N,1)


def create_rect(N,M):
     ''' this subroutine creates an ill-conditioned rectangular matrix'''
     a = np.linspace(1,10,M)
     d = 10**(-a)
     
     D2 = np.zeros((N,M))
     for j in range(0,M):
        D2[j,j] = d[j]
     
     '''' create matrices needed to manufacture the low rank matrix'''
     A = np.random.rand(N,N)
     Q1, R = la.qr(A)
     test = np.matmul(Q1,R)
     A =    np.random.rand(M,M)
     Q2,R = la.qr(A)
     test = np.matmul(Q2,R)
     
     B = np.matmul(Q1,D2)
     B = np.matmul(B,Q2)
     return B     
          
  
driver()       
