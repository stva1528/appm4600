import numpy as np
import numpy.linalg as la
import math

# Exercises 4.2
# exercise 1
def driver():
        # create two orthogonal vectors (x and y)
	n = 3
	x = np.linspace(0, 0, n)
	x[0] = 1
	x[2] = -1

	y = np.linspace(0, 0, n)
	y[0] = 1
	y[1] = 2**(1/2)
	y[2] = 1

        # find dot product of x and y
	dp = dotProduct(x,y,n)

	print('the dot product is : ', dp)
	return

def dotProduct(x,y,n):
	dp = 0.
	for j in range(n):
		dp = dp + x[j]*y[j]
	return dp

driver()

# exercise 2
def driver():
        # test matrixVectorMult with random matrix
	m = np.array([[4, 2, 3], [1, 6, 2], [4, 4, 4], [2, 8, 1]])
	v = np.array((2, 1, 3))
	
	solution = matrixVectorMult(m, v, len(v))
	print('the solution is : ', solution)
	return 

# matrix vector multiplication function 
def matrixVectorMult(m, v, n):
	sol = np.zeros(len(m))

	for i in range(len(m)):
		sol[i] = dotProduct(m[i], v, n) # uses provided dot product code

	return sol

driver()

# exercise 3
def driver():
        # compare my code to numpy commands
	n = 100
	x = np.linspace(0,np.pi,n)

	f = lambda x: x**2 + 4*x + 2*np.exp(x)
	g = lambda x: 6*x**3 + 2*np.sin(x)
	y = f(x)
	w = g(x)

	# compares dot product
	print('Dot product using Numpy : ', np.dot(y, w))
	print('Dot product using my code : ', dotProduct(y, w, n))
	
	n = 3
	m = np.array([[6, 8, 7], [4, 2, 5], [8, 4, 4], [1, 5, 9]])
	v = np.array((4, 7, 2))

        # compares matrix vector multiplication
	print('Matrix multiplication using Numpy : ', np.matmul(m, v))
	print('Matrix multiplication using my code : ', matrixVectorMult(m, v, n))

driver()


