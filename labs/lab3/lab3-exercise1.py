import numpy as np

def driver():

	# Exercise 1
	f = lambda x: x**2 * (x-1)
	tol = 1e-5

	# part (a)
	a = 0.5
	b = 2
	[astar, ier] = bisection(f, a, b, tol)
	print('for part (a) the error message reads:', ier)
	print('the approximate root is:', astar)
	print('f(astar) =', f(astar))
	# This choice of interval was successful for approximating the root

	# part (b)
	a = -1
	b = 0.5
	[astar, ier] = bisection(f, a, b, tol)
	print('for part (b) the error message reads:', ier)
	print('the approximate root is:', astar)
	print('f(astar) =', f(astar))
	# This choice of interval was not successful because the start
	# and endpoint are both negative

	# part (c)
	a = -1
	b = 2
	[astar, ier] = bisection(f, a, b, tol)
	print('for part (c) the error message reads:', ier)
	print('the approximate root is:', astar)
	print('f(astar) =', f(astar))
	# This choice of interval was again successful, yet it missed the root
	# at x = 0, and returned the root at x = 1

	# For this function, bisection will not be able to find the root
	# at x = 0 because the function is negative in the proximity of
	# x = 0

	return 


def bisection(f, a, b, tol):
# Inputs: 
#	f - function 
#	a,b - start and end points of function 
#	tol - function terminates when interval length < tol
# Returns:
#	astar - approximation of root
#	ier - error (1 = failure, 0 = success)

        fa = f(a)
        fb = f(b)
#       verify that there is a root to find in the interval
        if (fa*fb > 0):
                ier = 1
                astar = a
                return [astar, ier]

#       verify end points are not a root
        if (fa == 0):
                astar = a
                ier = 0
                return [astar, ier]

        if (fb == 0):
                astar = b
                ier = 0
                return [astar, ier]
        

        count = 0
#       initialize midpoint
        d = 0.5*(a+b)
        while (abs(d-a)> tol):
                fd = f(d)
                if (fd == 0):
                        astar = d
                        ier = 0
                        return [astar, ier]
                if (fa*fd < 0):
                        b = d
                else:
                        a = d
                        fa = fd
#               update midpoint
                d = 0.5*(a+b)
                count = count +1
      
        astar = d
        ier = 0
        return [astar, ier]

driver()
