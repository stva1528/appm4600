import numpy as np
import math

def driver():
        
	# Exercise 2
	tol = 1e-5
	
	# part (a)
	f = lambda x: (x-1)*(x-3)*(x-5)
	a = 0
	b = 2.4
	[astar, ier] = bisection(f, a, b, tol)
	print('for part (a) the error message reads:', ier)
	print('the approximate root is:', astar)
	print('f(astar) =', f(astar))
	# Code was successful and reached desired accuracy

	# part (b)
	f = lambda x: (x-1)**2*(x-3)
	a = 0
	b = 2
	[astar, ier] = bisection(f, a, b, tol)
	print('for part (b) the error message reads:', ier)
	print('the approximate root is:', astar)
	print('f(astar) =', f(astar))
	# Code was unsuccessful because a and b are both negative

        # part (c pt1)
	f = lambda x: math.sin(x)
	a = 0
	b = 0.1
	[astar, ier] = bisection(f, a, b, tol)
	print('for part (c) part 1 the error message reads:', ier)
	print('the approximate root is:', astar)
	print('f(astar) =', f(astar))
	# Code was successful and reached desired accuracy
	
	# part (c pt2)
	a = 0.5
	b = (3*math.pi)/4
	[astar, ier] = bisection(f, a, b, tol)
	print('for part (c) part 2 the error message reads:', ier)
	print('the approximate root is:', astar)
	print('f(astar) =', f(astar))
	# Code was successful in failing because there was no root

	# Overall the code behaved as expected with desired accuracy



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
