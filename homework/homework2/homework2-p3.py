import math

def driver():
    x = 9.999999995000000 * 10**(-10)
    soln = func(x)
    print('the solution that the algorithm yields is:', soln)

    poly = lambda x: x + (x**2)/2 + (x**3)/6 + (x**4)/24
    print('the solution that the polynomial yields is:', poly(x))
    
    return

def func(x):
    y = math.e**x
    return y - 1

driver()
