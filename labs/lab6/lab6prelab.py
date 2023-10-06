import numpy as np
import math
import matplotlib.pyplot as plt

f = lambda x: np.cos(x)
h = 0.01*2.0**(-np.arange(1,10))
x = math.pi/2

dapprox1 = lambda x: (f(x+h)-f(x))/h
dapprox2 = lambda x: (f(x+h)-f(x-h))/(2*h)

# exercise 1
estimate1 = dapprox1(x)
estimate2 = dapprox2(x)
print('the estimates from the first function are:', estimate1)
print('the estimates from the second function are', estimate2)
