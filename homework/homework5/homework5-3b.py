import numpy as np
import math

# initial values
x = 1
y = 1
z = 1

Nmax = 10

for i in range(Nmax):
    # calculate needed values
    f = x**2 + 4*y**2 + 4*z**2 - 16
    fx = 2*x
    fy = 8*y
    fz = 8*z

    # use iteration scheme to update x
    denom = (fx**2 + fy**2 + fz**2)
    x = x - (f*fx)/denom
    y = y - (f*fy)/denom
    z = z - (f*fz)/denom

    # show iterates
    print('*** iteration', i, '***')
    print('x:', x)
    print('y:', y)
    print('z:', z)
    
    
    
