import numpy as np
import math
import matplotlib.pyplot as plt

## Part (a)
t = np.linspace(0, math.pi, 31)
y = np.cos(t)
N = 31

sum = 0
for i in range(1,N+1):
    sum = sum + t[i]*y[i]

print('the sum is:', sum)


## Part(b)
R = 1.2
deltar = 0.1
f = 15
p = 0

x = lambda t: R*(1 + deltar*np.sin(f*t + p))*np.cos(t)
y = lambda t: R*(1 + deltar*np.sin(f*t + p))*np.sin(t)

theta = np.linspace(0, 2*math.pi, 1000)

plt.plot(theta, x(theta))
plt.plot(theta, y(theta))
plt.xlabel('theta')
plt.legend()
