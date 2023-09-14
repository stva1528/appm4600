import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import math

d = np.logspace(-15, 1, num=17, dtype=float)

f = lambda d: math.cos(math.pi + d) - math.cos(math.pi)

y = f(d)

plt.plot(d, y)
plt.xlabel('delt')
plt.title('x = pi')
plt.show()