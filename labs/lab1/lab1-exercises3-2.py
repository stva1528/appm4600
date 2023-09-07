import numpy as np
import matplotlib.pyplot as plt

# 3.2 Exercises
# exercise 1
# creates two numpy arrays of the same length using two different methods
x = np.linspace(0, 1, 5)
y = np.arange(1, 2, 0.2)

# exercise 2 and 3
print('the first three entries of x are', x[0:3])

# exercise 4
w = 10**(-np.linspace(1, 10, 10))
w

x = np.linspace(1, len(w), len(w)) #creates a vector of integers from 1 to length of w

# plot x versus w
plt.plot(x, w) 
plt.xlabel('x')
plt.ylabel('w')
plt.show()

# exercise 5
s = 3*w # vector equal to 3 times w

# plot x versus x and w
plt.plot(x, w) 
plt.plot(x, s)
plt.xlabel('x')
plt.ylabel('w')
plt.savefig('Downloads/appm4600/labs/lab1/Figure1') # saves image to file

exit()





