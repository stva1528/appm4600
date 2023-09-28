import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import special

# given values 
Ti = 20
Ts = -15
alpha = 0.138*10**(-6)
t = 5.184*10**6

# function 
T = lambda x: (Ti-Ts)*special.erf(x/(2*(alpha*t)**(1/2)))+Ts
x = np.linspace(0,10,100)

# plotting
plt.plot(x, T(x))
plt.xlabel('distance underground (meters)')
plt.ylabel('temperature (degrees celcius)')
plt.show()
