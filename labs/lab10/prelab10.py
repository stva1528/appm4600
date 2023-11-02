import matplotlib.pyplot as plt
import numpy as np
import math

def eval_legendre(n, x):
    p = np.zeros(n+1)
    p[0] = 1
    p[1] = x

    for i in range(1,n+1):
        p[i+1] = (1/(i+1))*(((2*i+1)*x*p[i])-i*p[i-1])

    return p
    
