import math 
import numpy as np 
import matplotlib.pyplot as plt

def driver():

    maclaurin = lambda x: x-((x**3)/6)+((x**5)/120)
    p1 = lambda x: (x-(7*(x**3)/60))/(1+((x**2)/20))
    p2 = lambda x: (x/(1+((x**2)/6))+(7*(x**4)/360))
    p3 = lambda x: (x-(7*(x**3)/60))/(1+((x**2)/20))

    f = lambda x: np.sin(x)
    x = np.linspace(0,5,100)

    maclaurin_error = abs(f(x)-maclaurin(x))
    p1_error = abs(f(x)-p1(x))
    p2_error = abs(f(x)-p2(x))
    p3_error = abs(f(x)-p3(x))

    plt.figure()
    plt.plot(x,maclaurin_error,label='6th Order Maclaurin')
    plt.plot(x,p1_error,label='Pade (3/3)')
    plt.plot(x,p2_error,label='Pade (2/4)')
    plt.plot(x,p3_error,label='Pade (4/2)')
    plt.title('Error of Approximations')
    plt.legend()
    plt.show()


driver()

