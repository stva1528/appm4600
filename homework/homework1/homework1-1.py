import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import math

def driver():
	x = np.arange(1.920, 2.080, 0.001)

	p1 = lambda x: x**9 - 18*x**8 + 144*x**7 - 672*x**6 + 2016*x**5 - 4032*x**4 + 5376*x**3 - 4608*x**2 + 2304*x - 512

	p2 = lambda x: (x-2)**9

	y = p1(x)
	g = p2(x)

	plt.plot(x, y, color='turquoise', label='expended form')
	plt.plot(x, g, color='hotpink', label='nonexpanded form')
	plt.xlabel('x')
	plt.ylabel('p(x)')
	plt.title('x versus p(x)')
	plt.legend()
	plt.savefig('Downloads/appm4600/Problem1')

	return
driver()

