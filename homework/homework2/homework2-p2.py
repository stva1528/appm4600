import math
def driver():
    A = [[1/2, 1/2],
         [(1/2)*(1+10**(-10)), (1/2)*(1-10**(-10))]]
    Ainv = [[1-10**10, 10**10],
            [1+10**10, -10**10]]

    Anorm = norm(A)
    Ainvnorm = norm(Ainv)
    print('The 2 norm of A is:', Anorm)
    print('The 2 norm of A inverse is:', Ainvnorm)
        
    return

def norm(A):
    f = 0

    for i in range(0, len(A)):
        for j in range(0, len(A)):
            f = f + abs(A[i][j])**2

    return math.sqrt(f)

driver()
