import numpy as np


def intGamma(x):
    '''
    integer gamma function
    '''
    if x == 1:
        return np.math.factorial(1)
    else:
        return np.math.factorial(x-1)

def beta(a, b, mu):
    '''
    beta distrubution
    '''
    normalization = intGamma(a+b)/(intGamma(a)*intGamma(b))
    conjugate_term = mu**(a-1) * (1-mu)**(b-1)
    return normalization * conjugate_term

def bino_likelihood(m, N, mu):
    norm = np.math.factorial(N)/(np.math.factorial(N-m)*np.math.factorial(m))
    body = mu**m * (1-mu)**(N-m)
    return norm * body
