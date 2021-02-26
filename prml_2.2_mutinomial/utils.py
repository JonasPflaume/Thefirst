import numpy as np
from scipy.special import gamma

def muti_likelihood(Data, mu):
    '''
    likelihood function of mutinomial distrubution
    Data is a Nx2 matrix
    len(mu) = 2
    '''
    # for k=1
    m1 = np.sum(Data[:,0])
    first = mu[0] ** m1
    m2 = np.sum(Data[:,1])
    second = mu[1] ** m2
    return first * second

def muti_distribution(m, mu, N):
    '''
    a two dimensional muti distribution: len(m),len(mu) = 2
    convinient for visualization
    '''
    assert m[0] + m[1] == N
    assert np.abs(mu[0] + mu[1] -1) <= 0.05

    normalization_term = np.math.factorial(N) / (np.math.factorial(m[0]) * np.math.factorial(m[1]))
    body = mu[0]**m[0] * mu[1]**m[1] 
    
    return normalization_term * body



def dir_distribution(alpha, mu):
    '''
    len(mu) = len(alpha) = 2
    '''
    if np.abs(mu[0] + mu[1]) < 1.05 and np.abs(mu[0] + mu[1]) > 0.95:
        alpha_0 = alpha[0] + alpha[1]
        normalization_term = Gamma(alpha_0) / (Gamma(alpha[0]) * Gamma(alpha[1]))
        body = mu[0]**(alpha[0]-1) * mu[1]**(alpha[1]-1)
        return normalization_term * body
    else:
        return 0
    
def Gamma(x):
    return gamma(x)