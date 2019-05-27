from mpmath import *
import numpy as np
from names import Names

codes = Names.coins

def f(y, mu, sig):
    return 1/(2*np.pi*sig)*mp.exp(-(y-mu)**2/(2*sig**2))

# To increase accuracy of calculation mpfmath libriary will be used
# mpfmath exclude nan values of MLF in case of abnormal returns in exponent

# Create function to estimate EM-algorithm

def EM(data):
    #Initiate parameters for Maximum Likelihood
    l_new, l_old = -10000, -100000
    epsilon = 0.5
    T = len(data)

    #Initial parameters
    mu_0, mu_1 = 1, 1.5
    sig_0, sig_1 = 0.5, 0.9
    P = [[0.9, 0.1], [0.1, 0.9]]
    dzeta_0 = 0.5

    #Variables for ksi, ksi_s, dzeta and tr
    ksi = np.zeros(len(data))
    ksi_s = np.zeros(len(data))
    dzeta = np.zeros(len(data))
    tr = np.zeros((T,2,2))

    while abs(l_new-l_old) > epsilon and l_new > l_old:
        l_old = l_new

        #Calculate ksi and dzeta for dzeta_0
        den = f(data[0], mu_0, sig_0) * dzeta_0 + f(data[0], mu_1, sig_1) * (1-dzeta_0)
        ksi[0] = f(data[0], mu_0, sig_0) * dzeta_0 / den
        dzeta[0] = P[0][0]*ksi[0] + P[0][1]*(1-ksi[0])
        for t in range(1,T):
            den = f(data[t], mu_0, sig_0)*dzeta[t-1] + f(data[t], mu_1, sig_1)*(1-dzeta[t-1])
            ksi[t] = f(data[t], mu_0, sig_0)*dzeta[t-1] / den
            dzeta[t] = P[0][0]*ksi[t] + P[0][1]*(1-ksi[t])

        #Calculate smoothed prob and prob of transitions
        ksi_s[-1] = ksi[-1]
        tr[-1][0][0], tr[-1][0][1] = ksi[-1]*P[0][0], (1-ksi[-1])*P[0][1]
        tr[-1][1][0], tr[-1][1][1] = ksi[-1]*P[1][0], (1-ksi[-1])*P[1][1]
        for t in reversed(range(0,T-1)):
            ksi_s[t] = ksi[t]*(P[0][0]*ksi_s[t+1]/dzeta[t] + P[1][0]*(1-ksi_s[t+1])/(1-dzeta[t]))
            tr[t][0][0] = ksi[t]*ksi_s[t+1]/dzeta[t]*P[0][0]
            tr[t][0][1] = (1-ksi[t])*ksi_s[t+1]/dzeta[t]*P[0][1]
            tr[t][1][0] = ksi[t]*(1-ksi_s[t+1])/(1-dzeta[t])*P[1][0]
            tr[t][1][1] = (1-ksi[t])*(1-ksi_s[t+1])/(1-dzeta[t])*P[1][1]

        #Calculate MLF
        l_new = ksi_s[0]*mp.log(dzeta[0]) + (1-ksi_s[0])*mp.log(1-dzeta[0]) +\
                ksi_s[0]*mp.log(f(data[0],mu_0,sig_0)) + (1-ksi_s[0])*mp.log(f(data[0],mu_1,sig_1))
        for t in range(1,T):
            l_new += ksi_s[t]*mp.log(f(data[t],mu_0,sig_0)) + (1-ksi_s[t])*mp.log(f(data[t],mu_1,sig_1)) +\
                     tr[t][0][0]*mp.log(P[0][0]) + tr[t][0][1]*mp.log(1-P[1][1])+\
                     tr[t][1][0]*mp.log(1-P[0][0]) +tr[t][1][1]*mp.log(P[1][1])

        #Calculate new parameters
        mu_0 = sum([x*y for x,y in zip(ksi_s,data)]) / sum(ksi_s)
        mu_1 = sum([x*y for x,y in zip([1-x for x in ksi_s], data)]) / sum([1-x for x in ksi_s])
        sig_0 = np.sqrt(sum(x*y**2 for x,y in zip(ksi_s, [r-mu_0 for r in data]))/sum(ksi_s))
        sig_1 = np.sqrt(sum(x*y**2 for x,y in zip([1-x for x in ksi_s], [r-mu_1 for r in data]))/\
                        sum([1-x for x in ksi_s]))
        P[0][0] = sum([x[0][0] for x in tr[1:]]) / sum(ksi_s[1:])
        P[0][1] = sum([x[0][1] for x in tr[1:]]) / sum(ksi_s[1:])
        P[1][0] = sum([x[1][0] for x in tr[1:]]) / sum([1-x for x in ksi_s[1:]])
        P[1][1] = sum([x[1][1] for x in tr[1:]]) / sum([1-x for x in ksi_s[1:]])

        #Calibrating
        sum_col_1 = P[0][0] + P[1][0]
        sum_col_2 = P[0][1] + P[1][1]
        P[0][0], P[1][0] = P[0][0]/ sum_col_1, P[1][0]/ sum_col_1
        P[0][1], P[1][1] = P[0][1]/ sum_col_1, P[1][1]/ sum_col_2
        dzeta_0 = ksi_s[0]

        if isinstance(l_new, complex):
            return (['complex', mu_0, mu_1, sig_0, sig_1, l_new, P[0][0], P[1][1], dzeta])


    return ([ksi_s, mu_0, mu_1, sig_0, sig_1, l_new, P[0][0], P[1][1], dzeta])