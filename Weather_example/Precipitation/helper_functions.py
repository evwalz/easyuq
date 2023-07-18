import numpy as np
from scipy.optimize import minimize_scalar
from scipy import stats
from statsmodels.distributions.empirical_distribution import ECDF
import properscoring as ps

def algo72_ensemble(fct_train, fct_test, y_train):
    n = len(fct_train)
    m = len(fct_test)
    sum_x = np.sum(fct_train)
    sum_x2 = np.sum(np.square(fct_train))
    XX = np.zeros((2, 2))
    XX[0, 0] = sum_x2
    XX[0, 1] = XX[1, 0] = -1 * sum_x
    XX[1, 1] = n
    XX = XX / (n * sum_x2 - (sum_x) ** 2)
    X_tr = np.ones((len(fct_train), 2))
    X_tr[:, 1] = fct_train
    X_tr_XX = X_tr @ XX
    H = X_tr_XX @ np.transpose(X_tr)
    C = np.zeros((m, n))
    g2 = XX[0, 0] + XX[1, 0] * fct_test + fct_test * (XX[0, 1] + fct_test * XX[1, 1])
    g = np.zeros(n + 1)
    Hbar = np.zeros((n, n))

    for j in range(m):
        g[0:n] = X_tr_XX[:, 0] + X_tr_XX[:, 1] * fct_test[j]
        g[n] = g2[j]
        Hbar = H - np.outer(g[0:n], g[0:n]) / (1 - g[n])
        g = g / (1 + g[n])
        gsrt = np.sqrt(1 - g[n])
        Hdiagsqrt = np.sqrt(1 - np.diag(Hbar)[0:n])
        B = gsrt + g[0:n] / Hdiagsqrt
        A = np.sum(g[0:n] * y_train) / gsrt + (y_train - np.sum(Hbar[0:n, 0:n].T * y_train, axis=1)) / Hdiagsqrt
        C[j, :] = A / B

    return C

def log_norm_optim(y, mean):
    y_scale = y - mean
    bb = np.max(y)
    def opt_log_sigma(sigma):
        return -1*np.mean(np.log(stats.norm.pdf(y_scale, scale = sigma)))
    res = minimize_scalar(opt_log_sigma, method = 'bounded', bounds =(0, bb))
    return res.x



def crps_censored_gaussian(y, mean, sigma):
    y_scale = y - mean
    lower = (0-mean) / sigma
    z = y_scale / sigma
    upper = float('Inf')
    crps_score = sigma * crps_cnorm(z, lower, upper)
    #crps_score = y_scale * (2* stats.norm.cdf(y_scale, scale = sigma)-1) + sigma * (np.sqrt(2) * np.exp(-0.5*z*z)-1) / np.sqrt(np.pi)
    return np.mean(crps_score)

def crps_cnorm(y, lower, upper):
    p_l = stats.norm.cdf(lower)
    out_u1 = 0
    out_u2 = 1
    out_l1 = -lower * p_l**2 - 2*stats.norm.pdf(lower)*p_l
    out_l2 = stats.norm.cdf(lower, scale = np.sqrt(0.5))
    z = np.maximum(y, lower)
    b = out_u2 - out_l2
    out_z = z*(2*stats.norm.cdf(z)-1) + 2*stats.norm.pdf(z)
    out = out_z + out_l1 + out_u1 - b/ np.sqrt(np.pi)
    return out + np.absolute(y - z)
