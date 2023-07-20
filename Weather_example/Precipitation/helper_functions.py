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


def norm_pdf(yval, thresholds, h, df):
    return stats.norm.pdf((yval - thresholds), scale=h)


def t_pdf(yval, thresholds, h, df):
    return stats.t.pdf((yval - thresholds), df, scale=h)


def onefit_h(preds, y, df):
    if df == None:
        fun = norm_pdf
    else:
        fun = t_pdf
    bb = np.max(y)
    
    def opt_ll(h):
        out = 0
        n = len(y)
        w = 0
        for i in range(n):
            yval = y[i]
            thresholds = preds.predictions[i].points
            y_help = preds.predictions[i].ecdf
            weights = np.diff(np.insert(y_help, 0, 0))
            if len(weights) > 1:
                arr = np.unique(weights)
                if arr[0] == 0 and arr[1] == 1:
                    w = w + 1
                else:
                    indx = np.where(thresholds == yval)[0]
                    weights[indx] = 0
                    weights = weights / np.sum(weights)

            dis = fun(yval, thresholds, h, df)
            f = np.sum(weights * dis)
            if f == 0 and df == None:
                thresholds_new = np.delete(thresholds, indx)
                cdf_new = np.cumsum(np.delete(weights, indx))
                f_log = fdense_update(cdf_new, thresholds_new, yval, h)
                out = out - f_log
            else:
                out = out - np.log(f)
        return out / n
    res = minimize_scalar(opt_ll, method='bounded', bounds=(0, bb))
    return res.x, res.fun

def optimize_paras_onefit(preds_train, y_train):
    hs, lls = [], []
    dfs = [None, 20, 10, 5, 4, 3, 2]
    for df in dfs:
        h, ll = onefit_h(preds_train, y_train, df)
        hs += [h]
        lls += [ll]

    ll_ix = np.nanargmin(lls)
    ll_min = lls[ll_ix]
    h_min = hs[ll_ix]
    df_min = dfs[ll_ix]
    return ll_min, h_min, df_min

def optim_norm(forecast, y):
    fun = norm_pdf
    bb = np.max(y)
    def opt_ll(h):
        out = 0
        n = len(y)
        for i in range(n):
            yval = y[i]
            ens = forecast[i,]
            dis = fun(yval, ens, h, df=None)
            f = np.mean(dis)
            out = out - np.log(f)
        return out / n
# boudns = 1e-2
    res = minimize_scalar(opt_ll, method='bounded', bounds=(1e-10, bb))
    return res.x, res.fun

def optim_t(forecast, y, df):
    fun = t_pdf
    bb = np.max(y)
    def opt_ll(h):
        out = 0
        n = len(y)
        for i in range(n):
            yval = y[i]
            ens = forecast[i,]
            dis = fun(yval, ens, h, df)
            f = np.mean(dis)
            out = out - np.log(f)
        return out / n

    res = minimize_scalar(opt_ll, method='bounded', bounds=(1e-10, bb))
    return res.x, res.fun

def optim_ll_dressing(forecast, y):
    h_norm, crps_norm = optim_norm(forecast, y)
    h_best = h_norm
    crps_best = crps_norm
    h, crps = optim_t(forecast, y, df = 20)
    dfs = [20, 10, 5, 4, 3, 2]
    if crps_norm < crps:
        return h_norm, None
    k = 0
    while (crps_best > crps and k < 5):
        crps_best = crps
        h_best = h
        df_best = dfs[k]
        k = k+1
        df = dfs[k]
        h, crps = optim_t(forecast, y, df=df)

    if k == 5:
        if crps_best < crps:
            return h_best, 3
        else:
            return h, 2
    else:
        return h_best, df_best
