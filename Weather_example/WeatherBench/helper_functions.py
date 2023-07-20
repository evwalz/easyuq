# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 06:29:11 2022

@author: walz
"""
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
from statsmodels.nonparametric.bandwidths import bw_silverman
from crpsmixture import smooth_crps
from statsmodels.distributions.empirical_distribution import ECDF


def t_pdf(yval, ens, h, df):
    return stats.t.pdf((yval - ens), df, scale = h)

def norm_pdf(yval, ens, h, df):
    return stats.norm.pdf((yval - ens), scale = h)

def llscore_ens_smoothing(forecast, y, h, df=None):
    if df == None:
        fun = norm_pdf
    else:
        fun = t_pdf

    out = 0
    n = len(y)
    for i in range(n):
        yval = y[i]
        ens = forecast[i,]
        dis = fun(yval, ens, h, df)
        f = np.mean(dis)
        out = out - np.log(f)
    return out / n

def llscore_cp(forecast, y, hs, df=None):
    if df == None:
        fun = norm_pdf
    else:
        fun = t_pdf

    out = 0
    n = len(y)
    for i in range(n):
        h = hs[i]
        yval = y[i]
        ens = forecast[i,]
        dis = fun(yval, ens, h, df)
        f = np.mean(dis)
        out = out - np.log(f)
    return out / n


def optim_h(X, y, df):
    if df == None:
        fun = norm_pdf
    else:
        fun = t_pdf
    bb = np.max(y)
    def opt_ll(h):
        n = len(y)
        out = 0
        for i in range(n):
            yval = y[i]
            ens = X[i,]
            dis = fun(yval, ens, h, df)
            f = np.mean(dis)
            out = out - np.log(f)
        return out / n
    res = minimize_scalar(opt_ll, method='bounded', bounds=(0, bb))
    return res.x, res.fun

        

def ensemble_smoothing(ensemble, y):
    dfs = [None, 2, 3, 4, 5, 10, 20]
    hs, lls = [], []
    for df in dfs:
        h, ll = optim_h(ensemble, y, df)
        hs += [h]
        lls += [ll]
        
    ll_ix = np.nanargmin(lls)
    ll_min = lls[ll_ix]
    h_min = hs[ll_ix]
    df_min = dfs[ll_ix]
    return ll_min, h_min, df_min

def log_norm(y, mean, sigma):
    return -1*np.mean(np.log(stats.norm.pdf(y - mean, scale = sigma)))

def single_gaussian_optim(y, mean):
    y_scale = y - mean
    bb = np.max(y)
    def opt_log_sigma(sigma):
        return -1*np.mean(np.log(stats.norm.pdf(y_scale, scale = sigma)))
    res = minimize_scalar(opt_log_sigma, method = 'bounded', bounds =(1e-8, bb))
    return res.x

def smooth_idr_dense_norm(y_help, thresholds ,grd, h, df=None):
    # only for grd is single value
    return np.sum(np.diff(np.insert(y_help, 0, 0)) * stats.norm.pdf((grd - thresholds), scale=h))


def smooth_idr_dense_t(y_help, thresholds, grd, h, df):
    # only for grd is single value
    return np.sum(np.diff(np.insert(y_help, 0, 0)) * stats.t.pdf((grd - thresholds), df, scale=h))

def fdense_update(y_help, thresholds, y_val, h):
    help_scale = np.round(-1 * np.max(-0.5 * ((y_val - thresholds) / h) ** 2))
    u = -0.5 * ((y_val - thresholds) / h) ** 2 + help_scale
    tPDF = 1 / np.sqrt(np.pi * 2) * np.exp(u)
    right_side = np.insert(tPDF, len(tPDF), 0)
    diff = (tPDF - right_side[1:])
    # print('extra')
    return np.log(np.sum(diff * y_help) * (1 / h)) - help_scale

# (idr_predict_test, y_test, h, degree_free=None)
def llscore_idr(idr_preds_validation, y_validation, h, df=None):
    if df == None:
        fun = smooth_idr_dense_norm
        # df = None
    else:
        fun = smooth_idr_dense_t
        # df = df
    s = 0
    for i in range(len(y_validation)):
        yval = y_validation[i]
        y_help = idr_preds_validation.predictions[i].ecdf
        thresholds = idr_preds_validation.predictions[i].points
        f = fun(y_help, thresholds, yval, h, df)
        if f == 0 and df == None:
            s = s - fdense_update(y_help, thresholds, yval, h)
        else:
            s = s - np.log(f)
    return s / len(y_validation)



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

#################################


def silver_rot(y):
    n = y.shape[0]
    IQR = np.quantile(y, 0.75, axis = 1) - np.quantile(y, 0.25, axis = 1)
    std_vec = np.std(y, axis = 1, ddof = 1)
    ix_0 = np.where(IQR == 0)[0]
    IQR[ix_0] = 1.34*IQR[ix_0]
    return 0.9*np.minimum(std_vec, IQR / 1.34) * n**(-1/5.)

def algo72_ensemble(fct_train, fct_test, y_train):
    
    n = len(fct_train)
    m = len(fct_test)
    sum_x = np.sum(fct_train)
    sum_x2 = np.sum(np.square(fct_train))
    XX = np.zeros((2, 2))
    XX[0, 0] = sum_x2
    XX[0, 1] = XX[1, 0] = -1*sum_x
    XX[1, 1] = n
    XX = XX / (n*sum_x2 - (sum_x)**2)
    X_tr = np.ones((len(fct_train), 2))
    X_tr[:, 1] = fct_train
    X_tr_XX = X_tr @ XX
    H = X_tr_XX @ np.transpose(X_tr)
    C = np.zeros((m, n))
    g2 = XX[0, 0] + XX[1, 0]* fct_test+ fct_test*( XX[0,1]+ fct_test*XX[1, 1])
    g = ghelp =np.zeros(n+1)
    Hbar = np.zeros((n, n))
    
    for j in range(m):
        g[0:n] = X_tr_XX[:, 0] + X_tr_XX[:, 1] * fct_test[j]
        g[n] = g2[j]
        Hbar = H - np.outer(g[0:n], g[0:n]) / (1-g[n])
        g = g / (1 + g[n]) 
        gsrt = np.sqrt(1 - g[n])
        Hdiagsqrt = np.sqrt(1 - np.diag(Hbar)[0:n])
        B = gsrt + g[0:n]/Hdiagsqrt
        A = np.sum(g[0:n]*y_train) / gsrt + (y_train - np.sum(Hbar[0:n, 0:n].T * y_train, axis = 1)) / Hdiagsqrt
        C[j, :] = A / B
    
    return C







    
