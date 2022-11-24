# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 06:29:11 2022

@author: walz
"""
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
from statsmodels.nonparametric.bandwidths import bw_silverman
import cpp_crpsmixw
import cpp_int_lims
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

def crps_tids2(forecast, y, h, df):
    if hasattr(y,  "__len__") == False:
        y = np.array([y])
    #if len(preds.predictions) != len(y):
     #   raise ValueError("preds same length as y")
    if h < 0:
        raise ValueError("h must be positive")
    if df < 0:
        raise ValueError("df must be positive") 
    #if hasattr(y,  "__len__"):
    mean = [np.unique(forecast[i,]) for i in range(len(y))]
    len_preds = [len(x) for x in mean]
    len_cumsum = np.cumsum(len_preds)
    len_cumsum = np.insert(len_cumsum, 0, 0)
    #mean = np.concatenate([x.points for x in preds.predictions])
    #ecdfs = [ECDF(forecast_test[i,])(mean[i]) for i in range(len(obs_test))]
    ecdfs = [ECDF(forecast[i,])(mean[i]) for i in range(len(y))]
    weights = np.concatenate([np.diff(np.insert(x, 0, 0)) for x in ecdfs])
    mean = np.concatenate(mean)
    crps = cpp_int_lims.cpp_int_lims(y,mean,weights, len_cumsum, h, df, float('-Inf'), float('Inf'))
    return crps

def crps_ens_smoothing(forecast, y, h, df=None):
    if df == None:
        n = len(y)
        crps_val = np.zeros(n)
        for i in range(n):
            #m = #preds.predictions[i].points
            m = np.sort(np.unique(forecast[i, ]))
            ecdf = ECDF(forecast[i,])
            w = np.diff(np.insert(ecdf(m), 0, 0)) 
            crps_val[i] = cpp_crpsmixw.crpsmixGw(m, w, y[i], h)
        return np.mean(crps_val)
    else:
        return crps_tids2(forecast, y, h, df)

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



#################################

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
def llscore(idr_preds_validation, y_validation, h, df=None):
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



def crps_tids_lims(preds, y, h, df):
    if hasattr(y,  "__len__") == False:
        y = np.array([y])
    if len(preds.predictions) != len(y):
        raise ValueError("preds same length as y")
    if h < 0:
        raise ValueError("h must be positive")
    if df < 0:
        raise ValueError("df must be positive")
    #if hasattr(y,  "__len__"):
    len_preds = [len(x.points) for x in preds.predictions]
    len_cumsum = np.cumsum(len_preds)
    len_cumsum = np.insert(len_cumsum, 0, 0)
    mean = np.concatenate([x.points for x in preds.predictions])
    weights = np.concatenate([np.diff(np.insert(x.ecdf, 0, 0)) for x in preds.predictions])
    crps = cpp_int_lims.cpp_int_lims(y,mean,weights, len_cumsum, h, df, -1*float('Inf'), float('Inf'))
    return crps

def smooth_crps(preds, y, h, df=None):
    if df == None:
        n = len(y)
        crps_val = np.zeros(n)
        for i in range(n):
            m = preds.predictions[i].points
            w = np.diff(np.insert(preds.predictions[i].ecdf, 0, 0))
            crps_val[i] = cpp_crpsmixw.crpsmixGw(m, w, y[i], h)
        return np.mean(crps_val)
    else:
        return crps_tids_lims(preds, y, h, df)



def norm_pdf(yval, thresholds, h, df):
    return stats.norm.pdf((yval - thresholds), scale=h)


def t_pdf(yval, thresholds, h, df):
    return stats.t.pdf((yval - thresholds), df, scale=h)


def optimize_ll(preds, y, df=None):
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

    # bracket = (0.5, 1.5)
    res = minimize_scalar(opt_ll, method='bounded', bounds=(0, bb))
    return res.x, res.fun


def optimize_ll2(preds, y, df=None, tol=1e-7):
    #if df == None:
    #    fun = norm_pdf
    #else:
    #    fun = t_pdf
    bb = np.max(y)

    def opt_ll(h):
        return llscore(preds, y, h, df)

    # bracket = (0.5, 1.5)
    res = minimize_scalar(opt_ll, method='bounded', bounds=(0, bb), tol=tol)
    return res.x, res.fun


def crps_norm_optim(y, mean):
    y_scale = y - mean
    bb = np.max(y)
    def opt_crps_sigma(sigma):
        z = y_scale / sigma
        crps_score = y_scale * (2* stats.norm.cdf(y_scale, scale = sigma)-1) + sigma * (np.sqrt(2) * np.exp(-0.5*z*z)-1) / np.sqrt(np.pi)
        return np.mean(crps_score)
    res = minimize_scalar(opt_crps_sigma, method = 'bounded', bounds =(0, bb))
    return res.x

def log_norm_optim(y, mean):
    y_scale = y - mean
    bb = np.max(y)
    aa = np.min(np.diff(np.sort(y)))/100
    def opt_log_sigma(sigma):
        return -1*np.mean(np.log(stats.norm.pdf(y_scale, scale = sigma)))
    res = minimize_scalar(opt_log_sigma, method = 'bounded', bounds =(aa, bb))
    return res.x

def crps_normal(y, mean, sigma):
    y_scale = y - mean
    z = y_scale / sigma
    crps_score = y_scale * (2* stats.norm.cdf(y_scale, scale = sigma)-1) + sigma * (np.sqrt(2) * np.exp(-0.5*z*z)-1) / np.sqrt(np.pi)
    return np.mean(crps_score)

def log_norm(y, mean, sigma):
    return -1*np.mean(np.log(stats.norm.pdf(y - mean, scale = sigma)))



def optimize_paras(idr_preds_validation, y_validation, y_train):
    h_rule = bw_silverman(y_train)
    tol = h_rule / 1000

    ll_deg1 = llscore(idr_preds_validation, y_validation, h = tol, df=2)
    ll_deg2 = llscore(idr_preds_validation, y_validation, h = tol / 100, df = 2)
    if ll_deg2 < ll_deg1:
        ll_deg1 = llscore(idr_preds_validation, y_validation, h = tol, df=None)
        ll_deg2 = llscore(idr_preds_validation, y_validation, h = tol / 100, df = None)
        df = None
        if ll_deg2 < ll_deg1:
            h = h_rule
            ll = llscore(idr_preds_validation, y_validation, h =h_rule, df = None)
        else:
            h, ll = optimize_ll2(idr_preds_validation, y_validation, df=None, tol = tol)
        return ll, h, df
    
    else:
        hs, lls = [], []
        dfs = [None, 20, 10, 5, 4, 3, 2]
        tol2 = tol / 100
        for df in dfs:
            h, ll = optimize_ll2(idr_preds_validation, y_validation, df, tol = tol2)
            hs += [h]
            lls += [ll]

        ll_ix = np.argmin(lls)
        ll_min = lls[ll_ix]
        h_min = hs[ll_ix]
        df_min = dfs[ll_ix]
        return ll_min, h_min, df_min







    
