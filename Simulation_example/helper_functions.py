# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 06:29:11 2022

@author: walz
"""
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
from statsmodels.nonparametric.bandwidths import bw_silverman
#import cpp_crpsmixw
#import cpp_int_lims
import random
from isodisreg import idr
import isodisreg
import pandas as pd
from crpsmixture import smooth_crps

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



#def crps_tids_lims(preds, y, h, df):
#    if hasattr(y,  "__len__") == False:
#        y = np.array([y])
#    if len(preds.predictions) != len(y):
#        raise ValueError("preds same length as y")
#    if h < 0:
#        raise ValueError("h must be positive")
#    if df < 0:
#        raise ValueError("df must be positive")
#    #if hasattr(y,  "__len__"):
#    len_preds = [len(x.points) for x in preds.predictions]
#    len_cumsum = np.cumsum(len_preds)
#    len_cumsum = np.insert(len_cumsum, 0, 0)
#    mean = np.concatenate([x.points for x in preds.predictions])
#    weights = np.concatenate([np.diff(np.insert(x.ecdf, 0, 0)) for x in preds.predictions])
#    crps = cpp_int_lims.cpp_int_lims(y,mean,weights, len_cumsum, h, df, -1*float('Inf'), float('Inf'))
#    return crps

#def smooth_crps(preds, y, h, df=None):
#    if df == None:
#        n = len(y)
#        crps_val = np.zeros(n)
#        for i in range(n):
#            m = preds.predictions[i].points
#            w = np.diff(np.insert(preds.predictions[i].ecdf, 0, 0))
#            crps_val[i] = cpp_crpsmixw.crpsmixGw(m, w, y[i], h)
#        return np.mean(crps_val)
#    else:
#        return crps_tids_lims(preds, y, h, df)



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


def cv_h_llo(x, y, df, cv):
    n = len(x)
    bb = np.max(y)
    if df == None:
        def opt_ll(h):
            out = 0
            for i in range(n):
                fit = idr(np.delete(y, i), pd.DataFrame({"fore": np.delete(x, i)}, columns=["fore"]))
                pred = fit.predict(pd.DataFrame({"fore": [x[i]]}, columns=["fore"])).predictions[0]
                f = smooth_idr_dense_norm(pred.ecdf, pred.points, y[i],h)
                if f == 0:
                    out = out - fdense_update(pred.ecdf, pred.points, y[i], h)
                else:
                    out = out - np.log(f)
            return out / n
    else:
        def opt_ll(h):
            out = 0
            for i in range(n):
                fit = idr(np.delete(y, i), pd.DataFrame({"fore": np.delete(x, i)}, columns=["fore"]))
                pred = fit.predict(pd.DataFrame({"fore": [x[i]]}, columns=["fore"])).predictions[0]
                out = out - np.log(smooth_idr_dense_t(pred.ecdf, pred.points, y[i], h, df))
            return out / n
    res = minimize_scalar(opt_ll, method='bounded', bounds=(0, bb))
    return res.x, res.fun   

def cv_h(x, y, df, cv):
    n = len(x)
    cv = int(cv)   
    bb = np.max(y)
    if df == None:  
        def opt_ll(h):
            out = 0
            indx_shuffle = np.arange(n)
            random.shuffle(indx_shuffle)
            indx_split = np.array_split(indx_shuffle, cv)
            original_list = indx_split.copy()
            for i in range(cv):
                indx_fold = indx_split.pop(i)
                y_fold = y[indx_fold]
                x_fold =x[indx_fold]
                y_cv = y[np.concatenate(indx_split)]
                x_cv = x[np.concatenate(indx_split)]
                fit = idr(y_cv, pd.DataFrame({"fore": x_cv}, columns=["fore"]))
                pred = fit.predict(pd.DataFrame({"fore": x_fold}, columns=["fore"]))

                out = out + llscore(pred, y_fold, h, df)
                indx_split = original_list.copy()
            return out / cv
    else:
        def opt_ll(h):
            out = 0
            indx_shuffle = np.arange(n)
            random.shuffle(indx_shuffle)
            indx_split = np.array_split(indx_shuffle, cv)
            original_list = indx_split.copy()
            for i in range(cv):
                indx_fold = indx_split.pop(i)
                y_fold = y[indx_fold]
                x_fold = x[indx_fold]
                y_cv = y[np.concatenate(indx_split)]
                x_cv = x[np.concatenate(indx_split)]
                fit = idr(y_cv, pd.DataFrame({"fore": x_cv}, columns=["fore"]))
                pred = fit.predict(pd.DataFrame({"fore": x_fold}, columns=["fore"]))
                out = out + llscore(pred, y_fold, h, df)
                indx_split = original_list.copy()
            return out / cv
        
    res = minimize_scalar(opt_ll, method='bounded', bounds=(0, bb))
    return res.x, res.fun 


def optimize_paras_cvfit(preds_train, y_train, cv):
    if cv == len(y_train):
        fun = cv_h_llo
    else:
        fun = cv_h
        
    hs, lls = [], []
    dfs = [None, 20, 10, 5, 4, 3, 2]
    for df in dfs:
        h, ll = fun(preds_train, y_train, df, cv)
        hs += [h]
        lls += [ll]

    ll_ix = np.nanargmin(lls)
    ll_min = lls[ll_ix]
    h_min = hs[ll_ix]
    df_min = dfs[ll_ix]
    return ll_min, h_min, df_min    







    
