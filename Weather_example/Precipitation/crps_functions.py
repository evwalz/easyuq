from statsmodels.distributions.empirical_distribution import ECDF
import numpy as np
import crps_lims
import pandas as pd
# Temperature:

# smooth CRPS for IDR type input
def crps_idr_mix(preds, y, h, df=None):
    if hasattr(y,  "__len__") == False:
        y = np.array([y])
    if len(preds.predictions) != len(y):
        raise ValueError("preds same length as y")
    if h < 0:
        raise ValueError("h must be positive")
    if df == None:
        return crps_idr_mixnorm(preds, y, h, df)
    else:
        if df < 0:
            raise ValueError("df must be positive")
        return crps_idr_mixt(preds, y, h, df)

def crps_idr_mixt(preds, y, h, df, low=-1 * float('Inf')):
    len_preds = [len(x.points) for x in preds.predictions]
    len_cumsum = np.cumsum(len_preds)
    len_cumsum = np.insert(len_cumsum, 0, 0)
    mean = np.concatenate([x.points for x in preds.predictions])
    weights = np.concatenate([np.diff(np.insert(x.ecdf, 0, 0)) for x in preds.predictions])
    yc = y.copy()
    crps = crps_lims.crps_t_lims(yc, mean, weights, len_cumsum, h, df, low, float('Inf'))
    return crps

def crps_idr_mixnorm(preds, y, h, low):
    len_preds = [len(x.points) for x in preds.predictions]
    len_cumsum = np.cumsum(len_preds)
    len_cumsum = np.insert(len_cumsum, 0, 0)
    mean = np.concatenate([x.points for x in preds.predictions])
    weights = np.concatenate([np.diff(np.insert(x.ecdf, 0, 0)) for x in preds.predictions])
    yc = y.copy()
    crps = crps_lims.crps_norm_lims(yc, mean, weights, len_cumsum, h, low, float('Inf'))
    return crps


# Precipitation with censoring
def crps_idr_censored(out, fct, y, h, df):
    n = len(y)
    y_zero = y[y == 0]
    y_sub = y[y > 0]
    n_pos = len(y_sub)
    n_zero = len(y_zero)
    if n_zero == 0:
        preds_sub = out.predict(pd.DataFrame({"X": fct[y > 0]}, columns=["X"]))
        return idr_tids_lims(preds_sub, y_sub, h, df, low = 0)
    elif n_pos == 0:
        preds_zero = out.predict(pd.DataFrame({"X": fct[y == 0]}, columns=["X"]))
        return idr_censored_tids_lims(preds_zero, y_zero, h, df, low = 0)
    else:
        preds_sub = out.predict(pd.DataFrame({"X": fct[y > 0]}, columns=["X"]))
        crps_pos = idr_tids_lims(preds_sub, y_sub, h, df, low = 0)
        preds_zero = out.predict(pd.DataFrame({"X": fct[y == 0]}, columns=["X"]))
        crps_zeros = idr_censored_tids_lims(preds_zero, y_zero, h, df, low = 0)
        return (crps_pos*n_pos + crps_zeros*n_zero) / n


def idr_censored_tids_lims(preds, y, h, df, low):
    len_preds = [len(x.points) for x in preds.predictions]
    len_cumsum = np.cumsum(len_preds)
    len_cumsum = np.insert(len_cumsum, 0, 0)
    mean = np.concatenate([x.points for x in preds.predictions])
    weights = np.concatenate([np.diff(np.insert(x.ecdf, 0, 0)) for x in preds.predictions])
    yc = y.copy()
    crps = crps_lims.crps_t_censored(yc,mean,weights, len_cumsum, h, df, low, float('Inf'))
    return crps

def idr_tids_lims(preds, y, h, df, low = -1*float('Inf')):
    len_preds = [len(x.points) for x in preds.predictions]
    len_cumsum = np.cumsum(len_preds)
    len_cumsum = np.insert(len_cumsum, 0, 0)
    mean = np.concatenate([x.points for x in preds.predictions])
    weights = np.concatenate([np.diff(np.insert(x.ecdf, 0, 0)) for x in preds.predictions])
    yc = y.copy()
    crps = crps_lims.crps_t_lims(yc,mean,weights, len_cumsum, h, df, low, float('Inf'))
    return crps


def crps_ens_precip_rot(forecast, y, h):
    below0 = crps_ens_below0_rot(forecast, y, h) 
    censored = crps_ens_censored_rot(forecast, y, h)
    classic = below0 + censored
    return classic, censored

def crps_ens_below0_rot(forecast, y, h,):
    mean = [np.unique(forecast[i,]) for i in range(len(y))]
    len_preds = [len(x) for x in mean]
    len_cumsum = np.cumsum(len_preds)
    len_cumsum = np.insert(len_cumsum, 0, 0)
    ecdfs = [ECDF(forecast[i,])(mean[i]) for i in range(len(y))]
    weights = np.concatenate([np.diff(np.insert(x, 0, 0)) for x in ecdfs])
    mean = np.concatenate(mean)
    yc = y.copy()
    crps =crps_lims.crps_norm_hvec_below0(yc,mean,weights, len_cumsum, h, -float('Inf'), 0)
    return crps

def crps_ens_censored_rot(ens, y, h):
    n = len(y)
    y_zero = y[y == 0]
    y_sub = y[y > 0]
    n_pos = len(y_sub)
    n_zero = len(y_zero)
    if n_zero == 0:
        ix_greater = np.where(y > 0)[0]
        ens_sub = ens[ix_greater , ]
        return ens_tids_lims_rot(ens_sub, y_sub, h, low = 0)
    elif n_pos == 0:
        ix_zero = np.where(y == 0)[0]
        ens_zero = ens[ix_zero , ]
        return ens_censored_tids_lims_rot(ens_zero, y_zero, h, low = 0)
    else:
        ix_greater = np.where(y > 0)[0]
        ix_zero = np.where(y == 0)[0]
        ens_sub = ens[ix_greater , ]
        ens_zero = ens[ix_zero , ]
        crps_pos = ens_tids_lims_rot(ens_sub, y_sub, h, low = 0)
        crps_zeros = ens_censored_tids_lims_rot(ens_zero, y_zero, h, low = 0)
        return (crps_pos*n_pos + crps_zeros*n_zero) / n

def ens_tids_lims_rot(forecast, y, h,  low):
    mean = [np.unique(forecast[i,]) for i in range(len(y))]
    len_preds = [len(x) for x in mean]
    len_cumsum = np.cumsum(len_preds)
    len_cumsum = np.insert(len_cumsum, 0, 0)
    ecdfs = [ECDF(forecast[i,])(mean[i]) for i in range(len(y))]
    weights = np.concatenate([np.diff(np.insert(x, 0, 0)) for x in ecdfs])
    mean = np.concatenate(mean)
    yc = y.copy()
    crps = crps_lims.crps_norm_hvec(yc,mean,weights, len_cumsum, h, low, float('Inf'))
    return crps

def ens_censored_tids_lims_rot(forecast, y, h, low):
    mean = [np.unique(forecast[i,]) for i in range(len(y))]
    len_preds = [len(x) for x in mean]
    len_cumsum = np.cumsum(len_preds)
    len_cumsum = np.insert(len_cumsum, 0, 0)
    ecdfs = [ECDF(forecast[i,])(mean[i]) for i in range(len(y))]
    weights = np.concatenate([np.diff(np.insert(x, 0, 0)) for x in ecdfs])
    mean = np.concatenate(mean)
    yc = y.copy()
    crps = crps_lims.crps_norm_censored_hvec(yc,mean,weights, len_cumsum, h, low, float('Inf'))
    return crps
