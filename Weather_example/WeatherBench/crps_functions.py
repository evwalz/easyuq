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


# smooth CRPS for ensemble type input with varying h for mixture of normal
def crps_ens_mixnorm_hvec(forecast, y, hs):
    if hasattr(y,  "__len__") == False:
        y = np.array([y])
    mean = [np.unique(forecast[i,]) for i in range(len(y))]
    len_preds = [len(x) for x in mean]
    len_cumsum = np.cumsum(len_preds)
    len_cumsum = np.insert(len_cumsum, 0, 0)
    ecdfs = [ECDF(forecast[i,])(mean[i]) for i in range(len(y))]
    weights = np.concatenate([np.diff(np.insert(x, 0, 0)) for x in ecdfs])
    mean = np.concatenate(mean)
    yc = y.copy()
    hsc = hs.copy()
    crps = crps_lims.crps_norm_hvec(yc,mean,weights, len_cumsum, hsc, float('-Inf'), float('Inf'))
    return crps
