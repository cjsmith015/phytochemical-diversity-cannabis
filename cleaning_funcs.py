import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import re
import time


def print_remove(df, remove, terp=False, terps=None):
    print('n samples dropped = '+str(remove.sum()))
    if terp:
        df.loc[remove, 'has_terps'] = False
        df.loc[remove, terps] = np.nan
    else:
        df = df[~remove].copy()
    return df


def decarb(df, cannab, acid, varin=False):
    if varin:
        decarb_val = 0.8668
    else:
        decarb_val = 0.877
    tot_cannab = (decarb_val*df[acid])+df[cannab]
    return tot_cannab


def MAD_outlier(x, thresh=5, return_bool=True):
    x = x.fillna(0)
    diff = np.abs(x - np.median(x))
    mad = np.median(diff[diff!=0])
    mod_z = 0.6745 * diff/mad
    if return_bool:
        return mod_z > thresh
    else:
        return mod_z