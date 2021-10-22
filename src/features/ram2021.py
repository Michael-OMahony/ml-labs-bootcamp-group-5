import math

import numpy as np
import pandas as pd
from scipy.stats import trim_mean
from src.features.common import postproc_default, read_chirp_sequence_from_file


def summary_stats(chirps, tunables={"trim_prop": 0.20}):

    q1 = [c.rssi.quantile(0.25) for c in chirps]
    q3 = [c.rssi.quantile(0.75) for c in chirps]
    summary = {
        "TrimmedMean": [trim_mean(c.rssi, tunables["trim_prop"]) for c in chirps],
        "Mean": [c.rssi.mean() for c in chirps],
        "Median": [c.rssi.median() for c in chirps],
        "Q1": q1,
        "Q3": q3,
        "Min": [c.rssi.min() for c in chirps],
        "Max": [c.rssi.max() for c in chirps],
        "Std": [c.rssi.std() for c in chirps],
        "Iqr": np.array(q3) - np.array(q1),
        "Kurtosis": [c.rssi.kurtosis() for c in chirps],
        "Skew": [c.rssi.skew() for c in chirps],
    }
    return summary


def replace_na_with_row_mean(summary):
    _mean = np.array([val for _, val in summary.items()
                     if not math.isnan(val)]).mean()
    for stat, value in summary.items():
        if math.isnan(value):
            summary[stat] = _mean
    return summary


def expand_summary(summary):
    expanded_summary = {}
    for name, vector in summary.items():
        for i, value in enumerate(vector):
            expanded_summary[f"{name}_{i}"] = value
    return expanded_summary


def histogram(chirps, tunables={"bin_size": 5}):
    rssi_values = pd.concat([c.rssi for c in chirps], axis=0).values
    bins = np.arange(-80, -40, tunables["bin_size"])
    counts, _ = np.histogram(rssi_values, bins=bins, density=True)
    center = (bins[:-1] + bins[1:]) / 2
    return {f"Hist_{-rssi}": count for count, rssi in zip(counts, center)}


def extract_features(filepath, key):
    chirps = read_chirp_sequence_from_file(filepath, max_chirps=2)

    feats = {
        'Distance': str(key.distance_in_meters),
        'CoarseGrain': key.coarse_grain,
        'fileid': key.fileid
    }
    summary = expand_summary(
        summary_stats(chirps, tunables={"trim_prop": 0.2}))
    feats.update(summary)
    feats.update(histogram(chirps, tunables={"bin_size": 3}))
    return feats


def postproc(feats, encoders={}):
    return postproc_default(feats)
