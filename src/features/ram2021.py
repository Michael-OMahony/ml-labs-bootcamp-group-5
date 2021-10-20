from scipy.stats import trim_mean
import numpy as np
import pandas as pd


def summary_stats(chirps):
    q1 = [c.rssi.quantile(0.25) for c in chirps]
    q3 = [c.rssi.quantile(0.75) for c in chirps]
    summary = {
        "TrimmedMean": [trim_mean(c.rssi) for c in chirps],
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


def histogram(chirps, tunables={"bin_size": 5}):
    rssi_values = pd.concat([c.rssi for c in chirps], axis=0).values
    bins = np.arange(-80, -40, tunables["bin_size"])
    counts, _ = np.histogram(rssi_values, bins=bins, density=True)
    center = (bins[:-1] + bins[1:]) / 2
    return {
        f"Hist_{-rssi}" : count for count, rssi in zip(counts, center)
    }
