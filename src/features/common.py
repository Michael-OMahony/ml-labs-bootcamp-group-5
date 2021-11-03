import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import RobustScaler, MinMaxScaler

TARGET = "Distance"


def read_chirp_sequence_from_file(filepath, max_chirps=None):
    """Return a sequence of values containing average RSSI reading of each chirp
    (Reads from file directly)
    """
    data, chirps, chirp = [], [], []
    for line in open(filepath).read().split("\n")[7:]:
        if "Bluetooth" in line:
            t, _, rssi = line.split(",")
            item = {"t": float(t), "rssi": float(rssi)}

            if len(data) > 0:
                if float(t) - data[-1]["t"] > 2.0:
                    chirps.append(chirp)
                    chirp = []
            chirp.append(item)
            data.append(item)
        if max_chirps:
            if len(chirps) > max_chirps:
                break
    if len(chirps) == 0:
        if not max_chirps:
            max_chirps = 1
        return [pd.DataFrame(chirp) for _ in range(max_chirps)]
    return [pd.DataFrame(chirp) for chirp in chirps]


def postproc_default(feats, pipe=None, tunables={}, verbose=False):
    return pd.DataFrame(feats), pipe


def get_predictors_default(dataset):
    return [col for col in dataset.columns
            if col not in ['Distance', 'fileid', 'CoarseGrain']]


def read_bluetooth_histogram_from_file(fp, _min=-100, _max=-30):
    rssi_list = read_bluetooth_from_file(fp)
    bins = np.arange(_min, _max)
    counts, _ = np.histogram(rssi_list, bins=bins, density=True)
    center = (bins[:-1] + bins[1:]) / 2
    return {f"Hist_{-rssi}": count for count, rssi in zip(counts, center)}


def to_histogram(_array, _min=-100, _max=-30, bin_count=50):
    bins = np.linspace(_min, _max, bin_count)
    counts, _ = np.histogram(_array, bins=bins)
    center = (bins[:-1] + bins[1:]) / 2
    return {f"Hist_{value:.2f}": count for count, value in zip(counts, center)}


def read_bluetooth_from_file(fp):
    rssi_list = []
    for line in open(fp).read().split('\n'):
        if 'Bluetooth' in line:
            t, _, rssi = line.split(",")
            rssi_list.append(float(rssi))
    return rssi_list


def postproc_bluetooth_histogram(feats, pipe=None, tunables={}, verbose=False):
    return scale_histogram_features(
        pd.DataFrame(feats), pipe=pipe
    )


def feature_bluetooth_histogram(fp, key, tunables={}):
    features = to_histogram(
        np.array(read_bluetooth_from_file(fp)),
        _min=-100, _max=-30, bin_count=70)
    features["Distance"] = str(key.distance_in_meters)
    return features


def scale_histogram_features(df, pipe=None):
    feat_cols = [col for col in df.columns if 'Hist' in col]
    if not pipe:
        pipe = Pipeline([("robustScalar", RobustScaler()),
                        ("minMaxScalar", MinMaxScaler())])
        df_scaled = pd.DataFrame(pipe.fit_transform(
            df[feat_cols]), columns=feat_cols)
    else:
        df_scaled = pd.DataFrame(pipe.transform(
            df[feat_cols]), columns=feat_cols)
    df_scaled["Distance"] = df["Distance"]
    return df_scaled, pipe
