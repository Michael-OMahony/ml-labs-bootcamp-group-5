import numpy as np
import pandas as pd


def distance_from_rssi(rssi, TX=-61.02, N=2.187):
    return np.power(10, (TX - rssi) / (10 * N))


# def distance_from_rssi(rssi, TX=-69., N=4.):
#    return np.power(10, (TX - rssi) / (10 * N))


def extract_feature(filepath, key, tunables={}):
    rssi = [float(line.split(',')[-1])
            for line in open(filepath).read().split('\n')
            if 'Bluetooth' in line]
    return {
        'PredictedDistance': distance_from_rssi(np.array(rssi).mean(),
                                                TX=-52 if key.coarse_grain == 'Y' else -54,
                                                N=2.6 if key.coarse_grain == 'Y' else 2.1),
        'Distance': str(key.distance_in_meters),
        'CoarseGrain': key.coarse_grain,
        'fileid': key.fileid
    }


def extract_feature2(filepath, key, tunables={}):
    rssi = [float(line.split(',')[-1])
            for line in open(filepath).read().split('\n')
            if 'Bluetooth' in line]
    _mean_rssi = np.array(rssi).mean()
    if len(tunables) == 0:
        tunables = {
            "TX": -61.02, "N": 2.187
        }
    predicted_distance = distance_from_rssi(
        _mean_rssi, tunables["TX"], tunables["N"])
    return {
        'PredictedDistance': predicted_distance,
        'Distance': str(key.distance_in_meters),
        'CoarseGrain': key.coarse_grain,
        'fileid': key.fileid
    }


def postproc_feature_dicts(feats, pipe=None, tunables={}, verbose=False):
    return pd.DataFrame(feats), pipe
