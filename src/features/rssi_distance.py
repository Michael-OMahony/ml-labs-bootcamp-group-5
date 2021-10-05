import numpy as np
import pandas as pd


def distance_from_rssi(rssi, TX=-61.02, N=2.187):
    return np.power(10, (TX - rssi) / (10 * N))


# def distance_from_rssi(rssi, TX=-69., N=4.):
#    return np.power(10, (TX - rssi) / (10 * N))


def extract_feature(filepath, key):
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


def extract_feature2(filepath, key):
    rssi = [float(line.split(',')[-1])
            for line in open(filepath).read().split('\n')
            if 'Bluetooth' in line]
    return {
        'PredictedDistance': distance_from_rssi(np.array(rssi).mean(),),
        'Distance': str(key.distance_in_meters),
        'CoarseGrain': key.coarse_grain,
        'fileid': key.fileid
    }


def postproc_feature_dicts(feats, encoders={}):
    return pd.DataFrame(feats), encoders
