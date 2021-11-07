import numpy as np
from src.features.common import read_bluetooth_from_file


def features(fp, key, tunables={}):
    rssi = np.array(read_bluetooth_from_file(fp))
    return {
        "RssiMin": rssi.min(),
        "RssiMax": rssi.max(),
        "RssiMean": rssi.mean(),
        "CoarseGrain": 0 if key.coarse_grain == "Y" else 1,
        "fileid": key.fileid,
        "Distance": str(key.distance_in_meters),
        "DistanceFloat": float(key.distance_in_meters),
    }
