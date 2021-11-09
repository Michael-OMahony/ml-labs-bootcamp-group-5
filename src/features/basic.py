import numpy as np
from src.features.common import read_bluetooth_from_file


def features(fp, key, tunables={}):
    rssi = np.array(read_bluetooth_from_file(fp))
    return {
        "RssiMin": rssi.min(),
        "RssiPercentile:1": np.percentile(rssi, 1.),
        "RssiPercentile:5": np.percentile(rssi, 5.),
        "RssiPercentile:10": np.percentile(rssi, 10.),
        "RssiPercentile:25": np.percentile(rssi, 25.),
        "RssiPercentile:40": np.percentile(rssi, 40.),
        "RssiMean": rssi.mean(),
        "RssiPercentile:65": np.percentile(rssi, 65.),
        "RssiPercentile:80": np.percentile(rssi, 80.),
        "RssiPercentile:90": np.percentile(rssi, 90.),
        "RssiPercentile:95": np.percentile(rssi, 95.),
        "RssiPercentile:99": np.percentile(rssi, 99.),
        "RssiMax": rssi.max(),
        "CoarseGrain": 0 if key.coarse_grain == "Y" else 1,
        "fileid": key.fileid,
        "Distance": str(key.distance_in_meters),
        "DistanceFloat": float(key.distance_in_meters),
    }


def percentile_features(fp, key, tunables={}):
    metadata = {
        "CoarseGrain": 0 if key.coarse_grain == "Y" else 1,
        "fileid": key.fileid,
        "Distance": str(key.distance_in_meters),
        "DistanceFloat": float(key.distance_in_meters),
    }
    rssi = np.array(read_bluetooth_from_file(fp))
    feats = {f"P:{i}": p
             for i, p in enumerate(
                 np.percentile(rssi, list(range(1, 100))))}
    feats.update({
        "P:min": rssi.min(),
        "P:max": rssi.max(),
        "P:mean": rssi.mean()
    })
    return feats | metadata
