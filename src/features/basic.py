import numpy as np
from src.features.common import read_bluetooth_from_file


def features(fp, key, tunables={}):
    features = {
        "CoarseGrain": 0 if key.coarse_grain == "Y" else 1,
        "fileid": key.fileid,
        "Distance": str(key.distance_in_meters),
        "DistanceFloat": float(key.distance_in_meters),
    }
    rssi = np.array(read_bluetooth_from_file(fp))
    features.update(bluetooth_extended_summary(rssi))
    return features


def bluetooth_extended_summary(rssi):
    return {
        "Bluetooth:Min": rssi.min(),
        "Bluetooth:Percentile_1": np.percentile(rssi, 1.),
        "Bluetooth:Percentile_5": np.percentile(rssi, 5.),
        "Bluetooth:Percentile_10": np.percentile(rssi, 10.),
        "Bluetooth:Percentile_25": np.percentile(rssi, 25.),
        "Bluetooth:Percentile_40": np.percentile(rssi, 40.),
        "Bluetooth:Mean": rssi.mean(),
        "Bluetooth:Percentile_65": np.percentile(rssi, 65.),
        "Bluetooth:Percentile_80": np.percentile(rssi, 80.),
        "Bluetooth:Percentile_90": np.percentile(rssi, 90.),
        "Bluetooth:Percentile_95": np.percentile(rssi, 95.),
        "Bluetooth:Percentile_99": np.percentile(rssi, 99.),
        "Bluetooth:Max": rssi.max(),
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
