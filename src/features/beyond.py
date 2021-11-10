import numpy as np
from src.features.basic import bluetooth_extended_summary
from src.features.common import read_sensors_from_file


def sensor_summary(fp, key, tunables={}):
    readings = read_sensors_from_file(fp)
    features = bluetooth_extended_summary(np.array(readings["Bluetooth"]))
    # remove bluetooth from readings
    del readings["Bluetooth"]
    for sensor in readings:
        features[f"{sensor}:Min"] = np.array(readings[sensor]).min()
        features[f"{sensor}:Mean"] = np.array(readings[sensor]).mean()
        features[f"{sensor}:Max"] = np.array(readings[sensor]).max()
    features.update({
        "CoarseGrain": 0 if key.coarse_grain == "Y" else 1,
        "fileid": key.fileid,
        "Distance": str(key.distance_in_meters),
        "DistanceFloat": float(key.distance_in_meters),
    })
    return features
