import os
import pandas as pd
from tqdm import tqdm
import numpy as np

dev_dir = "data/tc4tl_data_v5/tc4tl/data/dev/"
test_dir = "data/tc4tl_data_v5/tc4tl/data/test/"
train_dir = "data/tc4tl_training_data_v1/tc4tl/data/train/"
mitre_dir = "data/MITRE-Range-Angle-Structured/"
matrix_dir = "data/MIT-Matrix-Data/"
train_key_fp = "data/tc4tl_training_data_v1/tc4tl/docs/tc4tl_train_key.tsv"
dev_key_fp = "data/tc4tl_data_v5/tc4tl/docs/tc4tl_dev_key.tsv"
test_key_fp = "data/tc4tl_test_key/tc4tl/docs/tc4tl_test_key.tsv"


def read_ble_from_file(fp):
    return np.array(
        [
            float(line.split(",")[3])
            for line in open(fp).read().split("\n")
            if "Bluetooth" in line
        ]
    )


def read_sensors_from_file(fp):
    readings = {}
    lines = open(fp).read().split('\n')[10:]
    for line in lines:
        items = line.split(',')
        anathema = ["Activity", "scenario", "environment", "subject",
                    "session_id", "beacon", "receiver", "user_id", "Range",
                    "Angle", "beacon_subject", "partner_tester", "self_tester",
                    "partner_on_body_location", "partner_pose", "app_name",
                    "app_ver", "self_on_body_location", "self_pose", "self_activity",
                    "partner_activity", "app_name", "self_user_id", "Pedometer",
                    "partner_beacon_id", "self_beacon_id"]
        if not line.strip():
            continue
        if items[1] in anathema:
            continue
        if "Bluetooth" in line:
            if "Bluetooth" not in readings:
                readings["Bluetooth"] = []
            readings["Bluetooth"].append(float(items[3]))
        elif "Heading" in line:
            sensor_type = items[1]
            assert len(items) == 8
            suffixes = ["x1", "y1", "z1", "x2", "y2", "z2"]
            for suffix, scalar in zip(suffixes, items[2:]):
                name = sensor_type + "_" + suffix
                if name not in readings:
                    readings[name] = []
                readings[name].append(float(scalar))
        else:
            suffixes = ["x", "y", "z"]
            sensor_type = items[1]
            for suffix, scalar in zip(suffixes, items[2:]):
                name = sensor_type + "_" + suffix
                if name not in readings:
                    readings[name] = []
                if scalar.strip():
                    try:
                        readings[name].append(float(scalar))
                    except ValueError:
                        print(line)
    return readings


def apply_fn_to_data_files(_dir, fn, as_df=False):
    _apply_out = []
    folders = list(os.walk(_dir))
    for dirpath, dirnames, filenames in tqdm(folders, total=len(folders)):
        for filename in [f for f in filenames if f.endswith(".txt")]:
            fp = os.path.join(dirpath, filename)
            _apply_out.append(fn(fp))
    if as_df:
        return pd.DataFrame(_apply_out)
    return _apply_out


def summarize_sensor(readings, name):
    summary = []
    for event_readings in readings:
        _event_readings = np.array(event_readings)
        summary.append({
            f"{name}:Min": _event_readings.min(),
            f"{name}:Mean": _event_readings.mean(),
            f"{name}:Max": _event_readings.max()
        })
    return summary
