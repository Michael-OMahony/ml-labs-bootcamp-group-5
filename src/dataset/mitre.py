import pandas as pd
from src.dataset.common import (apply_fn_to_data_files, mitre_dir,
                                read_sensors_from_file, summarize_sensor)

MITRE_PROCESSED_DATA = "data/processed/MITRE-range_angle_structured.Bluetooth.csv"


def get_headers(fp):
    lines = open(fp).read().split("\n")
    devices = {}
    for line in lines[:18]:
        if "partner_tester" in line:
            a, b = line.split(",")[-2].strip(), line.split(",")[-1].strip()
            devices["TX"] = b if "iphone" not in a.lower() else a
        if "self_tester" in line:
            if "Clyneice’s iPhone".lower() in line.lower():
                devices["RX"] = line.split(",")[-3].strip()
                continue
            a, b = line.split(",")[-2].strip(), line.split(",")[-1].strip()
            devices["RX"] = b if "iphone" not in a.lower() else a
        if "environment" in line:
            _, _, e1, e2 = line.split(",")
            devices["Environment_1"] = e1.strip()
            devices["Environment_2"] = e2.strip()
        if "Range" in line:
            devices["Range"] = float(line.split(",")[-1])
    readings = read_sensors_from_file(fp)
    return devices | readings


def make_data():
    data = apply_fn_to_data_files(mitre_dir, get_headers, as_df=False)
    non_sensor_cols = ["Environment_1", "Environment_2", "TX", "RX", "Range"]
    sensor_cols = [col for col in data[0].keys() if col not in non_sensor_cols]
    df = pd.DataFrame(data)  # , columns=non_sensor_cols)
    for col in sensor_cols:
        _summary_cols = summarize_sensor(df[col].values, col)
        df = pd.concat([df, pd.DataFrame(_summary_cols)], axis=1)
    df = df.dropna()
    return df.drop(sensor_cols, axis=1)


def get_data():
    return pd.read_csv(MITRE_PROCESSED_DATA)
