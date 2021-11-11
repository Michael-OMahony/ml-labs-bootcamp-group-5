import pandas as pd
from src.dataset.common import (apply_fn_to_data_files, matrix_dir,
                                read_ble_from_file, read_sensors_from_file, summarize_sensor)

MATRIX_PROCESSED_DATA = "data/processed/MIT-Matrix-Data.Bluetooth.csv"


def get_headers(fp):
    lines = open(fp).read().split("\n")
    devices = {}
    for line in lines[:10]:
        if "beacon" in line and "beacon_subject" not in line:
            a, b = line.split(",")[-2].strip(), line.split(",")[-1].strip()
            devices["TX"] = b if a == "" else a
            devices["TX"] = devices["TX"].replace(" ", "")  # + ' | ' + line
        if "receiver" in line:
            a, b = line.split(",")[-2].strip(), line.split(",")[-1].strip()
            devices["RX"] = b if a == "" else a
            devices["RX"] = devices["RX"] .replace(" ", "")
        if "environment" in line:
            _, _, e1, e2 = line.split(",")
            devices["Environment_1"] = e1.strip()
            devices["Environment_2"] = e2.strip()
        if "Range" in line:
            devices["Range"] = float(line.split(",")[-1])
    readings = read_sensors_from_file(fp)
    return devices | readings


def make_data():
    data = apply_fn_to_data_files(
        matrix_dir, get_headers, as_df=False)
    non_sensor_cols = ["Environment_1", "Environment_2", "TX", "RX", "Range"]
    sensor_cols = [col for col in data[0].keys() if col not in non_sensor_cols]
    df = pd.DataFrame(data)  # , columns=non_sensor_cols)
    for col in sensor_cols:
        _summary_cols = summarize_sensor(df[col].values, col)
        df = pd.concat([df, pd.DataFrame(_summary_cols)], axis=1)
    df = df.dropna()
    # df.to_csv("data/processed/MIT-Matrix-Data.Bluetooth.csv", index=False)
    return df.drop(sensor_cols, axis=1)


def get_data():
    return pd.read_csv(MATRIX_PROCESSED_DATA)
