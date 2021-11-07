from src.data.common import apply_fn_to_data_files, matrix_dir, read_ble_from_file
import pandas as pd

MATRIX_PROCESSED_DATA = "data/processed/MIT-Matrix-Data.Bluetooth.csv"


def get_headers(fp):
    lines = open(fp).read().split("\n")
    devices = {}
    for line in lines[:10]:
        if "beacon" in line and "beacon_subject" not in line:
            a, b = line.split(",")[-2].strip(), line.split(",")[-1].strip()
            devices["TX"] = b if a == "" else a
            devices["TX"] = devices["TX"]  # + ' | ' + line
        if "receiver" in line:
            a, b = line.split(",")[-2].strip(), line.split(",")[-1].strip()
            devices["RX"] = b if a == "" else a
            devices["RX"] = devices["RX"]  # + ' | ' + line
        if "environment" in line:
            _, _, e1, e2 = line.split(",")
            devices["Environment_1"] = e1.strip()
            devices["Environment_2"] = e2.strip()
        if "Range" in line:
            devices["Range"] = float(line.split(",")[-1])
        if "Angle" in line:
            devices["Angle"] = float(line.split(",")[-1])
    devices["Bluetooth"] = read_ble_from_file(fp)
    return devices


def make_data():
    data = apply_fn_to_data_files(matrix_dir, get_headers, as_df=True)
    data.to_csv("data/processed/MIT-Matrix-Data.Bluetooth.csv", index=False)
    return data


def get_data():
    return pd.read_csv(MATRIX_PROCESSED_DATA)
