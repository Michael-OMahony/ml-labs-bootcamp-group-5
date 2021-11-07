from src.data.common import apply_fn_to_data_files, mitre_dir, read_ble_from_file
import pandas as pd

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
        if "Angle" in line:
            devices["Angle"] = float(line.split(",")[-1])
    devices["Bluetooth"] = read_ble_from_file(fp)
    return devices


def make_data():
    data = apply_fn_to_data_files(mitre_dir, get_headers, as_df=True)
    data.to_csv(MITRE_PROCESSED_DATA, index=False)
    return data


def get_data():
    return pd.read_csv(MITRE_PROCESSED_DATA)
