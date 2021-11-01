import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from src.features.radioprop import extract_features_from_array, postproc_row


MITRE_DIR = "data/MITRE-Range-Angle-Structured/"
MATRIX_DIR = "data/MIT-Matrix-Data/"
MITRE_PROCESSED_DATA = "data/processed/MITRE-range_angle_structured.Bluetooth.csv"
MATRIX_PROCESSED_DATA = "data/processed/MIT-Matrix-Data.Bluetooth.csv"


def read_ble_from_file(fp):
    return np.array([float(line.split(',')[3])
                     for line in open(fp).read().split('\n')
                     if 'Bluetooth' in line])


def apply_fn_to_data_files(dir_, fn, as_df=False):
    _apply_out = []
    folders = list(os.walk(dir_))
    for dirpath, dirnames, filenames in tqdm(folders, total=len(folders)):
        for filename in [f for f in filenames if f.endswith(".txt")]:
            fp = os.path.join(dirpath, filename)
            _apply_out.append(fn(fp))
    if as_df:
        return pd.DataFrame(_apply_out).dropna()
    return _apply_out


def apply_fn_to_bluetooth(row, tunables):
    if isinstance(row.Bluetooth, type("6, 9")):
        _array = np.array([float(value.strip())
                          for value in row.Bluetooth[1:-1].split(',')])
    else:
        _array = row.Bluetooth
    feats = extract_features_from_array(_array, tunables=tunables)
    feats = postproc_row(feats)
    feats.update({
        "CoarseGrain": 1.,
        "DistanceFloat": row.Range
    })
    return feats


def mitre():

    def _headers(fp):
        lines = open(fp).read().split('\n')
        devices = {}
        for line in lines[:18]:
            if 'partner_tester' in line:
                a, b = line.split(',')[-2].strip(), line.split(',')[-1].strip()
                devices['TX'] = b if 'iphone' not in a.lower() else a
            if 'self_tester' in line:
                if "Clyneice’s iPhone".lower() in line.lower():
                    devices['RX'] = line.split(',')[-3].strip()
                    continue
                a, b = line.split(',')[-2].strip(), line.split(',')[-1].strip()
                devices['RX'] = b if 'iphone' not in a.lower() else a
            if 'environment' in line:
                _, _, e1, e2 = line.split(',')
                devices['Environment_1'] = e1.strip()
                devices['Environment_2'] = e2.strip()
            if 'Range' in line:
                devices['Range'] = float(line.split(',')[-1]) / 3.281
            if 'Angle' in line:
                devices['Angle'] = float(line.split(',')[-1])
        devices['Bluetooth'] = read_ble_from_file(fp)
        return devices

    data = apply_fn_to_data_files(MITRE_DIR, _headers, as_df=True)
    data.to_csv(MITRE_PROCESSED_DATA, index=False)
    return data


def matrix():

    def _headers(fp):
        lines = open(fp).read().split('\n')
        devices = {}
        for line in lines[:10]:
            if 'beacon' in line and 'beacon_subject' not in line:
                a, b = line.split(',')[-2].strip(), line.split(',')[-1].strip()
                devices['TX'] = b if a == '' else a
                devices['TX'] = devices['TX']  # + ' | ' + line
            if 'receiver' in line:
                a, b = line.split(',')[-2].strip(), line.split(',')[-1].strip()
                devices['RX'] = b if a == '' else a
                devices['RX'] = devices['RX']  # + ' | ' + line
            if 'environment' in line:
                _, _, e1, e2 = line.split(',')
                devices['Environment_1'] = e1.strip()
                devices['Environment_2'] = e2.strip()
            if 'Range' in line:
                devices['Range'] = float(line.split(',')[-1]) / 3.281
            if 'Angle' in line:
                devices['Angle'] = float(line.split(',')[-1])
        devices['Bluetooth'] = read_ble_from_file(fp)
        return devices

    data = apply_fn_to_data_files(MATRIX_DIR, _headers, as_df=True)
    data.to_csv(MATRIX_PROCESSED_DATA, index=False)
    return data
