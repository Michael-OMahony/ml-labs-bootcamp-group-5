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
