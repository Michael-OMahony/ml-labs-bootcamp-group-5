import pandas as pd

TARGET = "Distance"


def read_chirp_sequence_from_file(filepath, max_chirps=None):
    """Return a sequence of values containing average RSSI reading of each chirp
    (Reads from file directly)
    """
    data, chirps, chirp = [], [], []
    for line in open(filepath).read().split("\n")[7:]:
        if "Bluetooth" in line:
            t, _, rssi = line.split(",")
            item = {"t": float(t), "rssi": float(rssi)}

            if len(data) > 0:
                if float(t) - data[-1]["t"] > 2.0:
                    chirps.append(chirp)
                    chirp = []
            chirp.append(item)
            data.append(item)
        if max_chirps:
            if len(chirps) > max_chirps:
                break
    if len(chirps) == 0:
        if not max_chirps:
            max_chirps = 1
        return [pd.DataFrame(chirp) for _ in range(max_chirps)]
    return [pd.DataFrame(chirp) for chirp in chirps]


def postproc_default(feats, encoders={}):
    return pd.DataFrame(feats), encoders


def get_predictors_default(dataset):
    return [col for col in dataset.columns
            if col not in ['Distance', 'fileid', 'CoarseGrain']]
