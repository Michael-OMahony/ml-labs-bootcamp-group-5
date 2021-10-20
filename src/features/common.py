import pandas as pd


def read_chirp_sequence_from_file(filepath):
    """Return a sequence of values containing average RSSI reading of each chirp
    (Reads from file directly)
    """
    data, chirps, chirp = [], [], []
    for line in (
        open("data/tc4tl_training_data_v1/tc4tl/data/train/aaadbuvp_tc4tl20.csv")
        .read()
        .split("\n")[7:]
    ):
        if "Bluetooth" in line:
            t, _, rssi = line.split(",")
            item = {"t": float(t), "rssi": float(rssi)}

            if len(data) > 0:
                if float(t) - data[-1]["t"] > 2.0:
                    chirps.append(chirp)
                    chirp = []
            chirp.append(item)
            data.append(item)
    return [pd.DataFrame(chirp) for chirp in chirps]


def postproc_default(feats, encoders={}):
    return pd.DataFrame(feats), encoders
