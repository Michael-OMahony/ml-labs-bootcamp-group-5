import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

TARGET = 'Distance'
MAX_CHIRPS = 32


def get_predictors(dataset):
    cols = dataset.columns
    return [col for col in cols if 'NormRssi' in col or 'CoarseGrainEnc' in col]


def get_unnormalized_predictors(dataset):
    cols = []
    for col in dataset.columns:
        if not 'Norm' in col and 'Rssi' in col:
            cols.append(col)
    return cols


def read_bluetooth_from_file(filepath):
    """Read bluetooth RSSI readings from file"""
    return pd.DataFrame([{'Time': float(line.split(',')[0]), 'Rssi': float(line.split(',')[-1])}
                         for line in open(filepath).read().split('\n')
                         if 'Bluetooth' in line])


def chirp_sequence(df):
    """Return a sequence of values containing average RSSI reading of each chirp"""
    chirps, rssi = [], []
    for idx, t in df.iterrows():
        if idx > 0 and t.Time - df.Time.loc[idx - 1] > 1:
            if len(rssi) > 0:
                chirps.append(np.array(rssi).mean())
            rssi = []
        else:
            rssi.append(t.Rssi)
    if len(rssi) > 0:
        chirps = chirps + [np.array(rssi).mean(), ]

    if len(chirps) > MAX_CHIRPS:    # NOTE: Is this a bad idea?
        return chirps[:MAX_CHIRPS]  # we will never know.

    return chirps


def extract_features_from_file(filepath, key):
    ble = read_bluetooth_from_file(filepath)
    chirps = chirp_sequence(ble)
    return {
        'fileid': key.fileid,
        'chirps': chirps,
        'Distance': str(key.distance_in_meters),
        'CoarseGrain': key.coarse_grain
    }


def postproc_feature_dicts(feats, encoders={}):
    assert isinstance(feats, type([6, 9]))
    assert isinstance(feats[0], type({6: 9, }))
    maxchirps = max([len(feat['chirps']) for feat in feats])
    if len(encoders) == 0:
        encoders = {
            'Rssi': MinMaxScaler(),
            'CoarseGrain': LabelEncoder()
        }
    rssi_values, featframe = [], []
    for feat in feats:
        chirps = feat['chirps']
        rssi_values.extend(chirps)
        rssi_cols = chirps + [0.]*(maxchirps - len(chirps))
        row = {f'Rssi{i}': rssi for i, rssi in enumerate(rssi_cols)}
        row.update(feat)
        del row['chirps']
        featframe.append(row)
    # make data frame
    featframe = pd.DataFrame(featframe)
    # fit Min-Max scalar
    encoders['Rssi'].fit(np.array([0.] + rssi_values).reshape(-1, 1))
    # fit Label Encoder
    encoders['CoarseGrain'].fit(featframe.CoarseGrain.values.reshape(-1, 1))
    # encode columns
    for col in featframe.columns:
        if 'Rssi' in col:
            featframe['Norm' + col] = encoders['Rssi'].transform(
                featframe[col].values.reshape(-1, 1)).reshape(-1)
        if col == 'CoarseGrain':
            featframe[col + 'Enc'] = encoders[col].transform(
                featframe[col].values.reshape(-1, 1)).reshape(-1)
    return featframe, encoders


def postproc_feature_dicts_unnormed(feats, encoders={}):
    assert isinstance(feats, type([6, 9]))
    assert isinstance(feats[0], type({6: 9, }))
    maxchirps = max([len(feat['chirps']) for feat in feats])

    rssi_values, featframe = [], []
    for feat in feats:
        chirps = feat['chirps']
        rssi_values.extend(chirps)
        rssi_cols = chirps + [0.]*(maxchirps - len(chirps))
        row = {f'Rssi{i}': rssi for i, rssi in enumerate(rssi_cols)}
        row.update(feat)
        del row['chirps']
        featframe.append(row)
    # make data frame
    featframe = pd.DataFrame(featframe)
    return featframe, None
