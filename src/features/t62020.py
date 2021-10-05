import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from src.datapoint import read_non_sensor_data

PREDICTORS = ['NormMeanRssi', 'NormPathLossAttenuation', 'PredictedDistance'] + \
    ['TXPower', 'TXDeviceEnc', 'RXDeviceEnc', 'TXCarryEnc', 'RXCarryEnc',
        'RXPoseEnc', 'TXPoseEnc', 'CoarseGrainEnc']
TARGET = 'Distance'


def distance_from_rssi(rssi, TX, N):
    return np.power(10, (TX - rssi) / (10 * N))


def _extract_features_from_file(filepath, coarse_grain, distance, fileid):
    filedump = open(filepath).read()
    ns = read_non_sensor_data(filedump)
    mean_rssi = np.array([float(line.split(',')[-1])
                         for line in filedump.split('\n')
                         if 'Bluetooth' in line]).mean()
    pla = float(ns['TXPower']) - 41 - mean_rssi
    pred_distance = distance_from_rssi(mean_rssi,
                                       TX=-52 if coarse_grain == 'Y' else -54,
                                       N=2.6 if coarse_grain == 'Y' else 2.1)
    ns.update({
        'PathLossAttenuation': pla,
        'MeanRssi': mean_rssi,
        'CoarseGrain': coarse_grain,
        'PredictedDistance': pred_distance,
        'Distance': str(distance),
        'fileid': fileid
    })
    return ns


def extract_features_from_file(filepath, key):
    return _extract_features_from_file(
        filepath, coarse_grain=key.coarse_grain,
        distance=key.distance_in_meters, fileid=key.fileid)


def postproc_feature_frame(feats, encoders={}):
    if isinstance(feats, type([6, 9])):
        feats = pd.DataFrame(feats)
    # min-max scaling
    if len(encoders) == 0:
        encoders = {
            'MeanRssi': MinMaxScaler(),
            'PathLossAttenuation': MinMaxScaler()
        }
    # TODO: separate fit and transform | include prebuilt encoder
    feats['NormMeanRssi'] = encoders['MeanRssi'].fit_transform(
        feats['MeanRssi'].values.reshape(-1, 1)).reshape(-1)
    feats['NormPathLossAttenuation'] = encoders['PathLossAttenuation'].fit_transform(
        feats['PathLossAttenuation'].values.reshape(-1, 1)).reshape(-1)
    # get categorical columns
    catcols = ['TXDevice', 'RXDevice', 'TXCarry',
               'RXCarry', 'RXPose', 'TXPose', 'CoarseGrain']
    # label encoding
    encoders.update({
        col: LabelEncoder() for col in catcols
    })
    for col in catcols:
        feats[col + 'Enc'] = encoders[col].fit_transform(
            feats[col].values.reshape(-1, 1)).reshape(-1)
    return feats, encoders
