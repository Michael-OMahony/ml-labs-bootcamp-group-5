import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from src.features.common import read_bluetooth_from_file, to_histogram
from tqdm import tqdm
from sklearn.pipeline import Pipeline


def linear_approximation_model(rssi, TX=-61.02, N=2.187):
    return np.power(10, (TX - rssi) / (10 * N))


# def linear_approx_model(rssi, K, A):
    # RSSI = -K logd + A
    # d = 10^((A - RSSI)/K)
#    return np.power(10, (A - rssi)/K)


def inverted_friis_free_space_model(rssi, Pt_dBm=0., Gt_dBi=1., Gr_dBi=1., f=2.4e9, L=1., n=2.):
    # P_r = RSSI : received signal power (dBm)
    # P_t : power of the transmitted signal (dBm)
    # G_r : gain of receiver antenna (dBi)
    # G_t : gain of transmitter antenna (dBi)
    # f   : Frequency (2.4 GHz) 2402-2480
    # lambda :Wavelength of the carrier (meters) 2.4 GHz => 0.124876 m
    # L   : Other losses (loss at the antenna, transmission line attenuation, loss at various
    #       filters etc.)
    #       L >= 1. | L = 1 => No such loss
    _lambda = 3 * np.power(10, 8) / f
    PL_dB = rssi - Pt_dBm
    return np.power(10, (Gt_dBi + Gr_dBi - PL_dB + 20 * np.log10(_lambda / (4. * np.pi)) - (10. * np.log10(L))) / (10 * n))


def inverted_log_normal_shadowing_model(rssi, Pt_dBm=-20., Gt_dBi=1., Gr_dBi=1.,
                                        f=2.4e9, d0=1., L=1., sigma=2., n=2):
    _lambda = 3 * np.power(10, 8) / f
    K = 20 * np.log10(_lambda / (4 * np.pi)) - \
        (10 * n * np.log10(d0)) - (10 * np.log10(L))
    X = sigma * np.random.randn(len(rssi))
    PL = rssi - Pt_dBm
    d = d0 * np.power(10, (Gt_dBi + Gr_dBi + K - PL)/(10 * n))
    return d


def extract_features(filepath, key, tunables={}):
    rssiv = np.array(read_bluetooth_from_file(filepath))
    rf_prop_models = {
        "LinearApprox": linear_approximation_model,
        "Friis": inverted_friis_free_space_model,
        "LogNormal": inverted_log_normal_shadowing_model
    }
    # if len(tunables) == 0:
    #    tunables = {
    #        "LinearApprox": {"TX": -61.02, "N": 2.187},
    #        "Friis": dict(Pt_dBm=0., Gt_dBi=1., Gr_dBi=1., f=2.4e9, L=1., n=2.),
    #        "LogNormal": dict(t_dBm=-20., Gt_dBi=1., Gr_dBi=1.,
    #                          f=2.4e9, d0=1., L=1., sigma=2., n=2)
    #    }
    features = {}
    for rf_model, fn in rf_prop_models.items():
        features[rf_model] = fn(rssiv, **tunables[rf_model])

    features.update({
        'DistanceFloat': float(key.distance_in_meters),
        'Distance': str(key.distance_in_meters),
        'CoarseGrain': 0 if key.coarse_grain == "Y" else 1,
        'fileid': key.fileid
    })
    return features


def postproc_feature_dicts(feats, pipe=None, tunables={}, verbose=False):
    postproc_features = []
    pbar = tqdm(feats) if verbose else feats
    for feat in pbar:
        row = {}
        for model_name in ["LinearApprox", "Friis", "LogNormal"]:
            _hist = to_histogram(
                feat[model_name], _min=0.1, _max=5., bin_count=50)
            _hist_updated = {
                f"{model_name}:{hkey}": rssi for hkey, rssi in _hist.items()}
            row.update(_hist_updated)
        row.update({
            fkey: feat[fkey] for fkey in ["CoarseGrain", "DistanceFloat"]
        })
        postproc_features.append(row)
        if verbose:
            pbar.set_description(f"Post-processing: {feat['fileid']}")

    df = pd.DataFrame(postproc_features).fillna(0.)
    distance_float = df["DistanceFloat"]
    feat_cols = [col for col in row.keys()]
    if not pipe:
        pipe = Pipeline([("robustScalar", RobustScaler()),
                        ("minMaxScalar", MinMaxScaler())])
    df_scaled = pd.DataFrame(pipe.fit_transform(
        df[feat_cols]), columns=feat_cols)
    df_scaled['DistanceFloat'] = distance_float
    return df_scaled, pipe


def compress_hyperparams(tunables):
    _extracted = {}
    for name, value in tunables.items():
        if "." in name:
            prefix, param_name = name.split(".")
            if prefix not in _extracted:
                _extracted[prefix] = {}
            _extracted[prefix][param_name] = value

    return _extracted
