import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler
from src.features.common import read_non_sensor_data, read_sensors_from_file


def linear_approximation_model(rssi, TX, N):
    return np.power(10, (TX - rssi) / (10 * N))


def features(fp, key, tunables={}):
    # [0] Non-sensor data
    nsdata = read_non_sensor_data(fp, key)
    # [2] meanRSSI
    rssi_list = read_sensors_from_file(fp)["Bluetooth"]
    rssi = np.array(rssi_list).mean()
    # [1] Predicted Distance
    if key.coarse_grain == "Y":
        params = tunables["cg=Y"]
    else:
        params = tunables["cg=N"]
    predicted_distance = linear_approximation_model(rssi, **params)
    # [3] Path Loss Attenuation
    path_loss = float(nsdata["Cat:TXPower"]) - 41 - rssi

    return {
        "Num:PredictedDistance": predicted_distance,
        "Num:MeanRssi": rssi,
        "Num:PathLossAttenuation": path_loss,
        "CoarseGrain": 0 if key.coarse_grain == "Y" else 1,
        "fileid": key.fileid,
        "Distance": str(key.distance_in_meters),
        "DistanceFloat": float(key.distance_in_meters)
    } | nsdata


def postproc(features, pipe=None, tunables={}, verbose=False):
    df = pd.DataFrame(features).fillna(0.)
    numcols = [col for col in df.columns if "Num:" in col]
    catcols = [col for col in df.columns if "Cat:" in col]
    if not pipe:
        pipe = {
            "Numerical": Pipeline(
                [("robustScalar", RobustScaler()),
                 ("minMaxScalar", MinMaxScaler())]
            ),
        }
        df_num_encoded = pd.DataFrame(pipe["Numerical"].fit_transform(
            df[numcols]), columns=numcols)
        cat_encoder = tunables["CategoricalEncoder"]
        catpipe = cat_encoder(cols=catcols)
        catpipe.fit(df[catcols], df["DistanceFloat"])
        df_cat_encoded = catpipe.transform(df[catcols])
        pipe["Categorical"] = catpipe
        df_encoded = pd.concat([df_num_encoded, df_cat_encoded])
    else:
        df_num_encoded = pd.DataFrame(pipe["Numerical"].transform(
            df[numcols]), columns=numcols)
        catpipe = pipe["Categorical"]
        df_cat_encoded = catpipe.transform(df[catcols])
        df_encoded = pd.concat([df_num_encoded, df_cat_encoded])

    for col in df.columns:
        if col not in numcols + catcols:
            df_encoded[col] = df[col]

    # return df_encoded, pipe
    return df, pipe
