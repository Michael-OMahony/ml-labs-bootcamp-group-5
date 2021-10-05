import os

from tqdm import tqdm


def aggregate_features_from_folder(folder, key, feat_fn, postproc_fn, predictors, target):
    features = []
    files = os.listdir(folder)
    for filename in tqdm(files, total=len(files)):
        filepath = os.path.join(folder, filename)
        features.append(feat_fn(filepath, key[key.fileid == filename]))
    df, encoders = postproc_fn(features)
    assert len(predictors) > 0
    assert df[predictors].shape[0] > 0
    assert target in df
    return df[predictors + [target]], encoders