import os
import pickle
import types

from tqdm import tqdm


def aggregate_features_from_folder(folder, key, feat_fn,
                                   postproc_fn, predictors=None,
                                   target=None, pipe={}, testing=0,
                                   tunables={}, verbose=False):
    features = []
    # for filename in tqdm(files, total=len(files)):
    sample_size = min(testing, key.shape[0])
    _key = key.sample(sample_size) if testing > 0 else key
    iterate_through = tqdm(
        _key.iterrows(), total=_key.shape[0]) if verbose else _key.iterrows()
    for _, row in iterate_through:
        filepath = os.path.join(folder, row.fileid)
        features.append(feat_fn(filepath, row, tunables=tunables))
    df, pipe = postproc_fn(
        features, pipe=pipe, tunables=tunables, verbose=verbose)
    if isinstance(predictors, type(None)):
        return df, pipe
    if not isinstance(predictors, type([6, 9])):
        assert isinstance(predictors, types.FunctionType)
        predictors = predictors(df)
    assert len(predictors) > 0
    assert df[predictors].shape[0] > 0
    assert target in df
    return df[['fileid'] + predictors + [target]], pipe


def save_encoders(encoders, filepath):
    with open(filepath, 'wb') as fd:
        pickle.dump(encoders, fd)


def pad_devset(devset, trainset):
    for col in trainset.columns:
        if col not in devset:
            devset[col] = trainset[col].min()
    return devset
