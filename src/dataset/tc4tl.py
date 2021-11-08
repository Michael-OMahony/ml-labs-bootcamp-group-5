from os import pipe
from src.featutils import aggregate_features_from_folder
from src.dataset.common import train_dir, test_dir, dev_dir
from src.dataset.common import train_key_fp, test_key_fp, dev_key_fp
import pandas as pd


def make_datasets(feature_fn, postproc_fn, tunables, verbose=False, testing=0):
    def make_dataset(folder, key, pipe=None):
        return aggregate_features_from_folder(
            folder,
            key,
            feature_fn,
            postproc_fn,
            pipe=pipe,
            tunables=tunables,
            verbose=verbose,
            testing=testing,
        )

    train, pipe = make_dataset(train_dir, pd.read_csv(train_key_fp, sep="\t"))
    dev, _ = make_dataset(dev_dir, pd.read_csv(dev_key_fp, sep="\t"), pipe=pipe)
    test, _ = make_dataset(test_dir, pd.read_csv(test_key_fp, sep="\t"), pipe=pipe)

    return train, dev, test
