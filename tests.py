from src.featutils import aggregate_features_from_folder
import src.features.t62020 as t62020
import src.features.baseline as fbase
from src.datapoint import DataPoint
import pandas as pd


def test_datapoint():
    filepath = "data/tc4tl_training_data_v1/tc4tl/data/train/bnqefvro_tc4tl20.csv"
    datapoint = DataPoint(filepath)


def test_20t6():
    folder = "data/tc4tl_training_data_v1/tc4tl/data/train/"
    key = pd.read_csv(
        "data/tc4tl_training_data_v1/tc4tl/docs/tc4tl_train_key.tsv", sep="\t"
    )
    aggregate_features_from_folder(
        folder,
        key,
        feat_fn=t62020.extract_features_from_file,
        postproc_fn=t62020.postproc_feature_frame,
        predictors=t62020.PREDICTORS,
        target=t62020.TARGET,
        testing=True,
    )


def test_devset():
    key = pd.read_csv(
        "data/tc4tl_data_v5/tc4tl/docs/tc4tl_dev_key.tsv", sep="\t")
    folder = "data/tc4tl_data_v5/tc4tl/data/dev/"
    aggregate_features_from_folder(
        folder,
        key,
        feat_fn=t62020.extract_features_from_file,
        postproc_fn=t62020.postproc_feature_frame,
        predictors=t62020.PREDICTORS,
        target=t62020.TARGET,
        testing=True,
    )


def test_baseline():
    folder = "data/tc4tl_training_data_v1/tc4tl/data/train/"
    key = pd.read_csv(
        "data/tc4tl_training_data_v1/tc4tl/docs/tc4tl_train_key.tsv", sep="\t"
    )
    aggregate_features_from_folder(
        folder,
        key,
        feat_fn=fbase.extract_features_from_file,
        postproc_fn=fbase.postproc_feature_dicts,
        predictors=fbase.get_predictors,
        target=fbase.TARGET,
        testing=True,
    )


def test_baseline_dev():
    key = pd.read_csv(
        "data/tc4tl_data_v5/tc4tl/docs/tc4tl_dev_key.tsv", sep="\t")
    folder = "data/tc4tl_data_v5/tc4tl/data/dev/"
    aggregate_features_from_folder(
        folder,
        key,
        feat_fn=fbase.extract_features_from_file,
        postproc_fn=fbase.postproc_feature_dicts,
        predictors=fbase.get_predictors,
        target=fbase.TARGET,
        testing=True,
    )


def test_baseline_train():
    folder = "data/tc4tl_training_data_v1/tc4tl/data/train/"
    key = pd.read_csv(
        "data/tc4tl_training_data_v1/tc4tl/docs/tc4tl_train_key.tsv", sep="\t"
    )
    aggregate_features_from_folder(
        folder,
        key,
        feat_fn=fbase.extract_features_from_file,
        postproc_fn=fbase.postproc_feature_dicts,
        predictors=fbase.get_predictors,
        target=fbase.TARGET,
        testing=True,
    )


def test_ram2021_summary_stats():
    from src.features.ram2021 import summary_stats
    from src.features.common import read_chirp_sequence_from_file

    filepath = "data/tc4tl_training_data_v1/tc4tl/data/train/bnqefvro_tc4tl20.csv"
    chirps = read_chirp_sequence_from_file(filepath)
    print(summary_stats(chirps))


def test_ram2021_histogram():
    from src.features.ram2021 import histogram
    from src.features.common import read_chirp_sequence_from_file

    filepath = "data/tc4tl_training_data_v1/tc4tl/data/train/bnqefvro_tc4tl20.csv"
    chirps = read_chirp_sequence_from_file(filepath)
    feats = histogram(chirps, tunables={"bin_size": 3})
    assert len(feats) == 13
    print(feats)


def test_ram2021():
    from src.features.ram2021 import extract_features

    filepath = "data/tc4tl_training_data_v1/tc4tl/data/train/bnqefvro_tc4tl20.csv"
    key = pd.read_csv(
        "data/tc4tl_training_data_v1/tc4tl/docs/tc4tl_train_key.tsv", sep="\t"
    )
    feats = extract_features(filepath, key)
    print(feats)


def test_ram2021_dev():
    from src.features.ram2021 import extract_features
    from src.features.ram2021 import postproc

    key = pd.read_csv(
        "data/tc4tl_data_v5/tc4tl/docs/tc4tl_dev_key.tsv", sep="\t")
    folder = "data/tc4tl_data_v5/tc4tl/data/dev/"
    aggregate_features_from_folder(
        folder,
        key,
        feat_fn=extract_features,
        postproc_fn=postproc,
        testing=True,
    )
