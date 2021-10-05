from src.featutils import aggregate_features_from_folder
import src.features.t62020 as t62020
from src.datapoint import DataPoint
import pandas as pd


def test_datapoint():
    filepath = "data/tc4tl_training_data_v1/tc4tl/data/train/bnqefvro_tc4tl20.csv"
    datapoint = DataPoint(filepath)


def test_20t6():
    folder = "data/tc4tl_training_data_v1/tc4tl/data/train/"
    key = pd.read_csv(
        "data/tc4tl_training_data_v1/tc4tl/docs/tc4tl_train_key.tsv", sep="\t")
    aggregate_features_from_folder(
        folder, key, feat_fn=t62020.extract_features_from_file,
        postproc_fn=t62020.postproc_feature_frame)
