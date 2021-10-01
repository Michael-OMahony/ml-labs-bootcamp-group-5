from src.datapoint import DataPoint


def test_datapoint():
    filepath = "data/tc4tl_training_data_v1/tc4tl/data/train/bnqefvro_tc4tl20.csv"
    datapoint = DataPoint(filepath)
