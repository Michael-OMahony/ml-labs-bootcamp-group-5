import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from src.hyperopt import optimize


def fit_rf(X, y, optimize_params=False):
    rf = RandomForestClassifier()
    if optimize_params:
        best_estim_ = optimize(rf, X, y)
        return best_estim_
    rf.fit(X, y)
    return rf


def evaluate(model, X, y):
    ypred = model.predict(X)
    return classification_report(y, ypred)


def generate_submission_output(trainset, devset, predictors, target, model=None):
    if model is None:
        model = fit_rf(trainset[predictors], trainset[target])
    ypred = model.predict(devset[predictors])
    devset_system_output = pd.DataFrame(
        {"fileid": devset["fileid"], "distance": ypred})
    devset_system_output.to_csv(
        "data/system_output/dev_system_output.tsv", sep="\t", index=False
    )
    return devset_system_output


def dual_evaluation(trainset, testset, predictors, target,
                    save_system_output=True, optimize_params=False):
    report, predictions = {}, {}
    for cg in [0, 1]:
        _trainset = trainset[trainset["CoarseGrain"] == cg]
        _testset = testset[testset["CoarseGrain"] == cg]

        _model = fit_rf(
            _trainset[predictors], _trainset[target], optimize_params=optimize_params)
        _ypred = _model.predict(_testset[predictors])
        predictions.update(
            {fileid: pred for fileid, pred in zip(
                _testset.fileid.values, _ypred)}
        )
        report[f"cg={cg}"] = classification_report(_testset[target], _ypred)
        report[f"model:cg={cg}"] = _model
        report[f"trainset:cg={cg}"] = _trainset
        report[f"testset:cg={cg}"] = _testset
    devset_system_output = pd.DataFrame(
        {
            "fileid": testset["fileid"],
            "distance": testset.apply(lambda row: predictions[row.fileid], axis=1),
        }
    )
    if save_system_output:
        devset_system_output.to_csv(
            "data/system_output/dev_system_output.tsv", sep="\t", index=False
        )
    return report, devset_system_output


def evaluate_prediction(devset, prediction, save_system_output=True):
    report = classification_report(devset.Distance, prediction)
    system_output = pd.DataFrame(
        {"fileid": devset["fileid"], "distance": prediction})
    if save_system_output:
        system_output.to_csv(
            "data/system_output/dev_system_output.tsv", sep="\t", index=False
        )
    return report, system_output
