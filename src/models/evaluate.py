import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from src.hyperopt import optimize


def fit_rf(X, y, optimize_params=False, seed=0, model=None):
    if model is None:
        model = RandomForestClassifier(random_state=seed)
    if optimize_params:
        best_estim_ = optimize(model, X, y)
        return best_estim_
    model.fit(X, y)
    return model


def evaluate(model, X, y):
    ypred = model.predict(X)
    return classification_report(y, ypred)


def generate_submission_output(trainset, testset, predictors, target, model=None):
    if model is None:
        model = fit_rf(trainset[predictors], trainset[target])
    ypred = model.predict(testset[predictors])
    testset_system_output = pd.DataFrame(
        {"fileid": testset["fileid"], "distance": ypred})
    testset_system_output.to_csv(
        "data/system_output/test_system_output.tsv", sep="\t", index=False
    )
    return testset_system_output


def dual_evaluation(trainset, testset, predictors, target, seed=0,
                    save_system_output=False, optimize_params=False, model=None):
    report, predictions = {}, {}
    for cg in [0, 1]:
        _trainset = trainset[trainset["CoarseGrain"] == cg]
        _testset = testset[testset["CoarseGrain"] == cg]

        _model = fit_rf(
            _trainset[predictors], _trainset[target], optimize_params=optimize_params,
            seed=seed, model=model)
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
        sysout_dir = "data/system_output/"
        if not os.path.isdir(sysout_dir):
            os.mkdir(sysout_dir)
        devset_system_output.to_csv(
            os.path.join(sysout_dir, "test_system_output.tsv"),
            sep="\t", index=False
        )
    return report, devset_system_output
