import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report


def fit_rf(X, y):
    rf = RandomForestClassifier()
    rf.fit(X, y)
    return rf


def evaluate(model, X, y):
    ypred = model.predict(X)
    return classification_report(y, ypred)


def generate_submission_output(trainset, devset, predictors, target, model=None):
    if model is None:
        model = fit_rf(trainset[predictors], trainset[target])
    ypred = model.predict(devset[predictors])
    devset_system_output = pd.DataFrame({
        'fileid': devset['fileid'],
        'distance': ypred
    })
    devset_system_output.to_csv(
        'data/system_output/dev_system_output.tsv', sep='\t', index=False)
    return devset_system_output


def dual_evaluation(trainset, devset, predictors, target, save_system_output=True):
    report, predictions = {}, {}
    for cg in ['Y', 'N']:
        _trainset = trainset[trainset['CoarseGrain'] == cg]
        _devset = devset[devset['CoarseGrain'] == cg]
        _model = fit_rf(_trainset[predictors], _trainset[target])
        _ypred = _model.predict(_devset[predictors])
        predictions.update({
            fileid: pred for fileid, pred in zip(_devset.fileid.values, _ypred)
        })
        report[f'cg={cg}'] = classification_report(_devset[target], _ypred)
    devset_system_output = pd.DataFrame({
        'fileid': devset['fileid'],
        'distance': devset.apply(lambda row: predictions[row.fileid], axis=1)
    })
    if save_system_output:
        devset_system_output.to_csv(
            'data/system_output/dev_system_output.tsv', sep='\t', index=False)
    return report, devset_system_output
