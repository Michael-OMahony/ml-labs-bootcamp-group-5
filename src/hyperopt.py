from hpsklearn import HyperoptEstimator
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model.ridge import RidgeClassifier
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.utils.testing import ignore_warnings
from tqdm import tqdm

from hyperopt import fmin, tpe

CLASSIFIERS = [
    ExtraTreeClassifier,
    DecisionTreeClassifier,
    MLPClassifier,
    KNeighborsClassifier,
    SGDClassifier,
    RidgeClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    RandomForestClassifier,
]


RF_PARAM_GRID = {
    "bootstrap": [True],
    "max_depth": [80, 90, 100, 110],
    "max_features": [2, 3],
    "min_samples_leaf": [3, 4, 5],
    "min_samples_split": [8, 10, 12],
    "n_estimators": [100, 200, 300, 1000],
}

RFP_PARAMS = {
    "LinearApprox": {"Pt_dBm": -61.02, "n": 2.187},
    "Friis": dict(Pt_dBm=0.0, Gt_dBi=1.0, Gr_dBi=1.0, f=2.4e9, L=1.0, n=2.0),
    "LogNormal": dict(
        Pt_dBm=-20.0, Gt_dBi=1.0, Gr_dBi=1.0, f=2.4e9, d0=1.0, L=1.0, sigma=2.0, n=2
    ),
}


def optimize(model, train_features, train_labels, param_grid=None):
    if param_grid is None:
        if isinstance(model, RandomForestClassifier) or isinstance(
            model, RandomForestRegressor
        ):
            param_grid = RF_PARAM_GRID
    grid_search = GridSearchCV(
        estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2
    )
    grid_search.fit(train_features, train_labels)
    return grid_search.best_estimator_


def optimize_hp(train_features, train_labels):
    # Create the estimator object
    estim = HyperoptEstimator()
    # Search the space of classifiers and preprocessing steps and their
    # respective hyperparameters in sklearn to fit a model to the data
    estim.fit(train_features, train_labels)
    return estim, estim.best_model()


def optimize_preproc(make_data, space):
    @ignore_warnings(category=ConvergenceWarning)
    def evaluate(features, labels, classifiers=[], verbose=False):
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.33, random_state=69
        )
        scores = []
        if len(classifiers) == 0:
            classifiers = CLASSIFIERS
        iterate_through = tqdm(classifiers) if verbose else classifiers
        for _classifier in iterate_through:
            model = _classifier()
            model.fit(X_train, y_train)
            scores.append((model.predict(X_test) == y_test).mean())
            if verbose:
                iterate_through.set_description(f"{_classifier.__name__}: {scores[-1]}")
        return max(scores)

    def objective(params):
        train_features, train_labels = make_data(params)
        return -evaluate(train_features, train_labels)

    best_params = fmin(fn=objective, algo=tpe.suggest, space=space, max_evals=20)

    return best_params
