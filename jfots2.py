from __future__ import annotations  # needed for older version of Python
from collections import Counter
from enum import Enum
from typing import Optional, Type, Union

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling.base import BaseOverSampler
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.algorithm import Algorithm
from pymoo.core.problem import ElementwiseProblem
from pymoo.indicators.hv import Hypervolume
from pymoo.optimize import minimize
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import NearestNeighbors

import metrics
import visualizers as vis
import itertools


class ObservationType(Enum):
    MAJORITY = -1
    SAFE = 0
    BORDERLINE = 1
    RARE = 2
    OUTLIER = 3


def _get_observation_types(
    X: np.ndarray, y: np.ndarray, majority_class: int
) -> np.ndarray:
    result = np.empty(y.shape, dtype=int)

    knn = NearestNeighbors(n_neighbors=6).fit(X)

    for i, (X_i, y_i) in enumerate(zip(X, y)):
        if y_i == majority_class:
            result[i] = ObservationType.MAJORITY.value
        else:
            indices = knn.kneighbors([X_i], return_distance=False)[0, 1:]
            n_majority_neighbors = sum(y[indices] == majority_class)

            if n_majority_neighbors in [0, 1]:
                result[i] = ObservationType.SAFE.value
            elif n_majority_neighbors in [2, 3]:
                result[i] = ObservationType.BORDERLINE.value
            elif n_majority_neighbors == 4:
                result[i] = ObservationType.RARE.value
            elif n_majority_neighbors == 5:
                result[i] = ObservationType.OUTLIER.value
            else:
                raise ValueError

    return result


def _get_feature_and_type_masks(
    individual: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    individual = np.round(individual).astype(bool)

    feature_mask = individual[:-4]
    type_mask = individual[-4:]

    return feature_mask, type_mask


def _use_individual_to_resample_dataset(
    individual: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    *,
    oversampler_class: Type[BaseOverSampler],
    oversampler_kwargs: Optional[dict] = None,
    observation_types: np.ndarray,
    minority_class: int,
    majority_class: int,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    feature_mask, type_mask = _get_feature_and_type_masks(individual)

    # fail to resample if no features or no types were selected
    if sum(feature_mask) == 0 or sum(type_mask) == 0:
        return None

    selected_types = [i for i in range(4) if type_mask[i]]
    unselected_types = [i for i in range(4) if not type_mask[i]]
    selected_minority = X[np.isin(observation_types, selected_types)]
    unselected_minority = X[np.isin(observation_types, unselected_types)]

    # fail to resample if there were no observations from the selected types
    if len(selected_minority) == 0:
        return None

    n_majority = sum(y == majority_class)
    n_minority = sum(y == minority_class)

    assert n_minority == len(selected_minority) + len(unselected_minority)

    sampling_strategy = {
        majority_class: n_majority,
        minority_class: n_majority - len(unselected_minority),
    }

    try:
        if oversampler_kwargs is None:
            oversampler_kwargs = {}

        oversampler = oversampler_class(
            sampling_strategy=sampling_strategy, **oversampler_kwargs
        )

        X_ = np.concatenate([X[y == majority_class], selected_minority])
        y_ = np.concatenate(
            [y[y == majority_class], minority_class * np.ones(len(selected_minority))]
        )

        X_ = X_[:, feature_mask]

        X_, y_ = oversampler.fit_resample(X_, y_)

        X_ = np.concatenate([X_, unselected_minority[:, feature_mask]])
        y_ = np.concatenate([y_, minority_class * np.ones(len(unselected_minority))])

        return X_, y_

    # fail to resample if oversampling algorithm raised any errors
    # (for example, when there was an insufficient number of observations
    # from selected types to create k-neighborhood for SMOTE)
    except ValueError:
        return None


def _get_minority_and_majority_class(y: np.ndarray) -> tuple[int, int]:
    minority_class = Counter(y).most_common()[1][0]
    majority_class = Counter(y).most_common()[0][0]

    return minority_class, majority_class


class _JFOTSProblem(ElementwiseProblem):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        observation_types: np.ndarray,
        estimator: BaseEstimator,
        oversampler_class: Type[BaseOverSampler],
        oversampler_kwargs: Optional[dict] = None,
        n_splits: int,
        type_var: object,
    ):
        self.X = X
        self.y = y
        self.observation_types = observation_types
        self.estimator = estimator
        self.oversampler_class = oversampler_class
        self.oversampler_kwargs = oversampler_kwargs
        self.n_splits = n_splits

        self.folds = []

        for train_index, test_index in RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=5).split(X, y):
            self.folds.append(
                (
                    (X[train_index], y[train_index], observation_types[train_index]),
                    (X[test_index], y[test_index]),
                )
            )

        self.n_variables = X.shape[1] + 4
        self.minority_class, self.majority_class = _get_minority_and_majority_class(y)

        super().__init__(n_var=self.n_variables, n_obj=2, xl=0, xu=1, type_var=type_var)

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        precisions = []
        recalls = []

        feature_mask, _ = _get_feature_and_type_masks(x)

        for (X_train, y_train, observation_types), (X_test, y_test) in self.folds:
            resampled_dataset = _use_individual_to_resample_dataset(
                x,
                X_train,
                y_train,
                oversampler_class=self.oversampler_class,
                oversampler_kwargs=self.oversampler_kwargs,
                observation_types=observation_types,
                minority_class=self.minority_class,
                majority_class=self.majority_class,
            )

            if resampled_dataset is None:
                out["F"] = np.column_stack([1.0, 1.0])

                return

            X_train_, y_train_ = resampled_dataset

            estimator = clone(self.estimator)
            estimator.fit(X_train_, y_train_)

            predictions = estimator.predict(X_test[:, feature_mask])

            precisions.append(metrics.precision(y_test, predictions))
            recalls.append(metrics.recall(y_test, predictions))

        out["F"] = np.column_stack([-np.mean(precisions), -np.mean(recalls)])


class JFOTSSolution:
    def __init__(
        self,
        data: Optional[tuple[np.ndarray, np.ndarray]],
        individual: np.ndarray,
        objectives: np.ndarray,
    ):
        self.data = data
        self.objectives = objectives

        self.feature_mask, self.type_mask = _get_feature_and_type_masks(individual)
        self.used_types = {ObservationType(i): self.type_mask[i] == 1 for i in range(4)}


class JFOTS:
    """
    Joint Feature and Oversampling Type Selection algorithm.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        oversampler_class: Optional[Type[BaseOverSampler]] = None,
        oversampler_kwargs: Optional[dict] = None,
        optimizer_class: Optional[Type[Algorithm]] = None,
        optimizer_kwargs: Optional[dict] = None,
        termination: Optional[tuple] = None,
        n_splits: int = 2,
        type_var: object = float,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        verbose: bool = False,
    ):
        self.estimator = estimator

        if oversampler_class is None:
            self.oversampler_class = SMOTE

            if oversampler_kwargs is None:
                self.oversampler_kwargs = {"k_neighbors": 1}
            else:
                self.oversampler_kwargs = oversampler_kwargs
        else:
            self.oversampler_class = oversampler_class

            if oversampler_kwargs is None:
                self.oversampler_kwargs = {}
            else:
                self.oversampler_kwargs = oversampler_kwargs

        if optimizer_class is None:
            self.optimizer_class = NSGA2
        else:
            self.optimizer_class = optimizer_class

        if optimizer_kwargs is None:
            self.optimizer_kwargs = {}
        else:
            self.optimizer_kwargs = optimizer_kwargs

        self.termination = termination
        self.n_splits = n_splits
        self.type_var = type_var
        self.random_state = random_state
        self.verbose = verbose

        self.type_counts: Optional[dict] = None
        self.objectives: Optional[list] = None
        self.history: Optional[list] = None
        self.solutions: Optional[list] = None

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(set(y)) != 2:
            raise ValueError(
                f"{self.__name__} only supports two-class classification, "
                f"received number of classes: {len(set(y))}."
            )

        if self.random_state is not None:
            np.random.seed(self.random_state)

        minority_class, majority_class = _get_minority_and_majority_class(y)

        observation_types = _get_observation_types(X, y, majority_class=majority_class)

        self.type_counts = {
            ObservationType(i): sum(observation_types == i) for i in range(4)
        }

        problem = _JFOTSProblem(
            X,
            y,
            observation_types=observation_types,
            estimator=self.estimator,
            oversampler_class=self.oversampler_class,
            oversampler_kwargs=self.oversampler_kwargs,
            n_splits=self.n_splits,
            type_var=self.type_var,
        )
        optimizer = self.optimizer_class(**self.optimizer_kwargs)
        result = minimize(
            problem,
            optimizer,
            termination=self.termination,
            seed=self.random_state,
            save_history=True,
            verbose=self.verbose,
        )

        order = np.argsort(result.F[:, 0])

        self.history = [-r.opt.get("F") for r in result.history]
        self.objectives = []
        self.solutions = []

        processed_binarized_individuals = {}

        for individual, objectives in zip(result.X[order], -result.F[order]):
            binarized_individual = tuple(np.round(individual).astype(bool))

            if processed_binarized_individuals.get(binarized_individual) is not None:
                continue

            processed_binarized_individuals[binarized_individual] = True

            data = _use_individual_to_resample_dataset(
                individual,
                X,
                y,
                oversampler_class=self.oversampler_class,
                oversampler_kwargs=self.oversampler_kwargs,
                observation_types=observation_types,
                minority_class=minority_class,
                majority_class=majority_class,
            )

            self.objectives.append(objectives)
            self.solutions.append(JFOTSSolution(data, individual, objectives))

        # Solutions from all population
        pop = result.pop
        pop_solutions = []
        for pop_row in pop.get("X"):
            pop_feature_mask, pop_type_mask = _get_feature_and_type_masks(pop_row)
            ps_row = [list(pop_feature_mask), list(pop_type_mask)]
            ps_row_flatten = list(itertools.chain.from_iterable(ps_row))
            pop_solutions.append(ps_row_flatten)
        self.pop_X = pop_solutions
        self.pop_F = -pop.get("F")

    def hv(self, ref_point=(0.0, 0.0), normalize=False):
        metric = Hypervolume(ref_point=ref_point, normalize=normalize)

        return np.array([metric.do(-f) for f in self.history])

    def plot_hv(self, ref_point=(0.0, 0.0), normalize=False, file_name=None):
        vis.hypervolume(self.hv(ref_point, normalize), file_name)

    def plot_pareto(self, file_name=None):
        vis.pareto_front(self.objectives, ("Precision", "Recall"), file_name)
