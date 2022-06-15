import os
import pandas as pd

import numpy as np
from scipy.stats import mode
import pickle

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestNeighbors

from imblearn.over_sampling import SMOTE

from sklearn.cluster import MeanShift


classifiers = {
    'CART': DecisionTreeClassifier(random_state=0),
    'KNN': KNeighborsClassifier(n_neighbors=3),
    'SVM': SVC(kernel='rbf', random_state=0),
    # 'MLP': MLPClassifier(random_state=RANDOM_STATE)
}

from sklearn.metrics import f1_score, precision_score, recall_score,  balanced_accuracy_score, roc_auc_score

from sklearn.base import clone
from datasets import load
from enum import Enum


# from tqdm import tqdm

import argparse

# file_list = [
#     "ecoli-0-1-3-7_vs_2-6", "glass-0-1-6_vs_2", "glass-0-1-6_vs_5", "glass2",
#              "glass4", "glass5", "page-blocks-1-3_vs_4", "yeast-0-5-6-7-9_vs_4", "yeast-1-2-8-9_vs_7",
#              "yeast-1-4-5-8_vs_7", "yeast-1_vs_7", "yeast-2_vs_4", "yeast-2_vs_8", "yeast4", "yeast5", "yeast6",
#              "cleveland-0_vs_4", "ecoli-0-1-4-7_vs_2-3-5-6",
#               "ecoli-0-1_vs_2-3-5", "ecoli-0-2-6-7_vs_3-5",
#              "ecoli-0-6-7_vs_3-5", "ecoli-0-6-7_vs_5", "glass-0-1-4-6_vs_2", "glass-0-1-5_vs_2",
#              "yeast-0-2-5-6_vs_3-7-8-9", "yeast-0-3-5-9_vs_7-8",
#              "abalone-17_vs_7-8-9-10", "abalone-19_vs_10-11-12-13",
#              "abalone-20_vs_8-9-10", "abalone-21_vs_8", "flare-F", "kddcup-buffer_overflow_vs_back",
#              "kddcup-rootkit-imap_vs_back",
#     "kr-vs-k-zero_vs_eight", "poker-8-9_vs_5", "poker-8-9_vs_6", "poker-8_vs_6",
#              "poker-9_vs_7", "winequality-red-3_vs_5", "winequality-red-4",
#              "winequality-red-8_vs_6-7",
#              "winequality-red-8_vs_6", "winequality-white-3-9_vs_5", "winequality-white-3_vs_7",
#              "winequality-white-9_vs_4","zoo-3", "ecoli1", "ecoli2", "ecoli3", "glass0", "glass1", "haberman",
#              "page-blocks0",
#     "pima", "vehicle1", "vehicle3", "yeast1", "yeast3", "abalone19",
#              "abalone9-18"
# ]

# file_list = ['glass1', 'pima', 'yeast1', 'haberman', 'vehicle1', 'vehicle3', 'ecoli1', 'yeast3', 'page-blocks0', 'yeast-0-2-5-6_vs_3-7-8-9']

file_list = ['ecoli-0-1-3-7_vs_2-6', 'glass2', 'vehicle3', 'yeast-1_vs_7', 'zoo-3']

metrics = [balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score]

# results_path = os.path.join('results', 'experiment_dt_moo_delete_for_ensemble')


class VotingPretrainedEnsemble:

    def __init__(self, estimators, classes):
        self.ensemble_ = estimators[estimators != np.array(None)]
        self.classes_ = classes

    def predict(self, X):
        predictions = np.array([member_clf.predict(X) for member_clf in self.ensemble_])
        prediction = np.squeeze(mode(predictions, axis=0)[0])

        return self.classes_[prediction.astype(int)]


def get_solutions(estimators, objectives, threshold):
    # i_balanced = objectives.index(min(objectives, key=lambda i: abs(i[0] - i[1])))
    i_balanced = np.argmin(abs(objectives[:,0] - objectives[:,1]))

    obj1_c = objectives[i_balanced, 0]
    obj2_c = objectives[i_balanced, 1]

    return estimators[np.where((abs(objectives[:, 0] - obj1_c) <= threshold) & (abs(objectives[:, 1] - obj2_c) <= threshold))]


def get_solutions_from_file(path, dataset, fold, classifier_name,  pareto_front):

    if pareto_front:
        solutions = pd.read_csv(os.path.join(path, 'solution_masks', f'{dataset}_{fold}_{classifier_name}_sol_X.csv'), header=None, sep=' ').values
        objectives = pd.read_csv(os.path.join(path, 'solution_masks', f'{dataset}_{fold}_{classifier_name}_sol_F.csv'), header=None, sep=' ').values
    else:
        solutions = pd.read_csv(os.path.join(path, 'population', f'{dataset}_{fold}_{classifier_name}_pop_X.csv'), header=None, sep=' ').values
        objectives = pd.read_csv(os.path.join(path, 'population', f'{dataset}_{fold}_{classifier_name}_pop_F.csv'), header=None, sep=' ').values

    return solutions, objectives

class ObservationType(Enum):
    MAJORITY = -1
    SAFE = 0
    BORDERLINE = 1
    RARE = 2
    OUTLIER = 3


class FeatureSubspaceClassifierWrapper():

    def __init__(self, classifier, column_mask):
        self._classifier = classifier
        self._col_mask = column_mask

    def fit(self, X, y):
        self._classifier.fit(X, y)

    def predict_proba(self, X):
        return self._classifier.predict_proba(X[:, self._col_mask])

    def predict(self, X):
        return self._classifier.predict(X[:,self._col_mask])

    def get_type(self):
        return type(self._classifier)


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
    individual,
    X,
    y: np.ndarray,
    oversampler,
    oversampler_kwargs,
    observation_types,
    minority_class,
    majority_class):

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

        oversampler = SMOTE(
            sampling_strategy=sampling_strategy, random_state=0, k_neighbors=1)

        X_ = np.concatenate([X[y == majority_class], selected_minority])
        y_ = np.concatenate(
            [y[y == majority_class], minority_class * np.ones(len(selected_minority))]
        )

        X_ = X_[:, feature_mask]

        X_, y_ = oversampler.fit_resample(X_, y_)

        X_ = np.concatenate([X_, unselected_minority[:, feature_mask]])
        y_ = np.concatenate([y_, minority_class * np.ones(len(unselected_minority))])

        return X_, y_
    except Exception as e:
        return None


def get_ensemble(base_classifier, solutions, X, y):

    ensemble = []

    classes = np.unique(y)
    sizes = [sum(y == c) for c in classes]
    majority_class = classes[np.argmax(sizes)]
    minority_class = classes[np.argmin(sizes)]


    for solution in solutions:
        resampled_data = _use_individual_to_resample_dataset(solution,X, y, SMOTE(random_state=0, k_neighbors=1), None,
                                                             _get_observation_types(X, y, majority_class),
                                                             minority_class, majority_class)
        if resampled_data is not None:
            clf = clone(base_classifier)
            clf = FeatureSubspaceClassifierWrapper(clf, _get_feature_and_type_masks(solution)[0])
            clf.fit(resampled_data[0], resampled_data[1])
            ensemble.append(clf)
        else:
            ensemble.append(None)

    return np.array(ensemble)


def pick_ensemble(ensembles,val_X, val_y, criteria):
    criteria_values = [criteria(val_y, e.predict(val_X)) for e in ensembles]
    return ensembles[criteria_values.index(max(criteria_values))]


def conduct_experiments_for_one_dataset(results_path, file, thresholds, pareto_front=True):
    print(file)
    folds = load(file)
    try:
        os.mkdir(os.path.join(results_path, 'ensembles_prediction', file))
    except:
        pass
    metrics_array = np.zeros((10, len(metrics)))
    # for i, fold in tqdm(enumerate(folds), total=len(folds)):
    for i, fold in enumerate(folds):
        try:
            os.mkdir(os.path.join(results_path, 'ensembles_prediction', file, f'fold_{i}'))
        except:
            pass
        for classifier in classifiers.keys():
            base_classifier = classifiers[classifier]
            solutions, objectives = get_solutions_from_file(results_path, file, i, classifier, pareto_front)
            estimators = get_ensemble(base_classifier, solutions, fold[0][0], fold[0][1])
            objectives = objectives[estimators != np.array(None)]
            estimators = estimators[estimators != np.array(None)]
            all_ensembles = []
            for threshold in thresholds:
                selected_estimators = get_solutions(estimators, objectives, threshold)
                ensemble = VotingPretrainedEnsemble(selected_estimators,
                                                        np.unique(fold[0][1]))
                all_ensembles.append(ensemble)
            # chosen_ensembles = [pick_ensemble(all_ensembles, fold[0][0], fold[0][1], metric) for metric in metrics]
            chosen_ensembles = all_ensembles
            try:
                os.mkdir(os.path.join(results_path, 'ensembles_prediction', file, f'fold_{i}', classifier))
            except:
                pass
            for im, ensemble in enumerate(chosen_ensembles):
                y_pred = ensemble.predict(fold[1][0])
                df = pd.DataFrame(y_pred)
                df.to_csv(os.path.join(results_path, 'ensembles_prediction', file, f'fold_{i}', classifier, f"ensemble_threshold_{round(thresholds[im],2)}_{'population' if not pareto_front else 'pareto_front'}.csv"))
    #     for im, metric in enumerate(metrics):
    #         metrics_array[i, im] = metric(fold[1][1], y_pred)
    #
    # values = np.mean(metrics_array, axis=0)
    # std = np.std(metrics_array, axis=0)
    # return values, std


def conduct_experiment(results_path, dataset, pareto):
    thresholds = np.arange(0.0, 1.01, 0.05)
    try:
        os.mkdir(os.path.join(results_path, 'ensembles_prediction'))
    except:
        pass
    pareto_front = [False]

    metrics_dfs = [pd.DataFrame(index=file_list) for metric in metrics]


    for dataset in file_list:
    # algorithm_name = f"{t}_{'no' if not pareto else ''}_pareto_front"
    #     try:
        conduct_experiments_for_one_dataset(results_path, dataset, thresholds, pareto)
        # except Exception as e:
        #     print("!!!!!!!!!!!!!!Blad w datasecie {}!!!!!!!!!!!!!!!!".format(dataset))
        #     print(e)
    # for im, metric in enumerate(metrics):
    #     metrics_dfs[im].at[file, algorithm_name] = values[im]
    #     metrics_dfs[im].at[file, algorithm_name + '_std'] = std[im]
    #     metrics_dfs[im].to_csv(os.path.join(results_path, 'results_std_{}.csv'.format(metric.__name__)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("result_directory")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--pareto", dest='pareto_front', action='store_true')
    parser.set_defaults(transformed=False)
    args = parser.parse_args()
    conduct_experiment(args.result_directory, args.dataset, args.pareto_front)
    # conduct_experiment()


