from pathlib import Path
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from joblib import Parallel, delayed
import time
import smote_variants as sv
from sklearn.base import clone

from pymoo.factory import get_decomposition

import datasets
import metrics

import logging


RANDOM_STATE = 0
resamplers = {
    'SMOTE': sv.SMOTE(random_state=RANDOM_STATE),
    'polynom-fit-SMOTE': sv.polynom_fit_SMOTE(random_state=RANDOM_STATE),
    'Lee': sv.Lee(random_state=RANDOM_STATE),
    'SMOBD': sv.SMOBD(random_state=RANDOM_STATE),
    'G-SMOTE': sv.G_SMOTE(random_state=RANDOM_STATE),
    'LVQ-SMOTE': sv.LVQ_SMOTE(random_state=RANDOM_STATE),
    'Assembled-SMOTE': sv.Assembled_SMOTE(random_state=RANDOM_STATE),
    'SMOTE-TomekLinks': sv.SMOTE_TomekLinks(random_state=RANDOM_STATE),
    'JFOTS': None
}

classifiers = {
    'CART': DecisionTreeClassifier(random_state=RANDOM_STATE),
    'KNN': KNeighborsClassifier(n_neighbors=3),
    'SVM': SVC(kernel='rbf', random_state=RANDOM_STATE),
    # 'MLP': MLPClassifier(random_state=RANDOM_STATE)
}

scoring_functions = {
    'Precision': metrics.precision,
    'Recall': metrics.recall,
    'AUC': metrics.auc,
    'G-mean': metrics.g_mean,
    'BAC': metrics.bac,
}

# Calculation uni weighted to promethee method


def uni_cal(solutions_col, criteria_min_max, preference_function, criteria_weights):
    uni = np.zeros((solutions_col.shape[0], solutions_col.shape[0]))
    uni_weighted = np.zeros((solutions_col.shape[0], solutions_col.shape[0]))
    for i in range(np.size(uni, 0)):
        for j in range(np.size(uni, 1)):
            if i == j:
                uni[i, j] = 0
            # Usual preference function
            elif preference_function == 'u':
                diff = solutions_col[j] - solutions_col[i]
                if diff > 0:
                    uni[i, j] = 1
                else:
                    uni[i, j] = 0
            uni_weighted[i][j] = criteria_weights * uni[i, j]
    # criteria min (0) or max (1) optimization array
    if criteria_min_max == 0:
        uni_weighted = uni_weighted
    elif criteria_min_max == 1:
        uni_weighted = uni_weighted.T
    return uni_weighted


# promethee method to choose one solution from the pareto front
def promethee_function(solutions, criteria_min_max, preference_function, criteria_weights):
    weighted_unis = []
    for i in range(solutions.shape[1]):
        weighted_uni = uni_cal(solutions[:, i:i + 1], criteria_min_max[i], preference_function[i], criteria_weights[i])
        weighted_unis.append(weighted_uni)
    agregated_preference = []
    uni_acc = weighted_unis[0]
    uni_cost = weighted_unis[1]
    # Combine two criteria into agregated_preference
    for (item1, item2) in zip(uni_acc, uni_cost):
        agregated_preference.append((item1 + item2)/sum(criteria_weights))
    agregated_preference = np.array(agregated_preference)

    n_solutions = agregated_preference.shape[0] - 1
    # Sum by rows - positive flow
    pos_flows = []
    pos_sum = np.sum(agregated_preference, axis=1)
    for element in pos_sum:
        pos_flows.append(element/n_solutions)
    # Sum by columns - negative flow
    neg_flows = []
    neg_sum = np.sum(agregated_preference, axis=0)
    for element in neg_sum:
        neg_flows.append(element/n_solutions)
    # Calculate net_flows
    net_flows = []
    for i in range(len(pos_flows)):
        net_flows.append(pos_flows[i] - neg_flows[i])
    return net_flows


def evaluate(dataset_name, classifier_name, resampler_name):
    _logger = logging.getLogger('smote_variants')
    _logger.setLevel(logging.CRITICAL)

    print(f"[{dataset_name}]: {classifier_name} + {resampler_name}")
    scores = np.zeros((len(scoring_functions), 10))
    scores_ = np.zeros((5, len(scoring_functions), 10))

    for fold in range(10):

        dataset = datasets.load(dataset_name)
        (X_train, y_train), (X_test, y_test) = dataset[fold][0], dataset[fold][1]

        RESULTS_PATH = Path(__file__).parents[0] / 'results_from_server'
        result_final_path = RESULTS_PATH / f'scores'
        result_final_path.mkdir(exist_ok=True, parents=True)

        if resampler_name == "JFOTS":
            result_path = RESULTS_PATH / f'raw' / f'{dataset_name}_{fold}.p'
            JFOTS_results = pickle.load(open(result_path, "rb"))
            max_rc = -10
            max_pr = -10
            max_gm = -10

            no_results = True

            solutions = []
            solutions_scores = []

            for solution_id in range(len(JFOTS_results)):
                if JFOTS_results[solution_id][7] is None:
                    pass
                else:
                    solutions.append(JFOTS_results[solution_id])
                    solutions_scores.append(JFOTS_results[solution_id][4])
                    no_results = False

                    # Best precision
                    pr = JFOTS_results[solution_id][4][0]
                    if pr > max_pr:
                        max_pr = pr
                        max_pr_id = solution_id
                    # Best recall
                    rc = JFOTS_results[solution_id][4][1]
                    if rc > max_rc:
                        max_rc = rc
                        max_rc_id = solution_id
                    # Best Gmean
                    gm = np.sqrt(pr * rc)
                    if gm > max_gm:
                        max_gm = gm
                        max_gm_id = solution_id

            if not no_results:
                X_train = JFOTS_results[max_pr_id][7][0]
                y_train = JFOTS_results[max_pr_id][7][1]

                # Prepare test set with features from feature_mask
                feature_mask = JFOTS_results[max_pr_id][5]
                X_test, y_test = dataset[fold][1]
                X_test = X_test[:, feature_mask]

                classifier = clone(classifiers[classifier_name])
                clf = classifier.fit(X_train, y_train)
                predictions = clf.predict(X_test)

                for sc_idx, scoring_function_name in enumerate(scoring_functions.keys()):
                    scores_[0, sc_idx, fold] = scoring_functions[scoring_function_name](y_test, predictions)

                X_train = JFOTS_results[max_rc_id][7][0]
                y_train = JFOTS_results[max_rc_id][7][1]

                # Prepare test set with features from feature_mask
                feature_mask = JFOTS_results[max_rc_id][5]
                X_test, y_test = dataset[fold][1]
                X_test = X_test[:, feature_mask]

                classifier = clone(classifiers[classifier_name])
                clf = classifier.fit(X_train, y_train)
                predictions = clf.predict(X_test)

                for sc_idx, scoring_function_name in enumerate(scoring_functions.keys()):
                    scores_[1, sc_idx, fold] = scoring_functions[scoring_function_name](y_test, predictions)

                solutions_scores = np.array(solutions_scores)

                # criteria min (0) or max (1) optimization array
                criteria_min_max = ([0, 0])
                criteria_weights = np.array([0.5, 0.5])
                # u - usual
                preference_function = (['u', 'u'])
                net_flows = promethee_function(solutions_scores, criteria_min_max, preference_function, criteria_weights)
                promethee_id = np.argmax(net_flows, axis=0)

                X_train = solutions[promethee_id][7][0]
                y_train = solutions[promethee_id][7][1]

                # Prepare test set with features from feature_mask
                feature_mask = solutions[promethee_id][5]
                X_test, y_test = dataset[fold][1]
                X_test = X_test[:, feature_mask]
                classifier = clone(classifiers[classifier_name])
                clf = classifier.fit(X_train, y_train)
                predictions = clf.predict(X_test)

                for sc_idx, scoring_function_name in enumerate(scoring_functions.keys()):
                    scores_[2, sc_idx, fold] = scoring_functions[scoring_function_name](y_test, predictions)

                X_train = JFOTS_results[max_gm_id][7][0]
                y_train = JFOTS_results[max_gm_id][7][1]

                # Prepare test set with features from feature_mask
                feature_mask = JFOTS_results[max_gm_id][5]
                X_test, y_test = dataset[fold][1]
                X_test = X_test[:, feature_mask]

                classifier = clone(classifiers[classifier_name])
                clf = classifier.fit(X_train, y_train)
                predictions = clf.predict(X_test)

                for sc_idx, scoring_function_name in enumerate(scoring_functions.keys()):
                    scores_[3, sc_idx, fold] = scoring_functions[scoring_function_name](y_test, predictions)

                # ASF
                weights = np.array([0.5, 0.5])
                decomp = get_decomposition("asf")
                asf_sol = decomp.do(solutions_scores, weights).argmin()

                X_train = JFOTS_results[asf_sol][7][0]
                y_train = JFOTS_results[asf_sol][7][1]

                # Prepare test set with features from feature_mask
                feature_mask = JFOTS_results[asf_sol][5]
                X_test, y_test = dataset[fold][1]
                X_test = X_test[:, feature_mask]

                classifier = clone(classifiers[classifier_name])
                clf = classifier.fit(X_train, y_train)
                predictions = clf.predict(X_test)

                for sc_idx, scoring_function_name in enumerate(scoring_functions.keys()):
                    scores_[4, sc_idx, fold] = scoring_functions[scoring_function_name](y_test, predictions)

        else:
            resampler = resamplers[resampler_name]
            X_train, y_train = resampler.sample(X_train, y_train)

            classifier = clone(classifiers[classifier_name])
            clf = classifier.fit(X_train, y_train)
            predictions = clf.predict(X_test)

            for sc_idx, scoring_function_name in enumerate(scoring_functions.keys()):
                scores[sc_idx, fold] = scoring_functions[scoring_function_name](y_test, predictions)

        if resampler_name == "JFOTS":
            for sc_idx, scoring_function_name in enumerate(scoring_functions.keys()):
                fpath = result_final_path / f'{dataset_name}' / f'{classifier_name}' / f'{scoring_function_name}'
                fpath.mkdir(exist_ok=True, parents=True)

                fpath_pr = fpath / f'{resampler_name}_pr.csv'
                np.savetxt(fpath_pr, scores_[0, sc_idx, :])

                fpath_rc = fpath / f'{resampler_name}_rc.csv'
                np.savetxt(fpath_rc, scores_[1, sc_idx, :])

                fpath_pro = fpath / f'{resampler_name}_prom.csv'
                np.savetxt(fpath_pro, scores_[2, sc_idx, :])

                fpath_pro = fpath / f'{resampler_name}_gm.csv'
                np.savetxt(fpath_pro, scores_[3, sc_idx, :])

                fpath_pro = fpath / f'{resampler_name}_asf.csv'
                np.savetxt(fpath_pro, scores_[4, sc_idx, :])
        else:
            for sc_idx, scoring_function_name in enumerate(scoring_functions.keys()):
                fpath = result_final_path / f'{dataset_name}' / f'{classifier_name}' / f'{scoring_function_name}'
                fpath.mkdir(exist_ok=True, parents=True)
                fpath = fpath / f'{resampler_name}.csv'
                np.savetxt(fpath, scores[sc_idx, :])


Parallel(n_jobs=-1)(
                delayed(evaluate)
                (dataset_name, classifier_name, resampler_name)
                for dataset_name in datasets.names()
                for classifier_name in classifiers.keys()
                for resampler_name in resamplers
                )
