from pathlib import Path
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from joblib import Parallel, delayed
import time
import smote_variants as sv

import datasets
import metrics

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


def evaluate(dataset_name, classifier_name, resampler_name):
    start = time.time()

    scores = np.zeros((len(scoring_functions), 10))
    scores_ = np.zeros((2, len(scoring_functions), 10))

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

            no_results = True

            for solution_id in range(len(JFOTS_results)):
                if JFOTS_results[solution_id][7] is None:
                    pass
                else:
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

            if not no_results:
                X_train = JFOTS_results[max_pr_id][7][0]
                y_train = JFOTS_results[max_pr_id][7][1]

                # Prepare test set with features from feature_mask
                feature_mask = JFOTS_results[max_pr_id][5]
                X_test, y_test = dataset[fold][1]
                X_test = X_test[:, feature_mask]

                classifier = classifiers[classifier_name]
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

                classifier = classifiers[classifier_name]
                clf = classifier.fit(X_train, y_train)
                predictions = clf.predict(X_test)

                for sc_idx, scoring_function_name in enumerate(scoring_functions.keys()):
                    scores_[1, sc_idx, fold] = scoring_functions[scoring_function_name](y_test, predictions)

        else:
            resampler = resamplers[resampler_name]
            X_train, y_train = resampler.sample(X_train, y_train)

            classifier = classifiers[classifier_name]
            clf = classifier.fit(X_train, y_train)
            predictions = clf.predict(X_test)

            for sc_idx, scoring_function_name in enumerate(scoring_functions.keys()):
                scores[sc_idx, fold] = scoring_functions[scoring_function_name](y_test, predictions)

        if resampler_name == "JFOTS":
            for sc_idx, scoring_function_name in enumerate(scoring_functions.keys()):
                fpath1 = result_final_path / f'{dataset_name}' / f'{classifier_name}' / f'{scoring_function_name}'
                fpath1.mkdir(exist_ok=True, parents=True)
                fpath1 = fpath1 / f'{resampler_name}_pr.csv'
                np.savetxt(fpath1, scores_[0, sc_idx, :])

                fpath2 = result_final_path / f'{dataset_name}' / f'{classifier_name}' / f'{scoring_function_name}'
                fpath2.mkdir(exist_ok=True, parents=True)
                fpath2 = fpath2 / f'{resampler_name}_rc.csv'
                np.savetxt(fpath2, scores_[1, sc_idx, :])
        else:
            for sc_idx, scoring_function_name in enumerate(scoring_functions.keys()):
                fpath = result_final_path / f'{dataset_name}' / f'{classifier_name}' / f'{scoring_function_name}'
                fpath.mkdir(exist_ok=True, parents=True)
                fpath = fpath / f'{resampler_name}.csv'
                np.savetxt(fpath, scores[sc_idx, :])


Parallel(n_jobs=4)(
                delayed(evaluate)
                (dataset_name, classifier_name, resampler_name)
                for dataset_name in datasets.names()
                for classifier_name in classifiers.keys()
                for resampler_name in resamplers
                )
