import argparse
import logging
from pathlib import Path
import pickle
import sys
import numpy as np
from imblearn.over_sampling import SMOTE
from pymoo.algorithms.moo.nsga2 import NSGA2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from joblib import Parallel, delayed
import time
import smote_variants as sv

from jfots import JFOTS
import datasets
import metrics

# Settings for printing whole array without comprehension
# np.set_printoptions(threshold=sys.maxsize)

logging.basicConfig(filename='textinfo/experiment.log', filemode="a", format='%(asctime).s - %(levelname)s: %(message)s', level='DEBUG')

RANDOM_STATE = 0
resamplers = {
    # 'SMOTE': sv.SMOTE(random_state=RANDOM_STATE),
    # 'polynom-fit-SMOTE': sv.polynom_fit_SMOTE(random_state=RANDOM_STATE),
    # 'Lee': sv.Lee(random_state=RANDOM_STATE),
    # 'SMOBD': sv.SMOBD(random_state=RANDOM_STATE),
    # 'G-SMOTE': sv.G_SMOTE(random_state=RANDOM_STATE),
    # 'LVQ-SMOTE': sv.LVQ_SMOTE(random_state=RANDOM_STATE),
    # 'Assembled-SMOTE': sv.Assembled_SMOTE(random_state=RANDOM_STATE),
    # 'SMOTE-TomekLinks': sv.SMOTE_TomekLinks(random_state=RANDOM_STATE),
    'JFOTS': None
}


# def evaluate(fold, dataset_name, resampler_name):
def evaluate(fold, dataset_name):
    print(f'START: {fold}, {dataset_name}')
    logging.info(f'START - {fold}, {dataset_name}')
    start = time.time()

    classifiers = {
        'CART': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'KNN': KNeighborsClassifier(n_neighbors=3),
        'SVM': SVC(kernel='rbf', random_state=RANDOM_STATE),
        # 'MLP': MLPClassifier(random_state=RANDOM_STATE)
    }

    dataset = datasets.load(dataset_name)
    (X_train, y_train), (X_test, y_test) = dataset[fold][0], dataset[fold][1]

    RESULTS_PATH = Path(__file__).parents[0] / 'results_from_server'
    scoring_functions = {
        'Precision': metrics.precision,
        'Recall': metrics.recall,
        'AUC': metrics.auc,
        'G-mean': metrics.g_mean,
        'BAC': metrics.bac,
    }
    result_final_path = RESULTS_PATH / f'scores'
    result_final_path.mkdir(exist_ok=True, parents=True)

    rows = []
    for resampler_name in resamplers:
        if resampler_name == "JFOTS":
            # Load X_train and y_train from pickle, where each row is one solution from pareto front and columns are: [dataset_name, fold, classifier_name, "JFOTS", solution.objectives, solution.feature_mask, solution.type_mask, solution.data]
            result_path = RESULTS_PATH / f'raw' / f'{dataset_name}_{fold}.p'
            JFOTS_results = pickle.load(open(result_path, "rb"))
            # rows = []
            # print(JFOTS_results, len(JFOTS_results))
            for solution_id in range(len(JFOTS_results)):
                if JFOTS_results[solution_id][7] is None:
                    row = [dataset_name, fold, JFOTS_results[solution_id][2], resampler_name, solution_id, "None", -1]
                    rows.append(row)
                else:
                    X_train = JFOTS_results[solution_id][7][0]
                    y_train = JFOTS_results[solution_id][7][1]
                    # Prepare test set with features from feature_mask
                    feature_mask = JFOTS_results[solution_id][5]
                    X_test, y_test = dataset[fold][1]
                    X_test = X_test[:, feature_mask]
                    for classifier_name in classifiers.keys():
                        classifier = classifiers[classifier_name]
                        clf = classifier.fit(X_train, y_train)
                        predictions = clf.predict(X_test)

                        for scoring_function_name in scoring_functions.keys():
                            score = scoring_functions[scoring_function_name](y_test, predictions)
                            row = [dataset_name, fold, classifier_name, resampler_name, solution_id, scoring_function_name, score]
                            rows.append(row)
                # print(rows)
                result_final_path_file = result_final_path / f'{dataset_name}_{fold}_JFOTS.p'
                # pickle.dump(rows, open(result_final_path_file, "wb"))
        else:
            resampler = resamplers[resampler_name]
            X_train, y_train = resampler.sample(X_train, y_train)

            for classifier_name in classifiers.keys():
                classifier = classifiers[classifier_name]
                clf = classifier.fit(X_train, y_train)
                predictions = clf.predict(X_test)

                for scoring_function_name in scoring_functions.keys():
                    score = scoring_functions[scoring_function_name](y_test, predictions)
                    row = [dataset_name, fold, classifier_name, resampler_name, scoring_function_name, score]
                    rows.append(row)
            result_final_path_file = result_final_path / f'{dataset_name}_{fold}.p'

    pickle.dump(rows, open(result_final_path_file, "wb"))
    end = round(time.time() - start)
    print(f'DONE - Fold {fold} {dataset_name} (Time: {end} [s])')
    logging.info(f'DONE - Fold {fold} {dataset_name} (Time: {end} [s])')


# print("======================")
# results_opt = pickle.load(open("results_from_server/raw/ecoli-0-1-3-7_vs_2-6_0.p", "rb"))
# print(results_opt)

# Multithread
# n_jobs - number of threads, where - 1 all threads, safe for my computer 2
Parallel(n_jobs=2)(
                delayed(evaluate)
                # (fold, dataset_name, resampler_name)
                (fold, dataset_name)
                for fold in range(10)
                # for fold in range(2)
                for dataset_name in datasets.names()
                # for dataset_name in datasets_test
                # for resampler_name in resamplers
                )

# results = pickle.load(open("results_from_server/scores/glass2_0_JFOTS.p", "rb"))
# print(results)
