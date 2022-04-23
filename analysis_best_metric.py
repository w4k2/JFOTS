

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
import os

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
    'JFOTS_pr': None,
    'JFOTS_rc': None,
    'JFOTS_prom': None,
    'JFOTS_gm': None,
    'JFOTS_asf': None,
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

rows = []
datas = []
selected_datasets = [2, 5, 12, 47, 57]
for id, dataset in enumerate(datasets.names()):
    for id_ in selected_datasets:
        if id == id_:
            datas.append(dataset)
n_datasets = len(datas)
# n_datasets = len(datasets.names())
n_metrics = len(scoring_functions)
n_methods = len(classifiers)
n_resamplers = len(resamplers)
n_folds = 10

data_np = np.zeros((n_datasets, n_metrics, n_methods, n_resamplers, n_folds))
mean_scores = np.zeros((n_datasets, n_metrics, n_methods, n_resamplers))
stds = np.zeros((n_datasets, n_metrics, n_methods, n_resamplers))

for dataset_id, dataset_name in enumerate(datas):
    for clf_id, clf_name in enumerate(classifiers.keys()):
        for metric_id, metric in enumerate(scoring_functions.keys()):
            for resampler_id, resampler in enumerate(resamplers.keys()):
                try:
                    filename = "results_from_server/scores/%s/%s/%s/%s.csv" % (dataset_name, clf_name, metric, resampler)
                    if not os.path.isfile(filename):
                        # print("File not exist - %s" % filename)
                        continue
                    scores = np.genfromtxt(filename, delimiter=',', dtype=np.float32)
                    data_np[dataset_id, metric_id, clf_id, resampler_id] = scores
                    mean_score = np.mean(scores)
                    mean_scores[dataset_id, metric_id, clf_id, resampler_id] = mean_score
                    std = np.std(scores)
                    stds[dataset_id, metric_id, clf_id, resampler_id] = std
                except:
                    print("Error loading data!", dataset_name, clf_name, metric)

for clf_id, clf_name in enumerate(classifiers.keys()):
    for metric_id, metric in enumerate(scoring_functions.keys()):
        if not os.path.exists("results_from_server/tables/"):
            os.makedirs("results_from_server/tables/")
        with open("results_from_server/tables/results_%s_%s.tex" % (metric, clf_name), "w+") as file:
            print("\\begin{table}[!ht]", file=file)
            print("\\centering", file=file)
            print("\\caption{%s -- %s}" % (clf_name, metric), file=file)
            columns = "r"
            for i in resamplers.keys():
                columns += " c"

            print("\\scalebox{0.4}{", file=file)
            print("\\begin{tabular}{%s}" % columns, file=file)
            print("\\hline", file=file)
            columns_names = "\\textbf{Dataset name} &"
            for name in resamplers.keys():
                name = name.replace("_", "-")
                columns_names += f'\\textbf{{{name}}} & '
            columns_names = columns_names[:-3]
            columns_names += "\\\\"
            print(columns_names, file=file)
            print("\\hline", file=file)

            for dataset_id, dataset_name in enumerate(datas):
                line = "$%s$" % (dataset_name)
                line_values = []
                line_values = mean_scores[dataset_id, metric_id, clf_id, :]
                max_value = np.amax(line_values)
                # for clf_id, clf_name in enumerate(resamplers.keys()):
                for resampler_id, resampler in enumerate(resamplers.keys()):
                    if mean_scores[dataset_id, metric_id, clf_id, resampler_id] == max_value:
                        line += " & \\textbf{%0.3f $\\pm$ %0.3f}" % (mean_scores[dataset_id, metric_id, clf_id, resampler_id], stds[dataset_id, metric_id, clf_id, resampler_id])
                    else:
                        line += " & %0.3f $\\pm$ %0.3f" % (mean_scores[dataset_id, metric_id, clf_id, resampler_id], stds[dataset_id, metric_id, clf_id, resampler_id])
                line += " \\\\"
                print(line, file=file)
            print("\\end{tabular}}", file=file)
            print("\\end{table}", file=file)
