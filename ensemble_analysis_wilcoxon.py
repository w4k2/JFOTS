# from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import smote_variants as sv
import numpy as np
import pandas as pd

import datasets
import metrics
from wilcoxon_ranking import pairs_metrics_multi_ensemble


RANDOM_STATE = 0
classifiers = {
    'CART': DecisionTreeClassifier(random_state=RANDOM_STATE),
    'KNN': KNeighborsClassifier(n_neighbors=3),
    'SVM': SVC(kernel='rbf', random_state=RANDOM_STATE),
    # 'MLP': MLPClassifier(random_state=RANDOM_STATE)
}

resamplers = {
    'JFOTS_en_pop': None,
    'JFOTS_en_pf': None,
    'JFOTS_prom': None,
    'JFOTS_pr': None,
    'JFOTS_rc': None,
    'JFOTS_gm': None,
    # 'JFOTS_asf': None,
    'SMOTE': sv.SMOTE(random_state=RANDOM_STATE),
    'polynom-fit-SMOTE': sv.polynom_fit_SMOTE(random_state=RANDOM_STATE),
    'Lee': sv.Lee(random_state=RANDOM_STATE),
    'SMOBD': sv.SMOBD(random_state=RANDOM_STATE),
    'G-SMOTE': sv.G_SMOTE(random_state=RANDOM_STATE),
    'LVQ-SMOTE': sv.LVQ_SMOTE(random_state=RANDOM_STATE),
    'Assembled-SMOTE': sv.Assembled_SMOTE(random_state=RANDOM_STATE),
    'SMOTE-TomekLinks': sv.SMOTE_TomekLinks(random_state=RANDOM_STATE),
}

datasets_list = []
for id, dataset in enumerate(datasets.names()):
    datasets_list.append(dataset)


scoring_functions = {
    'Precision': metrics.precision,
    'Recall': metrics.recall,
    'AUC': metrics.auc,
    'G-mean': metrics.g_mean,
    'BAC': metrics.bac,
}

thresholds = np.arange(0.0, 1.01, 0.05)
# for dataset_name in datasets_list:
#     for clf_name in classifiers.keys():
#         for metric in scoring_functions:

experiment_names = list(classifiers.keys())
# Load data
data = {}
for method_name in list(resamplers.keys()):
    for stream_name in datasets_list:
        for metric in scoring_functions:
            for experiment_name in experiment_names:
                if method_name == "JFOTS_en_pf" or method_name == "JFOTS_en_pop":
                    try:
                        # Thresholds chosen from colored excel with the best metric BAC:
                        # CART: ensemble_threshold_0.9_pareto_front
                        data[("JFOTS_en_pf", stream_name, metric, "CART")] = np.genfromtxt("results_cv52/ensemble_scores/%s/CART/%s/ensemble_threshold_0.9_pareto_front.csv" % (stream_name, metric))
                        # KNN: ensemble_threshold_0.4_pareto_front
                        data[("JFOTS_en_pf", stream_name, metric, "KNN")] = np.genfromtxt("results_cv52/ensemble_scores/%s/KNN/%s/ensemble_threshold_0.4_pareto_front.csv" % (stream_name, metric))
                        # SVM: ensemble_threshold_0.75_pareto_front
                        data[("JFOTS_en_pf", stream_name, metric, "SVM")] = np.genfromtxt("results_cv52/ensemble_scores/%s/SVM/%s/ensemble_threshold_0.75_pareto_front.csv" % (stream_name, metric))

                        # CART: ensemble_threshold_0.45_population
                        data[("JFOTS_en_pop", stream_name, metric, "CART")] = np.genfromtxt("results_cv52/ensemble_scores/%s/CART/%s/ensemble_threshold_0.45_population.csv" % (stream_name, metric))
                        # KNN: ensemble_threshold_0.65_population
                        data[("JFOTS_en_pop", stream_name, metric, "KNN")] = np.genfromtxt("results_cv52/ensemble_scores/%s/KNN/%s/ensemble_threshold_0.65_population.csv" % (stream_name, metric))
                        # SVM: ensemble_threshold_0.3_population
                        data[("JFOTS_en_pop", stream_name, metric, "SVM")] = np.genfromtxt("results_cv52/ensemble_scores/%s/SVM/%s/ensemble_threshold_0.3_population.csv" % (stream_name, metric))
                    except:
                        # print("Dataset is None")
                        pass

                else:
                    try:
                        data[(method_name, stream_name, metric, experiment_name)] = np.genfromtxt("results_cv52/scores/%s/%s/%s/%s.csv" % (stream_name, experiment_name, metric, method_name))
                    except:
                        print("None is ", method_name, stream_name, metric, experiment_name)
                        data[(method_name, stream_name, metric, experiment_name)] = None
                        print(data[(method_name, stream_name, metric, experiment_name)])

# Wilcoxon ranking plots
pairs_metrics_multi_ensemble(method_names=list(resamplers.keys()), dataset_names=list(datasets_list), metrics=list(scoring_functions.keys()), experiment_names=list(classifiers.keys()), data=data, methods_alias=list(resamplers.keys()), metrics_alias=list(scoring_functions.keys()), streams_alias=list(datasets_list)[0], title=True)
