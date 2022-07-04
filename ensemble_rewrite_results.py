from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

import datasets
import metrics


RANDOM_STATE = 0
classifiers = {
    'CART': DecisionTreeClassifier(random_state=RANDOM_STATE),
    'KNN': KNeighborsClassifier(n_neighbors=3),
    'SVM': SVC(kernel='rbf', random_state=RANDOM_STATE),
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

# It was needed to save ensemble results in the same way as previously
thresholds = np.arange(0.0, 1.01, 0.05)
for dataset_name in datasets_list:
    for clf_name in classifiers.keys():
        for metric in scoring_functions:
            for threshold_number in thresholds:
                threshold = round(threshold_number, 2)
                try:
                    filepath_pf = "results_cv52/ensemble_scores/%s/%s/%s/ensemble_threshold_%s_pareto_front.csv" % (dataset_name, clf_name, metric, threshold)
                    pareto_front_df = pd.read_csv(filepath_pf)
                    pareto_front_arr = pareto_front_df.to_numpy()[:, 1]
                    np.savetxt(filepath_pf, pareto_front_arr)
                except:
                    print(filepath_pf)
                    # pass
                try:
                    filepath_pop = "results_cv52/ensemble_scores/%s/%s/%s/ensemble_threshold_%s_population.csv" % (dataset_name, clf_name, metric, threshold)
                    population_df = pd.read_csv(filepath_pop)
                    population_arr = population_df.to_numpy()[:, 1]
                    np.savetxt(filepath_pop, population_arr)
                except:
                    print(filepath_pop)
                    # pass
