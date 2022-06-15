from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score, roc_auc_score

from imblearn.over_sampling import SMOTE, RandomOverSampler
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import numpy as np
import os
import argparse

from sklearn.base import clone
from datasets import load
import metrics

# path = os.path.join('results', 'experiment_clustering_dt_no_clusters')
path = r'C:\Users\Weronika Węgier\Desktop\results_cv52'
results_path = os.path.join(path, 'ensemble_scores')

try:
    os.mkdir(results_path)
except:
    print('Results directory already created')
classifiers = {
    'CART': DecisionTreeClassifier(random_state=0),
    'KNN': KNeighborsClassifier(n_neighbors=3),
    'SVM': SVC(kernel='rbf', random_state=0),
    # 'MLP': MLPClassifier(random_state=RANDOM_STATE)
}
scoring_functions = {
    'Precision': metrics.precision,
    'Recall': metrics.recall,
    'AUC': metrics.auc,
    'G-mean': metrics.g_mean,
    'BAC': metrics.bac,
}
file_list = [f for f in os.listdir(os.path.join(path, 'ensembles_prediction')) if os.path.isdir(os.path.join(path, 'ensembles_prediction', f))]

columns_name = []
for file in file_list:
    try:
        os.mkdir(os.path.join(results_path,file))
    except:
        pass
    folds = load(file)
    # metrics_array = np.zeros((10, len(classifiers), len(algorithms), len(scoring_functions.keys())))
    for ic, cl in enumerate(classifiers.keys()):
        try:
            os.mkdir(os.path.join(results_path, file, cl))
        except:
            pass
        algorithms = [f for f in os.listdir(os.path.join(path, 'ensembles_prediction', file_list[0], 'fold_0', cl)) if
                      not os.path.isdir(os.path.join(path, f))]
        for i in range(len(algorithms)):
            algorithms[i] = algorithms[i].replace('.csv', '')
        for ia, al in enumerate(algorithms):
            metrics_list = [[] for metric in scoring_functions.keys()]
            for i, fold in enumerate(folds):
                try:
                    y_pred = pd.read_csv(os.path.join(path,'ensembles_prediction',file, 'fold_{}'.format(i), cl, '{}.csv'.format(al)), index_col=0).values
                    for im, metric in enumerate(scoring_functions.keys()):
                        metrics_list[im].append(scoring_functions[metric](fold[1][1], y_pred))
                except Exception as e:
                    print(e)
            for im, metric in enumerate(scoring_functions.keys()):
                try:
                    os.mkdir(os.path.join(results_path, file, cl, metric))
                except:
                    pass
                metric_df = pd.DataFrame(metrics_list[im])
                metric_df.to_csv(os.path.join(results_path, file, cl, metric, '{}.csv'.format(al)))

