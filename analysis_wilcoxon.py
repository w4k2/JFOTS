from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import smote_variants as sv

import datasets
import metrics
from wilcoxon_ranking import pairs_metrics_multi


RANDOM_STATE = 0
resamplers = {
    # 'JFOTS_pr': None,
    # 'JFOTS_rc': None,
    'JFOTS_prom': None,
    'SMOTE': sv.SMOTE(random_state=RANDOM_STATE),
    'polynom-fit-SMOTE': sv.polynom_fit_SMOTE(random_state=RANDOM_STATE),
    'Lee': sv.Lee(random_state=RANDOM_STATE),
    'SMOBD': sv.SMOBD(random_state=RANDOM_STATE),
    'G-SMOTE': sv.G_SMOTE(random_state=RANDOM_STATE),
    'LVQ-SMOTE': sv.LVQ_SMOTE(random_state=RANDOM_STATE),
    'Assembled-SMOTE': sv.Assembled_SMOTE(random_state=RANDOM_STATE),
    'SMOTE-TomekLinks': sv.SMOTE_TomekLinks(random_state=RANDOM_STATE),
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

# Wilcoxon ranking plots
pairs_metrics_multi(method_names=list(resamplers.keys()), dataset_names=list(datasets.names()), metrics=list(scoring_functions.keys()), experiment_names=list(classifiers.keys()), methods_alias=list(resamplers.keys()), metrics_alias=list(scoring_functions.keys()), streams_alias=list(datasets.names())[0], title=False)
