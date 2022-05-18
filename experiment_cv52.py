import argparse
import logging
from pathlib import Path
import pickle
import numpy as np
from imblearn.over_sampling import SMOTE
from pymoo.algorithms.moo.nsga2 import NSGA2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from joblib import Parallel, delayed
import time
import itertools

from jfots2 import JFOTS
import datasets

logging.basicConfig(filename='textinfo/experiment.log', filemode="a", format='%(asctime).s - %(levelname)s: %(message)s', level='DEBUG')


def evaluate(fold, dataset_name):
    print(f'START: {fold}, {dataset_name}')
    logging.info(f'START - {fold}, {dataset_name}')
    start = time.time()
    RESULTS_PATH = Path(__file__).parents[0] / 'results_cv52'
    # Parameters for classifiers and optimization
    RANDOM_STATE = 0
    oversampler_class = SMOTE
    oversampler_kwargs = {"k_neighbors": 5, "random_state": 0}
    optimizer_class = NSGA2
    optimizer_kwargs = {"pop_size": 500}

    classifiers = {
        'CART': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'KNN': KNeighborsClassifier(n_neighbors=3),
        'SVM': SVC(kernel='rbf', random_state=RANDOM_STATE),
        # 'MLP': MLPClassifier(random_state=RANDOM_STATE)
    }
    result_name = f'{dataset_name}_{fold}.p'
    r_path = RESULTS_PATH / f'raw'
    r_path.mkdir(exist_ok=True, parents=True)
    result_path = r_path / result_name

    dataset = datasets.load(dataset_name)
    (X_train, y_train), (X_test, y_test) = dataset[fold][0], dataset[fold][1]
    pop_rows = []
    rows = []
    for classifier_name, classifier in classifiers.items():
        logging.info(f'Evaluating {result_name} for {classifier_name}...')
        jfots = JFOTS(
            estimator=classifier,
            oversampler_class=oversampler_class,
            oversampler_kwargs=oversampler_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            # termination=('n_gen', 500),  # we don't have to specify the termination criterion and can use pymoo default instead
            random_state=0,
            verbose=False,
        )
        jfots.fit_resample(X_train, y_train)

        figure_path = RESULTS_PATH / f'figures'
        figure_path.mkdir(exist_ok=True, parents=True)
        pareto_path = figure_path / f'pareto_{dataset_name}_{fold}_{classifier_name}'
        hv_path = figure_path / f'hv_{dataset_name}_{fold}_{classifier_name}'

        jfots.plot_pareto(file_name=pareto_path)
        jfots.plot_hv(file_name=hv_path)

        # Save solutions from all populations
        pop_path = RESULTS_PATH / f'population'
        pop_path.mkdir(exist_ok=True, parents=True)
        pop_X_path = pop_path / f'{dataset_name}_{fold}_{classifier_name}_pop_X.csv'
        np.savetxt(fname=pop_X_path, fmt="%f", X=jfots.pop_X)
        pop_F_path = pop_path / f'{dataset_name}_{fold}_{classifier_name}_pop_F.csv'
        np.savetxt(fname=pop_F_path, fmt="%f", X=jfots.pop_F)

        rows_solution_masks = []
        sol_objectives = []
        for solution in jfots.solutions:
            row = [dataset_name, fold, classifier_name, "JFOTS", solution.objectives, solution.feature_mask, solution.type_mask, solution.data]
            rows.append(row)
            row_solution_masks = [solution.feature_mask, solution.type_mask]
            row_solution_masks_flat = list(itertools.chain.from_iterable(row_solution_masks))
            rows_solution_masks.append(row_solution_masks_flat)
            sol_objectives.append(list(solution.objectives))
        # Save solutions masks from pareto to csv
        sol_path = RESULTS_PATH / f'solution_masks'
        sol_path.mkdir(exist_ok=True, parents=True)
        sol_X_path = sol_path / f'{dataset_name}_{fold}_{classifier_name}_sol_X.csv'
        np.savetxt(fname=sol_X_path, fmt="%f", X=rows_solution_masks)
        sol_F_path = sol_path / f'{dataset_name}_{fold}_{classifier_name}_sol_F.csv'
        np.savetxt(fname=sol_F_path, fmt="%f", X=sol_objectives)

        end = round(time.time() - start)
        print(f'DONE - Fold {fold} {dataset_name} (Time: {end} [s]) Classifier {classifier_name}')
        logging.info(f'DONE - Fold {fold} {dataset_name} (Time: {end} [s]) Classifier {classifier_name}')
    RESULTS_PATH.mkdir(exist_ok=True, parents=True)
    pickle.dump(rows, open(result_path, "wb"))


datas = []
selected_datasets = [2, 5, 12, 47, 57]
for id, dataset in enumerate(datasets.names()):
    for id_ in selected_datasets:
        if id == id_:
            datas.append(dataset)

# Multithread
# n_jobs - number of threads, where - 1 all threads, safe for my computer 2
Parallel(n_jobs=1)(
                delayed(evaluate)
                (fold, dataset_name)
                for fold in range(10)
                for dataset_name in datas
                )
