import argparse
import logging
from pathlib import Path
import pickle
from imblearn.over_sampling import SMOTE
from pymoo.algorithms.moo.nsga2 import NSGA2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from jfots import JFOTS
import datasets


def evaluate(fold, dataset_name):
    RESULTS_PATH = Path(__file__).parents[0] / 'results'
    # Parameters for classifiers and optimization
    RANDOM_STATE = 0
    oversampler_class = SMOTE
    oversampler_kwargs = {"k_neighbors": 1, "random_state": 0}
    optimizer_class = NSGA2
    optimizer_kwargs = {"pop_size": 10}  # 200

    # datasets_names = ["haberman"]
    # for dataset_name in datasets_names:
    # for dataset_name in datasets.names():
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

        for solution in jfots.solutions:
            row = [dataset_name, fold, classifier_name, "JFOTS", solution.objectives, solution.feature_mask, solution.type_mask, solution.data]
            rows.append(row)

    RESULTS_PATH.mkdir(exist_ok=True, parents=True)
    pickle.dump(rows, open(result_path, "wb"))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('-fold', type=int)
    parser.add_argument('-dataset_name', type=str)
    args = parser.parse_args()

    evaluate(args.fold, args.dataset_name)
    # przykładowe wywołanie w konsoli: python experiment.py -fold 8 -dataset_name "haberman"

    # evaluate(6, "haberman")

    # Wczytanie pickle
    # rows = pickle.load(open("results_final/haberman_1.p", "rb"))
    # print(rows)
