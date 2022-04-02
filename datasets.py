import pickle
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).parent / "data"
FOLDS_PATH = Path(__file__).parent / "folds"

RANDOM_SEED = 42

URLS = [
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1/abalone19.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1/abalone9-18.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1/ecoli-0-1-3-7_vs_2-6.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1/glass-0-1-6_vs_2.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1/glass-0-1-6_vs_5.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1/glass2.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1/glass4.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1/glass5.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1/page-blocks-1-3_vs_4.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1/yeast-0-5-6-7-9_vs_4.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1/yeast-1-2-8-9_vs_7.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1/yeast-1-4-5-8_vs_7.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1/yeast-1_vs_7.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1/yeast-2_vs_4.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1/yeast-2_vs_8.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1/yeast4.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1/yeast5.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1/yeast6.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p2/cleveland-0_vs_4.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p2/ecoli-0-1-4-7_vs_2-3-5-6.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p2/ecoli-0-1_vs_2-3-5.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p2/ecoli-0-2-6-7_vs_3-5.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p2/ecoli-0-6-7_vs_3-5.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p2/ecoli-0-6-7_vs_5.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p2/glass-0-1-4-6_vs_2.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p2/glass-0-1-5_vs_2.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p2/yeast-0-2-5-6_vs_3-7-8-9.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p2/yeast-0-3-5-9_vs_7-8.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/abalone-17_vs_7-8-9-10.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/abalone-19_vs_10-11-12-13.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/abalone-20_vs_8-9-10.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/abalone-21_vs_8.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/flare-F.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/kddcup-buffer_overflow_vs_back.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/kddcup-rootkit-imap_vs_back.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/kr-vs-k-zero_vs_eight.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/poker-8-9_vs_5.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/poker-8-9_vs_6.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/poker-8_vs_6.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/poker-9_vs_7.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/winequality-red-3_vs_5.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/winequality-red-4.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/winequality-red-8_vs_6-7.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/winequality-red-8_vs_6.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/winequality-white-3-9_vs_5.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/winequality-white-3_vs_7.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/winequality-white-9_vs_4.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/zoo-3.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRlowerThan9/ecoli1.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRlowerThan9/ecoli2.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRlowerThan9/ecoli3.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRlowerThan9/glass0.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRlowerThan9/glass1.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRlowerThan9/haberman.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRlowerThan9/page-blocks0.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRlowerThan9/pima.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRlowerThan9/vehicle1.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRlowerThan9/vehicle3.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRlowerThan9/yeast1.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRlowerThan9/yeast3.zip",
]


def download(url):
    name = url.split("/")[-1]
    download_path = DATA_PATH / name

    DATA_PATH.mkdir(exist_ok=True, parents=True)

    if not download_path.exists():
        urlretrieve(url, download_path)

    if not download_path.with_suffix(".dat").exists():
        if name.endswith(".zip"):
            with zipfile.ZipFile(download_path) as f:
                f.extractall(DATA_PATH)
        else:
            raise Exception("Unrecognized file type.")


def encode(X, y, encode_features=True):
    y = preprocessing.LabelEncoder().fit(y).transform(y)

    if encode_features:
        encoded = []

        for i in range(X.shape[1]):
            try:
                float(X[0, i])
                encoded.append(X[:, i])
            except ValueError:
                encoded.append(preprocessing.LabelEncoder().fit_transform(X[:, i]))

        X = np.transpose(encoded)

    return X.astype(np.float32), y.astype(np.float32)


def partition(X, y):
    partitions = []

    for i in range(5):
        folds = []
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_SEED + i)

        for train_idx, test_idx in skf.split(X, y):
            folds.append([train_idx, test_idx])

        partitions.append(folds)

    return partitions


def load(
    name,
    url=None,
    encode_features=True,
    remove_metadata=True,
    scale=True,
):
    file_name = "%s.dat" % name

    if url is not None:
        download(url)

    skiprows = 0

    if remove_metadata:
        with open(DATA_PATH / file_name) as f:
            for line in f:
                if line.startswith("@"):
                    skiprows += 1
                else:
                    break

    df = pd.read_csv(
        DATA_PATH / file_name,
        header=None,
        skiprows=skiprows,
        skipinitialspace=True,
        sep=" *, *",
        na_values="?",
        engine="python",
    )

    matrix = df.dropna().values

    X, y = matrix[:, :-1], matrix[:, -1]
    X, y = encode(X, y, encode_features)

    partitions_path = FOLDS_PATH / file_name.replace(".dat", ".folds.pickle")

    FOLDS_PATH.mkdir(exist_ok=True, parents=True)

    if partitions_path.exists():
        partitions = pickle.load(open(partitions_path, "rb"))
    else:
        partitions = partition(X, y)
        pickle.dump(partitions, open(partitions_path, "wb"))

    folds = []

    for i in range(5):
        for j in range(2):
            train_idx, test_idx = partitions[i][j]

            train_set = [X[train_idx], y[train_idx]]
            test_set = [X[test_idx], y[test_idx]]

            if scale:
                scaler = StandardScaler().fit(train_set[0])

                train_set[0] = scaler.transform(train_set[0])
                test_set[0] = scaler.transform(test_set[0])

            folds.append([train_set, test_set])

    return folds


def names():
    return [url.split("/")[-1].replace(".zip", "") for url in URLS]


def load_all():
    datasets = {}

    for url, name in zip(URLS, names()):
        datasets[name] = load(name, url)

    return datasets


if __name__ == "__main__":
    load_all()
