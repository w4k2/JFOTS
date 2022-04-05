import warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ALPHA = 0.9
BACKGROUND_COLOR = "#EEEEEE"
BORDER_COLOR = "#161921"
COLOR_MAJORITY = "#C44E52"
COLOR_MINORITY = "#4C72B0"
COLOR_PROTOTYPE = "#EEEEEE"
FIGURE_SIZE = (6, 6)
LIMITS = (-0.05, 1.05)
LINE_WIDTH = 1.0
MARKER_SIZE = 100
MARKER_SYMBOL = "o"
MARKER_PROTOTYPE_SYMBOL = "^"
ORIGINAL_EDGE_COLOR = "#F2F2F2"
OVERSAMPLED_EDGE_COLOR = "#262223"
PREDICTION_GRID_N = 250


def decision_border(clf, X, y, prototypes, file_name=None):
    if X.shape[1] != 2:
        raise ValueError(
            f"Expected 2-dimensional data, received X with shape = {X.shape}."
        )

    if X.min() < LIMITS[0] or X.max() > LIMITS[1]:
        warnings.warn(
            f"X outside of displayed limits: expected within {LIMITS}, received {(X.min(), X.max())}."
        )

    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    plt.xlim(LIMITS)
    plt.ylim(LIMITS)

    axis.grid(False)

    axis.set_xticks([])
    axis.set_yticks([])

    for key in axis.spines.keys():
        axis.spines[key].set_color(BORDER_COLOR)

    axis.tick_params(colors=BORDER_COLOR)
    axis.set_facecolor(BACKGROUND_COLOR)

    minority_class = Counter(y).most_common()[1][0]
    majority_class = Counter(y).most_common()[0][0]

    prediction_grid = np.zeros((PREDICTION_GRID_N + 1, PREDICTION_GRID_N + 1))

    for i, x1 in enumerate(np.linspace(LIMITS[0], LIMITS[1], PREDICTION_GRID_N + 1)):
        for j, x2 in enumerate(
            np.linspace(LIMITS[0], LIMITS[1], PREDICTION_GRID_N + 1)
        ):
            prediction_grid[i][j] = clf.predict(np.array([[x1, x2]]))[0]

    prediction_grid = np.swapaxes(prediction_grid, 0, 1)

    color_map = LinearSegmentedColormap.from_list(
        "heatmap", (COLOR_MAJORITY, COLOR_MINORITY), N=2
    )

    plt.imshow(
        prediction_grid,
        vmin=np.min(prediction_grid),
        vmax=np.max(prediction_grid),
        extent=[LIMITS[0], LIMITS[1], LIMITS[0], LIMITS[1]],
        origin="lower",
        cmap=color_map,
    )

    for cls, cls_color in [
        [majority_class, COLOR_MAJORITY],
        [minority_class, COLOR_MINORITY],
    ]:
        points = X[y == cls]

        if len(points) > 0:
            plt.scatter(
                points[:, 0],
                points[:, 1],
                facecolors=cls_color,
                s=MARKER_SIZE,
                marker=MARKER_SYMBOL,
                linewidths=LINE_WIDTH,
                alpha=ALPHA,
                edgecolors=ORIGINAL_EDGE_COLOR,
            )

    plt.scatter(
        prototypes[:, 0],
        prototypes[:, 1],
        facecolors=COLOR_PROTOTYPE,
        s=MARKER_SIZE,
        marker=MARKER_PROTOTYPE_SYMBOL,
        linewidths=LINE_WIDTH,
        alpha=ALPHA,
        edgecolors=OVERSAMPLED_EDGE_COLOR,
    )

    if file_name is None:
        plt.show()
    else:
        plt.savefig(file_name, bbox_inches="tight")


def hypervolume(values, file_name=None):
    x = np.arange(1, len(values) + 1)

    plt.plot(x, values, "-o", markersize=4, linewidth=2)

    plt.xlabel("Iteration")
    plt.ylabel("Hypervolume")

    if file_name is None:
        plt.show()
    else:
        plt.savefig(file_name, bbox_inches="tight")
    plt.close()


def pareto_front(objectives, labels, file_name=None):
    fig = plt.figure(figsize=FIGURE_SIZE)
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(
        [o[0] for o in objectives],
        [o[1] for o in objectives],
        facecolors=COLOR_MINORITY,
        s=MARKER_SIZE,
        marker=MARKER_SYMBOL,
        linewidths=LINE_WIDTH,
        alpha=ALPHA,
        edgecolors=OVERSAMPLED_EDGE_COLOR,
    )

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])

    if file_name is None:
        plt.show()
    else:
        plt.savefig(file_name, bbox_inches="tight")
    plt.close()


def metric_history(history, title=None, file_name=None):
    y_min = np.array([np.min(h) for h in history])
    y_avg = np.array([np.mean(h) for h in history])
    y_max = np.array([np.max(h) for h in history])

    x = np.arange(1, len(history) + 1)

    plt.plot(x, y_min, color="g")
    plt.plot(x, y_avg, color="y")
    plt.plot(x, y_max, color="r")

    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Score")
    plt.ylim([-0.1, 1.1])
    plt.legend(["min", "avg", "max"], loc=4)

    if file_name is None:
        plt.show()
    else:
        plt.savefig(file_name, bbox_inches="tight")
