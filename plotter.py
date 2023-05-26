import argparse
from collections import defaultdict

import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
plt.style.use('ggplot')


def save_fig(fig_path):
    if fig_path == '':
        plt.show()
    else:
        plt.savefig(fig_path)

def load_zen_scores(zen_scores_path):
    if zen_scores_path.endswith('zen_score.log'):
        with open(zen_scores_path) as zen_log:
            zen_scores = [float(line.rstrip('\n')) for line in zen_log.readlines()]
        # return zen_scores
        min_zen_score = min(zen_scores)
        return [zen - min_zen_score for zen in zen_scores]
    else:
        zen_scores = np.load(zen_scores_path, allow_pickle=True)
    # return zen_scores
    return zen_scores - min(zen_scores)

def load_all_features(all_features_path):
    return np.load(all_features_path, allow_pickle=True).reshape(-1, 1)


def plot_zen_scores(zen_scores, fig_path='plots/zen_dist.png'):
    max_bins = len(zen_scores)  # TODO: change to a smaller number

    x = np.round(zen_scores, decimals=4)
    plt.hist(x, bins=min(max_bins, len(set(x))))

    plt.xlabel('Sensitivity Score')
    plt.ylabel('#. of Samples')
    plt.title('Distribution of Sensitivity Scores')

    save_fig(fig_path)


def plot_tsne(X, zen_scores=None, fig_path='plots/tsne.png'):
    from utils import get_smallest_arg

    threshold = 0.15
    if zen_scores is not None:
        i = get_smallest_arg(zen_scores, int(threshold * len(zen_scores)))[-1]
        k = get_smallest_arg(- zen_scores, int(threshold * len(zen_scores)))[-1]
        lower = zen_scores < zen_scores[i]
        higher = zen_scores > zen_scores[k]
        hue = [f'Lowest {int(threshold * 100)}%' if lower[j] else f'Highest {int(threshold * 100)}%' if higher[j] else 'Others'
               for j in range(len(zen_scores))]
    else:
        hue = None

    tsne = TSNE()
    X_embedded = tsne.fit_transform(X)
    # plt.scatter(X_embedded[:, 0], X_embedded[:, 1])
    sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=hue)

    plt.title('tSNE all features')
    if zen_scores is not None:
        plt.legend(title=f'Zen score')

    save_fig(fig_path)


def plot_kmeans(X, fig_path='plots/kmeans.png'):
    from nas.nas_arch import SEED
    from utils import get_elbow, get_distortions

    max_K = 31
    K = range(1, max_K)
    distortions = get_distortions(X, K)

    elbow_k = K[0] + get_elbow(distortions, threshold=0.25)
    elbowModel = KMeans(n_clusters=elbow_k, random_state=SEED).fit(X)
    # elbow_clusters = [np.where(elbowModel.labels_ == i)[0] for i in range(elbow_k)]  # i-th: indices in cluster i

    tsne = TSNE()  # for 2d plotting
    X_embedded = tsne.fit_transform(X)
    # plt.scatter(X_embedded[:, 0], X_embedded[:, 1])
    sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=elbowModel.labels_)

    save_fig(fig_path)


def plot_performance_method(stats_file, *methods, fig_path='plots/ablation_method.png'):
    # if fig_path == 'plots/ablation.png':
    #     fig_path = f'plots/ablation_method{len(methods)}.png'

    from parse_stats import parse
    stats_dict = parse(stats_file)
    
    SPLITS = list(stats_dict.keys())  # assert == ['train', 'val', 'test']

    assert len(set(split_dict.keys() for split_dict in stats_dict.values())) == 1
    keys = stats_dict[SPLITS[0]].keys()
    X = list(map(int, keys))

    YY = {split: {method: [] for method in methods} for split in SPLITS}
    EE = {split: {method: [] for method in methods} for split in SPLITS}

    for split in SPLITS:
        for selection, method_to_scores in stats_dict[split].items():
            assert not selection.endswith('%')  # TODO: implement Selected %
            if len(methods) < 6 and int(selection) > 500:
                break

            for method, scores in method_to_scores.items():
                if method in methods:
                    scores = list(map(float, scores))

                    YY[split][method].append(sum(scores) / len(scores))
                    EE[split][method].append(max(scores) - min(scores))
    
    assert len(set(YY[split][methods[0]] for split in SPLITS)) == 1
    X = X[:len(YY[SPLITS[0]][methods[0]])]

    if 1 <= len(methods) <= 3:
        nrows = 1
        ncols = len(methods)
    elif 4 <= len(methods) <= 6:
        nrows = 2
        ncols = (len(methods) + 1) // nrows
    else:
        raise NotImplementedError
    fig, axs = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
    for i in range(nrows):
        axs_i = axs[i] if nrows > 1 else axs

        for j in range(ncols):
            if i * ncols + j >= len(methods):
                break
            axs_ij = axs_i[j] if ncols > 1 else axs_i

            for split in SPLITS:
                Y = YY[split][methods[i * ncols + j]]
                E = EE[split][methods[i * ncols + j]]
                axs_ij.plot(X, Y)
                axs_ij.errorbar(X, Y, E, label=split,
                                linestyle='-', marker=None, capsize=3)

            axs_ij.set_title(methods[i * ncols + j])
            axs_ij.legend()
            if i == nrows - 1:
                axs_ij.set_xlabel('Number of training samples')
            if j == 0:
                axs_ij.set_ylabel('Accuracy')
    
    axs_00 = axs
    if nrows > 1:
        axs_00 = axs_00[0]
    if ncols > 1:
        axs_00 = axs_00[0]
    for i in range(nrows):
        axs_i = axs[i] if nrows > 1 else axs
        for j in range(ncols):
            if i * ncols + j >= len(methods):
                break
            axs_ij = axs_i[j] if ncols > 1 else axs_i
            axs_ij.sharey(axs_00)
            axs_ij.sharex(axs_00)
            axs_ij.grid()
            
    
    # handles, labels = plt.gca().get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center')

    fig.tight_layout()
    # fig.suptitle("")
    fig.savefig(fig_path)


def plot_performance_size(stats_files, stats_pct_files, fig_path='plots/ablation_size.png'):
    METHODS = ['random', 'highest', 'lowest', 'proposed']
    split = 'test'

    from parse_stats import parse_txt, parse_csv

    test_dict = defaultdict(lambda: defaultdict(list))
    for stats_file in stats_files:
        if stats_file.endswith('.csv'):
            new_dict = parse_csv(stats_file)
        else:
            new_dict = parse_txt(stats_file)[split]
        for selection, method_to_scores in new_dict.items():
            for method, scores in method_to_scores.items():
                test_dict[selection][method] += scores

        X = list(map(int, test_dict.keys()))
    
    avgYY = {method: [] for method in METHODS}
    maxYY = {method: [] for method in METHODS}
    minYY = {method: [] for method in METHODS}
    for selection, method_to_scores in test_dict.items():
        assert not selection.endswith('%')

        for method, scores in method_to_scores.items():
            if method in METHODS:
                scores = list(map(float, scores))

                avgYY[method].append(sum(scores) / len(scores))
                maxYY[method].append(max(scores))
                minYY[method].append(min(scores))
    
    X = X[:len(avgYY[METHODS[0]])]

    nrows, ncols = 1, 2    
    fig, axs = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    
    for i, method in enumerate(METHODS):
        avgY = avgYY[method]
        maxY = maxYY[method]
        minY = minYY[method]
        axs[0].plot(X, avgY, color=f'C{i}', label=method)
        axs[0].fill_between(X, minY, maxY, alpha=0.3, facecolor=f'C{i}')

        # axs[0].set_title()
        axs[0].legend()
        axs[0].set_xlabel('Number of training samples')
        axs[0].set_ylabel(f'{split} performance'.capitalize())
    
    test_pct_dict = defaultdict(lambda: defaultdict(list))
    for stats_pct_file in stats_pct_files:
        if stats_pct_file.endswith('.csv'):
            new_dict = parse_csv(stats_pct_file)
        else:
            new_dict = parse_txt(stats_pct_file)[split]
        for selection, method_to_scores in new_dict.items():
            for method, scores in method_to_scores.items():
                test_pct_dict[selection][method] += scores
        X_pct = list(map(lambda pct_str: int(pct_str.rstrip('%')),
                         test_pct_dict.keys()))
    
    YY_pct = {method: [] for method in METHODS}
    for selection, method_to_scores in test_pct_dict.items():
        assert selection.endswith('%')

        for method, scores in method_to_scores.items():
            if method in METHODS:
                scores = list(map(float, scores))

                YY_pct[method].append(sum(scores) / len(scores))
    
    X_pct = X_pct[:len(YY_pct[METHODS[0]])]

    for method in METHODS:
        Y_pct = YY_pct[method]
        axs[1].plot(X_pct, Y_pct, label=method)

        # axs[1].set_title()
        axs[1].legend()
        axs[1].set_xlabel('Percent of training samples')
        axs[1].set_ylabel(f'{split} performance'.capitalize())
    
    axs[1].sharey(axs[0])
    # axs[0].grid()
    # axs[1].grid()
            
    # handles, labels = plt.gca().get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center')

    fig.tight_layout()
    # fig.suptitle("")
    fig.savefig(fig_path)

def main():
    METHODS = ['random', 'lowest', 'highest', 'middle', 'uniform', 'mix']
    DATASETS = ["IMDb", "MNLI", "QNLI", "RTE", "SST2", "CoLA"]

    parser = argparse.ArgumentParser()
    parser.add_argument('plot', choices=['kmeans', 'zen', 'tsne', 'performance_method', 'performance_size'])
    parser.add_argument('--zen_scores', type=str)
    parser.add_argument('--all_features', type=str)
    parser.add_argument('--stats', nargs='*', type=str, help='Selected #')
    parser.add_argument('--stats_pct', nargs='*', type=str, help='Selected %')
    parser.add_argument('--methods', nargs='*', default=METHODS, choices=METHODS, type=str)
    args = parser.parse_args()

    if args.plot == 'zen':
        zen_scores = load_zen_scores(args.zen_scores)

        fig_path = 'plots/zen_dist.png'
        for name in DATASETS:
            if name in args.zen_scores:
                fig_path = f'plots/{name}_zen_dist.png'
            # TODO: assert other names not in zen_scores path
        plot_zen_scores(zen_scores, fig_path=fig_path)
    
    elif args.plot == 'kmeans':
        all_features = load_all_features(args.all_features)
        plot_kmeans(all_features)

    elif args.plot == 'tsne':
        zen_scores = load_zen_scores(args.zen_scores) if args.zen_scores else None
        all_features = load_all_features(args.all_features)
        plot_tsne(all_features, zen_scores=zen_scores)
    
    elif args.plot == 'performance_method':
        plot_performance_method(args.stats, *args.methods)
    
    elif args.plot == 'performance_size':
        plot_performance_size(args.stats, args.stats_pct)

    else:
        raise NotImplementedError(f'unrecognized plot type: {args.plot}')
    

if __name__ == "__main__":
    main()
