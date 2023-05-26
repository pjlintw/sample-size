import argparse
import os

from matplotlib import pyplot as plt
plt.style.use('ggplot')
import numpy as np
import torch


model_path = 'model_test.pt'  # removed after execution
n_samples = 50
n_inits = 3  # random initializations
N_LAYERS = [1, 2]

def get_model(nlayers=2):
    from nas.nas_arch import build_model
    from datautils import TEXT

    input_dim = len(TEXT.vocab)
    emsize = 1000
    nhid = 1820
    nhead = 4
    model_arch = {
        'emsize': emsize,
        'nhid': nhid,
        'nlayers': nlayers,
        'nhead': nhead,
    }  
    model = build_model(model_arch=model_arch, model_type="transformer", input_dim=input_dim, nlabels=1)

    return model

def get_sample_zen_scores(train_samples):  # from sample_selector.py
    from torchtext.legacy import data
    from datautils import get_zen_score, FIELDS

    zen_scores = []

    for train_sample in train_samples:
        model = torch.load(model_path)
        train_subset = data.Dataset([train_sample], fields=FIELDS)
        sample_zen_score = get_zen_score(model, train_subset)

        zen_scores.append(sample_zen_score)

    return zen_scores

def plot(all_zen_scores, fig_path):
    from plotter import save_fig
    # def adjust_lightness(color, amount=0.5):
    #     import matplotlib.colors as mc
    #     import colorsys
    #     try:
    #         color = mc.cnames[color]
    #     except KeyError:
    #         pass
    #     h, l, s = colorsys.rgb_to_hls(*mc.to_rgb(color))
    #     l = max(0, min(1, amount * l))
    #     return colorsys.hls_to_rgb(h, l, s)

    all_ranks = []
    all_sorted_indices = []
    for zen_scores in all_zen_scores:
        sorted_indices = zen_scores.argsort()  # ascending
        all_sorted_indices.append(sorted_indices)

        ranks = np.empty(len(zen_scores), int)
        ranks[sorted_indices] = np.arange(len(zen_scores))
        
        all_ranks.append(ranks)

    # from sklearn import preprocessing
    # normalized = preprocessing.normalize(all_zen_scores)

    nrows, ncols = 1, len(N_LAYERS)
    fig, axs = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))

    X = np.arange(n_samples)

    for subplot, nlayers in enumerate(N_LAYERS):
        # for init in range(n_inits):
        #     Y = np.full(n_samples, init)
        #     colors = [adjust_lightness('C0', np.linspace(0.5, 1.8, num=n_samples)[rank])
        #                                            for rank in all_ranks[init + subplot*n_inits]]
        #     axs[subplot].scatter(X, Y, c=colors)
        for init in range(n_inits):
            if init == 0:
                rank0 = all_ranks[init + subplot*n_inits]
                Y = all_sorted_indices[init + subplot*n_inits][rank0]
                axs[subplot].plot(X, Y)
            Y = all_sorted_indices[init + subplot*n_inits][rank0]
            axs[subplot].scatter(X, Y, s=10, label=f'init-{init}')

        axs[subplot].set_title(f"{nlayers}-layer model")
        # axs[subplot].set_xlabel('Index of training samples')
        axs[subplot].set_ylabel('Relative rank')
        axs[subplot].legend(loc='lower right')
    plt.setp(axs, xticks=[], yticks=[])

    # for subplot in range(1, len(N_LAYERS)):
    #     axs[subplot].sharex(axs[0])

    fig.tight_layout()
    save_fig(fig_path)


def main():
    zen_path = 'zen_test.npy'
    try:
        all_zen_scores = list(np.load(zen_path))
        assert len(all_zen_scores) == n_inits * len(N_LAYERS)
        for i in range(len(all_zen_scores)):
            assert len(all_zen_scores[i]) >= n_samples
            if len(all_zen_scores[i]) > n_samples:
                all_zen_scores[i] = all_zen_scores[i][:n_samples]
    except (FileNotFoundError, AssertionError) as error:
        from datautils import train_data
        # from nas.model_profiler import count_parameters

        all_zen_scores = []

        for nlayers in N_LAYERS:
            for init in range(n_inits):
                model = get_model(nlayers=nlayers)
                # print(f'Saving model to {model_path}')
                torch.save(model, model_path)
                # print(f'The model has {count_parameters(model):,} trainable parameters')

                zen_scores = get_sample_zen_scores(train_samples=train_data[:n_samples])
                all_zen_scores.append(zen_scores)
        
        np.save(zen_path, np.array(all_zen_scores), allow_pickle=True)

    # import pdb; pdb.set_trace()
    plot(all_zen_scores, fig_path=args.fig_path)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fig_path', nargs='?', default=f'plots/IMDb_zen_rank_first{n_samples}.png', type=str)
    args = parser.parse_args()

    if os.path.isfile(model_path):
        os.system(f'rm {model_path}')
    
    main()

    if os.path.isfile(model_path):
        os.system(f'rm {model_path}')
