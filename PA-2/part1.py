import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import KMeans, EMGMM, MeanShift

np.random.seed(42)


def plot_points(data, labels, centers, ax, title):
    ax.scatter(
        x=data[:, 0],
        y=data[:, 1],
        c=labels,
        cmap='rainbow',
        s=20,
    )

    centers_scatter = ax.scatter(
        x=centers[:, 0],
        y=centers[:, 1],
        c=np.arange(len(centers)),
        cmap='rainbow',
        marker='+',
        s=200,
    )

    legend1 = ax.legend(*centers_scatter.legend_elements(),
                        loc="lower left", title="Clusters")
    ax.add_artist(legend1)

    ax.set_xlabel('$x_1$', fontsize=20)
    ax.set_ylabel('$x_2$', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_title(title, fontsize=20)


def get_kmeans_result(X, ax, title):
    model = KMeans(n_clusters=4)
    labels = model.fit_predict(X=X)
    plot_points(data=X, labels=labels, centers=model.centers_, ax=ax, title=title)


def get_gmm_result(X, ax, title):
    model = EMGMM(n_clusters=4)
    labels = model.fit_predict(X=X)
    plot_points(data=X, labels=labels, centers=model.params_['mu'], ax=ax, title=title)


def get_meanshift_result(X, ax, title, bandwidth):
    model = MeanShift(bandwidth=bandwidth)
    labels = model.fit_predict(X=X)
    plot_points(data=X, labels=labels, centers=model.centers_, ax=ax, title=title)


def question_b():
    fig = plt.figure(figsize=(20, 22), constrained_layout=True)
    subfigs = fig.subfigures(3, 1, height_ratios=[1, 1, 1], hspace=0.1).flatten()

    methods = ['K-means', 'EM-GMM', 'Mean-shift']
    for i, method in enumerate(methods):
        subfigs[i].suptitle(method, fontsize=20, fontweight='bold')
        axes = subfigs[i].subplots(1, 3).flatten()
        for j, (data_name, data) in enumerate(data_dic.items()):
            if method == 'K-means':
                get_kmeans_result(X=data.values, ax=axes[j], title=f'{data_name}; $K=4$')
                # if j == 2:
                #     axes[j].text(-9.5, 7.0, '$\mathbf{x}_a$', fontsize=20, color='r')
            elif method == 'EM-GMM':
                get_gmm_result(X=data.values, ax=axes[j], title=f'{data_name}; $K=4$')
            else:
                bandwidth = 1.7
                get_meanshift_result(X=data.values, ax=axes[j], title=f'{data_name}; $h={bandwidth}$', bandwidth=bandwidth)

    fig.savefig('./figs/part1_b.png')


def question_c():
    fig = plt.figure(figsize=(24, 20), constrained_layout=True)
    subfigs = fig.subfigures(3, 1, height_ratios=[1, 1, 1], hspace=0.1).flatten()

    for i, (data_name, data) in enumerate(data_dic.items()):
        if i in [0, 2]:
            bandwidths = [1, 2, 3, 4]
        else:
            bandwidths = [1, 1.5, 2, 2.5]
        subfigs[i].suptitle(data_name, fontsize=20, fontweight='bold')
        axes = subfigs[i].subplots(1, 4).flatten()
        for j, bandwidth in enumerate(bandwidths):
            get_meanshift_result(data.values, ax=axes[j], title=f'$h={bandwidth}$', bandwidth=bandwidth)

    fig.savefig('./figs/part1_c.png')


if __name__ == '__main__':
    data_dir = r'./PA2-cluster-data/cluster_data_text'
    df_X1 = pd.read_csv(os.path.join(data_dir, 'cluster_data_dataA_X.txt'), sep='\t', header=None).dropna(
        axis=1).T.reset_index(drop=True)

    df_X2 = pd.read_csv(os.path.join(data_dir, 'cluster_data_dataB_X.txt'), sep='\t', header=None).dropna(
        axis=1).T.reset_index(drop=True)

    df_X3 = pd.read_csv(os.path.join(data_dir, 'cluster_data_dataC_X.txt'), sep='\t', header=None).dropna(
        axis=1).T.reset_index(drop=True)

    data_dic = {
        'dataA': df_X1,
        'dataB': df_X2,
        'dataC': df_X3
    }

    """
    #####################(b)#####################
    """
    question_b()

    """
    #####################(c)#####################
    """
    question_c()
