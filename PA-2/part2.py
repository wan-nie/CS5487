import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PA2_cluster_python.python import pa2
from utils import KMeans, EMGMM, MeanShift, KMeans2, MeanShift2


np.random.seed(56)


def cluster_image(path, method, params, axes):
    # load and show image
    img = Image.open(path)
    X, L = pa2.getfeatures(img, 7)
    X = X.T  # (N, 4)

    if method == 'K-means':
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        model = KMeans(n_clusters=params['n_clusters'])
        labels = model.fit_predict(X=X)

    elif method == 'EM-GMM':
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        model = EMGMM(n_clusters=params['n_clusters'])
        labels = model.fit_predict(X=X)

    elif method == 'Mean-shift':
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        model = MeanShift(bandwidth=params['bandwidth'], max_iter=params['max_iter'])
        labels = model.fit_predict(X=X, cutoff=0.01)

    elif method == 'K-means2':
        model = KMeans2(n_clusters=params['n_clusters'], lambda_=params['lambda_'])
        labels = model.fit_predict(X=X)

    elif method == 'Mean-shift2':
        model = MeanShift2(bandwidth=params['bandwidth'], lambda_=params['lambda_'])
        labels = model.fit_predict(X=X)

    else:
        raise Exception('Unknown clustering method!')

    labels += 1  # to match matlab
    segm = pa2.labels2seg(labels, L)  # make segmentation image from labels
    csegm = pa2.colorsegms(segm, img)  # color the segmentation image
    axes[0].imshow(img)
    axes[1].imshow(segm)
    axes[2].imshow(csegm)


def question_a():
    methods = ['K-means', 'EM-GMM', 'Mean-shift']
    for method in methods:
        for img_path in img_paths:
            img_name = os.path.basename(img_path).replace('.jpg', '')
            fig = plt.figure(figsize=(14, 4), constrained_layout=True)
            subfigs = fig.subfigures(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], wspace=0.1).flatten()

            if method != 'Mean-shift':
                # try different Ks
                K_ls = [2, 3, 4, 5]
                for i, K in enumerate(K_ls):
                    axes = subfigs[i].subplots(1, 3)
                    axes = axes.flatten()
                    subfigs[i].suptitle(method + f'; $K={K}$', fontsize=16, y=1)
                    cluster_image(
                        img_path,
                        method=method,
                        params={'n_clusters': K},
                        axes=axes
                    )

            else:
                # try different bandwidths
                h_ls = [0.2, 0.5, 0.8, 1.2]
                for i, h in enumerate(h_ls):
                    axes = subfigs[i].subplots(1, 3)
                    axes = axes.flatten()
                    subfigs[i].suptitle(method + f'; $h={h}$', fontsize=16, y=1)
                    cluster_image(img_path, method=method, params={'bandwidth': h, 'max_iter': 200}, axes=axes)

            fig.savefig(f'./figs/part2_a_{method}_{img_name}.png')


def question_b():
    methods = ['K-means2', 'Mean-shift2']
    for method in methods:
        for img_path in img_paths:
            img_name = os.path.basename(img_path).replace('.jpg', '')
            fig = plt.figure(figsize=(14, 4), constrained_layout=True)
            subfigs = fig.subfigures(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], wspace=0.1).flatten()

            if method == 'K-means2':
                K = 4
                lambda_ls = [0.001, 0.01, 0.05, 0.1]
                for i, lambda_ in enumerate(lambda_ls):
                    axes = subfigs[i].subplots(1, 3)
                    axes = axes.flatten()
                    subfigs[i].suptitle(method + f'; $K={K}$, $\lambda={lambda_}$', fontsize=16, y=1)
                    cluster_image(
                        img_path,
                        method=method,
                        params={'n_clusters': K, 'lambda_': lambda_},
                        axes=axes
                    )

            else:
                h = 4
                lambda_ls = [20, 10, 5, 1]
                for i, lambda_ in enumerate(lambda_ls):
                    axes = subfigs[i].subplots(1, 3)
                    axes = axes.flatten()
                    subfigs[i].suptitle(method + f'; $h_c/h_p={1/lambda_}$', fontsize=16, y=1)
                    cluster_image(
                        img_path,
                        method=method,
                        params={'bandwidth': h, 'lambda_': lambda_},
                        axes=axes
                    )

            fig.savefig(f'./figs/part2_b_{method}_{img_name}.png')


if __name__ == '__main__':
    img_dir = './PA2-cluster-images/images/'
    img_ls = ['12003.jpg', '299086.jpg']
    img_paths = [os.path.join(img_dir, img) for img in img_ls]

    """
    #####################(a)#####################
    """
    # question_a()

    """
    #####################(b)#####################
    """
    question_b()





