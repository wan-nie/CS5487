import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import Regression2

np.random.seed(742)


def reg_and_plot(train_data_x, train_data_y, test_data_x, test_data_y, map_method, lambda_, bayesian_alpha,
                 bayesian_sigma_squared):
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)
    axes[2][1].set_visible(False)
    axes = axes.flatten()
    mse_ls = []
    mae_ls = []
    for i, method in enumerate(methods):
        ax = axes[i]
        # fit
        reg = Regression2(
            train_data_x,
            train_data_y,
            method=method,
            map_method=map_method,
            lambda_=lambda_,
            bayesian_alpha=bayesian_alpha,
            bayesian_sigma_squared=bayesian_sigma_squared
        )
        reg.fit()

        test_data_y = test_data_y.flatten()
        if method != 'bayesian_regression':
            pre_y = reg.predict(test_data_x)
            mse = ((pre_y - test_data_y) ** 2).mean()
            mae = (np.abs(pre_y - test_data_y)).mean()
            tmp_df = pd.DataFrame(
                {'test_count': test_data_y.round(0).astype(int), 'pre_count': pre_y.round(0).astype(int)})
        else:
            pre_mu, pre_sigma = reg.predict(test_data_x)
            mse = ((pre_mu - test_data_y) ** 2).mean()
            mae = (np.abs(pre_mu - test_data_y)).mean()
            tmp_df = pd.DataFrame(
                {'test_count': test_data_y.round(0).astype(int), 'pre_count': pre_mu.round(0).astype(int)})

        mse_ls.append(mse)
        mae_ls.append(mae)

        ax.scatter(range(len(tmp_df)), tmp_df.test_count, marker='o', label='true counts', s=3)
        ax.scatter(range(len(tmp_df)), tmp_df.pre_count, marker='o', label=f'predicted counts', s=3)

        ax.set_title(f'({i + 1}) {method}; MSE={round(mse, 2)}, MAE={round(mae, 2)}', fontsize=14)
        ax.set_xlabel('data', fontsize=12)
        ax.set_ylabel('count', fontsize=12)
        ax.legend()
    plt.show()
    return tmp_df, fig


if __name__ == '__main__':
    data_dir = './PA-1-data-text'
    fig_dir = './figs'

    df_trainx = pd.read_csv(os.path.join(data_dir, 'count_data_trainx.txt'), sep='\t', header=None)
    df_trainy = pd.read_csv(os.path.join(data_dir, 'count_data_trainy.txt'), sep='\t', header=None)
    df_trainx = df_trainx.dropna(axis=1)
    df_trainy = df_trainy.dropna(axis=1)

    df_testx = pd.read_csv(os.path.join(data_dir, 'count_data_testx.txt'), sep='\t', header=None)
    df_testy = pd.read_csv(os.path.join(data_dir, 'count_data_testy.txt'), sep='\t', header=None)
    df_testx = df_testx.dropna(axis=1)
    df_testy = df_testy.dropna(axis=1)

    methods = [
        'least_squares',
        'regularized_LS',
        'L1-regularized_LS',
        'robust_regression',
        'bayesian_regression'
    ]
    train_x = df_trainx.values
    train_y = df_trainy.values
    test_x = df_testx.values
    test_y = df_testy.values

    """
    #####################(a)#####################
    """
    _, plt_obj = reg_and_plot(
        train_x,
        train_y,
        test_x,
        test_y,
        map_method='identity',
        lambda_=1.,
        bayesian_alpha=1.,
        bayesian_sigma_squared=5.
    )
    plt_obj.savefig(os.path.join(fig_dir, 'part2-a.png'))

    """
    #####################(b)#####################
    """
    _, plt_obj = reg_and_plot(
        train_x,
        train_y,
        test_x,
        test_y,
        map_method='poly1',
        lambda_=1.,
        bayesian_alpha=1.,
        bayesian_sigma_squared=5.
    )
    plt_obj.savefig(os.path.join(fig_dir, 'part2-b1.png'))

    _, plt_obj = reg_and_plot(
        train_x,
        train_y,
        test_x,
        test_y,
        map_method='poly2',
        lambda_=1.,
        bayesian_alpha=1.,
        bayesian_sigma_squared=5.
    )
    plt_obj.savefig(os.path.join(fig_dir, 'part2-b2.png'))

