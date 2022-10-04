import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import Regression

np.random.seed(742)


def plot_reg(train_sample_x, train_sample_y, method_reg_record):
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)
    axes[2][1].set_visible(False)
    axes = axes.flatten()
    for i, (method, record) in enumerate(method_reg_record.items()):
        ax = axes[i]
        ax.scatter(x=train_sample_x, y=train_sample_y, color='k', s=8)  # plot samples
        plot_x = record[0]
        plot_y = record[1]
        mse = record[-1]
        if method != 'bayesian_regression':
            ax.plot(plot_x, plot_y, color='b', label=method)
        else:
            plot_sigma = record[2]
            ax.fill_between(plot_x, plot_y - plot_sigma, plot_y + plot_sigma, color='b',
                            label='bayesian_regression (one σ)', alpha=0.3)

        ax.legend()
        ax.set_title(f'({i + 1}) {method}; MSE={round(mse, 2)}', fontsize=14)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)

    fig.suptitle(f'# of data: {len(train_sample_x)}', fontsize=16, fontweight='bold')
    plt.show()
    return fig


def reg_and_plot(train_sample_x, train_sample_y, poly_order, lambda_, bayesian_alpha, bayesian_sigma_squared,
                 plot=True):
    mse_ls_ = []
    param_ls_ = []
    method_reg_record = {}
    figure = None
    for _, method in enumerate(methods):
        # fit
        reg = Regression(
            train_sample_x,
            train_sample_y,
            method=method,
            poly_order=poly_order,
            lambda_=lambda_,
            bayesian_alpha=bayesian_alpha,
            bayesian_sigma_squared=bayesian_sigma_squared
        )
        reg.fit()

        # get prediction and mse
        poly_x = poly_data.x.values
        poly_y = poly_data.y.values
        if method != 'bayesian_regression':
            pre_y = reg.predict(poly_x)
            mse = ((pre_y.flatten() - poly_y.flatten()) ** 2).mean()
            mse_ls_.append(mse)
            param_ls_.append([reg.theta_hat])

            method_reg_record[method] = [poly_x, pre_y, mse]
        else:
            pre_mu, pre_sigma = reg.predict(poly_x)
            mse = ((pre_mu.flatten() - poly_y.flatten()) ** 2).mean()
            mse_ls_.append(mse)
            param_ls_.append([reg.mu_hat, reg.Sigma_hat])

            method_reg_record[method] = [poly_x, pre_mu, pre_sigma, mse]

    if plot:
        figure = plot_reg(train_sample_x, train_sample_y, method_reg_record)

    return mse_ls_, param_ls_, figure


if __name__ == '__main__':
    data_dir = './PA-1-data-text'
    fig_dir = './figs'

    df_x = pd.read_csv(os.path.join(data_dir, 'polydata_data_sampx.txt'), sep='\t', header=None)
    df_y = pd.read_csv(os.path.join(data_dir, 'polydata_data_sampy.txt'), sep='\t', header=None)
    data = pd.DataFrame({'x': df_x.values[0][:-1], 'y': df_y.values[:, 0]})
    data = data.sort_values('x').reset_index(drop=True)

    df_poly_x = pd.read_csv(os.path.join(data_dir, 'polydata_data_polyx.txt'), sep='\t', header=None)
    df_poly_y = pd.read_csv(os.path.join(data_dir, 'polydata_data_polyy.txt'), sep='\t', header=None)
    poly_data = pd.DataFrame({
        'x': df_poly_x.values[0][:-1],
        'y': df_poly_y.values[:, 0]
    })

    methods = [
        'least_squares',
        'regularized_LS',
        'L1-regularized_LS',
        'robust_regression',
        'bayesian_regression'
    ]

    """
    #####################(b)#####################
    """
    sample_x = data.x.values
    sample_y = data.y.values
    _, _, plt_obj = reg_and_plot(
        sample_x,
        sample_y,
        poly_order=5,
        lambda_=5.,
        bayesian_alpha=1.,
        bayesian_sigma_squared=10.
    )
    plt_obj.savefig(os.path.join(fig_dir, 'part1-b.png'))

    """
    #####################(c)#####################
    """
    ratio_ls = [0.15, 0.25, 0.5, 0.75]  # I did not use 0.1 because of the rank issue of LP
    for ratio in ratio_ls:
        sub_index = sorted(np.random.choice(len(data), int(ratio * len(data)), replace=False))
        sub_data = data.iloc[sub_index, :]
        sample_x = sub_data.x.values
        sample_y = sub_data.y.values

        _, _, plt_obj = reg_and_plot(
            sample_x,
            sample_y,
            poly_order=5,
            lambda_=5.,
            bayesian_alpha=1.,
            bayesian_sigma_squared=10.
        )
        plt_obj.savefig(os.path.join(fig_dir, f'part1-c-{ratio}.png'))

    mse_record = {
        0.15: [],
        0.25: [],
        0.5: [],
        0.75: []
    }
    runs = 50
    for ratio in mse_record.keys():
        for _ in range(runs):
            sub_index = sorted(np.random.choice(len(data), int(ratio * len(data)), replace=False))
            sub_data = data.iloc[sub_index, :]
            sample_x = sub_data.x.values
            sample_y = sub_data.y.values
            mse_ls, _, _ = reg_and_plot(
                sample_x,
                sample_y,
                poly_order=5,
                lambda_=5.,
                bayesian_alpha=1.,
                bayesian_sigma_squared=10.,
                plot=False
            )
            mse_record[ratio].append(mse_ls)
        mse_record[ratio] = np.mean(mse_record[ratio], axis=0)
    mse_df = pd.DataFrame(mse_record, index=methods)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    for i, method in enumerate(methods):
        x = mse_df.loc[method].index
        y = mse_df.loc[method].values
        ax.plot(x, y, label=method, marker='.', markersize=6)
        ax.set_title(f'{method}', fontsize=14)
        ax.set_xlabel('Training size', fontsize=12)
        ax.set_ylabel('MSE', fontsize=12)
        ax.legend()
        ax.set_xticks(x)
    ax.set_ylim([0, 30])
    fig.suptitle(f'MSE ({runs} runs average) versus Training size ', fontsize=16, fontweight='bold')
    plt.show()
    fig.savefig(os.path.join(fig_dir, f'part1-c-MSEvsTrainingSize.png'))

    """
    #####################(d)#####################
    """
    outlier_df = pd.DataFrame({'x': [-1.0, -0.5, 0.2, 0.1], 'y': [200, 200, 200, 200]})
    outlier_df = pd.concat((data, outlier_df))
    outlier_df = outlier_df.sort_values('x').reset_index(drop=True)
    sample_x = outlier_df.x.values
    sample_y = outlier_df.y.values
    #
    _, _, plt_obj = reg_and_plot(
        sample_x,
        sample_y,
        poly_order=5,
        lambda_=5.,
        bayesian_alpha=1.,
        bayesian_sigma_squared=10.
    )
    plt_obj.savefig(os.path.join(fig_dir, 'part1-d.png'))

    """
    #####################(e)#####################
    """
    sample_x = data.x.values
    sample_y = data.y.values
    mse_ls, param_ls, plt_obj = reg_and_plot(
        sample_x,
        sample_y,
        poly_order=10,
        lambda_=5.,
        bayesian_alpha=1.,
        bayesian_sigma_squared=10.
    )
    plt_obj.savefig(os.path.join(fig_dir, 'part1-e.png'))

    param_dic = {}
    for k, method in enumerate(methods[:-1]):
        param_dic[method] = param_ls[k][0].flatten().round(3)
    param_dic[f'{methods[-1]}_mu'] = param_ls[-1][0].flatten().round(3)
    param_dic[f'{methods[-1]}_deviation'] = param_ls[-1][1].diagonal().flatten().round(3)
    param_df = pd.DataFrame(param_dic)
    param_df.to_csv('part1-e-params.csv', index=False)
