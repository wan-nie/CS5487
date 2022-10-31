import numpy as np
import numpy.linalg as LA
from scipy.stats import multivariate_normal


THRESHOLD = 1e-6
MAX_ITER = 300
N_INIT = 50
SIGMA_ALPHA = 1e-6  # trick from Antoni's lecture, used to prevent singularities of covariance
EPSILON = 1e-6


def get_distance_matrix(X):
    coords1 = np.expand_dims(X, 0)
    coords2 = np.expand_dims(X, 1)
    dm = LA.norm(coords1 - coords2, axis=-1)  # (N, N)
    return dm


def get_init_centers(X, K, method):
    if method == 'random':
        centers = X[np.random.choice(len(X), K, replace=False)]

    elif method == 'farthest':
        centers = []
        idx_ls = []
        dm = get_distance_matrix(X)

        # get the first two farthest points
        first2_idx = list(np.unravel_index(np.argmax(dm, axis=None), dm.shape))
        idx_ls += first2_idx
        centers += list(X[first2_idx])

        # get other K-2 centers
        for _ in range(K - 2):
            sort_ = np.argsort(dm[idx_ls].mean(axis=0))[::-1]
            sort_ = sort_[~np.isin(sort_, idx_ls)]
            idx = sort_[0]
            # the point farthest from all previous points
            idx_ls.append(idx)
            centers.append(X[idx])

        centers = np.stack(centers, axis=0)

    else:
        raise Exception('Unknown method for generating initial centers')

    return centers.copy()


class KMeans:
    def __init__(
            self,
            n_clusters=4,
            threshold=THRESHOLD,
            max_iter=MAX_ITER,
            n_init=N_INIT,
            init_method='random'
    ):
        self.K_ = n_clusters
        self.threshold = threshold
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_method = init_method
        self.centers_ = None

    def get_distance_to_centers(self, X, centers):
        coords1 = np.expand_dims(X, 1)
        coords2 = np.expand_dims(centers, 0)
        dc = LA.norm(coords1 - coords2, axis=-1)  # (N, K)
        return dc

    def fit(self, X):
        """
            From Antoni's lecture, random initialization is not very robust for K-means because it could start near a bad
            local minimum. Run several trials with different random initializations, and then select the trial that
            results in the smallest total distance to the centers.
        :param X: (N, d); d-dim data points
        """

        inertia_min = 1e36
        best_centers = None
        for _ in range(self.n_init):
            centers = get_init_centers(X, self.K_, method=self.init_method)  # (K, d)

            # update centers
            for i in range(self.max_iter):
                centers_old = centers.copy()

                # update self.centers according to Z
                dc = self.get_distance_to_centers(X, centers)  # (N, K)
                Z = dc.argmin(axis=1)  # (N, )
                for k in range(len(centers)):
                    centers[k] = X[Z == k].mean(axis=0)

                # check whether to stop according to the mean center change
                change_percent = LA.norm(centers - centers_old, axis=-1).mean() / LA.norm(centers_old, axis=-1).mean()
                # print('{:3}: mean center change percent {:.5f}'.format(i, change_percent))
                if change_percent < self.threshold:
                    break

            #
            dc = self.get_distance_to_centers(X, centers)
            # sum of distances of samples to their closest cluster center
            inertia = dc[range(len(dc)), dc.argmin(axis=1)].sum()

            if inertia < inertia_min:
                inertia_min = inertia
                best_centers = centers

        if best_centers is not None:
            self.centers_ = best_centers
        else:
            raise Exception('K-means failed, check your data!')

    def fit_predict(self, X):
        self.fit(X)
        dc = self.get_distance_to_centers(X, self.centers_)
        return dc.argmin(axis=1)


class EMGMM:
    def __init__(
            self,
            n_clusters=4,
            threshold=THRESHOLD,
            max_iter=MAX_ITER,
            n_init=N_INIT,
            init_method='random'
    ):
        self.K_ = n_clusters
        self.threshold = threshold
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_method = init_method
        self.params_ = None

    def get_init_params(self, X, variance):
        # pi
        pis = np.random.random(self.K_)  # (K, )
        pis /= pis.sum()

        # mu
        mus = get_init_centers(X, self.K_, method=self.init_method)  # (K, d)

        # sigma
        sigmas = np.array([np.eye(X.shape[1]) * variance for _ in range(self.K_)])  # (K, d, d)

        return pis, mus, sigmas

    def e_step(self, X, pis, mus, sigmas):
        """
        get the Z matrix (N, K), each element is the posterior, \hat{z}_{ij}
        :param X: (N, d)
        :param pis: (K, )
        :param mus: (K, d)
        :param sigmas: (K, d, d)
        """
        joint_prob_ls = []
        for j in range(self.K_):
            pi = pis[j]
            mu = mus[j]
            sigma = sigmas[j]
            #
            p_x_zj = pi * multivariate_normal.pdf(X, mu, sigma)  # the joint prob of all x with jth gaussian, (N, )
            joint_prob_ls.append(p_x_zj)

        joint_prob_matrix = np.stack(joint_prob_ls, axis=0).T  # (N, K), every element is the joint prob p(x_i, z_j)
        p_x = joint_prob_matrix.sum(axis=1, keepdims=True)  # (N, 1), every element is the likelihood p(x_i)
        p_x += EPSILON  # add epsilon for numeric stability
        Z = joint_prob_matrix / p_x  # (N, K)

        return Z, np.sum(np.log(p_x))

    def fit(self, X):
        """
            From Antoni's lecture, random initialization is not very robust for EM because it could start near a bad
            local minimum. Run several trials with different random initializations, and then select the trial that
            results in the largest data log-likelihood, i.e., logP(X).
        :param X: (N, d); d-dim data points
        """

        ll_max = -1e36
        best_params = None
        for _ in range(self.n_init):
            pis, mus, sigmas = self.get_init_params(X, variance=2)
            Z, ll = self.e_step(X=X, pis=pis, mus=mus, sigmas=sigmas)

            # update params
            for i in range(self.max_iter):
                ll_old = ll
                pi_new_ls = []
                mu_new_ls = []
                sigma_new_ls = []
                for j in range(self.K_):
                    z_j = Z[:, [j]]  # (N, 1)
                    N_j = z_j.sum()

                    #
                    pi_j_new = N_j / X.shape[0]
                    pi_new_ls.append(pi_j_new)
                    #
                    mu_j_new = (z_j * X).sum(axis=0, keepdims=True) / N_j  # (1, d)
                    mu_new_ls.append(mu_j_new.squeeze())  # (d, )
                    #
                    sigma = np.matmul(((X - mu_j_new) * z_j).T, X - mu_j_new) / N_j  # (d, d)
                    sigma += np.eye(sigma.shape[0]) * SIGMA_ALPHA  # trick from Antoni's lecture, to prevent singularities
                    sigma_new_ls.append(sigma)

                pis, mus, sigmas = np.array(pi_new_ls), np.array(mu_new_ls), np.array(sigma_new_ls)

                # check whether to stop according to the log-likelihood change percent
                Z, ll = self.e_step(X=X, pis=pis, mus=mus, sigmas=sigmas)
                change_percent = np.abs((ll - ll_old) / ll_old)
                # print('{:3}: delta-LL% {:.5f}'.format(i, change_percent))
                if change_percent < self.threshold:
                    break

            #
            if ll > ll_max:
                ll_max = ll
                best_params = {
                    'pi': pis,  # (K, )
                    'mu': mus,  # (K, d)
                    'sigma': sigmas  # (K, d, d)
            }

        if best_params is not None:
            self.params_ = best_params
        else:
            raise Exception('EM-GMM failed, check your data!')

    def fit_predict(self, X):
        self.fit(X)
        Z, _ = self.e_step(X, self.params_['pi'], self.params_['mu'], self.params_['sigma'])
        return Z.argmax(axis=1)


class MeanShift:
    def __init__(
            self,
            bandwidth=2,
            threshold=THRESHOLD,
            max_iter=MAX_ITER
    ):
        self.bandwidth = bandwidth
        self.threshold = threshold
        self.max_iter = max_iter
        self.centers_ = None

    @staticmethod
    def group_X_hat(X_hat, cutoff):
        """
        group data points in X_hat within a distance cutoff
        """
        X_hat = X_hat.copy()
        dm = get_distance_matrix(X_hat) / X_hat.shape[1]
        a1, a2 = np.where(dm < cutoff)
        indices = np.arange(len(dm))
        centers = []
        while len(indices) > 0:
            i = indices[0]
            group_idx = a2[a1 == i]
            mean_coord = X_hat[group_idx].mean(axis=0)  # mean coord of the group
            X_hat[group_idx] = mean_coord
            centers.append(mean_coord)
            indices = indices[~np.isin(indices, group_idx)]

        return X_hat, np.stack(centers, axis=0)

    def fit(self, X, cutoff):
        """
        :param X: (N, d); d-dim data points
        :param cutoff: distance below which to group modes
        """
        X_hat = X.copy()  # initialize \hat{x}^{(0)}}

        for i in range(self.max_iter):
            X_hat_old = X_hat.copy()  # \hat{x}^{(t)}}
            #
            coord1 = np.expand_dims(X_hat, 1) / self.bandwidth  # (N, 1, d)
            coord2 = np.expand_dims(X, 0) / self.bandwidth  # (1, N, d)
            exp = np.exp(
                -0.5 * np.sum((coord1 - coord2) ** 2, axis=-1))  # (N, N), element_ij is exp(-0.5*||(xi-xj)/h||^2)

            # calculate \hat{x}^{(t+1)}}
            X_hat = (np.expand_dims(exp.T, 2) * np.expand_dims(X, 1)).sum(axis=0) / exp.sum(axis=1, keepdims=True)  # (N, d)

            change_percent = LA.norm(X_hat - X_hat_old, axis=-1) / LA.norm(X_hat_old, axis=-1)
            # check whether to stop according to the number of stop_shifting
            if (change_percent < self.threshold).sum() == len(X_hat):
                break

        X_hat, self.centers_ = self.group_X_hat(X_hat, cutoff=cutoff)
        return X_hat

    def fit_predict(self, X, cutoff=0.01):
        X_hat = self.fit(X, cutoff)
        labels = (np.expand_dims(X_hat, 1) == np.expand_dims(self.centers_, 0)).sum(axis=-1).argmax(axis=-1)
        return labels


class KMeans2(KMeans):
    def __init__(
            self,
            n_clusters=4,
            lambda_=0.1,
            threshold=THRESHOLD,
            max_iter=MAX_ITER,
            n_init=N_INIT,
            init_method='random'
    ):
        super().__init__(
            n_clusters=n_clusters,
            threshold=threshold,
            max_iter=max_iter,
            n_init=n_init,
            init_method=init_method
        )
        self.lambda_ = lambda_

    def get_distance_to_centers(self, X, centers):
        coords1 = np.expand_dims(X, 1)
        coords2 = np.expand_dims(centers, 0)

        # apply a weighting between the feature types to calculate distance to centers
        dc = (coords1 - coords2) ** 2  # (N, K, d)
        dc = dc[:, :, :2].sum(axis=-1) + self.lambda_ * dc[:, :, 2:].sum(axis=-1)  # (N, K)
        dc = np.sqrt(dc)

        return dc


class MeanShift2(MeanShift):
    def __init__(
            self,
            bandwidth=2,
            lambda_=2,
            threshold=THRESHOLD,
            max_iter=MAX_ITER
    ):
        super().__init__(
            bandwidth=bandwidth,
            threshold=threshold,
            max_iter=max_iter
        )
        # lambda_ is the ratio of h_p and h_c, i.e., lambda_ = h_p/h_c
        self.bandwidth = np.array([bandwidth] * 2 + [bandwidth * lambda_] * 2)
