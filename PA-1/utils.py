import numpy as np
import numpy.linalg as LA
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False


class Regression:
    def __init__(self, x, y, method, poly_order=None, lambda_=None, bayesian_alpha=None, bayesian_sigma_squared=None):
        """
        x: numpy array; array of x
        y: numpy array; array of y
        method: string; regression method
        poly_order: integer; Kth order
        lambda_: float; lambda of regularized_LS and LASSO
        bayesian_alpha: float; variance of parameter theta
        bayesian_sigma_squared: float; variance of noise
        """

        self.x = x.flatten()
        self.y = y.flatten().reshape(-1, 1)  # N x 1, column vector
        self.method = method
        self.poly_order = poly_order
        self.lambda_ = lambda_
        self.bayesian_alpha = bayesian_alpha
        self.bayesian_sigma_squared = bayesian_sigma_squared
        self.map = self.poly_map

        self.theta_hat = None
        self.mu_hat = None  # only for Bayesian regression
        self.Sigma_hat = None  # only for Bayesian regression

    def poly_map(self, x):
        def map_x(x):
            return [x ** k for k in range(self.poly_order + 1)]

        return np.apply_along_axis(map_x, 0, x)

    def fit(self):
        theta = None
        mu = None
        Sigma = None

        # [phi(x1), ... , phi(xn)]
        phi = self.map(self.x)  # (K + 1) x N

        if self.method == 'least_squares':
            theta = LA.inv(phi @ phi.T) @ phi @ self.y

        elif self.method == 'regularized_LS':
            theta = LA.inv(phi @ phi.T + self.lambda_ * np.eye(len(phi))) @ phi @ self.y

        elif self.method == 'L1-regularized_LS':
            """
                min    1/2*x.T@P@x + q.T@x
                s.t.       G@x <= h
            """
            block1 = phi @ phi.T
            block2 = phi @ self.y

            #  [[phi@phiT, -phi@phi.T], [-phi@phi.T, phi@phiT]]
            P = np.block([
                [block1, -block1],
                [-block1, block1]
            ])

            # lambda*1 - [phi@y, -phi]
            q = self.lambda_ * np.ones(phi.shape[0] * 2).reshape(-1, 1) - np.block([[block2], [-block2]])

            # -I
            G = -1 * np.eye(phi.shape[0] * 2)

            # 0 vector
            h = np.zeros(phi.shape[0] * 2).reshape(-1, 1)

            sol = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))
            x = np.array(sol['x'])  # [theta_positive, theta_negative]

            # check result
            assert round((0.5 * x.T @ P @ x + q.T @ x).item(), 6) == round(sol['primal objective'], 6)

            # get theta from x
            theta = np.array(x[:phi.shape[0]] - x[phi.shape[0]:]).reshape(-1, 1)

        elif self.method == 'robust_regression':
            """
            min      c.T@x 
            s.t.     G@x <= h
            """
            # [0_(K+1), 1_(N)]
            c = np.concatenate((np.zeros(phi.shape[0]), np.ones(phi.shape[1])), axis=0).reshape(-1, 1)

            # [[-phi.T, -I], [phi.T, -I]]
            G = np.block([
                [-phi.T, -np.eye(phi.shape[1])],
                [phi.T, -np.eye(phi.shape[1])]
            ])

            # [[-y], [y]]
            h = np.block([
                [-self.y],
                [self.y]
            ])

            sol = solvers.lp(matrix(c), matrix(G), matrix(h))
            x = np.array(sol['x'])  # [theta, t]

            # check result
            assert round((c.T @ x).item(), 6) == round(sol['primal objective'], 6)

            # get theta from x
            theta = np.array(x[:phi.shape[0]])

        elif self.method == 'bayesian_regression':
            """
            posterior
            """
            Sigma = LA.inv(np.eye(phi.shape[0]) / self.bayesian_alpha + phi @ phi.T / self.bayesian_sigma_squared)
            mu = Sigma @ phi @ self.y / self.bayesian_sigma_squared

        # for other regression methods
        self.theta_hat = theta

        # for bayesian regression method
        self.mu_hat = mu
        self.Sigma_hat = Sigma

    def predict(self, x):
        if self.method != 'bayesian_regression':
            phi = self.map(x)
            y = phi.T @ self.theta_hat
            return y.flatten()
        else:
            phi = self.map(x)
            mu = phi.T @ self.mu_hat  # mean
            sigma = np.sqrt((phi.T @ self.Sigma_hat @ phi).diagonal())  # standard deviation
            return mu.flatten(), sigma.flatten()


class Regression2(Regression):
    def __init__(self, x, y, method, map_method='identity', lambda_=None, bayesian_alpha=None,
                 bayesian_sigma_squared=None):
        """
        x: numpy array; array of x
        y: numpy array; array of y
        method: string; regression method; select one from ['least_squares']
        map_method: string; 'identity' or 'poly'
        lambda_: float; lambda of regularized_LS and LASSO
        bayesian_alpha: float; variance of parameter theta
        bayesian_sigma_squared: float; variance of noise
        """
        super().__init__(
            x=x,
            y=y,
            method=method,
            lambda_=lambda_,
            bayesian_alpha=bayesian_alpha,
            bayesian_sigma_squared=bayesian_sigma_squared
        )
        self.x = x
        if map_method == 'identity':
            self.map = self.identity_map
        elif map_method == 'poly1':
            self.map = self.poly_map1
        elif map_method == 'poly2':
            self.map = self.poly_map2
        else:
            raise Exception("map_method not identified.")

    @staticmethod
    def identity_map(x):
        return x

    @staticmethod
    def poly_map1(x):
        def map_x(x):
            poly_term = x ** 2
            return np.concatenate((x, poly_term), axis=0)

        return np.apply_along_axis(map_x, 0, x)

    @staticmethod
    def poly_map2(x):
        def map_x(x):
            poly_term1 = x ** 2
            poly_term2 = x ** 3
            return np.concatenate((x, poly_term1, poly_term2), axis=0)

        return np.apply_along_axis(map_x, 0, x)



