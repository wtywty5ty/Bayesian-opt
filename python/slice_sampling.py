from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from sklearn.datasets import make_regression
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


class Gaussian_Process():
    def __init__(self, kernel):
        self.kernel = kernel
        self.gp_ = GaussianProcessRegressor(kernel=kernel,
                                            optimizer=None,
                                            normalize_y=True)

    def optimise__with_splice_sampling(self, initial_theta, num_iters, sigma, burn_in, X, y):
        self.fit(X, y)
        slice_sampler = Slice_sampler(num_iters=num_iters,
                                      sigma=sigma,
                                      burn_in=burn_in,
                                      gp=self)
        samples = slice_sampler.sample(init=initial_theta)

        theta_opt = [np.mean(samples_k) for samples_k in samples]

        # kernel__k1__noise_level = noise of the data
        # kernel__k2__k1__constant_value = aplitude
        # kernel__k2__k2__length_scale = Matern length scales

        self.gp_.set_params(**{"kernel__k1__noise_level": np.abs(theta_opt[0]),
                               "kernel__k2__k1__constant_value": np.abs(theta_opt[1]),
                               "kernel__k2__k2__length_scale": theta_opt[2:]})

    def log_marginal_likelihood(self, theta):
        theta = [theta[0], theta[1], theta[2:]]
        return self.gp_.log_marginal_likelihood(theta)

    def log_prior_parameters(self, theta):
        return np.sum(np.log([ss.norm(0, 1).pdf(theta_k) for theta_k in theta]))

    def log_joint_unnorm(self, theta):
        return self.log_marginal_likelihood(theta) + self.log_prior_parameters(theta)

    def fit(self, X, y):
        self.gp_.fit(X, y)

    def predict(self, X, return_std=True):
        mu, sigma = self.gp_.predict(X, return_std)
        return mu, sigma

    def get_params(self):
        return self.gp_.get_params()


class Slice_sampler():
    def __init__(self, num_iters, sigma, burn_in, gp):
        self.num_iters = num_iters
        self.sigma = sigma
        self.burn_in = burn_in
        self.gp = gp

    def sample(self, init, step_out=True):

        D = len(init)
        samples = np.zeros((D, self.num_iters))

        xx = init.copy()

        for i in xrange(self.num_iters + self.burn_in):
            perm = range(D)
            np.random.shuffle(perm)
            last_llh = self.gp.log_joint_unnorm(xx)

            for d in perm:
                llh0 = last_llh + np.log(np.random.rand())
                rr = np.random.rand(1)
                x_l = xx.copy()
                x_l[d] = x_l[d] - rr * self.sigma[d]
                x_r = xx.copy()
                x_r[d] = x_r[d] + (1 - rr) * self.sigma[d]

                if step_out:
                    llh_l = self.gp.log_joint_unnorm(x_l)
                    while llh_l > llh0:
                        x_l[d] = x_l[d] - self.sigma[d]
                        llh_l = self.gp.log_joint_unnorm(x_l)
                    llh_r = self.gp.log_joint_unnorm(x_r)
                    while llh_r > llh0:
                        x_r[d] = x_r[d] + self.sigma[d]
                        llh_r = self.gp.log_joint_unnorm(x_r)

                x_cur = xx.copy()
                while True:
                    xd = np.array(np.random.rand() * (x_r[d] - x_l[d]) + x_l[d])
                    x_cur[d] = xd.copy()
                    last_llh = self.gp.log_joint_unnorm(x_cur)
                    if last_llh > llh0:
                        xx[d] = xd.copy()
                        break
                    elif xd > xx[d]:
                        x_r[d] = xd
                    elif xd < xx[d]:
                        x_l[d] = xd
                    else:
                        raise RuntimeError('Slice sampler shrank too far.')

            if i == 0:
                print
                "burn-in"
            elif i > self.burn_in and i % 100 == 0:
                print
                'iteration', i - self.burn_in

            if i > self.burn_in:
                samples[:, i - self.burn_in] = xx.copy().ravel()

        # plt.hist(samples[0])

        return samples
