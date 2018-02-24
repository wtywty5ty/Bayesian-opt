from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from sklearn.datasets import make_regression
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss

'''
def log_marginal_likelihood(gp, theta):
    return gp.log_marginal_likelihood(theta)
    
def log_prior_parameters(theta):
    return np.sum(np.log([ss.norm(0, 1).pdf(theta_k) for theta_k in theta]))
    
def log_joint_unnorm(gp, theta):
    return log_marginal_likelihood(gp, theta) + log_prior_parameters(theta)
'''


class Gaussian_Process():
    
    def __init__(self, kernel, mode):
        self.kernel = kernel
        if mode == 'OPT':
            self.gp = GaussianProcessRegressor(kernel=kernel,
                                               n_restarts_optimizer=10,
                                               normalize_y=True)
        elif mode == 'MAP':
            self.gp = GaussianProcessRegressor(kernel=kernel,
                                               optimizer = None,
                                               n_restarts_optimizer=10,
                                               normalize_y=True)
        else:
            raise Exception('Wrong GP initial mode!!!')
    '''
    def optimise__with_splice_sampling(self, initial_theta, num_iters, sigma, burn_in,  X, y):
        self.fit(X, y)
        slice_sampler = Slice_sampler(num_iters = num_iters, 
                                      sigma = sigma, 
                                      burn_in = burn_in,
                                      gp = self)
        samples = slice_sampler.sample(init = initial_theta)
        theta_opt = [np.mean(samples[0]),np.mean(samples[1])]
        
        self.gp.set_params(**{"kernel__k1__length_scale": np.mean(samples[0]),
                            "kernel__k2__noise_level": np.power(np.mean(samples[0]),2)})
    '''
    
    
    def set_params(self, theta_set):
        self.gp.set_params(**{"kernel__k1__noise_level": np.abs(theta_set[0]),
                        "kernel__k2__k1__constant_value": np.abs(theta_set[1]),
                        "kernel__k2__k2__length_scale": theta_set[2:]})

    def log_marginal_likelihood(self,theta):
        return self.gp.log_marginal_likelihood(theta)
    
    def log_prior_parameters(self, theta):
        return np.sum(np.log([ss.norm(0, 1).pdf(theta_k) for theta_k in theta]))
    
    def log_joint_unnorm(self, theta):
        return self.log_marginal_likelihood(theta) + self.log_prior_parameters(theta)
        
    def fit(self, X, y):
        self.gp.fit(X, y)
    
    def predict(self, X):
        mu, sigma = self.gp.predict(X, return_std =True)
        return mu, sigma



class Slice_sampler():
    
    def __init__(self, num_iters, sigma, burn_in):
        self.num_iters = num_iters # specify number of samples we want to draw
        self.sigma = sigma
        self.burn_in = burn_in
        
    def sample(self, init, gp, step_out=True):

        D = len(init)
        samples = np.zeros((D, self.num_iters))
        
        xx = init.copy()

        for i in range(self.num_iters + self.burn_in):
            perm = list(range(D))
            np.random.shuffle(perm)
            last_llh = gp.log_joint_unnorm(xx)

            for d in perm:
                llh0 = last_llh + np.log(np.random.rand())
                rr = np.random.rand(1)
                x_l = xx.copy()
                x_l[d] = x_l[d] - rr * self.sigma[d]
                x_r = xx.copy()
                x_r[d] = x_r[d] + (1 - rr) * self.sigma[d]

                if step_out:
                    llh_l = gp.log_joint_unnorm(x_l)
                    while llh_l > llh0:
                        x_l[d] = x_l[d] - self.sigma[d]
                        llh_l = gp.log_joint_unnorm(x_l)
                    llh_r = gp.log_joint_unnorm(x_r)
                    while llh_r > llh0:
                        x_r[d] = x_r[d] + self.sigma[d]
                        llh_r = gp.log_joint_unnorm(x_r)

                x_cur = xx.copy()
                while True:
                    xd = np.array(np.random.rand() * (x_r[d] - x_l[d]) + x_l[d])
                    x_cur[d] = xd.copy()
                    last_llh = gp.log_joint_unnorm(x_cur)
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
                pass
                #print ("burn-in")
            elif i > self.burn_in and i % 100 == 0: 
                pass
                #print ('iteration', i - self.burn_in)
            
            if i >= self.burn_in:   
                samples[:, i-self.burn_in] = xx.copy().ravel() # index starting from 0
        
        #plt.hist(samples[0])

        return samples