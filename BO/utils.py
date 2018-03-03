from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from sklearn.datasets import make_regression
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss

def log_marginal_likelihood(gp, theta, likelihood_mode):
    if likelihood_mode == 'normal_likelihood':
        return gp.log_marginal_likelihood(theta)
    elif likelihood_mode == 'rho_likelihood':
        para = np.log(1.0 + np.exp(theta))
        return gp.log_marginal_likelihood(para)
    else:
        raise Exception('Wrong likelihood mode!!!')
    
def log_prior_parameters(theta, prior_mode):
    if prior_mode == 'normal_prior':
        return np.sum(np.log([ss.norm(0, 1).pdf(theta_k) for theta_k in theta]))
    elif prior_mode == 'exp_prior':
        log_prior = 0.0
        lambda_para = 1.5
        if np.all(theta>0):
            for num in theta:
                log_prior += (np.log(lambda_para) - lambda_para * num)
            return log_prior
        else:
            return -1 * float('inf')
    else:
        raise Exception('Wrong prior mode!!!')
    
def log_joint_unnorm(gp, theta, prior_mode, likelihood_mode):
    return log_marginal_likelihood(gp, theta, likelihood_mode) + log_prior_parameters(theta, prior_mode)


class Slice_sampler():
    
    def __init__(self, num_iters, sigma, burn_in, prior_mode, likelihood_mode):
        self.num_iters = num_iters # specify number of samples we want to draw
        self.sigma = sigma
        self.burn_in = burn_in
        self.prior_mode = prior_mode
        self.likelihood_mode = likelihood_mode
        
    def sample(self, init, gp, step_out=True):

        D = len(init)
        samples = np.zeros((D, self.num_iters))
        
        xx = init.copy()

        for i in range(self.num_iters + self.burn_in):
            perm = list(range(D))
            np.random.shuffle(perm)
            last_llh = log_joint_unnorm(gp, xx, self.prior_mode, self.likelihood_mode)

            for d in perm:
                llh0 = last_llh + np.log(np.random.rand())
                rr = np.random.rand(1)
                x_l = xx.copy()
                x_l[d] = x_l[d] - rr * self.sigma[d]
                x_r = xx.copy()
                x_r[d] = x_r[d] + (1 - rr) * self.sigma[d]

                if step_out:
                    llh_l = log_joint_unnorm(gp, x_l, self.prior_mode, self.likelihood_mode)
                    while llh_l > llh0:
                        x_l[d] = x_l[d] - self.sigma[d]
                        llh_l = log_joint_unnorm(gp, x_l, self.prior_mode, self.likelihood_mode)
                    llh_r = log_joint_unnorm(gp, x_r, self.prior_mode, self.likelihood_mode)
                    while llh_r > llh0:
                        x_r[d] = x_r[d] + self.sigma[d]
                        llh_r = log_joint_unnorm(gp, x_r, self.prior_mode, self.likelihood_mode)

                x_cur = xx.copy()
                while True:
                    xd = np.array(np.random.rand() * (x_r[d] - x_l[d]) + x_l[d])
                    x_cur[d] = xd.copy()
                    last_llh = log_joint_unnorm(gp, x_cur, self.prior_mode, self.likelihood_mode)
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