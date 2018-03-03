""" gp.py

Bayesian optimisation of loss functions.
"""

import numpy as np
import sklearn.gaussian_process as gp
from sklearn.gaussian_process import kernels

from scipy.stats import norm
from scipy.optimize import minimize

from utils import Slice_sampler
##################################################################################################################################################
def integrate_EI(x, sample_theta_list, evaluated_loss, greater_is_better=False, n_params=1):
    """ expected_improvement

    Expected improvement acquisition function.

    Arguments:
    ----------
        x: array-like, shape = [n_samples, n_hyperparams]
            The point for which the expected improvement needs to be computed.
        sample_theta_list: hyperparameter samples of the GP model, which will be used to 
            calculate integrated acquisition function
        evaluated_loss: Numpy array.
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        n_params: int.
            Dimension of the hyperparameter space.

    """
    # sample_theta_list contains all samples of hyperparameters
    ei_list = list()
    input_dimension = n_params
    init_length_scale = np.ones((input_dimension, ))
    kernel = kernels.Sum(kernels.WhiteKernel(),kernels.Product(kernels.ConstantKernel(),kernels.Matern(length_scale=init_length_scale, nu=5./2.)))
    for theta_set in sample_theta_list:
        model = gp.GaussianProcessRegressor(kernel=kernel, alpha=1e-5, optimizer = None, normalize_y=True)
        model.set_params(**{"kernel__k1__noise_level": np.abs(theta_set[0]),
                            "kernel__k2__k1__constant_value": np.abs(theta_set[1]),
                            "kernel__k2__k2__length_scale": theta_set[2:]})
        x_to_predict = x.reshape(-1, n_params)

        mu, sigma = model.predict(x_to_predict, return_std=True)

        if greater_is_better:
            loss_optimum = np.max(evaluated_loss)
        else:
            loss_optimum = np.min(evaluated_loss)

        scaling_factor = (-1) ** (not greater_is_better)

        # In case sigma equals zero
        with np.errstate(divide='ignore'):
            Z = scaling_factor * (mu - loss_optimum) / sigma
            expected_improvement = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
            expected_improvement[sigma == 0.0] == 0.0
        ei_list.append(expected_improvement[0])
    res_ei = np.max(ei_list)
    result = np.array([res_ei])
    return -1 * result


def integrate_sample(acquisition_func, sample_theta_list, evaluated_loss, greater_is_better=False,
                               bounds=(0, 10), n_restarts=25):
    """ sample_next_hyperparameter

    Proposes the next hyperparameter to sample the loss function for.

    Arguments:
    ----------
        acquisition_func: function.
            Acquisition function to optimise.
        sample_theta_list: hyperparameter samples of the GP model, which will be used to 
            calculate integrated acquisition function
        evaluated_loss: array-like, shape = [n_obs,]
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        bounds: Tuple.
            Bounds for the L-BFGS optimiser.
        n_restarts: integer.
            Number of times to run the minimiser with different starting points.

    """
    best_x = None
    best_acquisition_value = 1
    n_params = bounds.shape[0]

    for starting_point in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, n_params)):
        res = minimize(fun=acquisition_func,
                       x0=starting_point.reshape(1, -1),
                       bounds=bounds,
                       method='L-BFGS-B',
                       args=(sample_theta_list, evaluated_loss, greater_is_better, n_params))

        if res.fun < best_acquisition_value:
            best_acquisition_value = res.fun
            best_x = res.x

    return best_x

##################################################################################################################################################

def expected_improvement(x, gaussian_process, evaluated_loss, greater_is_better=False, n_params=1):
    """ expected_improvement

    Expected improvement acquisition function.

    Arguments:
    ----------
        x: array-like, shape = [n_samples, n_hyperparams]
            The point for which the expected improvement needs to be computed.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: Numpy array.
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        n_params: int.
            Dimension of the hyperparameter space.

    """

    x_to_predict = x.reshape(-1, n_params)

    mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)

    if greater_is_better:
        loss_optimum = np.max(evaluated_loss)
    else:
        loss_optimum = np.min(evaluated_loss)

    scaling_factor = (-1) ** (not greater_is_better)

    # In case sigma equals zero
    with np.errstate(divide='ignore'):
        Z = scaling_factor * (mu - loss_optimum) / sigma
        expected_improvement = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
        expected_improvement[sigma == 0.0] == 0.0
    return -1 * expected_improvement


def sample_next_hyperparameter(acquisition_func, gaussian_process, evaluated_loss, greater_is_better=False,
                               bounds=(0, 10), n_restarts=25):
    """ sample_next_hyperparameter

    Proposes the next hyperparameter to sample the loss function for.

    Arguments:
    ----------
        acquisition_func: function.
            Acquisition function to optimise.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: array-like, shape = [n_obs,]
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        bounds: Tuple.
            Bounds for the L-BFGS optimiser.
        n_restarts: integer.
            Number of times to run the minimiser with different starting points.

    """
    best_x = None
    best_acquisition_value = 1
    n_params = bounds.shape[0]

    for starting_point in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, n_params)):
        res = minimize(fun=acquisition_func,
                       x0=starting_point.reshape(1, -1),
                       bounds=bounds,
                       method='L-BFGS-B',
                       args=(gaussian_process, evaluated_loss, greater_is_better, n_params))

        if res.fun < best_acquisition_value:
            best_acquisition_value = res.fun
            best_x = res.x

    return best_x


def bayesian_optimisation(coor_sigma, burn_in, input_dimension,
                          n_iters, sample_loss, bounds, x0=None, n_pre_samples=5, acqui_eva_num = 10, 
                          alpha=1e-5, epsilon=1e-7, greater_is_better=False, mode = 'OPT', acqui_mode = 'MCMC', 
                          acqui_sample_num = 3, process_sample_mode = 'normal', prior_mode = 'normal_prior', likelihood_mode = 'normal_likelihood'):
    """ bayesian_optimisation
    Uses Gaussian Processes to optimise the loss function `sample_loss`.

    Arguments:
    ----------
        slice_sample_num: integer.
            how many samples we draw for each time of slice sampling
        coor_sigma: numpy array
            step-size for slice sampling of each coordinate, the dimension is equal to the number of 
            hyperparameters contained in the kernel
        burn_in: integer.
            how many iterations we want to wait before draw samples from slice sampling
        input_dimension: integer.
            dimension of input data
        n_iters: integer.
            Number of iterations to run the search algorithm.
        sample_loss: function.
            Function to be optimised.
        bounds: array-like, shape = [n_params, 2].
            Lower and upper bounds on the parameters of the function `sample_loss`.
        x0: array-like, shape = [n_pre_samples, n_params].
            Array of initial points to sample the loss function for. If None, randomly
            samples from the loss function.
        n_pre_samples: integer.
            If x0 is None, samples `n_pre_samples` initial points from the loss function.
        acqui_eva_num:
            when evaluating acquisition function, how many points we want to look into, number of restarts
        alpha: double.
            Variance of the error term of the GP.
        epsilon: double.
            Precision tolerance for floats.
        greater_is_better: boolean
            True: maximize the sample_loss function,
            False: minimize the sample_loss function
        mode: OPT means using optimizer to optimize the hyperparameters of GP
              MAP means using sample posterior mean to optimize the hyperparameters of GP
        acqui_mode: mode controlling the acquisition
            'OPT': using one prediction based on previously optimized model
            'MCMC': using several samples to sample the expected acquisition function
        acqui_sample_num:
            the number of hyperparameter samples we want to use for integrated acquisition function
        process_sample_mode:
            after getting sample, how to process it
            'normal': only accept positive sample and reject negative ones
            'abs': accept all samples after taking absolute value
            'rho': reparamization trick is used, the samples are rho
        prior_mode:
            the prior distribution we want to use
            'normal_prior': normal distribution
            'exp_prior': exponential distribution
        likelihood_mode: how to calculate likelihood
            'normal_likelihood': directly using input hyperparameter to calculate likelihood
            'rho_likelihood': using reparamization trick (theta = np.log(1.0 + np.exp(rho)))
    """

    # call slice sampler
    acqui_slice_sampler = Slice_sampler(1, coor_sigma, burn_in, prior_mode, likelihood_mode) # only sample one sample a time

    x_list = []
    y_list = []

    n_params = bounds.shape[0]

    print ('Start presampling...')
    if x0 is None:
        # random draw several points as GP prior
        for params in np.random.uniform(bounds[:, 0], bounds[:, 1], (n_pre_samples, bounds.shape[0])):
            x_list.append(params)
            y_list.append(sample_loss(params))
    else:
        for params in x0:
            x_list.append(params)
            y_list.append(sample_loss(params))
    print ('Presampling finished.')

    xp = np.array(x_list)
    yp = np.array(y_list)

    # Create the GP
    init_length_scale = np.ones((input_dimension, ))
    kernel = kernels.Sum(kernels.WhiteKernel(),kernels.Product(kernels.ConstantKernel(),kernels.Matern(length_scale=init_length_scale, nu=5./2.)))
    if mode == 'OPT':
        model = gp.GaussianProcessRegressor(kernel=kernel,
                                            alpha=alpha,
                                            n_restarts_optimizer=10,
                                            normalize_y=True)
    elif mode == 'MAP':
        model = gp.GaussianProcessRegressor(kernel=kernel,
                                            alpha=alpha,
                                            optimizer = None,
                                            n_restarts_optimizer=0,
                                            normalize_y=True)
    else:
        raise Exception('Wrong GP model initialization mode!!!')

    iter_num = 0
    for n in range(n_iters):
        iter_num += 1
        if iter_num % int(n_iters/2) == 0:
            print ('%d iterations have been run' % iter_num)
        else:
            pass
        # for each iteration, one sample will be drawn and used to train GP
        if mode == 'OPT':
            # for optimization mode, the hyperparameters are optimized during the process of fitting
            model.fit(xp, yp)
        elif mode == 'MAP':
            # for MAP mode, we use slice sampling to sample the posterior of hyperparameters and use the mean to update GP's hyperparameters
            model.fit(xp, yp)
            initial_theta = 10 * np.ones((input_dimension + 2, )) # input_dimension + 2 = number of length_scale + amplitude + noise_sigma
        else:
            raise Exception('Wrong GP model initialization mode!!!')
        # Sample next hyperparameter

        if acqui_mode == 'OPT':
            next_sample = sample_next_hyperparameter(expected_improvement, model, yp, greater_is_better=greater_is_better, bounds=bounds, n_restarts=acqui_eva_num)
        elif acqui_mode == 'MCMC':
            sample_theta_list = list()
            while(len(sample_theta_list) < acqui_sample_num): # all samples of theta must be valid
                one_sample = acqui_slice_sampler.sample(init = initial_theta, gp = model)
                if process_sample_mode == 'normal':
                    if np.all(one_sample[:,0]>0):
                        one_theta = [np.mean(samples_k) for samples_k in one_sample]
                        sample_theta_list.append(one_theta)
                    else:
                        continue
                elif process_sample_mode == 'abs':
                    one_theta = [np.abs(np.mean(samples_k)) for samples_k in one_sample]
                    sample_theta_list.append(one_theta)
                elif process_sample_mode == 'rho':
                    one_theta = [np.log(1.0 + np.exp((np.mean(samples_k)))) for samples_k in one_sample]
                    sample_theta_list.append(one_theta)
                else:
                    raise Exception('Wrong process sample mode!!!')

            next_sample = integrate_sample(integrate_EI, sample_theta_list, yp, greater_is_better=greater_is_better, bounds=bounds, n_restarts=acqui_eva_num)
        else:
            raise Exception('Wrong acquisition mode!!!')

        # Duplicates will break the GP. In case of a duplicate, we will randomly sample a next query point.
        if np.any(np.abs(next_sample - xp) <= epsilon):
            next_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], bounds.shape[0])

        # Sample loss for new set of parameters
        func_value = sample_loss(next_sample)

        # Update lists
        x_list.append(next_sample)
        y_list.append(func_value)

        # Update xp and yp
        xp = np.array(x_list)
        yp = np.array(y_list)

    return xp, yp