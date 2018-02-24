""" gp_mcmc.py

Bayesian optimisation of loss functions using slice sampling.
"""

import numpy as np
import sklearn.gaussian_process as gp

from scipy.stats import norm
from scipy.optimize import minimize

import slice_sampling as ssp


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
    gaussian_process = gaussian_process[0]
    #print(type(gaussian_process))

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

	
	
	
def expected_improvement_average(x, gaussian_process_list, evaluated_loss, greater_is_better=False, n_params=1):

	average = 0
	for i in range(len(gaussian_process_list)):
		average = average + expected_improvement(x, gaussian_process_list[i], evaluated_loss, greater_is_better, n_params)
		
	average = average/len(gaussian_process_list)
	return average

	
	
def sample_next_hyperparameter(acquisition_func, gaussian_process_list, evaluated_loss, greater_is_better=False,
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
    print(gaussian_process_list)
    print(np.array(gaussian_process_list))
    print(len(gaussian_process_list))

    for starting_point in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, n_params)):

        res = minimize(fun=acquisition_func,
                       x0=starting_point.reshape(1, -1),
                       bounds=bounds,
                       method='L-BFGS-B',
                       args=(np.array(gaussian_process_list).reshape(1, len(gaussian_process_list)), 
                             evaluated_loss, 
                             greater_is_better, 
                             n_params))

        if res.fun < best_acquisition_value:
            best_acquisition_value = res.fun
            best_x = res.x

    return best_x


def bayesian_optimisation(n_iters, sample_loss, bounds, x0=None, n_pre_samples=5,
                          gp_params=None, random_search=False, alpha=1e-5, epsilon=1e-7, burn_in=200, num_samples=3):
    """ bayesian_optimisation

    Uses Gaussian Processes to optimise the loss function `sample_loss`.

    Arguments:
    ----------
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
        gp_params: dictionary.
            Dictionary of parameters to pass on to the underlying Gaussian Process.
        random_search: integer.
            Flag that indicates whether to perform random search or L-BFGS-B optimisation
            over the acquisition function.
        alpha: double.
            Variance of the error term of the GP.
        epsilon: double.
            Precision tolerance for floats.
    """

    x_list = []
    y_list = []

    n_params = bounds.shape[0]

    if x0 is None:
        for params in np.random.uniform(bounds[:, 0], bounds[:, 1], (n_pre_samples, bounds.shape[0])):
            x_list.append(params)
            y_list.append(sample_loss(params))
    else:
        for params in x0:
            x_list.append(params)
            y_list.append(sample_loss(params))

    xp = np.array(x_list)
    yp = np.array(y_list)

    # Create the GP
#    if gp_params is not None:
#        model = gp.GaussianProcessRegressor(**gp_params)
#    else:
#        kernel = gp.kernels.Matern()
#        model = gp.GaussianProcessRegressor(kernel=kernel,
#                                            alpha=alpha,
#                                            n_restarts_optimizer=10,
#                                            normalize_y=True)
        
    kernel = gp.kernels.Sum(gp.kernels.WhiteKernel(),gp.kernels.Product(gp.kernels.ConstantKernel(),gp.kernels.Matern(nu=5./2.)))
    model = ssp.Gaussian_Process(kernel = kernel)
    
    for n in range(n_iters):

        print("Iteration")
        print(n)
        model.fit(xp, yp)
        
        slice_sampler = ssp.Slice_sampler(num_iters = 3, 
                                      sigma = np.ones(n_params+2), 
                                      burn_in = burn_in,
                                      gp = model)
    
        sample_list = []
        gaussian_list = []
        for i in range(num_samples):
            samples = slice_sampler.sample(init = np.ones(n_params+2))
            sample_list.append(samples)
            gauss = ssp.Gaussian_Process(kernel = kernel)
            gauss.gp_.set_params(**{"kernel__k1__noise_level": np.abs(samples[0]),
                              "kernel__k2__k1__constant_value": np.abs(samples[1]),
                              "kernel__k2__k2__length_scale": samples[2:]})
            gaussian_list.append(gauss)
       
        #theta_opt = [np.mean(samples_k) for samples_k in samples]
        
        # kernel__k1__noise_level = noise of the data
        # kernel__k2__k1__constant_value = aplitude
        # kernel__k2__k2__length_scale = Matern length scales
        
        #for 
        #model = Gaussian_Process(kernel = kernel)
        #self.gp_.set_params(**{"kernel__k1__noise_level": np.abs(theta_opt[0]),
        #                      "kernel__k2__k1__constant_value": np.abs(theta_opt[1]),
        #                      "kernel__k2__k2__length_scale": theta_opt[2:]})
        
        

        # Sample next hyperparameter
        if random_search:
            x_random = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(random_search, n_params))
            ei = -1 * expected_improvement_average(x_random, model, yp, greater_is_better=True, n_params=n_params)
            next_sample = x_random[np.argmax(ei), :]
        else:
            next_sample = sample_next_hyperparameter(expected_improvement_average, gaussian_list, yp, greater_is_better=True, bounds=bounds, n_restarts=10)

        # Duplicates will break the GP. In case of a duplicate, we will randomly sample a next query point.
        if np.any(np.abs(next_sample - xp) <= epsilon):
            next_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], bounds.shape[0])

        # Sample loss for new set of parameters
        cv_score = sample_loss(next_sample)

        # Update lists
        x_list.append(next_sample)
        y_list.append(cv_score)

        # Update xp and yp
        xp = np.array(x_list)
        yp = np.array(y_list)

    return xp, yp
