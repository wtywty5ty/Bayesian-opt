import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from matplotlib import pyplot as plt

def log_gaussian(x, mu, sigma):
    """
        Returns the log gaussian likelihood of an observation x given a mean and a std. deviation
        
        Keyword arguments:
            - x: observation.
            - mu: mean.
            - sigma: std. deviation.
    """
    return tf.cast(-0.5 * tf.log(2 * np.pi) - tf.log(sigma) - tf.square(x - mu) / (2 * tf.square(sigma)), tf.float64)


def reparametrise_weights(mu, rho):
    """
        Petforms the reparameterisation trick: w = mu + log(1+exp(rho))*e
        
        Keyword arguments:
            - mu: variational paramter for the mean.
            - rho: variational parameter for the std. deviation.
    """
    epsilon = tf.random_normal(mu.shape, mean=0., stddev=1.)
    return mu + tf.multiply(tf.log(1. + tf.exp(rho)), epsilon)



def sample_network(weights_mu, biases_mu, weights_rho, biases_rho, n_hidden_layers):
    """
        Samples the weights and biases from the posterior distribution.
        
        Keyword arguments: 
            - weights_mu: variational paramters for mu for the weights.
            - biases_mu: variational paramters for mu for the biases.
            - weights_rho: variational paramters for rho for the weights.
            - biases_rho: variational paramters for rho for the biases.
    """
    weights = {}
    biases = {}
    for i in range(n_hidden_layers+1):
        weights[i] = tf.random_normal(weights_mu[i].shape, 
                                      mean = weights_mu[i], 
                                      stddev=tf.log(1. + tf.exp(weights_rho[i])))
        biases[i] = tf.random_normal(biases_mu[i].shape, 
                                     mean = biases_mu[i], 
                                     stddev=tf.log(1. + tf.exp(biases_rho[i])))
    return weights, biases






def build_network(x, weights, biases, n_hidden_layers):
    """
        Builds network recursively.
        
        Keyword arguments:
            - weights: weights of the network.
            - biases: biases of the network.
    """
    network = []
    for i in range(n_hidden_layers+1):
        if i == 0:
            network.append(tf.nn.relu(tf.matmul(x, weights[i]) + biases[i]))
        else:
            network.append(tf.nn.relu(tf.matmul(network[i-1], weights[i]) + biases[i]))
    output = tf.matmul(network[n_hidden_layers-1], weights[n_hidden_layers]) + biases[n_hidden_layers]
    return output


def log_prior_gaussian(x, sigma):
    """
        Returns the log probability of an observation under a guassian.
        
        Keyword arguments:
            - x: samples.
            - sigma: the std. devaition.
    """
    return log_gaussian(x = x, mu = 0., sigma = sigma)


def sample(x, y, n_samples, weights_mu, weights_rho, biases_mu, biases_rho, n_hidden_layers, prior_sigma):   
    """
        Computes the approximation of the log prior, posterior and likelihood
        
        Keyword arguments:
            - n_samples: number of samples to be used to compute the approximation.
            - weights_mu: variational paramters mu for the weights.
            - weights_rho: variational paramters rho for the weights.
            - biases_mu: variational paramters mu for the biases.
            - biases_rho: variational paramters rho for the biases.
    """
    approximate_log_prior = 0.
    approximate_log_posterior = 0.
    approximate_log_likelihood = 0.
    for _ in range(n_samples):
        # Reparametrise weights
        weights = {}
        biases = {}
        for i in range(n_hidden_layers+1):
            weights[i] = reparametrise_weights(mu = weights_mu[i], rho = weights_rho[i])
            biases[i] = reparametrise_weights(mu = biases_mu[i], rho = biases_rho[i])
        # Build network
        output = build_network(x = x, weights = weights, biases = biases, n_hidden_layers = n_hidden_layers)
        # Initialise samples to 0
        sample_log_prior = 0.
        sample_log_posterior = 0.
        sample_log_likelihood = 0.
        # For each layer 
        for i in range(n_hidden_layers+1):
            # Sample log prior
            sample_log_prior += tf.reduce_sum(log_prior_gaussian(x = weights[i],
                                                                sigma = prior_sigma))
            sample_log_prior += tf.reduce_sum(log_prior_gaussian(x = biases[i],
                                                                sigma = prior_sigma))
            # Sample log posterior
            sample_log_posterior += tf.reduce_sum(log_gaussian(x = weights[i], 
                                                               mu = weights_mu[i], 
                                                               sigma = tf.log(1. + tf.exp(weights_rho[i]))))
            sample_log_posterior += tf.reduce_sum(log_gaussian(x = biases[i], 
                                                               mu = biases_mu[i], 
                                                               sigma = tf.log(1. + tf.exp(biases_rho[i]))))
        # Sample log likelihood
        sample_log_likelihood += tf.reduce_sum(-tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = output))

        approximate_log_prior += sample_log_prior
        approximate_log_posterior += sample_log_posterior
        approximate_log_likelihood += sample_log_likelihood

    approximate_log_prior  = tf.cast(tf.div(approximate_log_prior, n_samples), tf.float64)
    approximate_log_posterior = tf.cast(tf.div(approximate_log_posterior, n_samples), tf.float64)
    approximate_log_likelihood = tf.cast(tf.div(approximate_log_likelihood, n_samples), tf.float64)
    
    return approximate_log_prior, approximate_log_posterior, approximate_log_likelihood



def initialise_variational_parametes(n_hidden_layers, hidden_layers_dim, n_input, n_output, init_params):
    """
        Initialises the variational parameters mu and rho.
        
        Keyword arguments:
            - n_hidden_layers: number of hidden layers
            - hidden_layers_dim: the number of neurons per in each hidden layer
            - n_input: dimension of the input 
            - n_output: dimention of the output
    """
    neurons = [n_input] + hidden_layers_dim + [n_output]
    weights_mu={}
    biases_mu={}
    weights_rho={}
    biases_rho={}
    
    init_sigma_weights_mu = init_params[0]
    init_sigma_biases_mu = init_params[1]
    init_sigma_weights_rho = init_params[2]
    init_sigma_biases_rho = init_params[3]
    
    for i in range(n_hidden_layers+1):
            weights_mu[i] = tf.Variable(tf.random_normal((neurons[i], neurons[i+1]), 
                                                        mean = 0., 
                                                        stddev=init_sigma_weights_mu),
                                       tf.float64)
            biases_mu[i] = tf.Variable(tf.random_normal((neurons[i+1],), 
                                                        mean = 0., 
                                                        stddev=init_sigma_biases_mu),
                                       tf.float64)
            weights_rho[i] = tf.Variable(tf.random_normal((neurons[i], neurons[i+1]), 
                                                          mean = 0., 
                                                          stddev=init_sigma_weights_rho),
                                         tf.float64)
            biases_rho[i] = tf.Variable(tf.random_normal((neurons[i+1],), 
                                                         mean = 0., 
                                                         stddev = init_sigma_biases_rho),
                                        tf.float64)
    return weights_mu, biases_mu, weights_rho, biases_rho


class BN():

	def __init__(self, mnist):
		N = mnist.data.shape[0]
		data = np.float32(mnist.data[:]) / 255.
		target = mnist.target.reshape(N, 1)
		train_idx, test_idx = train_test_split(np.array(range(N)), test_size=0.15)
		self.train_data, self.test_data = data[train_idx], data[test_idx]
		self.train_target, self.test_target = target[train_idx], target[test_idx]
		self.train_target = np.float32(preprocessing.OneHotEncoder(sparse=False).fit_transform(self.train_target))

	def train_bayesian_nn(self,params):
		learning_rate_log = params[0]
		n_epochs = int(params[1])
		batch_size = int(params[2])
		n_samples = int(params[3])
		prior_sigma = np.float32(params[4])
		init_sigma_weights_mu = np.float32(params[5])
		init_sigma_biases_mu = np.float32(params[6])
		init_sigma_weights_rho = np.float32(params[7])
		init_sigma_biases_rho =np.float32(params[8])
		
		learning_rate = float(np.exp(learning_rate_log))
		print ("\tLearning rate: " + str(learning_rate) +", training epochs: " + str(n_epochs) + ", batch size: "+ str(batch_size) + ", n_samples: " + str(n_samples) + ", prior_sigma: " +str(prior_sigma) +", init_sigma_weights_mu:  " + str(init_sigma_weights_mu) +", init_sigma_biases_mu: "+ str(init_sigma_biases_mu) +", init_sigma_weights_rho: "+ str(init_sigma_weights_rho) +", init_sigma_biases_rho: "+ str(init_sigma_biases_rho))
		        
		n_batches =  int(self.train_data.shape[0]/ float(batch_size))

		ops.reset_default_graph()

		# Define input and ouput dimension
		n_input = 784
		n_output = 10
		# Define input and output placeholders
		x = tf.placeholder(tf.float32, shape = [None, n_input])
		y = tf.placeholder(tf.float32, shape = [None, n_output])        

		# Set number of hidden layers
		n_hidden_layers = 2

		# Initialise variational paramters: mus and rhos for weights and biases
		weights_mu, biases_mu, weights_rho, biases_rho = initialise_variational_parametes(n_hidden_layers = n_hidden_layers, 
		                                                                          hidden_layers_dim = [200, 200],
		                                                                          n_input = n_input,
		                                                                          n_output = n_output,
		                                                                          init_params = [init_sigma_weights_mu,
		                                                                                        init_sigma_biases_mu,
		                                                                                        init_sigma_weights_rho,
		                                                                                        init_sigma_biases_rho,])
		 # Sample prior, posterior and likelihood
		log_prior, log_posterior, log_likelihood = sample(x = x,
		                                                  y = y,
		                                                  n_samples = n_samples,
		                                                  weights_mu = weights_mu, 
		                                                  weights_rho = weights_rho, 
		                                                  biases_mu = biases_mu, 
		                                                  biases_rho = biases_rho,
		                                                  n_hidden_layers = n_hidden_layers,
		                                                  prior_sigma = prior_sigma)
		# Set the scaling factor for log_posterior - log_prior to account for the fact that we are
		# using bacthes of data
		scaling_factor = tf.placeholder(tf.float64, shape = None, name = 'scaling_factor')
		# Create loss function
		loss = tf.reduce_sum(scaling_factor*(log_posterior - log_prior) - log_likelihood)
		# Create optimiser
		optimiser = tf.train.AdamOptimizer(learning_rate)
		optimise = optimiser.minimize(loss)
		# Sample the weights and biases of the network
		weights, biases = sample_network(weights_mu, biases_mu, weights_rho, biases_rho, n_hidden_layers= n_hidden_layers)
		# Build the netwotk
		output = tf.nn.softmax(build_network(x=x, weights = weights, biases = biases, n_hidden_layers = n_hidden_layers))
		# Store predictions
		pred = tf.argmax(output, 1)
		# Initialise all variables
		init = tf.global_variables_initializer()

		results = {'loss':[], 'test_set_accuracy':[]}
		with tf.Session() as sess:
		    sess.run(init)
		    for epoch in range(n_epochs):
		        #print ("Epoch: %03d/%03d" % (epoch+1, n_epochs))      
		        for i in range(n_batches):
		            ob = sess.run([loss, optimise, log_prior], feed_dict={x: self.train_data[i * batch_size: (i + 1) * batch_size],
		                                                       y: self.train_target[i * batch_size: (i + 1) * batch_size],
		                                                       scaling_factor: (2 ** (n_batches - (i + 1))) / ((2 ** n_batches) - 1 )})
		    predictions = sess.run(pred, feed_dict={x: self.test_data})
		    test_accuracy = np.count_nonzero(predictions == np.int32(self.test_target.ravel())) / float(self.test_data.shape[0]) ; print ("Accuracy " + str(test_accuracy))
		return 1 - test_accuracy

        
