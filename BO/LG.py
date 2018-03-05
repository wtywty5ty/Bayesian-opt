import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from gp import *
import numpy as np
import matplotlib.pyplot as plt

def return_mean_std(y_list):
    mean_list = list()
    std_list = list()
    for one_list in y_list:
        mean_list.append(np.mean(one_list))
        std_list.append(np.std(one_list))
    return mean_list, std_list

def train_logisitc_regression_(mnist):
    def train_logisitc_regression_(params):
        learning_rate_log = params[0]
        training_epochs = int(params[1])
        batch_size = int(params[2])
        beta = params[3]
        learning_rate = np.exp(learning_rate_log)
        print ("\tLearning rate: " + str(learning_rate) + ", training epochs: " + str(training_epochs) + ", batch size: "+ str(batch_size) + ", beta " + str(beta))
        display_step = 10
        # tf Graph Input
        x = tf.placeholder(tf.float32, [None, 784])
        y = tf.placeholder(tf.float32, [None, 10])
        # Set model weights
        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))
        # Construct model
        pred = tf.nn.softmax(tf.matmul(x, W) + b)
        # Create regulariser
        regularizer = tf.nn.l2_loss(W)
        # Minimize error using cross entropy
        cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1) + beta * regularizer)
        # Gradient Descent
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
        # Initialize the variables
        init = tf.global_variables_initializer()
        # Start training
        with tf.Session() as sess:
            # Run the initializer
            sess.run(init)
            # Training cycle
            for epoch in range(training_epochs):
                avg_cost = 0.
                total_batch = int(mnist.train.num_examples/batch_size)
                # Loop over all batches
                for i in range(total_batch):
                    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                                  y: batch_ys})
                    # Compute average loss
                    avg_cost += c / total_batch
                # Display logs per epoch step
                #if (epoch+1) % display_step == 0:
                #    print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
            # Test model
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            accuracy_result = accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
            print ("\t\t Accuracy: " + str(accuracy_result))
        return 1 - accuracy_result
    return train_logisitc_regression_