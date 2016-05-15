
import numpy as np
# this is the same data as in slide 37, lecture 3

# create 3 X 4 matrix of weights
W = np.array([[0.01, -0.05, 0.1, 0.05],
              [0.7, 0.2, 0.05, 0.16],
              [0.0, -.45, -0.2, 0.03]])

print '\n    The Weights are: \n\n{}'.format(W)

# and a vector b of biases; b is a 3 X 1 column vector
b = np.array([0.0, 0.2, -0.3])
#b.shape = (3,1)
print 'The bias vector is: \n{}'.format(b)

# create a test image; imagine an image of  2 X 2 pixels stretched into
# a 4 X 1 column vector
Xi = np.array([-15, 22, -44, 56])
#Xi.shape = (4,1)
print 'The input image is: \n{}'.format(Xi)

# compute the vector of scores
s = W.dot(Xi) + b
print 'The scores then are:\n{}'.format(s)

########################################################################
# Now that we have some test data, let's see how to compute hinge loss #
########################################################################

# First, the svm_loss_naive functions expects the data to come in as rows
# So we reshape the data as follows:

Xi.shape = (1,4)   # an image is reshaped into a row

# W is expected to have shape (#of_pixels X,  #of_classes) so we transpose it
W = W.transpose()


# We are going to debug the svm_loss_naive function
# note that SVM

from cs231n.classifiers.linear_svm import svm_loss_naive
loss, grad = svm_loss_naive(W, Xi, 1, 0.00001)

















