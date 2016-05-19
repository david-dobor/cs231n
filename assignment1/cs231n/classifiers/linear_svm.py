import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization parameter

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    n_classes = W.shape[0]
    n_train = X.shape[1]
    loss = 0.0
    for i in xrange(n_train):
        scores = W.dot(X[:, i])
        correct_class_score = scores[y[i]]
        for j in xrange(n_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1
            if margin > 0:
                loss += margin

                dW[y[i],:] -= X[:,i] # sum over j != y_i
                dW[j,:] += X[:,i]    # sum of the x_i's contributions

    loss /= n_train

    # Gradient is averaged, like loss above
    dW /= n_train

    # Add regularization
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg*W

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################

    num_classes = W.shape[0]
    num_train = X.shape[1]
    scores = W.dot(X)

    correct_scores = scores[y, np.arange(num_train)]

    margins = scores - correct_scores + 1 # like above, delta = 1
    margins[y, np.arange(num_train)] = 0 # accounting for the j=y_i term we shouldn't count (subtracting 1 makes up for it since w_j = w_{y_j} in this case)

    # Compute max
    thresh = np.maximum(np.zeros((num_classes,num_train)), margins)

    # Compute loss as double sum
    loss = np.sum(thresh)
    loss /= num_train

    # Add regularization
    loss += 0.5 * reg * np.sum(W * W)

    # Binarize into integers
    binary = thresh
    binary[thresh > 0] = 1

    # Perform the two operations simultaneously
    # (1) for all j: dW[j,:] = sum_{i, j produces positive margin with i} X[:,i].T
    # (2) for all i: dW[y[i],:] = sum_{j != y_i, j produces positive margin with i} -X[:,i].T
    col_sum = np.sum(binary, axis=0)
    binary[y, range(num_train)] = -col_sum[range(num_train)]
    dW = np.dot(binary, X.T)

    # Divide
    dW /= num_train

    # Regularize
    dW += reg*W

    return loss, dW