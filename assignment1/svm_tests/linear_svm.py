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
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in xrange(num_train):
      scores = X[i].dot(W)
      correct_class_score = scores[y[i]]
      for j in xrange(num_classes):
        if j == y[i]:
          continue
        margin = scores[j] - correct_class_score + 1 # note delta = 1
        if margin > 0:
          loss += margin

          #####  MAKE A VIDEO ABOUT THIS! #### <--- COME BACK!
          dW[:,y[i]] -= X[i,:]
          dW[:,j] += X[i,:]


    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by n_train.
    loss /= num_train
    dW /= num_train


    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg*W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################


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

    # Get dims
    D = X.shape[0]
    num_classes = W.shape[0]
    num_train = X.shape[1]
    scores = W.dot(X)

    # Construct correct_scores vector that is Dx1 (or 1xD) so we can subtract out
    # where we append the "true" scores: [W*X]_{y_1, 1}, [W*X]_{y_2, 2}, ..., [W*X]_{y_D, D}
    # Using advanced indexing into scores: http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
    # Slow, sadly:
    # correct_scores = np.diag(scores[y,:])
    # Fast (index in both directions):
    correct_scores = scores[y, np.arange(num_train)] # using the fact that all elements in y are < C == n_classes

    mat = scores - correct_scores + 1 # like above, delta = 1
    mat[y, np.arange(num_train)] = 0 # accounting for the j=y_i term we shouldn't count (subtracting 1 makes up for it since w_j = w_{y_j} in this case)

    # Compute max
    thresh = np.maximum(np.zeros((num_classes,num_train)), mat)

    # Compute loss as double sum
    loss = np.sum(thresh)
    loss /= num_train

    # Add regularization
    loss += 0.5 * reg * np.sum(W * W)

    #############################################################################
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################

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