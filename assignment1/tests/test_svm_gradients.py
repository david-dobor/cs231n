import numpy as np


def svm_loss_naive(W, X, y, reg=0):
    n_classes = W.shape[0]
    n_train = X.shape[1]

    dW = np.zeros(W.shape) # initialize gradient to zero
    losses = []
    for i in xrange(n_train):
        scores_i = W.dot(X[:,i])
        correct_class_score = scores_i[y[i]]

        loss_i = 0.0
        for j in xrange(n_classes):
            if j == y[i]:
                continue

            margin = scores_i[j] - correct_class_score + 1
            if margin > 0:
                loss_i += margin

                dW[y[i],:] -= X[:,i]
                dW[j,:] += X[:,i]

        losses.append(loss_i)

    #loss = np.sum(losses) / len(losses)
    dW /= n_train

    return losses, dW


def svm_loss_vectorized(W, X, y, reg=0):
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    D = X.shape[0]
    num_classes = W.shape[0]
    num_train = X.shape[1]
    scores = W.dot(X)

    correct_scores = scores[np.arange(len(y)), y]

    margins = scores - correct_scores + 1

    # set margins for correct class entries to zero:
    margins[np.arange(len(y)), y] = 0

    # set margins for negative entries to zero:
    margins[np.where(margins <= 0)] = 0


    # Compute loss as double sum
    loss = np.sum(margins)
    loss /= num_train

    # Add regularization
    loss += 0.5 * reg * np.sum(W * W)

    # Binarize entries in margins
    binary = margins
    binary[margins > 0] = 1


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




##################################################################################
#     create some artificial data: matrix W, input vector X. Compute scores.     #
##################################################################################

#  row i in W is the classifier for class i
W = np.array([[0.01, -0.05, 0.1,  0.05, 0.01],
              [0.7,   0.2,  0.05, 0.16, 0.7 ],
              [0.0,  -0.45, -0.2, 0.03, 0.5]])
print '\n    The weights are: \n\n{}'.format(W)
#print '\n      W.shape is {}'.format(W.shape)

# each row in X is an image
X = np.array([[-15, 22, -44, 56, 44],
              [-34, 55,  19, 22, 23]]).T
#print '\nX.shape is {}'.format(X.shape)
print '\nThe input vector(s) X: \n{}'.format(X); print '\n'

# ith element in y is the correct class label for the ith image
y = np.array([1, 0])

loss, grad = svm_loss_naive(W, X, y)
print 'loss is: {}'.format(loss)
print 'gradeint is: \n{}'.format(grad)

# # Apply W to X to compute the score vector (i.e. W X = s)
# s = W.dot(X.T)
# print '\n    The vector s of scores works out to be: \n{}'.format(s); print '\n'
# print '\n      s.shape is {}\n'.format(s.shape)

loss, grad = svm_loss_vectorized(W, X, y)
print 'loss is: {}'.format(loss)
print 'gradeint is: \n{}'.format(grad)