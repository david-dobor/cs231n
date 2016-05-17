""" Implements svm hinge loss, tests implementation (David) """
import numpy as np
import time


def svm_loss_naive(scores, y):
    """ Computes svm hinge loss based on scores and class labels - naive implementation.

    Args:
        scores: Array of scores. Each row corresponds to a training example.
                Columns correspond to class labels. E.g.: entry (2, 3) in the array
                is the score for the third class of the second training example.
        y:      The correct class labels corresponding to the training examples.
                y has the same length as the number of rows in array scores.


    Returns:
        losses:    list of losses for training examples (i.e. for each row in the scores array,
                                                        compute the loss and store it in this list)
        avg_loss:  average of losses across all training examples

    """
    n_classes = scores.shape[1]
    n_train = scores.shape[0]

    losses = []
    for i in xrange(n_train):
        scores_i = scores[i]
        correct_class_score = scores_i[y[i]]

        loss_i = 0.0
        for j in xrange(n_classes):
            if j == y[i]:
                continue

            margin = scores_i[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss_i += margin

        losses.append(loss_i)

    return losses, np.mean(losses)


def svm_loss_vectorized(scores, y):
    """ Computes svm hinge loss based on scores and class labels - vectorized implementation.
    Args:
        scores: same as in svm_loss_naive
        y:      same as in svm_loss_naive

    Returns:   same as in svm_loss_naive

    """
    n_classes = scores.shape[1]
    n_train = scores.shape[0]

    # in each row of score matrix, pick off the entry corresponding to the correct class
    correct_scores = scores[np.arange(len(y)), y]

    margins = scores - correct_scores[:, np.newaxis] + 1

    # set margins for correct class entries to zero:
    margins[np.arange(len(y)), y] = 0

    # set margins for negative entries to zero:
    #thresh = np.maximum(np.zeros((n_classes,n_train)), margins)
    margins[np.where(margins <= 0)] = 0

    losses = np.sum(margins, axis=1)
    return losses, np.mean(losses)

##################################################################################
#               Create data to test the svm loss implementations.                #
#         The scores array is exactly the same as in the lecture notes           #
##################################################################################

scores = np.array([[3.2, 5.1, -1.7],   # first row: scores for input image 'cat'
                   [1.3, 4.9, 2.0],    # second row: scores for 'car'
                   [2.2, 2.5, -3.1]])  # third row:  scores for 'frog'

# correct labels for training example scores: | 0: cat | 1: car | 2: frog |
y = np.array([0, 1, 2])


losses, total_loss = svm_loss_naive(scores, y)
print '\n   losses for all examples: {}\n'.format(losses)
print '       Our measure of unhappiness with the current scores: {}'.format(total_loss)

losses, total_loss = svm_loss_vectorized(scores, y)
print '\n   losses for all examples: {}\n'.format(losses)
print '       Our measure of unhappiness with the current scores: {}'.format(total_loss)

##################################################################################
# Now that this works, we time the two different implementations to see how much #
# improvement in speed we get by using the vectorized implementations over the   #
# naive implementaion.                                                           #
##################################################################################

n_train = 50000
n_classes = 10
scores = np.random.randn(n_train, n_classes) * 5 + 2
y = np.random.randint(n_classes, size=n_train)

tic = time.time()
loss_naive, temp = svm_loss_naive(scores, y)
toc = time.time()
t_naive = toc - tic
print '\n =====> Naive loss computed in %fs\n' % t_naive
print 'result is: {}'.format(temp)


tic = time.time()
loss_vec, temp = svm_loss_vectorized(scores, y)
toc = time.time()
t_vectorized = toc - tic
print '\n =====> Vectorized loss computed in %fs\n' % t_vectorized
print 'result is: {}'.format(temp)

print 'ratio of times (naive over vevtorized): {}'.format(t_naive / t_vectorized)

























