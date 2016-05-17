import linear_svm2 as svm2
import linear_svm as svm
import numpy as np


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
              [-34, 55,  19, 22, 23]]).T; #X.shape = (2,4)
#print '\nX.shape is {}'.format(X.shape)
print '\nThe input vector(s) X: \n{}'.format(X); print '\n'

# ith element in y is the correct class label for the ith image
y = np.array([1, 0])

#loss, grad = svm2.svm_loss_naive(W, X, y, 0)
loss, grad = svm.svm_loss_naive(W.T, X.T, y, 0)
print 'loss is: {}'.format(loss)
print 'gradeint is: \n{}'.format(grad)

#loss, grad = svm2.svm_loss_vectorized(W, X, y, 0)
loss, grad = svm.svm_loss_vectorized(W.T, X.T, y, 0)
print 'loss is: {}'.format(loss)
print 'gradeint is: \n{}'.format(grad)