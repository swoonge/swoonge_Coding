from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]
    scores = X.dot(W)
    for i in range(num_train):
      scores_i = scores[i]
      loss_i = np.exp(scores_i)/np.sum(np.exp(scores_i))
      loss += -np.log(loss_i[y[i]])

      # loss_i를 정리하면 이렇게 할 수 있다.
      # loss += np.log(np.sum(np.exp(scores_i))) - scores_i[y[i]]

      dW[:,y[i]] -= X[i]
      dW += X[i].reshape(X.shape[1],1).dot(loss_i.reshape(1, num_classes))
    


      # for j in range(num_classes):
      #   dW[:,j] += X[i] * loss_i[j]
    
    loss /= num_train
    dW /= num_train

    loss += reg * np.sum(W * W)
    dW += reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]
    scores = X.dot(W)

    # print(np.exp(scores)/np.sum(np.exp(scores), axis = 1).reshape(num_train,1))
    loss_table = np.exp(scores)/(np.sum(np.exp(scores), axis = 1).reshape(num_train,1))
    loss = np.sum(-np.log(loss_table[range(num_train),y]))

    yi_scores = np.zeros(loss_table.shape)
    yi_scores[range(num_train), y] = 1

    dW = X.transpose().dot(loss_table - yi_scores)
    
    loss /= num_train
    dW /= num_train

    loss += reg * np.sum(W * W)
    dW += reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
