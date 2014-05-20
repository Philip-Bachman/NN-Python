import numpy as np
import theano
import theano.tensor as T

class LogisticRegression(object):
    """Multi-class Logistic Regression loss dangler."""

    def __init__(self, linear_layer):
        """Dangle a logistic regression from the given linear layer.

        The given linear layer should be a HiddenLayer (or subclass) object,
        for HiddenLayer as defined in LayerNet.py."""
        self.input_layer = linear_layer

    def loss_func(self, y):
        """Return the multiclass logistic regression loss for y.

        The class labels in y are assumed to be in correspondence with the
        set of column indices for self.input_layer.linear_output.
        """
        p_y_given_x = T.nnet.softmax(self.input_layer.linear_output)
        loss = -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]),y])
        return loss

    def errors(self, y):
        """Compute the number of wrong predictions by self.input_layer.

        Predicted class labels are computed as the indices of the columns of
        self.input_layer.linear_output which are maximal. Wrong predictions are
        those for which max indices do not match their corresponding y values.
        """
        # Compute class memberships predicted by self.input_layer
        y_pred = T.argmax(self.input_layer.linear_output, axis=1)
        errs = 0
        # check if y has same dimension of y_pred
        if y.ndim != y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            errs = T.sum(T.neq(y_pred, y))
        else:
            raise NotImplementedError()
        return errs

class LogRegSS(object):
    """Multi-class semi-supervised Logistic Regression loss dangler."""

    def __init__(self, linear_layer):
        """Dangle a logistic regression from the given linear layer.

        The given linear layer should be a HiddenLayer (or subclass) object,
        for HiddenLayer as defined in LayerNet.py."""
        self.input_layer = linear_layer

    def safe_softmax_ss(self, x):
        """Softmax that shouldn't overflow."""
        e_x = T.exp(x - T.max(x, axis=1, keepdims=True))
        x_sm = e_x / T.sum(e_x, axis=1, keepdims=True)
        return x_sm

    def loss_func(self, y):
        """Return the multiclass logistic regression loss for y.

        The class labels in y are assumed to be in correspondence with the
        set of column indices for self.input_layer.linear_output.
        """
        row_idx = T.arange(y.shape[0])
        row_mask = T.neq(y, 0).reshape((y.shape[0], 1))
        p_y_given_x = self.safe_softmax_ss(self.input_layer.linear_output)
        wacky_mat = (p_y_given_x * row_mask) + (1. - row_mask)
        loss = -T.sum(T.log(wacky_mat[row_idx,y])) / T.sum(row_mask)
        return loss

    def errors(self, y):
        """Compute the number of wrong predictions by self.input_layer.

        Predicted class labels are computed as the indices of the columns of
        self.input_layer.linear_output which are maximal. Wrong predictions are
        those for which max indices do not match their corresponding y values.
        """
        # Compute class memberships predicted by self.input_layer
        y_pred = T.argmax(self.input_layer.linear_output[:,1:], axis=1)
        y_pred = y_pred + 1
        errs = 0
        # check if y has same dimension of y_pred
        if y.ndim != y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            errs = T.sum(T.neq(y_pred, y) * T.neq(y, 0))
        else:
            raise NotImplementedError()
        return errs

class MCL2Hinge(object):
    """Multi-class one-vs-all L2 hinge loss dangler."""

    def __init__(self, linear_layer):
        """Dangle a squred hinge loss from the given linear layer.

        The given linear layer should be a HiddenLayer (or subclass) object,
        for HiddenLayer as defined in LayerNet.py."""
        self.input_layer = linear_layer

    def loss_func(self, y):
        """Return the multiclass squared hinge loss for y.

        The class labels in y are assumed to be in correspondence with the
        set of column indices for self.input_layer.linear_output.
        """
        y_hat = self.input_layer.linear_output
        margin_pos = T.maximum(0.0, (1.0 - y_hat))
        margin_neg = T.maximum(0.0, (1.0 + y_hat))
        obs_idx = T.arange(y.shape[0])
        loss_pos = T.sum(margin_pos[obs_idx,y]**2.0)
        loss_neg = T.sum(margin_neg**2.0) - T.sum(margin_neg[obs_idx,y]**2.0)
        loss = (loss_pos + loss_neg) / y.shape[0]
        return loss

    def errors(self, y):
        """Compute the number of wrong predictions by self.input_layer.

        Predicted class labels are computed as the indices of the columns of
        self.input_layer.linear_output which are maximal. Wrong predictions are
        those for which max indices do not match their corresponding y values.
        """
        # Compute class memberships predicted by self.input_layer
        y_pred = T.argmax(self.input_layer.linear_output, axis=1)
        errs = 0
        # check if y has same dimension of y_pred
        if y.ndim != y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            errs = T.sum(T.neq(y_pred, y))
        else:
            raise NotImplementedError()
        return errs

class MCL2HingeSS(object):
    """Multi-class one-vs-all L2 hinge loss dangler.

    For this loss, class index 0 is never penalized, and errors for inputs
    with class index 0 are similarly ignored. This is for semi-supervised
    training, constrained by Theano's programming model."""

    def __init__(self, linear_layer):
        """Dangle a squred hinge loss from the given linear layer.

        The given linear layer should be a HiddenLayer (or subclass) object,
        for HiddenLayer as defined in LayerNet.py."""
        self.input_layer = linear_layer

    def loss_func(self, y):
        """Return the multiclass squared hinge loss for y.

        The class labels in y are assumed to be in correspondence with the
        set of column indices for self.input_layer.linear_output.
        """
        y_hat = self.input_layer.linear_output
        row_idx = T.arange(y.shape[0])
        row_mask = T.neq(y, 0).reshape((y_hat.shape[0], 1))
        margin_pos = T.maximum(0.0, (1.0 - y_hat)) * row_mask
        margin_neg = T.maximum(0.0, (1.0 + y_hat)) * row_mask
        loss_pos = T.sum(margin_pos[row_idx,y]**2.0)
        loss_neg = T.sum(margin_neg**2.0) - T.sum(margin_neg[row_idx,y]**2.0)
        loss = (loss_pos + loss_neg) / T.sum(row_mask)
        return loss

    def errors(self, y):
        """Compute the number of wrong predictions by self.input_layer.

        Predicted class labels are computed as the indices of the columns of
        self.input_layer.linear_output which are maximal. Wrong predictions are
        those for which max indices do not match their corresponding y values.
        """
        # Compute class memberships predicted by self.input_layer
        y_pred = T.argmax(self.input_layer.linear_output[:,1:], axis=1)
        y_pred = y_pred + 1
        errs = 0
        # check if y has same dimension of y_pred
        if y.ndim != y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            errs = T.sum(T.neq(y_pred, y) * T.neq(y, 0))
        else:
            raise NotImplementedError()
        return errs
