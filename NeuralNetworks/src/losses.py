import numpy as np
from sklearn.metrics import mean_squared_error, log_loss


def get_loss_function_by_name(name):
    if name == 'mse':
        return mse, d_mse
    elif name == 'log_loss':
        return cross_entropy, d_cross_entropy


def mse(y_true, y_predicted):
    return mean_squared_error(y_true.T, y_predicted.T)


def d_mse(y_true, y_predicted):
    return 2 * (y_predicted - y_true) / np.size(y_true)


def cross_entropy(y_true, y_predicted):
    return log_loss(y_true.T, y_predicted.T)


def d_cross_entropy(y_true, y_predicted):
    eps = np.finfo(y_predicted.dtype).eps
    y_predicted = np.maximum(eps, np.minimum(1 - eps, y_predicted))
    return ((1 - y_true) / (1 - y_predicted) - y_true / y_predicted) / np.size(y_true)
