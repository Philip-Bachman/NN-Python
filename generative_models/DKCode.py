###################################################################
# Code adapted from Durk Kingma's Github repository: "nips14-ssl" #
###################################################################

from collections import OrderedDict
import numpy as np
import theano as theano
import theano.tensor as T
from theano.ifelse import ifelse

# Pre-processing routines

def PCA_theano(x_in, cutoff=0.99, global_sd=True):
    """
    Given input matrix x_in in numpy form, compute transform functions for
    reducing the dimensionality of inputs. Make the transform functions and
    all their parameters based around theano shared variables, for GPU use.
    """
    x_center = x_in.mean(axis=0)
    x = x_in - x_center
    if not global_sd:
        x_sd = x.std(axis=0) + 1e-5
    else:
        x_sd = x.std() + 1e-5
    # normalize to either unit standard deviation "globally" or
    # per-feature
    x = x / x_sd
    # compute covariance matrix and its eigen-decomposition
    print "Performing eigen-decomposition for PCA..."
    x_cov = np.dot(x.T, x) / x.shape[0]
    eigval, eigvec = np.linalg.eig(x_cov)
    #
    #eigval = np.ones(eigval.shape)
    #
    print "Done."
    if cutoff <= 1:
        # pick the number of dimensions to keep based on recovered variance
        n_used = ((eigval.cumsum() / eigval.sum()) < cutoff).sum()
        print 'PCA cutoff:', cutoff, 'n_used:', n_used
    else:
        # pick the number of dimensions to keep by user-provided value
        n_used = int(cutoff)
    eigval = eigval[:n_used].reshape((n_used,))
    eigvec = eigvec[:,:n_used]
    # construct functions for applying PCA
    f_enc, f_dec, pca_shared_params = \
            PCA_encdec_theano(eigvec, eigval, x_center, x_sd)
    pca_shared_params['pca_dim'] = n_used
    return f_enc, f_dec, pca_shared_params
        
def PCA_encdec_theano(eigvec, eigval, x_mean, x_sd):
    """
    Construct PCA encoder/decoder functions based around Theano shared
    variables. Return the function handles and a dict containing the relevant
    shared variables (well, symbolic references to them, at least).
    """
    # construct the shared variables to use in the encoder/decoder functions
    fx = theano.config.floatX
    eigval_shared = theano.shared(value=eigval.astype(fx), name='eigval')
    eigvec_shared = theano.shared(value=eigvec.astype(fx), name='eigvec')
    x_mean_shared = theano.shared(value=x_mean.astype(fx), name='x_mean')
    x_sd_shared = theano.shared(value=x_sd.astype(fx), name='x_sd')
    pca_shared_params = {'eigval': eigval_shared, 'eigvec': eigvec_shared, \
            'x_mean':x_mean_shared, 'x_sd':x_sd_shared}
    # construct the encoder/decoder functions using the shared variables
    def f_enc( x ):
        x_sands = (x - x_mean_shared) / x_sd_shared
        result = T.dot(x_sands, eigvec_shared) / T.sqrt(eigval_shared)
        return result
    def f_dec( x ):
        result = (T.dot((x * T.sqrt(eigval_shared)), eigvec_shared.T) * \
                x_sd_shared) + x_mean_shared
        return result
    return f_enc, f_dec, pca_shared_params


def norm_clip(dW, max_l2_norm=10.0):
    """
    Clip theano symbolic var dW to have some max l2 norm.
    """
    dW_l2_norm = T.sqrt(T.sum(dW**2.0))
    norm_ratio = (max_l2_norm / dW_l2_norm)
    clip_factor = ifelse(T.lt(norm_ratio, 1.0), norm_ratio, 1.0)
    dW_clipped = dW * clip_factor
    return dW_clipped

def get_adam_updates(params=None, grads=None, \
        alpha=None, beta1=None, beta2=None, \
        mom2_init=1e-3, smoothing=1e-6, max_grad_norm=10000.0):
    """
    Get the Theano updates to perform ADAM optimization of the shared-var
    parameters in params, given the shaared-var gradients in grads.

    params should be an iterable containing "keyable" values, grads should be
    a dict containing the grads for all values in params, and the remaining
    arguments should be theano shared variable arrays.
    """

    # make an OrderedDict to hold the updates
    updates = OrderedDict()
    
    for p in params:
        # initialize update the iteration counter
        zero_ary = np.zeros((1,)).astype(theano.config.floatX)
        it_count = theano.shared(value=zero_ary)
        it_count_new = it_count + 1.

        # apply a bias correction factor to the learning rate
        fix1 = 1. - beta1[0]**(it_count[0] + 1.)
        fix2 = 1. - beta2[0]**(it_count[0] + 1.)
        lr_t = alpha[0] * (T.sqrt(fix2) / fix1)

        # get gradient for parameter p
        grad_p = norm_clip(grads[p], max_grad_norm)

        # mean_squared_grad := E[g^2]_{t-1}
        mom1_ary = 0.0 * p.get_value(borrow=False)
        mom2_ary = (0.0 * p.get_value(borrow=False)) + mom2_init
        mom1 = theano.shared(mom1_ary)
        mom2 = theano.shared(mom2_ary)
        
        # update moments
        mom1_new = (beta1[0] * mom1) + ((1. - beta1[0]) * grad_p)
        mom2_new = (beta2[0] * mom2) + ((1. - beta2[0]) * T.sqr(grad_p))
        
        # compute the effective gradient
        effgrad = mom1_new / (T.sqrt(mom2_new) + smoothing)
        
        # do update
        p_new = p - (lr_t * effgrad)
            
        # apply updates
        updates[p] = p_new
        updates[mom1] = mom1_new
        updates[mom2] = mom2_new
        updates[it_count] = it_count_new
        
    return updates

def get_adadelta_updates(params=None, grads=None, \
        alpha=None, beta1=None, max_grad_norm=10000.0):
    """
    Get the Theano updates to perform AdaDelta optimization of the shared-var
    parameters in params, given the shaared-var gradients in grads.

    params should be an iterable containing "keyable" values, grads should be
    a dict containing the grads for all values in params, and the remaining
    arguments should be theano shared variable arrays.
    """

    # make an OrderedDict to hold the updates
    updates = OrderedDict()
    lr_t = alpha[0]
    
    for p in params:
        # get gradient for parameter p
        grad_p = norm_clip(grads[p], max_grad_norm)

        # initialize squared gradient accumulator
        mom_ary = (0.0 * p.get_value(borrow=False)) + 1.0
        mom1 = theano.shared(mom_ary)
        
        # update moments
        mom1_new = (beta1[0] * mom1) + ((1. - beta1[0]) * T.sqr(grad_p))
        
        # compute the effective gradient
        effgrad = grad_p / (T.sqrt(mom1_new) + 1e-6)
        
        # do update
        p_new = p - (lr_t * clipped_grad)
            
        # apply update
        updates[p] = p_new
        updates[mom1] = mom1_new
        
    return updates