import time
import utils as utils
import numpy as np
import numpy.random as npr
import theano
import theano.tensor as T

from load_data import load_udm, load_udm_ss, load_mnist, load_tfd
from PeaNet import PeaNet, load_peanet_from_file
from InfNet import InfNet, load_infnet_from_file
from VCGLoop import VCGLoop
from OneStageModel import OneStageModel
from NetLayers import relu_actfun, softplus_actfun, \
                      safe_softmax, row_shuffle

#####################################
# HELPER FUNCTIONS FOR DATA MASKING #
#####################################

def sample_masks(X, drop_prob=0.3):
    """
    Sample a binary mask to apply to the matrix X, with rate mask_prob.
    """
    probs = npr.rand(*X.shape)
    mask = 1.0 * (probs > drop_prob)
    return mask.astype(theano.config.floatX)

def sample_patch_masks(X, im_shape, patch_shape):
    """
    Sample a random patch mask for each image in X.
    """
    obs_count = X.shape[0]
    rs = patch_shape[0]
    cs = patch_shape[1]
    off_row = npr.randint(1,high=(im_shape[0]-rs-1), size=(obs_count,))
    off_col = npr.randint(1,high=(im_shape[1]-cs-1), size=(obs_count,))
    dummy = np.zeros(im_shape)
    mask = np.zeros(X.shape)
    for i in range(obs_count):
        dummy = (0.0 * dummy) + 1.0
        dummy[off_row[i]:(off_row[i]+rs), off_col[i]:(off_col[i]+cs)] = 0.0
        mask[i,:] = dummy.ravel()
    return mask.astype(theano.config.floatX)

def posterior_klds(IN, Xtr, batch_size, batch_count):
    """
    Get posterior KLd cost for some inputs from Xtr.
    """
    post_klds = []
    for i in range(batch_count):
        batch_idx = npr.randint(low=0, high=Xtr.shape[0], size=(batch_size,))
        X = Xtr.take(batch_idx, axis=0)
        post_klds.extend([k for k in IN.kld_func(X)])
    return post_klds

def collect_obs_costs(batch_costs, batch_reps):
    """
    Collect per-observation costs from a cost vector containing the cost for
    multiple repetitions of each observation.
    """
    obs_count = int(batch_costs.shape[0] / batch_reps)
    obs_costs = np.zeros((obs_count,))
    obs_idx = -1
    for i in range(batch_costs.shape[0]):
        if ((i % batch_reps) == 0):
            obs_idx = obs_idx + 1
        obs_costs[obs_idx] = obs_costs[obs_idx] + batch_costs[i]
    obs_costs = obs_costs / batch_reps
    return obs_costs