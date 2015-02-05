import time
import utils as utils
import numpy as np
import numpy.random as npr
import theano
import theano.tensor as T

from load_data import load_svhn, load_svhn_gray, load_svhn_all_gray_zca
from PeaNet import PeaNet, load_peanet_from_file
from InfNet import InfNet, load_infnet_from_file
from GenNet import GenNet, load_gennet_from_file
from PeaNetSeq import PeaNetSeq
from VCGLoop import VCGLoop
from GIPair import GIPair
from NetLayers import relu_actfun, softplus_actfun, \
                      safe_softmax, safe_log
import GenNet as GNet
import InfNet as INet
import PeaNet as PNet
from DKCode import PCA_theano

import sys, resource
resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
sys.setrecursionlimit(10**6)

# DERP
RESULT_PATH = "SVHN_WALKOUT_TEST_KLD/"

####################
# HELPER FUNCTIONS #
####################

def shift_and_scale_into_01(X):
    X = X - np.min(X, axis=1, keepdims=True)
    X = X / np.max(X, axis=1, keepdims=True)
    return X

def train_valid_split(X, valid_count=1000):
    """
    Split the observations in the rows of X into train/validate sets.
    """
    obs_count = X.shape[0]
    idx = np.arange(obs_count)
    npr.shuffle(idx)
    va_idx = idx[:valid_count]
    tr_idx = idx[valid_count:]
    Xtr = X.take(tr_idx, axis=0)
    Xva = X.take(va_idx, axis=0)
    return Xtr, Xva

##########################################
##########################################
## TEST SEMISUPERVISED LEARNING ON SVHN ##
##########################################
##########################################

def test_semisupervised():
    import utils as utils
    from load_data import load_udm, load_udm_ss, load_mnist
    from NetLayers import relu_actfun

    # Initialize a source of randomness
    rng = np.random.RandomState(123)

    sup_count = 1000
    va_count = 10000
    # Load some data to train/validate/test with
    tr_file = 'data/svhn_train_gray.pkl'
    te_file = 'data/svhn_test_gray.pkl'
    ex_file = 'data/svhn_extra_gray.pkl'
    data = load_svhn_gray(tr_file, te_file, ex_file=ex_file, ex_count=200000)
    X_mean = np.mean(data['Xtr'], axis=0, keepdims=True)
    X_std = np.std(data['Xtr'], axis=0, keepdims=True)
    data['Xtr'] = (data['Xtr'] - X_mean) / X_std
    data['Xte'] = (data['Xte'] - X_mean) / X_std
    data['Xex'] = (data['Xex'] - X_mean) / X_std
    idx = np.arange(data['Xtr'].shape[0])
    npr.shuffle(idx)
    Xva = data['Xte'][:,:] #[idx[0:va_count],:]
    Yva = data['Yte'][:,:].astype(np.int32) # [idx[0:va_count],:].astype(np.int32)
    Xtr_su = data['Xtr'][idx[va_count:(va_count+sup_count)], :]
    Ytr_su = data['Ytr'][idx[va_count:(va_count+sup_count)], :].astype(np.int32)
    Xtr_un = np.vstack([data['Xtr'][idx[va_count:], :], data['Xex']])
    Ytr_un = np.zeros((Xtr_un.shape[0],1)).astype(np.int32)
    print("unique(Ytr_su): {0:s}".format(str(np.unique(Ytr_su))))
    print("unique(Ytr_un): {0:s}".format(str(np.unique(Ytr_un))))
    print("Xtr_su.shape: {0:s}, Ytr_su.shape: {1:s}".format(str(Xtr_su.shape), str(Ytr_su.shape)))
    print("Xva.shape: {0:s}, Yva.shape: {1:s}".format(str(Xva.shape), str(Yva.shape)))

    un_samples = Xtr_un.shape[0]
    su_samples = Xtr_su.shape[0]
    va_samples = Xva.shape[0]

    # set up some symbolic variables for input to the GITrip
    Xd = T.matrix('Xd_base')
    Yd = T.icol('Yd_base')
    # set some "shape" parameters for the networks
    data_dim = Xtr_un.shape[1]
    label_dim = 10
    batch_size = 200 # we'll take 2x this per batch, for sup and unsup

    # choose some parameters for the categorical inferencer
    pn_params = {}
    pc0 = [data_dim, 800, 800, label_dim]
    pn_params['proto_configs'] = [pc0]
    # Set up some spawn networks
    sc0 = {'proto_key': 0, 'input_noise': 0.1, 'bias_noise': 0.2, 'do_dropout': True}
    pn_params['spawn_configs'] = [ sc0 ]
    pn_params['spawn_weights'] = [ 1.0 ]
    # Set remaining params
    pn_params['activation'] = relu_actfun
    pn_params['init_scale'] = 0.5
    pn_params['lam_l2a'] = 1e-3
    pn_params['vis_drop'] = 0.2
    pn_params['hid_drop'] = 0.5

    # Initialize the base network for this PNSeq
    PN = PeaNet(rng=rng, Xd=Xd, params=pn_params)
    PN.init_biases(0.1)

    # Initialize the PeaNetSeq
    PNS = PeaNetSeq(rng=rng, pea_net=PN, seq_len=2, seq_Xd=None, params=None)

    # set weighting parameters for the various costs...
    PNS.set_lam_class(1.0)
    PNS.set_lam_pea_su(0.0)
    PNS.set_lam_pea_un(1.0)
    PNS.set_lam_ent(0.0)
    PNS.set_lam_l2w(1e-5)

    out_file = open("SVHN_SS_TEST.txt", 'wb')
    cost_1 = [0. for i in range(10)]
    learn_rate = 0.02
    PNS.set_pn_sgd_params(lr_pn=learn_rate, mom_1=0.9, mom_2=0.999)
    for i in range(300000):
        # get some data to train with
        su_idx = npr.randint(low=0,high=su_samples,size=(batch_size,))
        Xd_su = Xtr_su.take(su_idx, axis=0)
        Yd_su = Ytr_su.take(su_idx, axis=0)
        un_idx = npr.randint(low=0,high=un_samples,size=(batch_size,))
        Xd_un = Xtr_un.take(un_idx, axis=0)
        Yd_un = Ytr_un.take(un_idx, axis=0)
        Xd_batch = np.vstack((Xd_su, Xd_un))
        Yd_batch = np.vstack((Yd_su, Yd_un))
        # set learning parameters for this update
        PNS.set_pn_sgd_params(lr_pn=learn_rate, mom_1=0.9, mom_2=0.999)
        # do a minibatch update of all PeaNet parameters
        outputs = PNS.train_joint(Xd_batch, Xd_batch, Yd_batch, Yd_batch)
        cost_1 = [(cost_1[k] + 1.*outputs[k]) for k in range(len(outputs))]
        if ((i % 1000) == 0):
            cost_1 = [(v / 1000.) for v in cost_1]
            o_str = "batch: {0:d}, joint: {1:.4f}, class: {2:.4f}, pea: {3:.4f}, ent: {4:.4f}, other_reg: {5:.4f}".format( \
                    i, cost_1[0], cost_1[1], cost_1[2], cost_1[3], cost_1[4])
            print(o_str)
            out_file.write(o_str+"\n")
            out_file.flush()
            cost_1 = [0. for v in cost_1]
            # check classification error on training and validation set
            train_err = PNS.classification_error(Xtr_su, Ytr_su)
            va_err = PNS.classification_error(Xva, Yva)
            o_str = "    tr_err: {0:.4f}, va_err: {1:.4f}".format(train_err, va_err)
            print(o_str)
            out_file.write(o_str+"\n")
            out_file.flush()
        if ((i % 1000) == 0):
            # draw the main PeaNet's first-layer filters/weights
            file_name = "SVHN_SS_PN_WEIGHTS.png".format(i)
            utils.visualize_net_layer(PNS.PN.proto_nets[0][0], file_name)
    print("TESTING COMPLETE!")



if __name__=="__main__":
    test_semisupervised()