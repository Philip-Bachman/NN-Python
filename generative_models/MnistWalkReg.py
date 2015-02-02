import time
import utils as utils
import numpy as np
import numpy.random as npr
import theano
import theano.tensor as T

from load_data import load_udm, load_udm_ss, load_mnist
from PeaNet import PeaNet, load_peanet_from_file
from InfNet import InfNet, load_infnet_from_file
from GenNet import GenNet, load_gennet_from_file
from PeaNetSeq import PeaNetSeq
from GIPair import GIPair
from NetLayers import relu_actfun, softplus_actfun, \
                      safe_softmax, safe_log
import GenNet as GNet
import InfNet as INet
import PeaNet as PNet
from DKCode import PCA_theano
from MCSampler import MCSampler, resample_chain_steps

def manifold_walk_regularization():
    # Initialize a source of randomness
    rng = np.random.RandomState(123)

    # Load some data to train/validate/test with
    sup_count = 600
    dataset = 'data/mnist.pkl.gz'
    datasets = load_udm_ss(dataset, sup_count, rng, zero_mean=False)
    Xtr_su = datasets[0][0].get_value(borrow=False)
    Ytr_su = datasets[0][1].get_value(borrow=False).astype(np.int32)
    Xtr_un = datasets[1][0].get_value(borrow=False)
    Ytr_un = datasets[1][1].get_value(borrow=False).astype(np.int32)

    # get the joint labeled and unlabeled data
    Xtr_un = np.vstack([Xtr_su, Xtr_un]).astype(theano.config.floatX)
    Ytr_un = np.vstack([Ytr_su[:,np.newaxis], Ytr_un[:,np.newaxis]])
    Ytr_un = 0 * Ytr_un # KEEP CATS FIXED OR FREE? YES/NO?
    Xtr_mean = np.mean(Xtr_un, axis=0, keepdims=True)
    # get the labeled data
    Xtr_su = Xtr_su.astype(theano.config.floatX)
    Ytr_su = Ytr_su[:,np.newaxis]
    # get observations and labels for the validation set
    Xva = datasets[2][0].get_value(borrow=False).astype(theano.config.floatX)
    Yva = datasets[2][1].get_value(borrow=False).astype(np.int32)
    Yva = Yva[:,np.newaxis] # numpy is dumb
    # get size information for the data and training batches
    un_samples = Xtr_un.shape[0]
    su_samples = Xtr_su.shape[0]
    va_samples = Xva.shape[0]
    data_dim = Xtr_su.shape[1]
    label_dim = 10
    batch_size = 100

    # Symbolic inputs
    Xd = T.matrix(name='Xd')
    Xc = T.matrix(name='Xc')
    Xm = T.matrix(name='Xm')
    Xt = T.matrix(name='Xt')
    Xp = T.matrix(name='Xp')
    Yd = T.icol('Yd')

    # Load inferencer and generator from saved parameters
    gn_fname = "MMS_RESULTS_32D/pt60k_walk_params_b250000_GN.pkl"
    in_fname = "MMS_RESULTS_32D/pt60k_walk_params_b250000_IN.pkl"
    IN = INet.load_infnet_from_file(f_name=in_fname, rng=rng, Xd=Xd, Xc=Xc, Xm=Xm)
    GN = GNet.load_gennet_from_file(f_name=gn_fname, rng=rng, Xp=Xp)
    prior_dim = GN.latent_dim

    MCS = MCSampler(rng=rng, Xd=Xd, i_net=IN, g_net=GN, chain_len=5, \
                    data_dim=data_dim, prior_dim=prior_dim)
    full_chain_len = MCS.chain_len + 1

    # setup "chain" versions of the labeled/unlabeled/validate sets
    Xtr_su_chains = [Xtr_su.copy() for i in range(full_chain_len)]
    Xtr_un_chains = [Xtr_un.copy() for i in range(full_chain_len)]
    Ytr_su_chains = [Ytr_su for i in range(full_chain_len)]
    Ytr_un_chains = [Ytr_un for i in range(full_chain_len)]
    Xva_chains = [Xva for i in range(full_chain_len)]
    Yva_chains = [Yva for i in range(full_chain_len)]

    # choose some parameters for the categorical inferencer
    pn_params = {}
    pc0 = [data_dim, 800, 800, label_dim]
    pn_params['proto_configs'] = [pc0]
    # Set up some spawn networks
    sc0 = {'proto_key': 0, 'input_noise': 0.1, 'bias_noise': 0.1, 'do_dropout': True}
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

    print("Initializing PNS...")
    # Initialize the PeaNetSeq
    PNS = PeaNetSeq(rng=rng, pea_net=PN, seq_len=full_chain_len, \
    		seq_Xd=None, params=None)

    # set weighting parameters for the various costs...
    PNS.set_lam_class(1.0)
    PNS.set_lam_pea_su(0.0)
    PNS.set_lam_pea_un(2.0)
    PNS.set_lam_ent(0.0)
    PNS.set_lam_l2w(1e-5)

    out_file = open("MWR_TEST_RESULT.txt", 'wb')
    learn_rate = 0.1
    PNS.set_pn_sgd_params(lr_pn=learn_rate, mom_1=0.9, mom_2=0.999)
    for i in range(300000):
        if i < 5000:
            scale = float(i + 1) / 5000.0
        if ((i+1 % 100000) == 0):
            learn_rate = learn_rate * 0.5
        if ((i % 250) == 0):
        	Xtr_su_chains = resample_chain_steps(MCS, Xtr_su_chains)
        	Xtr_un_chains = resample_chain_steps(MCS, Xtr_un_chains)
        # get some data to train with
        su_idx = npr.randint(low=0,high=su_samples,size=(batch_size,))
        xsuc = [(x.take(su_idx, axis=0) - Xtr_mean) for x in Xtr_su_chains]
        ysuc = [y.take(su_idx, axis=0) for y in Ytr_su_chains]
        un_idx = npr.randint(low=0,high=un_samples,size=(batch_size,))
        xunc = [(x.take(un_idx, axis=0) - Xtr_mean) for x in Xtr_un_chains]
        yunc = [y.take(un_idx, axis=0) for y in Ytr_un_chains]
        Xb_chains = [np.vstack((xsu, xun)) for (xsu, xun) in zip(xsuc, xunc)]
        Yb_chains = [np.vstack((ysu, yun)) for (ysu, yun) in zip(ysuc, yunc)]
        # set learning parameters for this update
        PNS.set_pn_sgd_params(lr_pn=learn_rate, mom_1=0.9, mom_2=0.999)
        # do a minibatch update of all PeaNet parameters
        outputs = PNS.train_joint(*(Xb_chains + Yb_chains))
        joint_cost = 1.0 * outputs[0]
        class_cost = 1.0 * outputs[1]
        pea_cost = 1.0 * outputs[2]
        ent_cost = 1.0 * outputs[3]
        other_reg_cost = 1.0 * outputs[4]
        assert(not (np.isnan(joint_cost)))
        if ((i % 500) == 0):
            o_str = "batch: {0:d}, joint: {1:.4f}, class: {2:.4f}, pea: {3:.4f}, ent: {4:.4f}, other_reg: {5:.4f}".format( \
                    i, joint_cost, class_cost, pea_cost, ent_cost, other_reg_cost)
            print(o_str)
            out_file.write(o_str+"\n")
            out_file.flush()
            # check classification error on training and validation set
            train_err = PNS.classification_error(Xtr_su-Xtr_mean, Ytr_su)
            va_err = PNS.classification_error(Xva-Xtr_mean, Yva)
            o_str = "    tr_err: {0:.4f}, va_err: {1:.4f}".format(train_err, va_err)
            print(o_str)
            out_file.write(o_str+"\n")
            out_file.flush()
        if ((i % 1000) == 0):
            # draw the main PeaNet's first-layer filters/weights
            file_name = "PNS_WEIGHTS_ZMUV.png".format(i)
            utils.visualize_net_layer(PNS.PN.proto_nets[0][0], file_name)
    print("TESTING COMPLETE!")

if __name__ == "__main__":
	manifold_walk_regularization()