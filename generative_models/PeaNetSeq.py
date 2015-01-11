###############################################################################
# Code for managing a sequence of clones of a "base" PeaNet.   ################
#                                                                             #
# The reason to put the networks in a sequence is to apply PEA-type           #
# regularization (for perturbation resistance) between pairs of networks that #
# are adjacent in the sequence. How to generate samples that will be useful   #
# for regularization is an interesting question....                           #
#                                                                             #
# These networks are trained by providing equal-sized input matrices for      #
# each clone network in the sequence. Each clone network will compute the     #
# outputs for its own input matrix and the PEA regularization cost will be    #
# computed between the outputs of each pair of sequentially-adjacent clones.  #
#                                                                             #
# Labels must be provided for all inputs to all clones. When the label is 0   #
# we will treat the input as "unsupervised" and when the label is > 0 we will #
# treat the input as "supervised". The PEA cost for unsupervised and          #
# supervised inputs is weighted by different "lambdas", to permit switching   #
# the cost on or off for each input type.                                     #
###############################################################################

# basic python
import numpy as np
import numpy.random as npr
from collections import OrderedDict

# theano business
import theano
import theano.tensor as T
#from theano.tensor.shared_randomstreams import RandomStreams as RandStream
from theano.sandbox.cuda.rng_curand import CURAND_RandomStreams as RandStream

# phil's sweetness
from NetLayers import safe_log, safe_softmax
from PeaNet import PeaNet
from DKCode import get_adam_updates, get_adadelta_updates

######################################################
# HELPER FUNCTIONS FOR PEAR AND CLASSIFICATION COSTS #
######################################################

def smooth_softmax(p, lam_smooth=1e-3):
    """
    Give a "smoothed" softmax distribution form for p. This is to counter
    the tendency for KL-divergence, cross-entropy, etc. to get a bit wonky
    when comparing strongly peaked categorical distributions.
    """
    dist_size = T.cast(p.shape[1], 'floatX')
    p_sm = (safe_softmax(p) + lam_smooth) / (1.0 + (dist_size * lam_smooth))
    return p_sm

def smooth_entropy(p, lam_smooth=1e-3):
    """
    Measure the entropy of distribution p after converting it from log-space
    into a normalized, sum-to-one distribution.
    """
    dist_size = T.cast(p.shape[1], 'floatX')
    p_sm = smooth_softmax(p, lam_smooth=lam_smooth)
    ent_sm = -T.sum((safe_log(p_sm) * p_sm), axis=1, keepdims=True)
    return ent_sm

def smooth_kl_divergence(p, q, lam_smooth=1e-3):
    """
    Measure the KL-divergence from "approximate" distribution q to "true"
    distribution p. Use smoothed softmax to convert p and q from log-space
    encodings into normalized, sum-to-one distributions.
    """
    dist_size = T.cast(p.shape[1], 'floatX')
    p_sm = smooth_softmax(p, lam_smooth=lam_smooth)
    q_sm = smooth_softmax(q, lam_smooth=lam_smooth)
    # This term is: cross_entropy(p, q) - entropy(p)
    kl_sm = T.sum(((safe_log(p_sm) - safe_log(q_sm)) * p_sm), axis=1, keepdims=True)
    return kl_sm

def smooth_cross_entropy(p, q):
    """
    Measure the cross-entropy between "approximate" distribution q and
    "true" distribution p. Use smoothed softmax to convert p and q from
    log-space encodings into normalized, sum-to-one distributions.
    """
    dist_size = T.cast(p.shape[1], 'floatX')
    p_sm = smooth_softmax(p, lam_smooth=lam_smooth)
    q_sm = smooth_softmax(q, lam_smooth=lam_smooth)
    # This term is: entropy(p) + kl_divergence(p, q)
    ce_sm = -T.sum((p_sm * safe_log(q_sm)), axis=1, keepdims=True)
    return ce_sm

def cat_entropy(p):
    """
    Compute the entropy of (row-wise) categorical distributions given by
    taking the softmax transform of p.
    """
    p_sm = smooth_softmax(p, lam_smooth=1e-4)
    row_ents = -T.sum((p_sm * safe_log(p_sm)), axis=1, keepdims=True)
    return row_ents

def cat_cost_xent(Yp, Yd):
    """
    Compute cross-entropy classification cost given the target labels in Yd
    and the log-space predictions in Yp. Unsupervised points are indicated by
    Yd[i] == 0, and labels for supervised points need to be shifted -1.
    """
    Yp_sm = smooth_softmax(Yp, lam_smooth=1e-5)
    # construct a mask that zeros-out unsupervised rows
    row_idx = T.arange(Yd.shape[0])
    row_mask = T.neq(Yd, 0).reshape((Yd.shape[0], 1))
    # make a matrix that has the predicted class probs for supervised
    # inputs' rows and is all ones for rows that are unsupervised
    wacky_mat = (Yp_sm * row_mask) + (1. - row_mask)
    # compute cross-entropy (classification) loss
    cat_cost = -T.sum(safe_log(wacky_mat[row_idx,(Yd.flatten()-1)])) \
            / (T.sum(row_mask) + 1e-4)
    return cat_cost

def one_hot(Yc, cat_dim=None):
    """
    Given a tensor Yc of dimension d with integer values from range(cat_dim),
    return new tensor of dimension d + 1 with values 0/1, where the last
    dimension gives a one-hot representation of the values in Yc.

    if cat_dim is not given, cat_dim is set to max(Yc) + 1
    """
    if cat_dim is None:
        cat_dim = T.max(Yc) + 1
    ranges = T.shape_padleft(T.arange(cat_dim), Yc.ndim)
    Yoh = T.eq(ranges, T.shape_padright(Yc, 1))
    return Yoh

def one_hot_np(Yc, cat_dim=None):
    """
    Given a numpy integer column vector Yc, generate a matrix Yoh in which
    Yoh[i,:] is a one-hot vector -- Yoh[i,Yc[i]] = 1.0 and other Yoh[i,j] = 0
    """
    if cat_dim is None:
        cat_dim = np.max(Yc) + 1
    Yoh = np.zeros((Yc.size, cat_dim))
    Yoh[np.arange(Yc.size),Yc.flatten()] = 1.0
    return Yoh

###############################################
# IMPLEMENTATION OF THE SEQUENTIAL PEAR CLASS #
###############################################

class PeaNetSeq(object):
    """
    TODO: preliminary documentation.

    Parameters:
        rng: numpy.random.RandomState (for reproducibility)
        pea_net: The PeaNet instance that will spawn the clone sequence
        seq_len: length of the sequence w.r.t. to compute PEAR cost
        seq_Xd: optional list of symbolic inputs to the clone sequence
        no_noise: boolean switch for whether to use noisy PeaNets
        params: dict for passing additional parameters
        shared_param_dict: dict for making clones with shared lambdas
    """
    def __init__(self, rng=None, pea_net=None, seq_len=2, seq_Xd=None, \
            no_noise=False, params=None, shared_param_dict=None):
        assert(not (rng is None))
        assert(not (pea_net is None))
        assert(seq_len >= 2)
        if not (seq_Xd is None):
            # if symbolic inputs for the sequence to receive are given when
            # the sequence is created, check if it's the right amount.
            assert(len(seq_Xd) == seq_len)
        self.params = params
        # setup a rng for this PeaNetSeq
        self.rng = RandStream(rng.randint(100000))
        if shared_param_dict is None:
            self.is_clone = False
            self.shared_param_dict = {}
        else:
            print("Inititalizing a PeaNetSeq clone...")
            self.is_clone = True
            self.shared_param_dict = shared_param_dict
        # make param dict for a noiseless version of the PNSeq
        new_pn_params = pea_net.params.copy()
        if no_noise:
            for sc in new_pn_params['spawn_configs']:
                sc['input_noise'] = 0.0
                sc['bias_noise'] = 0.0
                sc['do_dropout'] = False

        # setup the sequence of PeaNet clones
        self.seq_len = seq_len
        self.Xd_seq = []
        self.Yd_seq = []
        self.PN_seq = []
        for i in range(self.seq_len):
            if seq_Xd is None:
                # make new symbolic inputs if none were given
                Xd_i = T.matrix(name="Xd_{0:d}".format(i))
            else:
                # otherwise, use the given symbolic inputs
                Xd_i = seq_Xd[i]
            # create a label vector to be associated with this clone
            Yd_i = T.icol(name="Yd_{0:d}".format(i))
            # add observation/label inputs and the clone to the sequence
            self.Xd_seq.append(Xd_i)
            self.Yd_seq.append(Yd_i)
            self.PN_seq.append(pea_net.shared_param_clone(rng=rng, Xd=Xd_i, \
                    params=new_pn_params))
        self.PN = self.PN_seq[0]
        # create the full list of symbolic inputs required for training
        self.seq_inputs = self.Xd_seq + self.Yd_seq

        if not self.is_clone:
            # shared var learning rate for the base network
            zero_ary = np.zeros((1,)).astype(theano.config.floatX)
            self.lr_pn = theano.shared(value=zero_ary, name='pnseq_lr_pn')
            # shared var momentum parameters for the base network
            self.mom_1 = theano.shared(value=zero_ary, name='pnseq_mom_1')
            self.mom_2 = theano.shared(value=zero_ary, name='pnseq_mom_2')
            self.it_count = theano.shared(value=zero_ary, name='pnseq_it_count')
            # init parameters for controlling learning dynamics
            self.set_pn_sgd_params()
            # init shared var for weighting PEA cost on supervised inputs
            self.lam_pea_su = theano.shared(value=zero_ary, name='pnseq_lam_pea_su')
            self.set_lam_pea_su(lam_pea_su=1.0)
            # init shared var for weighting PEA cost on unsupervised inputs
            self.lam_pea_un = theano.shared(value=zero_ary, name='pnseq_lam_pea_un')
            self.set_lam_pea_un(lam_pea_un=1.0)
            # init shared var for weighting entropy cost on unsupervised inputs
            self.lam_ent = theano.shared(value=zero_ary, name='pnseq_lam_ent')
            self.set_lam_ent(lam_ent=0.0)
            # init shared var for weighting classification cost on supervised inputs
            self.lam_class = theano.shared(value=zero_ary, name='pnseq_lam_class')
            self.set_lam_class(lam_class=1.0)
            # init shared var for controlling l2 regularization on params
            self.lam_l2w = theano.shared(value=zero_ary, name='pnseq_lam_l2w')
            self.set_lam_l2w(1e-4)
            # make the dict for passing around shared lambdas
            self.shared_param_dict['lr_pn'] = self.lr_pn
            self.shared_param_dict['mom_1'] = self.mom_1
            self.shared_param_dict['mom_2'] = self.mom_2
            self.shared_param_dict['it_count'] = self.it_count
            self.shared_param_dict['lam_pea_su'] = self.lam_pea_su
            self.shared_param_dict['lam_pea_un'] = self.lam_pea_un
            self.shared_param_dict['lam_ent'] = self.lam_ent
            self.shared_param_dict['lam_class'] = self.lam_class
            self.shared_param_dict['lam_l2w'] = self.lam_l2w
        else:
            # copy shared lambdas from the cloning dict
            self.lr_pn = self.shared_param_dict['lr_pn']
            self.mom_1 = self.shared_param_dict['mom_1']
            self.mom_2 = self.shared_param_dict['mom_2']
            self.it_count = self.shared_param_dict['it_count']
            self.lam_pea_su = self.shared_param_dict['lam_pea_su']
            self.lam_pea_un = self.shared_param_dict['lam_pea_un']
            self.lam_ent = self.shared_param_dict['lam_ent']
            self.lam_class = self.shared_param_dict['lam_class']
            self.lam_l2w = self.shared_param_dict['lam_l2w']

        # grab the full set of "optimizable" parameters from the base network
        self.mlp_params = [p for p in self.PN.proto_params]

        ###################################
        # CONSTRUCT THE COSTS TO OPTIMIZE #
        ###################################
        self.pea_su_cost, self.pea_un_cost = self._construct_pea_costs()
        self.pea_cost = (self.lam_pea_su[0] * self.pea_su_cost) + \
                (self.lam_pea_un[0] * self.pea_un_cost)
        self.ent_cost = self.lam_ent[0] * self._construct_ent_cost()
        self.class_cost = self.lam_class[0] * self._construct_class_cost()
        self.other_reg_cost = self._construct_other_reg_cost()
        self.joint_cost = self.pea_cost + self.ent_cost + self.class_cost + \
                self.other_reg_cost

        ######################################################
        # Construct updates for the shared PeaNet parameters #
        ######################################################
        self.mlp_grads = OrderedDict()
        for p in self.mlp_params:
            self.mlp_grads[p] = T.grad(self.joint_cost, p).clip(-1.0,1.0)
        # Construct the updates for the generator and inferencer networks
        self.mlp_updates = get_adam_updates(params=self.mlp_params, \
                grads=self.mlp_grads, alpha=self.lr_pn, \
                beta1=self.mom_1, beta2=self.mom_2, it_count=self.it_count, \
                mom2_init=1e-3, smoothing=1e-8)

        # Construct a function for training the base network to minimize the
        # sequential PEAR cost
        self.train_joint = self._construct_train_joint()
        # make a function for computing outputs of the main PeaNet
        self.get_pn_output = theano.function([self.PN.Xd], \
                outputs=self.PN.output_proto)
        return

    def set_pn_sgd_params(self, lr_pn=0.01, mom_1=0.9, mom_2=0.999):
        """
        Set learning rate and momentum parameters for base network updates.
        """
        zero_ary = np.zeros((1,))
        new_lr = zero_ary + lr_pn
        self.lr_pn.set_value(new_lr.astype(theano.config.floatX))
        new_mom_1 = zero_ary + mom_1
        self.mom_1.set_value(new_mom_1.astype(theano.config.floatX))
        new_mom_2 = zero_ary + mom_2
        self.mom_2.set_value(new_mom_2.astype(theano.config.floatX))
        return

    def set_lam_pea_su(self, lam_pea_su=1.0):
        """
        Set weight for the cost of PEAR on supervised inputs.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_pea_su
        self.lam_pea_su.set_value(new_lam.astype(theano.config.floatX))
        return

    def set_lam_pea_un(self, lam_pea_un=1.0):
        """
        Set weight for the cost of PEAR on unsupervised inputs.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_pea_un
        self.lam_pea_un.set_value(new_lam.astype(theano.config.floatX))
        return

    def set_lam_ent(self, lam_ent=1.0):
        """
        Set weight for the entropy cost on unsupervised inputs.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_ent
        self.lam_ent.set_value(new_lam.astype(theano.config.floatX))
        return

    def set_lam_class(self, lam_class=1.0):
        """
        Set weight for the classification cost on supervised inputs.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_class
        self.lam_class.set_value(new_lam.astype(theano.config.floatX))
        return

    def set_lam_l2w(self, lam_l2w=1e-3):
        """
        Set the relative strength of l2 regularization on network params.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_l2w
        self.lam_l2w.set_value(new_lam.astype(theano.config.floatX))
        return

    def _construct_pea_costs(self):
        """
        Construct the sequence-based PEA regularization cost. Use the output
        of the first spawn net of each clone of the base PeaNet instance.
        """
        pea_costs_su = []
        pea_costs_un = []
        for i in range(self.seq_len-1):
            # get the user-provided labels for each input to this clone
            Yd_i = self.Yd_seq[i]
            # get the outputs from a pair of sequentially-adjacent clones
            x1 = self.PN_seq[i].output_spawn[0]
            x2 = self.PN_seq[i+1].output_spawn[0]
            # construct a mask that zeros-out unsupervised rows
            row_idx = T.arange(Yd_i.shape[0])
            row_mask = T.neq(Yd_i, 0).reshape((Yd_i.shape[0], 1))
            # compute PEAR costs for supervised and unsupervised points
            pea_costs_i = (smooth_kl_divergence(x1, x2, lam_smooth=1e-3) + \
                    smooth_kl_divergence(x2, x1, lam_smooth=1e-3)) / 2.0
            pea_cost_su = T.sum(row_mask * pea_costs_i) / \
                    (T.sum(row_mask) + 1e-4)
            pea_cost_un = T.sum((1.0 - row_mask) * pea_costs_i) / \
                    (T.sum(1.0 - row_mask) + 1e-4)
            # derp, record the computed values, derp
            pea_costs_su.append(pea_cost_su)
            pea_costs_un.append(pea_cost_un)
        pea_cost_su = sum(pea_costs_su) / float(self.seq_len-1)
        pea_cost_un = sum(pea_costs_un) / float(self.seq_len-1)
        return [pea_cost_su, pea_cost_un]

    def _construct_ent_cost(self):
        """
        Construct the sequence-based entropy cost. Use the output of
        the first spawn net of each clone of the base PeaNet instance.
        """
        ent_costs = []
        for i in range(self.seq_len):
            # get the user-provided labels for each input to this clone
            Yd_i = self.Yd_seq[i]
            # get the clone-provided predicted class probs for each input
            Yp_i = self.PN_seq[i].output_spawn[0]
            # get the entropy cost for all inputs
            ent_costs_i = cat_entropy(Yp_i)
            # count only the entropy costs for unsupervised inputs
            row_mask = T.eq(Yd_i, 0).reshape((Yd_i.shape[0], 1))
            ent_cost_i = T.sum(ent_costs_i * row_mask) / T.sum(row_mask)
            ent_costs.append(ent_cost_i)
        ent_cost = sum(ent_costs) / float(self.seq_len)
        return ent_cost

    def _construct_class_cost(self):
        """
        Construct the sequence-based classification cost. Use the output of
        the first spawn net of each clone of the base PeaNet instance.
        """
        class_costs = []
        for i in range(self.seq_len):
            # get the user-provided labels for each input to this clone
            Yd_i = self.Yd_seq[i]
            # get the clone-provided predicted class probs for each input
            Yp_i = self.PN_seq[i].output_spawn[0]
            # get the classification cost on supervised points
            cat_cost = cat_cost_xent(Yp_i, Yd_i)
            class_costs.append(cat_cost)
        class_cost = sum(class_costs) / float(self.seq_len)
        return class_cost

    def _construct_other_reg_cost(self):
        """
        Construct the cost for low-level basic regularization. E.g. for
        applying l2 regularization to the network activations and parameters.
        """
        obs_count = T.cast(self.Xd_seq[0].shape[0], 'floatX')
        act_reg_costs = []
        for i in range(self.seq_len):
            act_reg_costs.append(self.PN_seq[i].act_reg_cost)
        pn_param_cost = sum([T.sum(par**2.0) for par in self.mlp_params])
        param_reg_cost = self.lam_l2w[0] * pn_param_cost
        act_reg_cost = sum(act_reg_costs) / (float(self.seq_len) * obs_count)
        other_reg_cost = act_reg_cost + param_reg_cost
        return other_reg_cost

    def _construct_train_joint(self):
        """
        Construct theano function to train inferencer and generator jointly.
        """
        outputs = [self.joint_cost, self.class_cost, self.pea_cost, \
                self.ent_cost, self.other_reg_cost]
        func = theano.function(inputs=self.seq_inputs, \
                outputs=outputs, \
                updates=self.mlp_updates)
        return func

    def classification_error(self, X_d, Y_d):
        """
        Compute classification error for a set of observations X_d with known
        labels Y_d, based on passing X_d through the unperturbed prototype
        network of the PeaNet underlying this PeaNetSeq.
        """
        # first, convert labels to account for semi-supervised labeling
        Y_mask = 1.0 * (Y_d != 0)
        Y_d = Y_d - 1
        # compute the main PeaNet's output for X_d
        Y_p = self.get_pn_output(X_d)
        # get the implied class labels
        Y_c = np.argmax(Y_p, axis=1).reshape((Y_d.shape[0],1))
        # compute the classification error for points with valid labels
        err_rate = np.sum(((Y_d != Y_c) * Y_mask)) / np.sum(Y_mask)
        return err_rate

    def shared_param_clone(self, rng=None, seq_len=2, seq_Xd=None, \
            no_noise=False):
        """
        Create a clone of this PeaNetSeq that shares its cost lambdas.
        """
        pns_clone = PeaNetSeq(rng=rng, pea_net=self.PN, seq_len=seq_len, \
                seq_Xd=seq_Xd, no_noise=no_noise, params=self.params, \
                shared_param_dict=self.shared_param_dict)
        return pns_clone

def zmuv(X, axis=1):
    X = X - np.mean(X, axis=axis, keepdims=True)
    X = X / np.std(X, axis=axis, keepdims=True)
    return X

if __name__=="__main__":
    import utils as utils
    from load_data import load_udm, load_udm_ss, load_mnist
    from NetLayers import relu_actfun

    # Initialize a source of randomness
    rng = np.random.RandomState(123)

    # Load some data to train/validate/test with
    sup_count = 600
    dataset = 'data/mnist.pkl.gz'
    datasets = load_udm_ss(dataset, sup_count, rng, zero_mean=True)
    Xtr_su = datasets[0][0].get_value(borrow=False)
    Ytr_su = datasets[0][1].get_value(borrow=False).astype(np.int32)
    Xtr_un = datasets[1][0].get_value(borrow=False)
    Ytr_un = datasets[1][1].get_value(borrow=False).astype(np.int32)
    # get the joint labeled and unlabeled data
    Xtr_un = np.vstack([Xtr_su, Xtr_un]).astype(theano.config.floatX)
    Ytr_un = np.vstack([Ytr_su[:,np.newaxis], Ytr_un[:,np.newaxis]])
    Ytr_un = 0 * Ytr_un # KEEP CATS FIXED OR FREE? YES/NO?
    # get the labeled data
    Xtr_su = Xtr_su.astype(theano.config.floatX)
    Ytr_su = Ytr_su[:,np.newaxis]
    # get observations and labels for the validation set
    Xva = datasets[2][0].get_value(borrow=False).astype(theano.config.floatX)
    Yva = datasets[2][1].get_value(borrow=False).astype(np.int32)
    Yva = Yva[:,np.newaxis] # numpy is dumb
    # set observations to zero mean and unit variance
    #Xtr_su = zmuv(Xtr_su, axis=1)
    #Xtr_un = zmuv(Xtr_un, axis=1)
    #Xva = zmuv(Xva, axis=1)
    # get size information for the data
    un_samples = Xtr_un.shape[0]
    su_samples = Xtr_su.shape[0]
    va_samples = Xva.shape[0]

    # set up some symbolic variables for input to the GITrip
    Xd = T.matrix('Xd_base')
    Yd = T.icol('Yd_base')
    # set some "shape" parameters for the networks
    data_dim = Xtr_un.shape[1]
    label_dim = 10
    batch_size = 100 # we'll take 2x this per batch, for sup and unsup

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
    pn_params['lam_l2a'] = 1e-3
    pn_params['vis_drop'] = 0.2
    pn_params['hid_drop'] = 0.5

    # Initialize the base network for this PNSeq
    PN = PeaNet(rng=rng, Xd=Xd, params=pn_params)
    PN.init_biases(0.1)

    # Initialize the PeaNetSeq
    PNS = PeaNetSeq(rng=rng, pea_net=PN, seq_len=2, seq_Xd=None, params=None)
    PNS1 = PNS.shared_param_clone(rng=rng, seq_len=2, seq_Xd=None)

    # set weighting parameters for the various costs...
    PNS.set_lam_class(1.0)
    PNS.set_lam_pea_su(0.0)
    PNS.set_lam_pea_un(2.0)
    PNS.set_lam_ent(0.0)
    PNS.set_lam_l2w(1e-5)

    out_file = open("PNS_TEST_ADAM.txt", 'wb')
    learn_rate = 0.1
    PNS.set_pn_sgd_params(lr_pn=learn_rate, mom_1=0.9, mom_2=0.999)
    for i in range(300000):
        if i < 5000:
            scale = float(i + 1) / 5000.0
        if ((i+1 % 100000) == 0):
            learn_rate = learn_rate * 0.5
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
        outputs = PNS1.train_joint(Xd_batch, Xd_batch, Yd_batch, Yd_batch)
        joint_cost = 1.0 * outputs[0]
        class_cost = 1.0 * outputs[1]
        pea_cost = 1.0 * outputs[2]
        ent_cost = 1.0 * outputs[3]
        other_reg_cost = 1.0 * outputs[4]
        assert(not (np.isnan(joint_cost)))
        if ((i % 1000) == 0):
            o_str = "batch: {0:d}, joint: {1:.4f}, class: {2:.4f}, pea: {3:.4f}, ent: {4:.4f}, other_reg: {5:.4f}".format( \
                    i, joint_cost, class_cost, pea_cost, ent_cost, other_reg_cost)
            print(o_str)
            out_file.write(o_str+"\n")
            out_file.flush()
            # check classification error on training and validation set
            train_err = PNS.classification_error(Xtr_su, Ytr_su)
            va_err = PNS.classification_error(Xva, Yva)
            o_str = "    tr_err: {0:.4f}, va_err: {1:.4f}".format(train_err, va_err)
            print(o_str)
            out_file.write(o_str+"\n")
            out_file.flush()
        if ((i % 1000) == 0):
            # draw the main PeaNet's first-layer filters/weights
            file_name = "PNS_WEIGHTS_ZMUV.png".format(i)
            utils.visualize_net_layer(PNS.PN.proto_nets[0][0], file_name)
    print("TESTING COMPLETE!")




##############
# EYE BUFFER #
##############
