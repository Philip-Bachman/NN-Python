###############################################################################
# Code for managing and training a generator/inferencer pair that seeks to    #
# generate examples that maximize the joint cost for a PeaNetSeq.             #
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
from NetLayers import HiddenLayer, DiscLayer, relu_actfun, softplus_actfun, \
                      safe_log
from GenNet import GenNet
from InfNet import InfNet
from PeaNet import PeaNet
from PeaNetSeq import PeaNetSeq
from DKCode import get_adam_updates, get_adadelta_updates

#
#
# Important symbolic variables:
#   Xd: Xd represents input to which to apply perturbations
#
#

def max_normalize(x, axis=1):
    """
    Normalize matrix x to max unit (L2) norm along some axis.
    """
    row_norms = T.sqrt(T.sum(x**2., axis=axis, keepdims=1))
    row_scales = T.maximum(1.0, row_norms)
    x_normed = x / row_scales
    return x_normed


class AEDPair(object):
    """
    Controller for training an adversarial example generator.

    The generator must be an instance of the GenNet class implemented in
    "GenNet.py". The inferencer must be an instance of the InfNet class
    implemented in "InfNet.py".

    For now, this will learn to conditionaly generate perturbations of the
    input to a PeaNetSeq so as to maximally increase its "joint_cost".

    Parameters:
        rng: numpy.random.RandomState (for reproducibility)
        Xd: symbolic "data" input to be perturbed by the adversarial policy
        g_net: The GenNet instance that will serve as the base generator
        i_net: The InfNet instance that will serve as the base inferencer
        pn_seq: The PeaNetSeq whose cost the perturbations should maximize
        data_dim: dimension of the "observable data" variables
        prior_dim: dimension of the "latent prior" variables
        params: dict for passing additional parameters
            match_type: "grad_sign" matches the sign of the PEA gradient and
                        "grad_dir" matches the direction of the PEA gradient
    """
    def __init__(self, rng=None, Xd=None, \
            g_net=None, i_net=None, pn_seq=None, \
            data_dim=None, prior_dim=None, \
            params=None):
        # setup a rng for this AEDPair
        self.rng = RandStream(rng.randint(100000))

        if (params is None):
            self.params = {}
        else:
            self.params = params
        if 'match_type' in params:
            self.match_type = params['match_type']
        else:
            self.match_type = 'grad_sign'
        # we can only try to match sign or direction...
        assert((self.match_type == 'grad_dir') or \
                (self.match_type == 'grad_sign'))
        if self.match_type == 'grad_dir':
            # we match the direction of the gradient under the assumption
            # of gaussian observation noise
            self.mean_transform = lambda x: max_normalize(x, axis=1)
            assert(g_net.out_type == 'gaussian')
        else:
            # we match the sign of the gradient as if it were a collection
            # of independent binary variables
            self.mean_transform = lambda x: 2.0 * (x - 0.5)
            assert(g_net.out_type == 'bernoulli')

        # record the symbolic variables that will provide inputs to the
        # computation graph created to describe this AEDPair
        self.Xd = Xd
        self.Yd = T.icol('adp_Yd') # labels to pass to the PeaNetSeq
        self.Xc = 0.0 * self.Xd
        self.Xm = 0.0 * self.Xd
        self.obs_count = T.cast(Xd.shape[0], 'floatX')

        # create a "shared-parameter" clone of the inferencer, set up to
        # receive input from the appropriate symbolic variables.
        self.IN = i_net.shared_param_clone(rng=rng, \
                Xd=self.Xd, Xc=self.Xc, Xm=self.Xm)
        self.policy_mean = self.IN.output_mean
        self.policy_logvar = self.IN.output_logvar
        # capture a handle for samples from the variational posterior
        self.Xp = self.IN.output
        # create a "shared-parameter" clone of the generator, set up to
        # receive input from samples from the variational posterior
        self.GN = g_net.shared_param_clone(rng=rng, Xp=self.IN.output)
        # set up a var for controlling the max-norm bound on perturbations
        zero_ary = np.zeros((1,)).astype(theano.config.floatX)
        self.lam_mnb = theano.shared(value=zero_ary, \
                name='adp_lam_mnb')
        self.set_lam_mnb(lam_mnb=0.1)

        # get the perturbations output by the generator network
        self.Pg = self.mean_transform(self.GN.output)
        self.Pg_samples = self.mean_transform(self.GN.output_samples)

        # record and validate the data dimensionality parameters
        self.data_dim = data_dim
        self.prior_dim = prior_dim
        # output of the generator and input to the inferencer should both be
        # equal to self.data_dim
        assert(self.data_dim == self.GN.mlp_layers[-1].out_dim)
        assert(self.data_dim == self.IN.shared_layers[0].in_dim)
        # input of the generator and mu/sigma outputs of the inferencer should
        # both be equal to self.prior_dim
        assert(self.prior_dim == self.GN.mlp_layers[0].in_dim)
        assert(self.prior_dim == self.IN.mu_layers[-1].out_dim)
        assert(self.prior_dim == self.IN.sigma_layers[-1].out_dim)

        # make a clone of the target PeaNetSeq that takes perturbed inputs
        self.PNS = pn_seq.shared_param_clone(rng=rng, seq_len=2, \
                seq_Xd=[self.Xd, self.Xd], seq_Yd=[self.Yd, self.Yd], \
                no_funcs=True)
        self.grad_pea_Xd = T.grad(self.PNS.joint_cost, self.Xd)
        if self.match_type == 'grad_dir':
            # turn gradient into a unit max-normalized vector
            self.match_target = max_normalize(self.grad_pea_Xd)
        else:
            # transform gradient into binary indicators of sign
            self.match_target = (self.grad_pea_Xd > 0.0)
        # get the symbolic vars for passing inputs to self.PNS
        self.Xd_seq = self.PNS.Xd_seq
        self.Yd_seq = self.PNS.Yd_seq
        self.seq_inputs = self.Xd_seq + self.Yd_seq

        # shared var learning rate for generator and inferencer
        self.lr_gn = theano.shared(value=zero_ary, name='adp_lr_gn')
        self.lr_in = theano.shared(value=zero_ary, name='adp_lr_in')
        # shared var momentum parameters for generator and inferencer
        self.mom_1 = theano.shared(value=zero_ary, name='adp_mom_1')
        self.mom_2 = theano.shared(value=zero_ary, name='adp_mom_2')
        self.it_count = theano.shared(value=zero_ary, name='adp_it_count')
        # init parameters for controlling learning dynamics
        self.set_all_sgd_params()
        # init shared var for weighting nll of data given posterior sample
        self.lam_adv = theano.shared(value=zero_ary, name='adp_lam_adv')
        self.set_lam_adv(lam_adv=1.0)
        # init shared vars for weighting a penalty on the norms of our learned
        # policies and a reward to encourage maximizing their entropy.
        self.lam_kld = theano.shared(value=zero_ary, name='adp_lam_kld')
        self.set_lam_kld(lam_kld=0.1)
        # init shared var for controlling l2 regularization on params
        self.lam_l2w = theano.shared(value=zero_ary, name='adp_lam_l2w')
        self.set_lam_l2w(1e-4)

        # Grab the full set of "optimizable" parameters from the generator
        # and inferencer networks that we'll be working with.
        self.in_params = [p for p in self.IN.mlp_params]
        self.gn_params = [p for p in self.GN.mlp_params]
        self.joint_params = self.in_params + self.gn_params

        ###################################
        # CONSTRUCT THE COSTS TO OPTIMIZE #
        ###################################
        self.adv_cost = self.lam_adv[0] * self._construct_adv_cost()
        self.kld_cost = self.lam_kld[0] * self._construct_kld_cost()
        self.other_reg_cost = self._construct_other_reg_cost()
        self.joint_cost = self.adv_cost + self.kld_cost + \
                self.other_reg_cost

        # Get the gradient of the joint cost for all optimizable parameters
        self.joint_grads = OrderedDict()
        for p in self.joint_params:
            self.joint_grads[p] = T.grad(self.joint_cost, p).clip(-0.1, 0.1)

        # Construct the updates for the generator and inferencer networks
        self.gn_updates = get_adam_updates(params=self.gn_params, \
                grads=self.joint_grads, alpha=self.lr_gn, \
                beta1=self.mom_1, beta2=self.mom_2, it_count=self.it_count, \
                mom2_init=1e-3, smoothing=1e-8)
        self.in_updates = get_adam_updates(params=self.in_params, \
                grads=self.joint_grads, alpha=self.lr_in, \
                beta1=self.mom_1, beta2=self.mom_2, it_count=self.it_count, \
                mom2_init=1e-3, smoothing=1e-8)
        self.joint_updates = OrderedDict()
        for k in self.gn_updates:
            self.joint_updates[k] = self.gn_updates[k]
        for k in self.in_updates:
            self.joint_updates[k] = self.in_updates[k]

        # Construct a function for jointly training the generator/inferencer
        self.train_joint = self._construct_train_joint()

        # Construct a function for computing the outputs of the generator
        # network for a batch of noise. Presumably, the noise will be drawn
        # from the same distribution that was used in training....
        self.sample_from_gn = self.GN.sample_from_model
        self.sample_from_Xd = self._construct_sample_from_Xd()
        return

    def set_all_sgd_params(self, lr_gn=0.01, lr_in=0.01, \
                mom_1=0.9, mom_2=0.999):
        """
        Set learning rate and momentum parameter for all updates.
        """
        zero_ary = np.zeros((1,))
        # set learning rates
        new_lr_gn = zero_ary + lr_gn
        self.lr_gn.set_value(new_lr_gn.astype(theano.config.floatX))
        new_lr_in = zero_ary + lr_in
        self.lr_in.set_value(new_lr_in.astype(theano.config.floatX))
        # set momentums
        new_mom_1 = zero_ary + mom_1
        self.mom_1.set_value(new_mom_1.astype(theano.config.floatX))
        new_mom_2 = zero_ary + mom_2
        self.mom_2.set_value(new_mom_2.astype(theano.config.floatX))
        return

    def set_lam_mnb(self, lam_mnb=0.1):
        """
        Set a bound on the max norm of the perturbations we will generate.
        """
        zero_ary = np.zeros((1,))
        new_mnb = zero_ary + lam_mnb
        self.lam_mnb.set_value(new_mnb.astype(theano.config.floatX))
        return

    def set_lam_adv(self, lam_adv=0.1):
        """
        Set the relative weight of the adversarial reward function.
        """
        zero_ary = np.zeros((1,))
        new_adv = zero_ary + lam_adv
        self.lam_adv.set_value(new_adv.astype(theano.config.floatX))
        return

    def set_lam_kld(self, lam_kld=1.0):
        """
        Set the weight of cost for KL-divergence from the prior policy.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_kld
        self.lam_kld.set_value(new_lam.astype(theano.config.floatX))
        return

    def set_lam_l2w(self, lam_l2w=1e-3):
        """
        Set the relative strength of l2 regularization on network params.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_l2w
        self.lam_l2w.set_value(new_lam.astype(theano.config.floatX))
        return

    def _construct_adv_cost(self):
        """
        Construct the adversarial cost to minimize. Minimizing this cost
        should be roughly like maximizing the PeaNetSeq's cost.
        """
        match_cost = self.GN.compute_log_prob(Xd=self.match_target)
        adv_cost = -T.sum(match_cost) / self.obs_count
        return adv_cost

    def _construct_kld_cost(self):
        """
        Construct the penalty derived from the Gaussian prior over policies.
        """
        kld_cost = T.sum(self.IN.kld_cost) / self.obs_count
        return kld_cost

    def _construct_other_reg_cost(self):
        """
        Construct the cost for low-level basic regularization. E.g. for
        applying l2 regularization to the network activations and parameters.
        """
        act_reg_cost = (self.IN.act_reg_cost + self.GN.act_reg_cost)
        gp_cost = sum([T.sum(par**2.0) for par in self.gn_params])
        ip_cost = sum([T.sum(par**2.0) for par in self.in_params])
        param_reg_cost = self.lam_l2w[0] * (gp_cost + ip_cost)
        other_reg_cost = (act_reg_cost / self.obs_count) + param_reg_cost
        return other_reg_cost

    def _construct_train_joint(self):
        """
        Construct theano function to train inferencer and generator jointly.
        """
        outputs = [self.joint_cost, self.adv_cost, self.kld_cost, \
                self.other_reg_cost]
        func = theano.function(inputs=[self.Xd, self.Yd], \
                outputs=outputs, \
                updates=self.joint_updates)
        return func

    def _construct_sample_from_Xd(self):
        """
        Construct a theano function for sampling adversarial perturbations
        conditioned on some set of inputs.
        """
        bounded_perturbation = self.lam_mnb[0] * self.Pg
        samp_func = theano.function([self.Xd], outputs=bounded_perturbation)
        return samp_func


if __name__=="__main__":
    import utils as utils
    from load_data import load_udm, load_udm_ss, load_mnist
    from NetLayers import relu_actfun
    import PeaNet as PNet
    import GenNet as GNet
    import InfNet as INet

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
    # get size information for the data
    un_samples = Xtr_un.shape[0]
    su_samples = Xtr_su.shape[0]
    va_samples = Xva.shape[0]

    # set up some symbolic variables for input to the PeaNetSeq
    Xp = T.matrix('Xp_base')
    Xd = T.matrix('Xd_base')
    Xc = T.matrix('Xc_base')
    Xm = T.matrix('Xm_base')
    Yd = T.icol('Yd_base')
    # set some "shape" parameters for the networks
    data_dim = Xtr_un.shape[1]
    label_dim = 10
    prior_dim = 25
    prior_sigma = 1.0
    batch_size = 100 # we'll take 2x this per batch, for sup and unsup

    ####################################
    # Setup parameters for the AEDPair #
    ####################################
    aedp_params = {}
    #aedp_params['match_type'] = 'grad_dir'
    aedp_params['match_type'] = 'grad_sign'

    #################################################################
    # Construct the generator and inferencer to use for conditional #
    # generation of adversarial examples.                           #
    #################################################################
    # Choose some parameters for the generator network
    gn_params = {}
    gn_config = [prior_dim, 800, 800, data_dim]
    gn_params['mlp_config'] = gn_config
    gn_params['activation'] = relu_actfun
    if aedp_params['match_type'] == 'grad_dir':
        gn_params['out_type'] = 'gaussian'
        gn_params['mean_transform'] = lambda x: max_normalize(x, axis=1)
    if aedp_params['match_type'] == 'grad_sign':
        gn_params['out_type'] = 'bernoulli'
    gn_params['init_scale'] = 1.3
    gn_params['lam_l2a'] = 1e-3
    gn_params['vis_drop'] = 0.0
    gn_params['hid_drop'] = 0.0
    gn_params['bias_noise'] = 0.0
    # choose some parameters for the continuous inferencer
    in_params = {}
    shared_config = [data_dim, 800, 800]
    top_config = [shared_config[-1], prior_dim]
    in_params['shared_config'] = shared_config
    in_params['mu_config'] = top_config
    in_params['sigma_config'] = top_config
    in_params['activation'] = relu_actfun
    in_params['init_scale'] = 1.3
    in_params['lam_l2a'] = 1e-3
    in_params['vis_drop'] = 0.0
    in_params['hid_drop'] = 0.0
    in_params['bias_noise'] = 0.0
    in_params['input_noise'] = 0.0
    # Initialize the base networks for this GIPair
    IN = InfNet(rng=rng, Xd=Xd, Xc=Xc, Xm=Xm, prior_sigma=prior_sigma, \
            params=in_params, shared_param_dicts=None)
    GN = GenNet(rng=rng, Xp=Xp, prior_sigma=prior_sigma, \
            params=gn_params, shared_param_dicts=None)
    # Initialize biases in IN and GN
    IN.init_biases(0.1)
    GN.init_biases(0.1)


    ##################################################
    # Initialize and train a PeaNetSeq to antagonize #
    ##################################################
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

    LOAD_FROM_FILE = True
    PN_PARAM_FILE = 'AEDP_TEST_PN_PARAMS.pkl'
    # Initialize the base network for this PNSeq
    if LOAD_FROM_FILE:
        PN = PNet.load_peanet_from_file(f_name=PN_PARAM_FILE, \
                rng=rng, Xd=Xd)
    else:
        PN = PeaNet(rng=rng, Xd=Xd, params=pn_params)
        PN.init_biases(0.1)

    # Initialize the PeaNetSeq
    PNS = PeaNetSeq(rng=rng, pea_net=PN, seq_len=2, seq_Xd=None, \
            no_noise=False, params=None)

    # set weighting parameters for the various costs...
    PNS.set_lam_class(1.0)
    PNS.set_lam_pea_su(0.0)
    PNS.set_lam_pea_un(2.0)
    PNS.set_lam_ent(0.0)
    PNS.set_lam_l2w(1e-5)

    # initialize an AEDPair that antagonizes PNS
    print("Initializing adversarial generator...")
    AEDP = AEDPair(rng=rng, Xd=Xd, g_net=GN, i_net=IN, pn_seq=PNS, \
            data_dim=data_dim, prior_dim=prior_dim, params=aedp_params)
    AEDP.set_lam_mnb(1.0)
    AEDP.set_lam_l2w(1e-4)

    if not LOAD_FROM_FILE:
        # train the PeaNetSeq for some number of updates
        learn_rate = 0.1
        PNS.set_pn_sgd_params(lr_pn=learn_rate, mom_1=0.9, mom_2=0.999)
        print("Learning  weights....")
        for i in range(10001):
            if i < 5000:
                scale = float(i + 1) / 5000.0
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
                # check classification error on training and validation set
                train_err = PNS.classification_error(Xtr_su, Ytr_su)
                va_err = PNS.classification_error(Xva, Yva)
                o_str = "    tr_err: {0:.4f}, va_err: {1:.4f}".format(train_err, va_err)
                print(o_str)
            if ((i % 1000) == 0):
                # draw the main PeaNet's first-layer filters/weights
                file_name = "AEDP_TEST_PNS_WEIGHTS.png".format(i)
                utils.visualize_net_layer(PNS.PN.proto_nets[0][0], file_name)
        PN.save_to_file(f_name=PN_PARAM_FILE)

    # train the adversarial generator for number of iterations
    learn_rate = 0.0003
    AEDP.set_all_sgd_params(lr_gn=learn_rate, lr_in=learn_rate, \
            mom_1=0.9, mom_2=0.999)
    mean_costs = [0. for i in range(10)]
    print("TRAINING ADVERSARIAL GENERATOR....")
    for i in range(100000):
        if i < 5000:
            scale = float(i + 1) / 5000.0
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
        AEDP.set_all_sgd_params(lr_gn=learn_rate, lr_in=learn_rate, \
                mom_1=0.9, mom_2=0.999)
        AEDP.set_lam_adv(lam_adv=1.0)
        AEDP.set_lam_kld(lam_kld=0.05 + 0.95*scale)
        # do a minibatch update of all AEDPair parameters
        outputs = AEDP.train_joint(Xd_batch, Yd_batch)
        mean_costs = [(mean_costs[k] + 1.*outputs[k]) for k in range(len(outputs))]
        if ((i % 1000) == 0):
            mean_costs = [(v / 1000.) for v in mean_costs]
            o_str = "batch: {0:d}, joint: {1:.4f}, adv: {2:.4f}, kld: {3:.4f}, other_reg: {4:.4f}".format( \
                    i, mean_costs[0], mean_costs[1], mean_costs[2], mean_costs[3])
            print(o_str)
            mean_costs = [0. for v in mean_costs]
        if ((i % 1000) == 0):
            tr_idx = npr.randint(low=0,high=un_samples,size=(20,))
            Xd_batch = Xtr_un.take(tr_idx, axis=0)
            file_name = "AEDP_ADVERSARIAL_SAMPLES_b{0:d}.png".format(i)
            sample_lists = [Xd_batch]
            for j in range(9):
                sample_lists.append(AEDP.sample_from_Xd(Xd_batch))
            Xs = np.vstack(sample_lists)
            utils.mat_to_img(Xs, file_name, (28,28), num_rows=10, \
                    scale=True, colorImg=False, tile_spacing=(1,1))
    print("TESTING COMPLETE!")




##############
# EYE BUFFER #
##############
