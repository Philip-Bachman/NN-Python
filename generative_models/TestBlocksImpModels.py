##################################################################
# Code for testing the variational Multi-Stage Generative Model. #
##################################################################

from __future__ import print_function, division

# basic python
import cPickle as pickle
from PIL import Image
import numpy as np
import numpy.random as npr
from collections import OrderedDict

# theano business
import theano
import theano.tensor as T

# blocks stuff
from blocks.initialization import Constant, IsotropicGaussian, Orthogonal
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.roles import PARAMETER
from blocks.model import Model
from blocks.bricks import Tanh, Identity, Rectifier
from blocks.bricks.cost import BinaryCrossEntropy
from blocks.bricks.recurrent import SimpleRecurrent, LSTM

# phil's sweetness
import utils
from BlocksModels import *
from DKCode import get_adam_updates, get_adadelta_updates
from load_data import load_udm, load_tfd, load_mnist, load_svhn_gray
from HelperFuncs import construct_masked_data, shift_and_scale_into_01, \
                        row_shuffle, to_fX

###################################
###################################
## HELPER FUNCTIONS FOR SAMPLING ##
###################################
###################################

def scale_norm(arr):
    arr = arr - arr.min()
    scale = (arr.max() - arr.min())
    return scale * arr

def img_grid(arr, global_scale=True):
    N, height, width = arr.shape

    rows = int(np.sqrt(N))
    cols = int(np.sqrt(N))

    if rows*cols < N:
        cols = cols + 1

    if rows*cols < N:
        rows = rows + 1

    total_height = rows * height
    total_width  = cols * width

    if global_scale:
        arr = scale_norm(arr)

    I = np.zeros((total_height, total_width))

    for i in xrange(N):
        r = i // cols
        c = i % cols

        if global_scale:
            this = arr[i]
        else:
            this = scale_norm(arr[i])

        offset_y, offset_x = r*height, c*width
        I[offset_y:(offset_y+height), offset_x:(offset_x+width)] = this
    
    I = (255*I).astype(np.uint8)
    return Image.fromarray(I)

###########################
###########################
## TEST MNIST IMPUTATION ##
###########################
###########################

def test_imocld_imp_mnist(step_type='add', occ_dim=14, drop_prob=0.0, attention=False):
    ##########################
    # Get some training data #
    ##########################
    rng = np.random.RandomState(1234)
    dataset = 'data/mnist.pkl.gz'
    datasets = load_udm(dataset, as_shared=False, zero_mean=False)
    Xtr = datasets[0][0]
    Xva = datasets[1][0]
    Xtr = to_fX(shift_and_scale_into_01(Xtr))
    Xva = to_fX(shift_and_scale_into_01(Xva))
    tr_samples = Xtr.shape[0]
    va_samples = Xva.shape[0]
    batch_size = 250

    ############################################################
    # Setup some parameters for the Iterative Refinement Model #
    ############################################################
    x_dim = Xtr.shape[1]
    write_dim = 300
    enc_dim = 300
    dec_dim = 300
    mix_dim = 20
    z_dim = 100
    n_iter = 16
    dp_int = int(100.0 * drop_prob)
    
    rnninits = {
        'weights_init': IsotropicGaussian(0.01),
        'biases_init': Constant(0.),
    }
    inits = {
        'weights_init': IsotropicGaussian(0.01),
        'biases_init': Constant(0.),
    }

    att_tag = "NA" # attention not implemented yet

    # setup the reader and writer (shared by primary and guide policies)
    read_dim = 2*x_dim # dimension of output from reader_mlp
    reader_mlp = Reader(x_dim=x_dim, dec_dim=dec_dim, **inits)
    writer_mlp = MLP([None, None], [dec_dim, write_dim, x_dim], \
                     name="writer_mlp", **inits)
    
    # mlps for setting conditionals over z_mix
    mix_var_mlp = CondNet([Tanh()], [x_dim, 250, mix_dim], \
                          name="mix_var_mlp", **inits)
    mix_enc_mlp = CondNet([Tanh()], [x_dim, 250, mix_dim], \
                          name="mix_enc_mlp", **inits)
    # mlp for decoding z_mix into a distribution over initial LSTM states
    mix_dec_mlp = MLP([Tanh(), Tanh()], \
                      [mix_dim, 250, (2*enc_dim + 2*dec_dim + 2*enc_dim)], \
                      name="mix_dec_mlp", **inits)
    # mlps for processing inputs to LSTMs
    var_mlp_in = MLP([Identity()], [(read_dim + dec_dim), 4*enc_dim], \
                     name="var_mlp_in", **inits)
    enc_mlp_in = MLP([Identity()], [(read_dim + dec_dim), 4*enc_dim], \
                     name="enc_mlp_in", **inits)
    dec_mlp_in = MLP([Identity()], [               z_dim, 4*dec_dim], \
                     name="dec_mlp_in", **inits)
    # mlps for turning LSTM outputs into conditionals over z_gen
    var_mlp_out = CondNet([], [enc_dim, z_dim], name="var_mlp_out", **inits)
    enc_mlp_out = CondNet([], [enc_dim, z_dim], name="enc_mlp_out", **inits)
    # LSTMs for the actual LSTMs (obviously, perhaps)
    var_rnn = BiasedLSTM(dim=enc_dim, ig_bias=2.0, fg_bias=2.0, \
                         name="var_rnn", **rnninits)
    enc_rnn = BiasedLSTM(dim=enc_dim, ig_bias=2.0, fg_bias=2.0, \
                         name="enc_rnn", **rnninits)
    dec_rnn = BiasedLSTM(dim=dec_dim, ig_bias=2.0, fg_bias=2.0, \
                         name="dec_rnn", **rnninits)

    draw = IMoCLDrawModels(
                n_iter,
                step_type=step_type, # step_type can be 'add' or 'jump'
                reader_mlp=reader_mlp,
                writer_mlp=writer_mlp,
                mix_enc_mlp=mix_enc_mlp,
                mix_dec_mlp=mix_dec_mlp,
                mix_var_mlp=mix_var_mlp,
                enc_mlp_in=enc_mlp_in,
                enc_mlp_out=enc_mlp_out,
                enc_rnn=enc_rnn,
                dec_mlp_in=dec_mlp_in,
                dec_rnn=dec_rnn,
                var_mlp_in=var_mlp_in,
                var_mlp_out=var_mlp_out,
                var_rnn=var_rnn)
    draw.initialize()

    # build the cost gradients, training function, samplers, etc.
    draw.build_model_funcs()

    #draw.load_model_params(f_name="TBCLM_IMP_PARAMS_OD{}_DP{}_{}_{}.pkl".format(occ_dim, dp_int, step_type, att_tag))

    ################################################################
    # Apply some updates, to check that they aren't totally broken #
    ################################################################
    print("Beginning to train the model...")
    out_file = open("TBCLM_IMP_RESULTS_OD{}_DP{}_{}_{}.txt".format(occ_dim, dp_int, step_type, att_tag), 'wb')
    out_file.flush()
    costs = [0. for i in range(10)]
    learn_rate = 0.0002
    momentum = 0.5
    batch_idx = np.arange(batch_size) + tr_samples
    for i in range(250000):
        scale = min(1.0, ((i+1) / 1000.0))
        if (((i + 1) % 10000) == 0):
            learn_rate = learn_rate * 0.95
        if (i > 10000):
            momentum = 0.90
        else:
            momentum = 0.50
        # get the indices of training samples for this batch update
        batch_idx += batch_size
        if (np.max(batch_idx) >= tr_samples):
            # we finished an "epoch", so we rejumble the training set
            Xtr = row_shuffle(Xtr)
            batch_idx = np.arange(batch_size)
        # set sgd and objective function hyperparams for this update
        zero_ary = np.zeros((1,))
        draw.lr.set_value(to_fX(zero_ary + learn_rate))
        draw.mom_1.set_value(to_fX(zero_ary + momentum))
        draw.mom_2.set_value(to_fX(zero_ary + 0.99))

        # perform a minibatch update and record the cost for this batch
        Xb = to_fX(Xtr.take(batch_idx, axis=0))
        _, Xb, Mb = construct_masked_data(Xb, drop_prob=drop_prob, \
                                    occ_dim=occ_dim, data_mean=None)
        result = draw.train_joint(Xb, Mb)

        costs = [(costs[j] + result[j]) for j in range(len(result))]
        if ((i % 200) == 0):
            costs = [(v / 200.0) for v in costs]
            str1 = "-- batch {0:d} --".format(i)
            str2 = "    total_cost: {0:.4f}".format(costs[0])
            str3 = "    nll_bound : {0:.4f}".format(costs[1])
            str4 = "    nll_term  : {0:.4f}".format(costs[2])
            str5 = "    kld_q2p   : {0:.4f}".format(costs[3])
            str6 = "    kld_p2q   : {0:.4f}".format(costs[4])
            str7 = "    reg_term  : {0:.4f}".format(costs[5])
            joint_str = "\n".join([str1, str2, str3, str4, str5, str6, str7])
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()
            costs = [0.0 for v in costs]
        if ((i % 1000) == 0):
            draw.save_model_params("TBCLM_IMP_PARAMS_OD{}_DP{}_{}_{}.pkl".format(occ_dim, dp_int, step_type, att_tag))
            # compute a small-sample estimate of NLL bound on validation set
            Xva = row_shuffle(Xva)
            Xb = to_fX(Xva[:5000])
            _, Xb, Mb = construct_masked_data(Xb, drop_prob=drop_prob, \
                                    occ_dim=occ_dim, data_mean=None)
            va_costs = draw.compute_nll_bound(Xb, Mb)
            str1 = "    va_nll_bound : {}".format(va_costs[1])
            str2 = "    va_nll_term  : {}".format(va_costs[2])
            str3 = "    va_kld_q2p   : {}".format(va_costs[3])
            joint_str = "\n".join([str1, str2, str3])
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()
            # # draw some independent samples from the model
            # Xb = to_fX(Xva[:256])
            # _, Xb, Mb = construct_masked_data(Xb, drop_prob=drop_prob, \
            #                         occ_dim=occ_dim, data_mean=None)
            # samples = draw.do_sample(Xb, Mb)
            # n_iter, N, D = samples.shape
            # samples = samples.reshape( (n_iter, N, 28, 28) )
            # for j in xrange(n_iter):
            #     img = img_grid(samples[j,:,:,:])
            #     img.save("TBCLM-IMP-samples-%03d.png" % (j,))

##########################
##########################
## TEST SVHN IMPUTATION ##
##########################
##########################

def test_imocld_imp_svhn(step_type='add', occ_dim=14, drop_prob=0.0, attention=False):
    ##########################
    # Get some training data #
    ##########################
    tr_file = 'data/svhn_train_gray.pkl'
    te_file = 'data/svhn_test_gray.pkl'
    ex_file = 'data/svhn_extra_gray.pkl'
    data = load_svhn_gray(tr_file, te_file, ex_file=ex_file, ex_count=200000)
    Xtr = to_fX( shift_and_scale_into_01(np.vstack([data['Xtr'], data['Xex']])) )
    Xva = to_fX( shift_and_scale_into_01(data['Xte']) )
    tr_samples = Xtr.shape[0]
    va_samples = Xva.shape[0]
    batch_size = 250
    all_pix_mean = np.mean(np.mean(Xtr, axis=1))
    data_mean = to_fX( all_pix_mean * np.ones((Xtr.shape[1],)) )

    ############################################################
    # Setup some parameters for the Iterative Refinement Model #
    ############################################################
    x_dim = Xtr.shape[1]
    write_dim = 600
    enc_dim = 600
    dec_dim = 600
    mix_dim = 20
    z_dim = 200
    n_iter = 25
    dp_int = int(100.0 * drop_prob)
    
    rnninits = {
        'weights_init': IsotropicGaussian(0.01),
        'biases_init': Constant(0.),
    }
    inits = {
        'weights_init': IsotropicGaussian(0.01),
        'biases_init': Constant(0.),
    }

    att_tag = "NA" # attention not implemented yet

    # setup the reader and writer (shared by primary and guide policies)
    read_dim = 2*x_dim # dimension of output from reader_mlp
    reader_mlp = Reader(x_dim=x_dim, dec_dim=dec_dim, **inits)
    writer_mlp = MLP([None, None], [dec_dim, write_dim, x_dim], \
                     name="writer_mlp", **inits)
    
    # mlps for setting conditionals over z_mix
    mix_var_mlp = CondNet([Tanh()], [x_dim, 250, mix_dim], \
                          name="mix_var_mlp", **inits)
    mix_enc_mlp = CondNet([Tanh()], [x_dim, 250, mix_dim], \
                          name="mix_enc_mlp", **inits)
    # mlp for decoding z_mix into a distribution over initial LSTM states
    mix_dec_mlp = MLP([Tanh(), Tanh()], \
                      [mix_dim, 250, (2*enc_dim + 2*dec_dim + 2*enc_dim)], \
                      name="mix_dec_mlp", **inits)
    # mlps for processing inputs to LSTMs
    var_mlp_in = MLP([Identity()], [(read_dim + dec_dim), 4*enc_dim], \
                     name="var_mlp_in", **inits)
    enc_mlp_in = MLP([Identity()], [(read_dim + dec_dim), 4*enc_dim], \
                     name="enc_mlp_in", **inits)
    dec_mlp_in = MLP([Identity()], [               z_dim, 4*dec_dim], \
                     name="dec_mlp_in", **inits)
    #dec_mlp_in = MLP([Identity()], [   (enc_dim + z_dim), 4*dec_dim], \
    #                 name="dec_mlp_in", **inits)
    # mlps for turning LSTM outputs into conditionals over z_gen
    var_mlp_out = CondNet([], [enc_dim, z_dim], name="var_mlp_out", **inits)
    enc_mlp_out = CondNet([], [enc_dim, z_dim], name="enc_mlp_out", **inits)
    # LSTMs for the actual LSTMs (obviously, perhaps)
    var_rnn = BiasedLSTM(dim=enc_dim, ig_bias=2.0, fg_bias=2.0, \
                         name="var_rnn", **rnninits)
    enc_rnn = BiasedLSTM(dim=enc_dim, ig_bias=2.0, fg_bias=2.0, \
                         name="enc_rnn", **rnninits)
    dec_rnn = BiasedLSTM(dim=dec_dim, ig_bias=2.0, fg_bias=2.0, \
                         name="dec_rnn", **rnninits)

    draw = IMoCLDrawModels(
                n_iter,
                step_type=step_type, # step_type can be 'add' or 'jump'
                reader_mlp=reader_mlp,
                writer_mlp=writer_mlp,
                mix_enc_mlp=mix_enc_mlp,
                mix_dec_mlp=mix_dec_mlp,
                mix_var_mlp=mix_var_mlp,
                enc_mlp_in=enc_mlp_in,
                enc_mlp_out=enc_mlp_out,
                enc_rnn=enc_rnn,
                dec_mlp_in=dec_mlp_in,
                dec_rnn=dec_rnn,
                var_mlp_in=var_mlp_in,
                var_mlp_out=var_mlp_out,
                var_rnn=var_rnn)
    draw.initialize()

    # build the cost gradients, training function, samplers, etc.
    draw.build_model_funcs()

    draw.load_model_params(f_name="TBCLM_IMP_SVHN_PARAMS_OD{}_DP{}_{}_{}.pkl".format(occ_dim, dp_int, step_type, att_tag))

    ################################################################
    # Apply some updates, to check that they aren't totally broken #
    ################################################################
    print("Beginning to train the model...")
    out_file = open("TBCLM_IMP_SVHN_RESULTS_OD{}_DP{}_{}_{}.txt".format(occ_dim, dp_int, step_type, att_tag), 'wb')
    out_file.flush()
    costs = [0. for i in range(10)]
    learn_rate = 0.0002
    momentum = 0.5
    batch_idx = np.arange(batch_size) + tr_samples
    for i in range(250000):
        scale = min(1.0, ((i+1) / 1000.0))
        if (((i + 1) % 10000) == 0):
            learn_rate = learn_rate * 0.95
        if (i > 10000):
            momentum = 0.90
        else:
            momentum = 0.50
        # get the indices of training samples for this batch update
        batch_idx += batch_size
        if (np.max(batch_idx) >= tr_samples):
            # we finished an "epoch", so we rejumble the training set
            Xtr = row_shuffle(Xtr)
            batch_idx = np.arange(batch_size)
        # set sgd and objective function hyperparams for this update
        zero_ary = np.zeros((1,))
        draw.lr.set_value(to_fX(zero_ary + learn_rate))
        draw.mom_1.set_value(to_fX(zero_ary + momentum))
        draw.mom_2.set_value(to_fX(zero_ary + 0.99))

        # perform a minibatch update and record the cost for this batch
        Xb = to_fX(Xtr.take(batch_idx, axis=0))
        _, Xb, Mb = construct_masked_data(Xb, drop_prob=drop_prob, \
                                    occ_dim=occ_dim, data_mean=None)
        result = draw.train_joint(Xb, Mb)

        costs = [(costs[j] + result[j]) for j in range(len(result))]
        if ((i % 200) == 0):
            costs = [(v / 200.0) for v in costs]
            str1 = "-- batch {0:d} --".format(i)
            str2 = "    total_cost: {0:.4f}".format(costs[0])
            str3 = "    nll_bound : {0:.4f}".format(costs[1])
            str4 = "    nll_term  : {0:.4f}".format(costs[2])
            str5 = "    kld_q2p   : {0:.4f}".format(costs[3])
            str6 = "    kld_p2q   : {0:.4f}".format(costs[4])
            str7 = "    reg_term  : {0:.4f}".format(costs[5])
            joint_str = "\n".join([str1, str2, str3, str4, str5, str6, str7])
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()
            costs = [0.0 for v in costs]
        if ((i % 1000) == 0):
            #draw.save_model_params("TBCLM_IMP_SVHN_PARAMS_OD{}_DP{}_{}_{}.pkl".format(occ_dim, dp_int, step_type, att_tag))
            # compute a small-sample estimate of NLL bound on validation set
            Xva = row_shuffle(Xva)
            Xb = to_fX(Xva[:5000])
            _, Xb, Mb = construct_masked_data(Xb, drop_prob=drop_prob, \
                                    occ_dim=occ_dim, data_mean=None)
            va_costs = draw.compute_nll_bound(Xb, Mb)
            str1 = "    va_nll_bound : {}".format(va_costs[1])
            str2 = "    va_nll_term  : {}".format(va_costs[2])
            str3 = "    va_kld_q2p   : {}".format(va_costs[3])
            joint_str = "\n".join([str1, str2, str3])
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()
            # draw some independent samples from the model
            Xb = to_fX(Xva[:100])
            _, Xb, Mb = construct_masked_data(Xb, drop_prob=drop_prob, \
                                    occ_dim=occ_dim, data_mean=None)
            samples = draw.do_sample(Xb, Mb)
            n_iter, N, D = samples.shape
            samples = samples.reshape( (n_iter, N, 32, 32) )
            for j in xrange(n_iter):
                img = img_grid(samples[j,:,:,:])
                img.save("TBCLM-IMP-SVHN-OD{0:d}-DP{1:d}-{2:s}-samples-{3:03d}.png".format(occ_dim, dp_int, step_type, j))

#########################
#########################
## TEST TFD IMPUTATION ##
#########################
#########################

def test_imocld_imp_tfd(step_type='add', occ_dim=14, drop_prob=0.0, attention=False):
    ##########################
    # Get some training data #
    ##########################
    data_file = 'data/tfd_data_48x48.pkl'
    dataset = load_tfd(tfd_pkl_name=data_file, which_set='unlabeled', fold='all')
    Xtr_unlabeled = dataset[0]
    dataset = load_tfd(tfd_pkl_name=data_file, which_set='train', fold='all')
    Xtr_train = dataset[0]
    Xtr = np.vstack([Xtr_unlabeled, Xtr_train])
    dataset = load_tfd(tfd_pkl_name=data_file, which_set='valid', fold='all')
    Xva = dataset[0]
    Xtr = to_fX(shift_and_scale_into_01(Xtr))
    Xva = to_fX(shift_and_scale_into_01(Xva))
    tr_samples = Xtr.shape[0]
    va_samples = Xva.shape[0]
    batch_size = 250
    all_pix_mean = np.mean(np.mean(Xtr, axis=1))
    data_mean = to_fX( all_pix_mean * np.ones((Xtr.shape[1],)) )

    ############################################################
    # Setup some parameters for the Iterative Refinement Model #
    ############################################################
    x_dim = Xtr.shape[1]
    write_dim = 600
    enc_dim = 600
    dec_dim = 600
    mix_dim = 20
    z_dim = 200
    n_iter = 16
    dp_int = int(100.0 * drop_prob)
    
    rnninits = {
        'weights_init': IsotropicGaussian(0.01),
        'biases_init': Constant(0.),
    }
    inits = {
        'weights_init': IsotropicGaussian(0.01),
        'biases_init': Constant(0.),
    }

    att_tag = "NA" # attention not implemented yet

    # setup the reader and writer (shared by primary and guide policies)
    read_dim = 2*x_dim # dimension of output from reader_mlp
    reader_mlp = Reader(x_dim=x_dim, dec_dim=dec_dim, **inits)
    writer_mlp = MLP([None, None], [dec_dim, write_dim, x_dim], \
                     name="writer_mlp", **inits)
    
    # mlps for setting conditionals over z_mix
    mix_var_mlp = CondNet([Tanh()], [x_dim, 250, mix_dim], \
                          name="mix_var_mlp", **inits)
    mix_enc_mlp = CondNet([Tanh()], [x_dim, 250, mix_dim], \
                          name="mix_enc_mlp", **inits)
    # mlp for decoding z_mix into a distribution over initial LSTM states
    mix_dec_mlp = MLP([Tanh(), Tanh()], \
                      [mix_dim, 250, (2*enc_dim + 2*dec_dim + 2*enc_dim)], \
                      name="mix_dec_mlp", **inits)
    # mlps for processing inputs to LSTMs
    var_mlp_in = MLP([Identity()], [(read_dim + dec_dim), 4*enc_dim], \
                     name="var_mlp_in", **inits)
    enc_mlp_in = MLP([Identity()], [(read_dim + dec_dim), 4*enc_dim], \
                     name="enc_mlp_in", **inits)
    dec_mlp_in = MLP([Identity()], [               z_dim, 4*dec_dim], \
                     name="dec_mlp_in", **inits)
    #dec_mlp_in = MLP([Identity()], [   (enc_dim + z_dim), 4*dec_dim], \
    #                 name="dec_mlp_in", **inits)
    # mlps for turning LSTM outputs into conditionals over z_gen
    var_mlp_out = CondNet([], [enc_dim, z_dim], name="var_mlp_out", **inits)
    enc_mlp_out = CondNet([], [enc_dim, z_dim], name="enc_mlp_out", **inits)
    # LSTMs for the actual LSTMs (obviously, perhaps)
    var_rnn = BiasedLSTM(dim=enc_dim, ig_bias=2.0, fg_bias=2.0, \
                         name="var_rnn", **rnninits)
    enc_rnn = BiasedLSTM(dim=enc_dim, ig_bias=2.0, fg_bias=2.0, \
                         name="enc_rnn", **rnninits)
    dec_rnn = BiasedLSTM(dim=dec_dim, ig_bias=2.0, fg_bias=2.0, \
                         name="dec_rnn", **rnninits)

    draw = IMoCLDrawModels(
                n_iter,
                step_type=step_type, # step_type can be 'add' or 'jump'
                reader_mlp=reader_mlp,
                writer_mlp=writer_mlp,
                mix_enc_mlp=mix_enc_mlp,
                mix_dec_mlp=mix_dec_mlp,
                mix_var_mlp=mix_var_mlp,
                enc_mlp_in=enc_mlp_in,
                enc_mlp_out=enc_mlp_out,
                enc_rnn=enc_rnn,
                dec_mlp_in=dec_mlp_in,
                dec_rnn=dec_rnn,
                var_mlp_in=var_mlp_in,
                var_mlp_out=var_mlp_out,
                var_rnn=var_rnn)
    draw.initialize()

    # build the cost gradients, training function, samplers, etc.
    draw.build_model_funcs()

    #draw.load_model_params(f_name="TBCLM_IMP_PARAMS_OD{}_DP{}_{}_{}.pkl".format(occ_dim, dp_int, step_type, att_tag))

    ################################################################
    # Apply some updates, to check that they aren't totally broken #
    ################################################################
    print("Beginning to train the model...")
    out_file = open("TBCLM_IMP_TFD_RESULTS_OD{}_DP{}_{}_{}.txt".format(occ_dim, dp_int, step_type, att_tag), 'wb')
    out_file.flush()
    costs = [0. for i in range(10)]
    learn_rate = 0.0002
    momentum = 0.5
    batch_idx = np.arange(batch_size) + tr_samples
    for i in range(250000):
        scale = min(1.0, ((i+1) / 1000.0))
        if (((i + 1) % 10000) == 0):
            learn_rate = learn_rate * 0.95
        if (i > 10000):
            momentum = 0.90
        else:
            momentum = 0.50
        # get the indices of training samples for this batch update
        batch_idx += batch_size
        if (np.max(batch_idx) >= tr_samples):
            # we finished an "epoch", so we rejumble the training set
            Xtr = row_shuffle(Xtr)
            batch_idx = np.arange(batch_size)
        # set sgd and objective function hyperparams for this update
        zero_ary = np.zeros((1,))
        draw.lr.set_value(to_fX(zero_ary + learn_rate))
        draw.mom_1.set_value(to_fX(zero_ary + momentum))
        draw.mom_2.set_value(to_fX(zero_ary + 0.99))

        # perform a minibatch update and record the cost for this batch
        Xb = to_fX(Xtr.take(batch_idx, axis=0))
        _, Xb, Mb = construct_masked_data(Xb, drop_prob=drop_prob, \
                                    occ_dim=occ_dim, data_mean=None)
        result = draw.train_joint(Xb, Mb)

        costs = [(costs[j] + result[j]) for j in range(len(result))]
        if ((i % 200) == 0):
            costs = [(v / 200.0) for v in costs]
            str1 = "-- batch {0:d} --".format(i)
            str2 = "    total_cost: {0:.4f}".format(costs[0])
            str3 = "    nll_bound : {0:.4f}".format(costs[1])
            str4 = "    nll_term  : {0:.4f}".format(costs[2])
            str5 = "    kld_q2p   : {0:.4f}".format(costs[3])
            str6 = "    kld_p2q   : {0:.4f}".format(costs[4])
            str7 = "    reg_term  : {0:.4f}".format(costs[5])
            joint_str = "\n".join([str1, str2, str3, str4, str5, str6, str7])
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()
            costs = [0.0 for v in costs]
        if ((i % 1000) == 0):
            draw.save_model_params("TBCLM_IMP_TFD_PARAMS_OD{}_DP{}_{}_{}.pkl".format(occ_dim, dp_int, step_type, att_tag))
            # compute a small-sample estimate of NLL bound on validation set
            Xva = row_shuffle(Xva)
            Xb = to_fX(Xva[:5000])
            _, Xb, Mb = construct_masked_data(Xb, drop_prob=drop_prob, \
                                    occ_dim=occ_dim, data_mean=None)
            va_costs = draw.compute_nll_bound(Xb, Mb)
            str1 = "    va_nll_bound : {}".format(va_costs[1])
            str2 = "    va_nll_term  : {}".format(va_costs[2])
            str3 = "    va_kld_q2p   : {}".format(va_costs[3])
            joint_str = "\n".join([str1, str2, str3])
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()
            # draw some independent samples from the model
            Xb = to_fX(Xva[:256])
            _, Xb, Mb = construct_masked_data(Xb, drop_prob=drop_prob, \
                                    occ_dim=occ_dim, data_mean=None)
            samples = draw.do_sample(Xb, Mb)
            n_iter, N, D = samples.shape
            samples = samples.reshape( (n_iter, N, 48, 48) )
            for j in xrange(n_iter):
                img = img_grid(samples[j,:,:,:])
                img.save("TBCLM-IMP-TFD-samples-%03d.png" % (j,))

if __name__=="__main__":
    #test_imocld_imp_mnist(step_type='add', occ_dim=14, drop_prob=0.0)
    #test_imocld_imp_mnist(step_type='jump', occ_dim=14, drop_prob=0.0)
    #test_imocld_imp_mnist(step_type='add', occ_dim=16, drop_prob=0.0)
    #test_imocld_imp_mnist(step_type='jump', occ_dim=16, drop_prob=0.0)
    #test_imocld_imp_mnist(step_type='add', occ_dim=0, drop_prob=0.6)
    #test_imocld_imp_mnist(step_type='jump', occ_dim=0, drop_prob=0.6)
    #test_imocld_imp_mnist(step_type='add', occ_dim=0, drop_prob=0.8)
    #test_imocld_imp_mnist(step_type='jump', occ_dim=0, drop_prob=0.8)
    #test_imocld_imp_tfd(step_type='add', occ_dim=0, drop_prob=0.8)
    #test_imocld_imp_tfd(step_type='add', occ_dim=25, drop_prob=0.0)
    test_imocld_imp_svhn(step_type='add', occ_dim=0, drop_prob=0.8)
    #test_imocld_imp_svhn(step_type='add', occ_dim=17, drop_prob=0.0)