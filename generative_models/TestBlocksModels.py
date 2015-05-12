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
from NetLayers import apply_mask, binarize_data, row_shuffle, to_fX
from DKCode import get_adam_updates, get_adadelta_updates
from load_data import load_udm, load_udm_ss, load_mnist, load_binarized_mnist

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


########################################
########################################
## TEST WITH MODEL-BASED INITIAL STEP ##
########################################
########################################

def test_with_model_init():
    ##########################
    # Get some training data #
    ##########################
    rng = np.random.RandomState(1234)
    Xtr, Xva, Xte = load_binarized_mnist(data_path='./data/')
    del Xte
    tr_samples = Xtr.shape[0]
    va_samples = Xva.shape[0]
    batch_size = 250


    ############################################################
    # Setup some parameters for the Iterative Refinement Model #
    ############################################################
    x_dim = Xtr.shape[1]
    write_dim = 200
    enc_dim = 250
    dec_dim = 250
    mix_dim = 20
    z_dim = 100
    n_iter = 15
    
    rnninits = {
        'weights_init': IsotropicGaussian(0.01),
        'biases_init': Constant(0.),
    }
    inits = {
        'weights_init': IsotropicGaussian(0.01),
        'biases_init': Constant(0.),
    }

    # setup the reader and writer
    read_dim = 2*x_dim
    reader_mlp = Reader(x_dim=x_dim, dec_dim=dec_dim, **inits)
    writer_mlp = MLP([None, None], [dec_dim, write_dim, x_dim], \
                     name="writer_mlp", **inits)
    
    # setup the mixture weight sampler
    mix_enc_mlp = CondNet([Tanh()], [x_dim, 250, mix_dim], \
                          name="mix_enc_mlp", **inits)
    mix_dec_mlp = MLP([Tanh(), Tanh()], \
                      [mix_dim, 250, (2*enc_dim + 2*dec_dim + mix_dim)], \
                      name="mix_dec_mlp", **inits)
    # setup the components of the generative DRAW model
    enc_mlp_in = MLP([Identity()], [(read_dim + dec_dim + mix_dim), 4*enc_dim], \
                     name="enc_mlp_in", **inits)
    dec_mlp_in = MLP([Identity()], [             (z_dim + mix_dim), 4*dec_dim], \
                     name="dec_mlp_in", **inits)
    enc_mlp_out = CondNet([], [enc_dim, z_dim], name="enc_mlp_out", **inits)
    dec_mlp_out = CondNet([], [dec_dim, z_dim], name="dec_mlp_out", **inits)
    enc_rnn = BiasedLSTM(dim=enc_dim, ig_bias=2.0, fg_bias=2.0, \
                         name="enc_rnn", **rnninits)
    dec_rnn = BiasedLSTM(dim=dec_dim, ig_bias=2.0, fg_bias=2.0, \
                         name="dec_rnn", **rnninits)

    draw = IMoDrawModels(
                n_iter,
                step_type='jump', # step_type can be 'add' or 'jump'
                mix_enc_mlp=mix_enc_mlp,
                mix_dec_mlp=mix_dec_mlp,
                reader_mlp=reader_mlp,
                enc_mlp_in=enc_mlp_in,
                enc_mlp_out=enc_mlp_out,
                enc_rnn=enc_rnn,
                dec_mlp_in=dec_mlp_in,
                dec_mlp_out=dec_mlp_out,
                dec_rnn=dec_rnn,
                writer_mlp=writer_mlp)
    draw.initialize()

    # some symbolic vars to represent various inputs/outputs
    x_in_sym = T.matrix('x_in_sym')
    x_out_sym = T.matrix('x_out_sym')

    # collect reconstructions of x produced by the IMoDRAW model
    x_recons, kl_q2p, kl_p2q = draw.reconstruct(x_in_sym, x_out_sym)

    # get the expected NLL part of the VFE bound
    nll_term = BinaryCrossEntropy().apply(x_out_sym, x_recons)
    nll_term.name = "nll_term"

    # get KL(q || p) and KL(p || q)
    kld_q2p_term = kl_q2p.sum(axis=0).mean()
    kld_q2p_term.name = "kld_q2p_term"
    kld_p2q_term = kl_p2q.sum(axis=0).mean()
    kld_p2q_term.name = "kld_p2q_term"

    # get the proper VFE bound on NLL
    nll_bound = nll_term + kld_q2p_term
    nll_bound.name = "nll_bound"

    # grab handles for all the optimizable parameters in our cost
    cg = ComputationGraph([nll_bound])
    joint_params = VariableFilter(roles=[PARAMETER])(cg.variables)

    # apply some l2 regularization to the model parameters
    reg_term = (1e-5 * sum([T.sum(p**2.0) for p in joint_params]))
    reg_term.name = "reg_term"

    # compute the full cost w.r.t. which we will optimize
    total_cost = nll_term + (0.9 * kld_q2p_term) + \
                 (0.1 * kld_p2q_term) + reg_term
    total_cost.name = "total_cost"

    # Get the gradient of the joint cost for all optimizable parameters
    print("Computing gradients of total_cost...")
    joint_grads = OrderedDict()
    grad_list = T.grad(total_cost, joint_params)
    for i, p in enumerate(joint_params):
        joint_grads[p] = grad_list[i]
    
    # shared var learning rate for generator and inferencer
    zero_ary = to_fX( np.zeros((1,)) )
    lr_shared = theano.shared(value=zero_ary, name='tbm_lr')
    # shared var momentum parameters for generator and inferencer
    mom_1_shared = theano.shared(value=zero_ary, name='tbm_mom_1')
    mom_2_shared = theano.shared(value=zero_ary, name='tbm_mom_2')
    # construct the updates for the generator and inferencer networks
    joint_updates = get_adam_updates(params=joint_params, \
            grads=joint_grads, alpha=lr_shared, \
            beta1=mom_1_shared, beta2=mom_2_shared, \
            mom2_init=1e-4, smoothing=1e-6, max_grad_norm=10.0)

    # collect the outputs to return from this function
    outputs = [total_cost, nll_bound, nll_term, kld_q2p_term, \
               kld_p2q_term, reg_term]
    # compile the theano function
    print("Compiling model training/update function...")
    train_joint = theano.function(inputs=[ x_in_sym, x_out_sym ], \
                                  outputs=outputs, updates=joint_updates)
    print("Compiling NLL bound estimator function...")
    compute_nll_bound = theano.function(inputs=[ x_in_sym, x_out_sym], \
                                        outputs=outputs)
    print("Compiling model sampler...")
    n_samples = T.iscalar("n_samples")
    samples = draw.sample(n_samples)
    do_sample = theano.function([n_samples], outputs=samples, allow_input_downcast=True)

    ################################################################
    # Apply some updates, to check that they aren't totally broken #
    ################################################################
    print("Beginning to train the model...")
    out_file = open("TBM_RESULTS.txt", 'wb')
    costs = [0. for i in range(10)]
    learn_rate = 0.0002
    momentum = 0.5
    fresh_idx = np.arange(batch_size) + tr_samples
    for i in range(250000):
        scale = min(1.0, ((i+1) / 1000.0))
        if (((i + 1) % 10000) == 0):
            learn_rate = learn_rate * 0.95
        if (i > 10000):
            momentum = 0.90
        else:
            momentum = 0.50
        # get the indices of training samples for this batch update
        fresh_idx += batch_size
        if (np.max(fresh_idx) >= tr_samples):
            # we finished an "epoch", so we rejumble the training set
            Xtr = row_shuffle(Xtr)
            fresh_idx = np.arange(batch_size)
        batch_idx = fresh_idx
        # set sgd and objective function hyperparams for this update
        zero_ary = np.zeros((1,))
        lr_shared.set_value(to_fX(zero_ary + learn_rate))
        mom_1_shared.set_value(to_fX(zero_ary + momentum))
        mom_2_shared.set_value(to_fX(zero_ary + 0.99))

        # perform a minibatch update and record the cost for this batch
        Xb = to_fX( Xtr.take(batch_idx, axis=0) )
        result = train_joint(Xb, Xb)

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
            # compute a small-sample estimate of NLL bound on validation set
            Xva = row_shuffle(Xva)
            Xb = to_fX(Xva[:2000])
            va_costs = compute_nll_bound(Xb, Xb)
            str1 = "    va_nll_bound : {}".format(va_costs[1])
            str2 = "    va_nll_term  : {}".format(va_costs[2])
            str3 = "    va_kld_q2p   : {}".format(va_costs[3])
            joint_str = "\n".join([str1, str2, str3])
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()
            # draw some independent samples from the model
            samples = do_sample(16*16)
            n_iter, N, D = samples.shape
            samples = samples.reshape( (n_iter, N, 28, 28) )
            for j in xrange(n_iter):
                img = img_grid(samples[j,:,:,:])
                img.save("TBM-samples-b%06d-%03d.png" % (i, j))

if __name__=="__main__":
    test_with_model_init()