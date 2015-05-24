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
from blocks.model import Model
from blocks.bricks import Tanh, Identity, Rectifier
from blocks.bricks.recurrent import SimpleRecurrent, LSTM

# phil's sweetness
import utils
from BlocksModels import *
from load_data import load_udm, load_mnist, load_binarized_mnist
from HelperFuncs import row_shuffle, to_fX

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

def test_imoold_generation(step_type='add', attention=False):
    ##########################
    # Get some training data #
    ##########################
    rng = np.random.RandomState(1234)
    Xtr, Xva, Xte = load_binarized_mnist(data_path='./data/')
    Xtr = np.vstack((Xtr, Xva))
    Xva = Xte
    #del Xte
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
    if attention:
        n_iter = 64
    else:
        n_iter = 16
    
    rnninits = {
        'weights_init': IsotropicGaussian(0.01),
        'biases_init': Constant(0.),
    }
    inits = {
        'weights_init': IsotropicGaussian(0.01),
        'biases_init': Constant(0.),
    }

    # setup the reader and writer
    if attention:
        read_N, write_N = (2, 5) # resolution of reader and writer
        read_dim = 2*read_N**2   # total number of "pixels" read by reader
        reader_mlp = AttentionReader(x_dim=x_dim, dec_dim=dec_dim,
                                 width=28, height=28,
                                 N=read_N, **inits)
        writer_mlp = AttentionWriter(input_dim=dec_dim, output_dim=x_dim,
                                 width=28, height=28,
                                 N=write_N, **inits)
        att_tag = "YA"
    else:
        read_dim = 2*x_dim
        reader_mlp = Reader(x_dim=x_dim, dec_dim=dec_dim, **inits)
        writer_mlp = MLP([None, None], [dec_dim, write_dim, x_dim], \
                         name="writer_mlp", **inits)
        att_tag = "NA"
    
    # setup the infinite mixture initialization model
    mix_enc_mlp = CondNet([Tanh()], [x_dim, 250, mix_dim], \
                          name="mix_enc_mlp", **inits)
    mix_dec_mlp = MLP([Tanh(), Tanh()], \
                      [mix_dim, 250, (2*enc_dim + 2*dec_dim)], \
                      name="mix_dec_mlp", **inits)
    # setup the components of the sequential generative model
    enc_mlp_in = MLP([Identity()], [(read_dim + dec_dim), 4*enc_dim], \
                     name="enc_mlp_in", **inits)
    dec_mlp_in = MLP([Identity()], [               z_dim, 4*dec_dim], \
                     name="dec_mlp_in", **inits)
    enc_mlp_out = CondNet([], [enc_dim, z_dim], name="enc_mlp_out", **inits)
    dec_mlp_out = CondNet([], [dec_dim, z_dim], name="dec_mlp_out", **inits)
    enc_rnn = BiasedLSTM(dim=enc_dim, ig_bias=2.0, fg_bias=2.0, \
                         name="enc_rnn", **rnninits)
    dec_rnn = BiasedLSTM(dim=dec_dim, ig_bias=2.0, fg_bias=2.0, \
                         name="dec_rnn", **rnninits)

    draw = IMoOLDrawModels(
                n_iter,
                step_type=step_type, # step_type can be 'add' or 'jump'
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

    # build the cost gradients, training function, samplers, etc.
    draw.build_model_funcs()

    #draw.load_model_params(f_name="TBOLM_GEN_PARAMS_{}_{}.pkl".format(step_type, att_tag))

    ################################################################
    # Apply some updates, to check that they aren't totally broken #
    ################################################################
    print("Beginning to train the model...")
    out_file = open("TBOLM_GEN_RESULTS_{}_{}.txt".format(step_type, att_tag), 'wb')
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
        result = draw.train_joint(Xb, Xb)
        costs = [(costs[j] + result[j]) for j in range(len(result))]

        # diagnostics
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
            draw.save_model_params("TBOLM_GEN_PARAMS_{}_{}.pkl".format(step_type, att_tag))
            # compute a small-sample estimate of NLL bound on validation set
            Xva = row_shuffle(Xva)
            Xb = to_fX(Xva[:5000])
            va_costs = draw.compute_nll_bound(Xb, Xb)
            str1 = "    va_nll_bound : {}".format(va_costs[1])
            str2 = "    va_nll_term  : {}".format(va_costs[2])
            str3 = "    va_kld_q2p   : {}".format(va_costs[3])
            joint_str = "\n".join([str1, str2, str3])
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()
            # draw some independent samples from the model
            samples = draw.do_sample(16*16)
            n_iter, N, D = samples.shape
            samples = samples.reshape( (n_iter, N, 28, 28) )
            for j in xrange(n_iter):
                img = img_grid(samples[j,:,:,:])
                img.save("TBOLM-gen-samples-%03d.png" % (j,))

if __name__=="__main__":
    test_imoold_generation(step_type='add', attention=False)
    #test_imoold_generation(step_type='jump', attention=False)