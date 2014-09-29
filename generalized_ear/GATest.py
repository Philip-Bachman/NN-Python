import time
import numpy as np
import numpy.random as npr
import theano
import theano.tensor as T
from theano.ifelse import ifelse
from load_data import load_udm, load_udm_ss, load_mnist
from EarNet import EAR_NET
from GAPair import GEN_NET, GA_PAIR, projected_moments

# Simple test code, to check that everything is basically functional.
print("TESTING...")

# Initialize a source of randomness
rng = np.random.RandomState(1234)

# Load some data to train/validate/test with
dataset = 'data/mnist.pkl.gz'
datasets = load_udm(dataset, zero_mean=False)
Xtr = datasets[0][0]
tr_samples = Xtr.get_value(borrow=True).shape[0]
data_dim = Xtr.get_value(borrow=True).shape[1]
mm_proj_dim = 256

# Do moment matching in some transformed space
#P = np.identity(data_dim)
P = npr.randn(data_dim, mm_proj_dim) / np.sqrt(float(mm_proj_dim))
P = theano.shared(value=P.astype(theano.config.floatX))

target_mean, target_cov = projected_moments(Xtr, P, ary_type='theano')
P = P.get_value(borrow=False)

###########################
# Setup generator network #
###########################

# Choose some parameters for the generative network
gn_params = {}
gn_config = [75, 150, 150, 28*28]
gn_params['mlp_config'] = gn_config
gn_params['lam_l2a'] = 1e-2
gn_params['use_bias'] = 1
gn_params['vis_drop'] = 0.0
gn_params['hid_drop'] = 0.0
gn_params['bias_noise'] = 0.1
gn_params['out_noise'] = 0.1

# Symbolic input matrix to generator network
X_noise_sym = T.dmatrix(name='X_noise_sym')
X_data_sym = T.dmatrix(name='X_data_sym')

# Initialize a generator network object
GN = GEN_NET(rng=rng, input_noise=X_noise_sym, input_data=X_data_sym, \
        params=gn_params)


###############################
# Setup discriminator network #
###############################

# Set some reasonable mlp parameters
dn_params = {}
# Set up some proto-networks
pc0 = [28*28, 100, 100, 11]
dn_params['proto_configs'] = [pc0]
# Set up some spawn networks
sc0 = {'proto_key': 0, 'input_noise': 0.1, 'bias_noise': 0.1, 'do_dropout': False}
sc1 = {'proto_key': 0, 'input_noise': 0.1, 'bias_noise': 0.1, 'do_dropout': False}
dn_params['spawn_configs'] = [sc0, sc1]
dn_params['spawn_weights'] = [0.5, 0.5]
# Set remaining params
dn_params['ear_type'] = 2
dn_params['ear_lam'] = 2.0
dn_params['lam_l2a'] = 1e-3
dn_params['use_bias'] = 1
dn_params['reg_all_obs'] = True

# Initialize a discriminator network object
DN = EAR_NET(rng=rng, input=GN.output, params=dn_params)

########################################################################
# Initialize the joint controller for the generator/discriminator pair #
########################################################################

gap_params = {}
gap_params['d_net'] = DN
gap_params['g_net'] = GN
gap_params['mom_mix_rate'] = 0.05
gap_params['mom_match_weight'] = 0.02
gap_params['mom_match_proj'] = P
gap_params['target_mean'] = target_mean
gap_params['target_cov'] = target_cov

# Initialize a GA_PAIR instance using the previously constructed generator and
# discriminator networks.
GAP = GA_PAIR(rng=rng, d_net=DN, g_net=GN, params=gap_params)
GAP.set_gn_sgd_params(learn_rate=0.100, momentum=0.8)
GAP.set_dn_sgd_params(learn_rate=0.002, momentum=0.8)
# Init generator's mean and covariance estimates with many samples
Xn = npr.randn(5000, GN.latent_dim)
GAP.init_moments(Xn)

for i in range(5000):
    tr_idx = npr.randint(low=0,high=tr_samples,size=(100,))
    Xn = 4.0 * npr.randn(200, GAP.GN.latent_dim).astype(theano.config.floatX)
    Xd = Xtr.take(tr_idx, axis=0).eval()
    all_idx = np.arange(300)
    data_idx = all_idx[:100]
    noise_idx = all_idx[100:]
    if (i < 200):
        GAP.set_disc_weights(dweight_gn=0.01)
        outputs = GAP.train_gn(Xd, Xn, data_idx, noise_idx)
    else:
        GAP.set_disc_weights(dweight_gn=1.01)
        outputs = GAP.train_joint(Xd, Xn, data_idx, noise_idx)
    mom_match_cost = 1.0 * outputs[0]
    disc_cost_gn = 1.0 * outputs[1]
    disc_cost_dn = 1.0 * outputs[2]
    if ((i % 10) == 0):
        print("batch: {0:d}, mom_match_cost: {1:.4f}, disc_cost_gn: {2:.4f}, disc_cost_dn: {3:.4f}".format( \
                i, mom_match_cost, disc_cost_gn, disc_cost_dn))


print("TESTING COMPLETE!")





##############
# EYE BUFFER #
##############
