##################################################################
# CODE FOR EFFICIENTLY SAMPLING A (SMALL) FIXED-LENGTH VAE CHAIN #
##################################################################

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
from GIPair import GIPair


class MCSampler(object):
    """
    Class for quickly sampling some small fixed number of steps from the
    Markov chain constructed by self-looping a variational auto-encoder.

    Parameters:
        rng: numpy.random.RandomState (for reproducibility)
        Xd: symbolic var for providing points for starting the Markov Chain
        i_net: The InfNet instance that will serve as the inferencer
        g_net: The GenNet instance that will serve as the generator
        d_net: The PeaNet instance that will serve as the discriminator
        chain_len: number of steps to unroll the VAE Markov Chain
        data_dim: dimension of the generated data
        prior_dim: dimension of the model prior
    """
    def __init__(self, rng=None, Xd=None, \
                 i_net=None, g_net=None, chain_len=None, \
                 data_dim=None, prior_dim=None):
        # Do some stuff!
        self.rng = RandStream(rng.randint(100000))
        self.data_dim = data_dim
        self.prior_dim = prior_dim

        # symbolic var for inputting samples for initializing the VAE chain
        self.Xd = Xd
        # symbolic var for masking subsets of the state variables
        self.Xm = T.zeros_like(self.Xd)
        # symbolic var for controlling subsets of the state variables
        self.Xc = T.zeros_like(self.Xd)
        # integer number of times to cycle the VAE loop
        self.chain_len = chain_len

        # get a clone of the desired VAE, for easy access
        self.GIP = GIPair(rng=rng, Xd=self.Xd, Xc=self.Xc, Xm=self.Xm, \
                g_net=g_net, i_net=i_net, data_dim=self.data_dim, \
                prior_dim=self.prior_dim, params=None, shared_param_dicts=None)
        self.IN = self.GIP.IN
        self.GN = self.GIP.GN
        self.use_encoder = self.IN.use_encoder
        assert(self.use_encoder == self.GN.use_decoder)
        # self-loop some clones of the main VAE into a chain.
        # ** All VAEs in the chain share the same Xc and Xm, which are the
        #    symbolic inputs for providing the observed portion of the input
        #    and a mask indicating which part of the input is "observed".
        #    These inputs are used for training "reconstruction" policies.
        self.IN_chain = []
        self.GN_chain = []
        self.Xg_chain = []
        _Xd = self.Xd
        for i in range(self.chain_len):
            if (i == 0):
                # start the chain with data provided by user
                _IN = self.IN.shared_param_clone(rng=rng, Xd=_Xd, \
                        Xc=self.Xc, Xm=self.Xm)
                _GN = self.GN.shared_param_clone(rng=rng, Xp=_IN.output)
            else:
                # continue the chain with samples from previous VAE
                _IN = self.IN.shared_param_clone(rng=rng, Xd=_Xd, \
                        Xc=self.Xc, Xm=self.Xm)
                _GN = self.GN.shared_param_clone(rng=rng, Xp=_IN.output)
            if self.use_encoder:
                # use the "decoded" output of the previous generator as input
                # to the next inferencer, which will re-encode it prior to
                # inference
                _Xd = _GN.output_decoded
            else:
                # use the "encoded" output of the previous generator as input
                # to the next inferencer, as the inferencer won't try to 
                # re-encode it prior to inference
                _Xd = _GN.output
            self.IN_chain.append(_IN)
            self.GN_chain.append(_GN)
            self.Xg_chain.append(_Xd)

        # construct the function for training on training data
        self.sample_from_chain = self._construct_sample_from_chain()
        return

    def _construct_sample_from_chain(self):
        """
        Sample for several steps of a self-looped VAE.
        """
        outputs = [Xg for Xg in self.Xg_chain]
        sample_func = theano.function([self.Xd], outputs=outputs)
        return sample_func

def resample_chain_steps(MCS, Xtr_chains):
    # get and set some basic dataset information
    assert(len(Xtr_chains) == (MCS.chain_len + 1))
    Xtr = Xtr_chains[0]
    for Xc in Xtr_chains:
        assert(Xc.shape[0] == Xtr.shape[0])
        assert(Xc.shape[1] == Xtr.shape[1])
    tr_samples = Xtr.shape[0]
    data_dim = Xtr.shape[1]
    batch_size = 5000
    batch_count = int(np.ceil(tr_samples / float(batch_size)))
    # print("Resampling {0:d} batches of {1:d} chains with {2:d} steps...".format(batch_count, batch_size, MCS.chain_len))
    for i in range(batch_count):
        batch_start = i * batch_size
        batch_end = min(tr_samples, (batch_start + batch_size))
        batch_Xd = Xtr[batch_start:batch_end]
        batch_chains = MCS.sample_from_chain(batch_Xd)
        for j in range(len(batch_chains)):
            Xtr_chains[j+1][batch_start:batch_end] = batch_chains[j]
    return Xtr_chains


if __name__=="__main__":
    import utils
    import time
    from load_data import load_udm
    import InfNet as INet
    import GenNet as GNet
    # Initialize a source of randomness
    rng = npr.RandomState(12345)
    # Load some data to train/validate/test with
    dataset = 'data/mnist.pkl.gz'
    datasets = load_udm(dataset, zero_mean=False)
    Xtr = datasets[0][0]
    Xtr = Xtr.get_value(borrow=False)
    Xva = datasets[1][0]
    Xva = Xva.get_value(borrow=False)
    print("Xtr.shape: {0:s}, Xva.shape: {1:s}".format(str(Xtr.shape),str(Xva.shape)))

    # get and set some basic dataset information
    tr_samples = Xtr.shape[0]
    data_dim = Xtr.shape[1]
    batch_size = 2000
    batch_count = int(np.ceil(tr_samples / float(batch_size)))

    # Symbolic inputs
    Xd = T.matrix(name='Xd')
    Xc = T.matrix(name='Xc')
    Xm = T.matrix(name='Xm')
    Xt = T.matrix(name='Xt')
    Xp = T.matrix(name='Xp')

    # Load inferencer and generator from saved parameters
    gn_fname = "MNIST_WALKOUT_TEST_VAE/pt_walk_params_b30000_GN.pkl"
    in_fname = "MNIST_WALKOUT_TEST_VAE/pt_walk_params_b30000_IN.pkl"
    IN = INet.load_infnet_from_file(f_name=in_fname, rng=rng, Xd=Xd, Xc=Xc, Xm=Xm)
    GN = GNet.load_gennet_from_file(f_name=gn_fname, rng=rng, Xp=Xp)
    IN.set_sigma_scale(1.5)
    prior_dim = GN.latent_dim

    MCS = MCSampler(rng=rng, Xd=Xd, i_net=IN, g_net=GN, chain_len=9, \
                    data_dim=data_dim, prior_dim=prior_dim)

    Xtr_chains = [Xtr]
    for i in range(MCS.chain_len):
        Xtr_chains.append(0.0*Xtr)

    print("Testing chain sampler....")
    loop_times = []
    # TESTING SAMPLING SPEED!
    for i in range(batch_count):
        start_time = time.clock()
        batch_start = i * batch_size
        batch_end = min(tr_samples, (batch_start + batch_size))
        Xd_batch = Xtr[batch_start:batch_end]
        Xd_chain = MCS.sample_from_chain(Xd_batch)
        Xs = [Xd_batch[0:50]]
        Xs.extend([xd[0:50] for xd in Xd_chain])
        file_name = "MCS_TEST_{0:d}.png".format(i)
        utils.visualize_samples(np.vstack(Xs), file_name, num_rows=10)
        loop_times.append((time.clock() - start_time))
    total_time = sum(loop_times)
    mean_time = total_time / batch_count
    time_std = sum([(t - mean_time)**2.0 for t in loop_times]) / batch_count
    print("total_time: {0:.4f}".format(total_time))
    print("mean_time: {0:.4f}, time_std: {1:.4f}".format(mean_time, time_std))
    start_time = time.clock()
    Xtr_chains = resample_chain_steps(MCS, Xtr_chains)
    total_time = time.clock() - start_time
    print("total_time: {0:.4f}".format(total_time))





##############
# EYE BUFFER #
##############
