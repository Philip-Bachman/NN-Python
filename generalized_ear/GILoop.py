###################################################################
# Code for managing and training a generator/discriminator pair.  #
###################################################################

# basic python
import numpy as np
import numpy.random as npr
from collections import OrderedDict

# theano business
import theano
import theano.tensor as T
from theano.ifelse import ifelse
import theano.tensor.shared_randomstreams
#from theano.sandbox.cuda.rng_curand import CURAND_RandomStreams

# phil's sweetness
from NetLayers import HiddenLayer, DiscLayer
from GenNet import projected_moments

class GILoop(object):
    """
    Controller for propagating through a generate<->inference loop.

    The generator must be an instance of the GEN_NET class implemented in
    "GINets.py". The discriminator must be an instance of the EarNet class,
    as implemented in "EarNet.py".

    Parameters:
        rng: numpy.random.RandomState (for reproducibility)
        g_net: The GenNet instance that will serve as the generator
        i_net: The InfNet instance that will serve as the infer(er)
        data_dim: Dimensions of generated data
        loop_iters: The number of loop cycles to unroll
    """
    def __init__(self, rng=None, g_net=None, i_net=None, data_dim=None, \
            loop_iters=1):
        # Do some stuff!
        self.rng = theano.tensor.shared_randomstreams.RandomStreams( \
                rng.randint(100000))
        self.GN_base = g_net
        self.IN_base = i_net
        self.input_noise = self.GN.input_var
        self.input_data = data_var
        self.sample_data = self.GN.output
        self.data_dim = data_dim

        # symbolic var data input
        self.Xd = T.matrix(name='gil_Xd')
        # symbolic var noise input
        self.Xn = T.matrix(name='gil_Xn')
        # symbolic matrix of indices for data inputs
        self.Id = T.lvector(name='gil_Id')
        # symbolic matrix of indices for noise inputs
        self.In = T.lvector(name='gil_In')
        # shared var learning rate for generator and discriminator
        zero_ary = np.zeros((1,)).astype(theano.config.floatX)
        self.lr_gn = theano.shared(value=zero_ary, name='gil_lr_gn')
        self.lr_in = theano.shared(value=zero_ary, name='gil_lr_in')
        # shared var momentum parameters for generator and discriminator
        self.mo_gn = theano.shared(value=zero_ary, name='gil_mo_gn')
        self.mo_in = theano.shared(value=zero_ary, name='gil_mo_in')
        # init parameters for controlling learning dynamics
        self.set_gn_sgd_params() # init SGD rate/momentum for GN
        self.set_in_sgd_params() # init SGD rate/momentum for IN


        # Grab the full set of "optimizable" parameters from the generator
        # and inference networks that we'll be working with.
        self.in_params = []
        for pn in self.DN.proto_nets:
            for pnl in pn[0:-1]:
                self.dn_params.extend(pnl.params)
        self.gn_params = [p for p in self.GN.mlp_params]
        # Now construct a binary discriminator layer for each proto-net in the
        # discriminator network. And, add their params to optimization list.
        self._construct_disc_layers(rng)
        self.disc_reg_cost = self.lam_l2d[0] * \
                T.sum([dl.act_l2_sum for dl in self.disc_layers])

        # Construct costs for the generator and discriminator networks based 
        # on adversarial binary classification
        self.disc_cost_dn, self.disc_cost_gn = self._construct_disc_costs()

        # Cost w.r.t. discriminator parameters is only the adversarial binary
        # classification cost. Cost w.r.t. comprises an adversarial binary
        # classification cost and the (weighted) moment matching cost.
        self.dn_cost = self.disc_cost_dn + self.DN.act_reg_cost + self.disc_reg_cost
        self.gn_cost = self.disc_cost_gn + self.mom_match_cost + self.GN.act_reg_cost

        # Initialize momentums for mini-batch SGD updates. All parameters need
        # to be safely nestled in their lists by now.
        self.joint_moms = OrderedDict()
        self.dn_moms = OrderedDict()
        self.gn_moms = OrderedDict()
        for p in self.dn_params:
            p_mo = np.zeros(p.get_value(borrow=True).shape)
            self.dn_moms[p] = theano.shared(value=p_mo.astype(theano.config.floatX))
            self.joint_moms[p] = self.dn_moms[p]
        for p in self.gn_params:
            p_mo = np.zeros(p.get_value(borrow=True).shape)
            self.gn_moms[p] = theano.shared(value=p_mo.astype(theano.config.floatX))
            self.joint_moms[p] = self.gn_moms[p]

        # Construct the updates for the generator and discriminator network
        self.joint_updates = OrderedDict()
        self.dn_updates = OrderedDict()
        self.gn_updates = OrderedDict()
        for var in self.dn_params:
            # these updates are for trainable params in the discriminator net...
            # first, get gradient of cost w.r.t. var
            var_grad = T.grad(self.dn_cost, var)
            # get the momentum for this var
            var_mom = self.dn_moms[var]
            # update the momentum for this var using its grad
            self.dn_updates[var_mom] = (self.mo_dn[0] * var_mom) + \
                    ((1.0 - self.mo_dn[0]) * var_grad)
            self.joint_updates[var_mom] = self.dn_updates[var_mom]
            # make basic update to the var
            var_new = var - (self.lr_dn[0] * var_mom)
            if ((var in self.DN.clip_params) and \
                    (var in self.DN.clip_norms) and \
                    (self.DN.clip_params[var] == 1)):
                # clip the basic updated var if it is set as clippable
                clip_norm = self.DN.clip_norms[var]
                var_norms = T.sum(var_new**2.0, axis=1, keepdims=True)
                var_scale = T.clip(T.sqrt(clip_norm / var_norms), 0., 1.)
                self.dn_updates[var] = var_new * var_scale
            else:
                # otherwise, just use the basic updated var
                self.dn_updates[var] = var_new
            # add this var's update to the joint updates too
            self.joint_updates[var] = self.dn_updates[var]
        for var in self.mom_updates:
            # these updates are for the generator distribution's running first
            # and second-order moment estimates
            self.gn_updates[var] = self.mom_updates[var]
            self.joint_updates[var] = self.gn_updates[var]
        for var in self.gn_params:
            # these updates are for trainable params in the generator net...
            # first, get gradient of cost w.r.t. var
            var_grad = T.grad(self.gn_cost, var, \
                    consider_constant=[self.GN.dist_mean, self.GN.dist_cov])
            # get the momentum for this var
            var_mom = self.gn_moms[var]
            # update the momentum for this var using its grad
            self.gn_updates[var_mom] = (self.mo_gn[0] * var_mom) + \
                    ((1.0 - self.mo_gn[0]) * var_grad)
            self.joint_updates[var_mom] = self.gn_updates[var_mom]
            # make basic update to the var
            var_new = var - (self.lr_gn[0] * var_mom)
            if ((var in self.GN.clip_params) and \
                    (var in self.GN.clip_norms) and \
                    (self.GN.clip_params[var] == 1)):
                # clip the basic updated var if it is set as clippable
                clip_norm = self.GN.clip_norms[var]
                var_norms = T.sum(var_new**2.0, axis=1, keepdims=True)
                var_scale = T.clip(T.sqrt(clip_norm / var_norms), 0., 1.)
                self.gn_updates[var] = var_new * var_scale
            else:
                # otherwise, just use the basic updated var
                self.gn_updates[var] = var_new
            # add this var's update to the joint updates too
            self.joint_updates[var] = self.gn_updates[var]

        # Construct batch-based training functions for the generator and
        # discriminator networks, as well as a joint training function.
        self.train_gn = self._construct_train_gn()
        self.train_dn = self._construct_train_dn()
        self.train_joint = self._construct_train_joint()

        # Construct a function for computing the ouputs of the generator
        # network for a batch of noise. Presumably, the noise will be drawn
        # from the same distribution that was used in training....
        self.sample_from_gn = self._construct_gn_sampler()
        return

    def set_gn_sgd_params(self, learn_rate=0.02, momentum=0.9):
        """
        Set learning rate and momentum parameter for generator updates.
        """
        zero_ary = np.zeros((1,))
        new_lr = zero_ary + learn_rate
        self.lr_gn.set_value(new_lr.astype(theano.config.floatX))
        new_mo = zero_ary + momentum
        self.mo_gn.set_value(new_mo.astype(theano.config.floatX))
        return

    def set_in_sgd_params(self, learn_rate=0.02, momentum=0.9):
        """
        Set learning rate and momentum parameter for discriminator updates.
        """
        zero_ary = np.zeros((1,))
        new_lr = zero_ary + learn_rate
        self.lr_in.set_value(new_lr.astype(theano.config.floatX))
        new_mo = zero_ary + momentum
        self.mo_in.set_value(new_mo.astype(theano.config.floatX))
        return

    def _construct_train_gn(self):
        """
        Construct theano function to train generator on its own.
        """
        outputs = [self.mom_match_cost, self.disc_cost_gn, self.disc_cost_dn]
        func = theano.function(inputs=[ self.Xd, self.Xn, self.Id, self.In ], \
                outputs=outputs, \
                updates=self.gn_updates, \
                givens={self.input_data: self.Xd, \
                        self.input_noise: self.Xn})
        theano.printing.pydotprint(func, \
            outfile='gn_func_graph.png', compact=True, format='png', with_ids=False, \
            high_contrast=True, cond_highlight=None, colorCodes=None, \
            max_label_size=70, scan_graphs=False, var_with_name_simple=False, \
            print_output_file=True, assert_nb_all_strings=-1)
        return func

    def _construct_train_dn(self):
        """
        Construct theano function to train discriminator on its own.
        """
        outputs = [self.mom_match_cost, self.disc_cost_gn, self.disc_cost_dn]
        func = theano.function(inputs=[ self.Xd, self.Xn, self.Id, self.In ], \
                outputs=outputs, \
                updates=self.dn_updates, \
                givens={self.input_data: self.Xd, \
                        self.input_noise: self.Xn})
        theano.printing.pydotprint(func, \
            outfile='dn_func_graph.png', compact=True, format='png', with_ids=False, \
            high_contrast=True, cond_highlight=None, colorCodes=None, \
            max_label_size=70, scan_graphs=False, var_with_name_simple=False, \
            print_output_file=True, assert_nb_all_strings=-1)
        return func

    def _construct_train_joint(self):
        """
        Construct theano function to train generator and discriminator jointly.
        """
        outputs = [self.mom_match_cost, self.disc_cost_gn, self.disc_cost_dn]
        func = theano.function(inputs=[ self.Xd, self.Xn, self.Id, self.In ], \
                outputs=outputs, \
                updates=self.joint_updates, \
                givens={self.input_data: self.Xd, \
                        self.input_noise: self.Xn})
        return func

    def _construct_gn_sampler(self):
        """
        Construct theano function to sample from the generator network.
        """
        Xn_sym = T.matrix('gn_sampler_input')
        theano_func = theano.function( \
               inputs=[ Xn_sym ], \
               outputs=[ self.sample_data ], \
               givens={ self.input_noise: Xn_sym })
        sample_func = lambda Xn: theano_func(Xn)[0]
        return sample_func

if __name__=="__main__":
    NOT_DONE = True

    print("TESTING COMPLETE!")




##############
# EYE BUFFER #
##############
