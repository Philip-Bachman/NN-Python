from sys import stdout as stdout
from time import clock
import numpy as np
import numpy.random as npr
import LNLayers as lnl


class KMaxNet:
    """KMaxNet provides a variant of the Dynamic Convolutional Neural Network.

    The KMaxNet class implements a modified form of the Dynamic Convolutional
    Neural Network from "A Convolutional Neural Network for Modelling
    Sentences", by Kalchbrenner et. al, appearing in ACL 2014. The network
    architecture implemented here differs in its approach to convolution, which
    has been adapted to be more similar to the way convolutions are applied in
    typical deep neural networks for vision problems.

    The network is implemented as a series of layers. The first network layer
    is always a LUTLayer, which converts lists of index lists into lists of
    matrices, where the rows of each matrix represent a sequence of word
    embeddings. TODO: more docs.
    """
    def __init__(self, opts={}):
        # Validate the given options
        self.net_opts = self.check_opts(opts)
        self.k_max = self.net_opts['k_max']
        self.class_count = self.net_opts['class_count']
        # Construct the layer chain 
        self.all_layers = []
        self.drop_layers = []
        self.conv_layers = []
        self.conv_moms = []
        self.kmax_layers = []
        self.full_layers = []
        self.full_moms = []
        # Prepare the LUTLayer
        lut_opts = self.net_opts['lut_layer']
        self.lut_layer = lnl.LUTLayer(lut_opts['max_key'], lut_opts['embed_dim'])
        self.lut_moms = {}
        self.lut_moms['W'] = np.zeros(self.lut_layer.params['W'].shape)
        self.lut_layer.max_norm = lut_opts['max_norm']
        self.all_layers.append(self.lut_layer)
        # Add a dropout layer between the look-up-table/embedding layer and
        # the first convolutional layer.
        drop_layer = lnl.DropLayer(drop_rate=0.5, in_layer=self.all_layers[-1])
        self.all_layers.append(drop_layer)
        self.drop_layers.append(drop_layer)
        # Prepare the KMaxLayers (and compute some dependent options)
        kml_opts_list = self.net_opts['kmax_layers']
        kml_opts_list[0]['filt_dim'] = lut_opts['embed_dim']
        for kml_opts in kml_opts_list:
            # Make the convolutional part
            conv_layer = lnl.C1DLayer(num_filt=kml_opts['num_filt'], \
                                      filt_len=kml_opts['filt_len'], \
                                      filt_dim=kml_opts['filt_dim'], \
                                      in_layer=self.all_layers[-1])
            conv_layer.max_norm = kml_opts['max_norm']
            self.conv_layers.append(conv_layer)
            moms = {}
            moms['W'] = np.zeros(conv_layer.params['W'].shape)
            moms['b'] = np.zeros(conv_layer.params['b'].shape)
            self.conv_moms.append(moms)
            self.all_layers.append(conv_layer)
            # Make the k-max part
            km_layer = lnl.KMaxLayer(in_layer=self.all_layers[-1])
            self.kmax_layers.append(km_layer)
            self.all_layers.append(km_layer)
        # Add a reshaping layer between the convolutional layer stack and the
        # fully-connected layer stack.
        final_conv = self.conv_layers[-1]
        final_kml = self.kmax_layers[-1]
        reshape_layer = lnl.ML2VMLayer(in_shape=(self.k_max, final_conv.num_filt), \
                                       out_shape=(1, self.k_max * final_conv.num_filt), \
                                       in_layer=final_kml)
        self.all_layers.append(reshape_layer)
        # Prepare the FullLayer options to accomodate preceding layers
        fcl_opts_list = self.net_opts['full_layers']
        fcl_opts_list[0]['in_dim'] = reshape_layer.out_shape[1]
        fcl_opts_list[-1]['out_dim'] = self.class_count
        self.full_layers = []
        # Add a sequence of fully-cnonected mlp layers, with the input to each
        # such layer subject to dropout. Apply ReLU non-linearity to the output
        # of all fully-connected layers except the last one, which serves as
        # the output/prediction layer for this conv-type-thing.
        for (i, fcl_opts) in enumerate(fcl_opts_list):
            # Instantiate and append a drop layer
            drop_layer = lnl.DropLayer(drop_rate=0.5, in_layer=self.all_layers[-1])
            self.all_layers.append(drop_layer)
            self.drop_layers.append(drop_layer)
            # Instantiate and append a fully-connected layer
            fc_layer = lnl.FullLayer(in_dim=fcl_opts['in_dim'], \
                                     out_dim=fcl_opts['out_dim'], \
                                     in_layer=self.all_layers[-1])
            fc_layer.max_norm = fcl_opts['max_norm']
            moms = {}
            moms['W'] = np.zeros(fc_layer.params['W'].shape)
            moms['b'] = np.zeros(fc_layer.params['b'].shape)
            self.full_moms.append(moms)
            self.all_layers.append(fc_layer)
            self.full_layers.append(fc_layer)
            # Add a ReLU layer, unless this is the output layer
            if (i < (len(fcl_opts_list) - 1)):
                relu_layer = lnl.ReluLayer(in_layer=self.all_layers[-1])
                self.all_layers.append(relu_layer)
        return

    def check_opts(self, options={}):
        """Make a valid set of network configuration options."""
        # Check fundamental network parameters
        assert ('class_count' in options)
        if not 'k_max' in options:
            options['k_max'] = 5
        # Check configuration options for LUTLayer (i.e. embedding layer)
        if not 'lut_layer' in options:
            options['lut_layer'] = self.default_lut_opts()
        else:
            options['lut_layer'] = self.default_lut_opts(options['lut_layer'])
        # Check configuration options for KMaxLayers
        if not 'kmax_layers' in options:
            options['kmax_layers'] = self.default_kml_opts()
        else:
            options['kmax_layers'] = self.check_kml_opts(options['kmax_layers'])
        # Check configuration options for FullLayers
        if not 'full_layers' in options:
            options['full_layers'] = self.default_fcl_opts()
        else:
            options['full_layers'] = self.check_fcl_opts(options['full_layers'])
        return options

    def default_lut_opts(self, lut_opts={}):
        """Set simple LUTLayer options."""
        if not ('max_key' in lut_opts):
            lut_opts['max_key'] = 1000
        if not ('embed_dim' in lut_opts):
            lut_opts['embed_dim'] = 25
        if not ('max_norm' in lut_opts):
            lut_opts['max_norm'] = 2.0
        return lut_opts

    def default_kml_opts(self):
        """Set simple KMaxLayer options."""
        kml_opts = []
        kml_1 = {}
        kml_1['num_filt'] = 20
        kml_1['filt_len'] = 5
        kml_1['filt_dim'] = 0
        kml_1['max_norm'] = 2.0
        kml_2 = {}
        kml_2['num_filt'] = 25
        kml_2['filt_len'] = 5
        kml_2['filt_dim'] = kml_1['num_filt']
        kml_2['max_norm'] = 2.0
        return [kml_1, kml_2]

    def check_kml_opts(self, kml_opts={}):
        """Check for a valid set of KMaxLayer options."""
        kml_opts = self.default_kml_opts()
        return kml_opts

    def default_fcl_opts(self):
        """Set simple FullLayer options."""
        fcl_opts = []
        fcl_1 = {}
        fcl_1['in_dim'] = 0
        fcl_1['out_dim'] = 128
        fcl_1['max_norm'] = 2.0
        fcl_2 = {}
        fcl_2['in_dim'] = fcl_1['out_dim']
        fcl_2['out_dim'] = 128
        fcl_2['max_norm'] = 2.0
        fcl_3 = {}
        fcl_3['in_dim'] = fcl_2['out_dim']
        fcl_3['out_dim'] = 0
        fcl_3['max_norm'] = 2.0
        return [fcl_1, fcl_2, fcl_3]

    def check_fcl_opts(self, fcl_opts={}):
        """Check for a valid set of FullLayer options."""
        fcl_opts = self.default_fcl_opts()
        return fcl_opts

    def init_weights(self, w_scale=0.1, b_shift=0.0):
        """Do random initialization of the weights for this network.

        Note: currently all layer types (that have any parameters) have one
              parameter matrix/tensor 'W' and maybe a bias vector 'b'.
        """
        for layer in self.all_layers:
            if layer.has_params:
                if 'W' in layer.params:
                    w_shape = layer.params['W'].shape
                    if len(w_shape) == 2:
                        layer.params['W'] = w_scale * npr.randn(w_shape[0], w_shape[1])
                    elif len(w_shape) == 3:
                        layer.params['W'] = w_scale * npr.randn(w_shape[0], w_shape[1], w_shape[2])
                if 'b' in layer.params:
                    b_shape = layer.params['b'].shape
                    layer.params['b'] = np.zeros(b_shape) + b_shift
        return

    def reset_moms(self, ada_init=0.0, clear_moms=False):
        """Reset all of the adagrad sum-of-squared-gradients accumulators."""
        # Reset adagrad sum-of-squared-gradient accumulators (only for LUT)
        self.lut_moms['W'] = np.zeros(self.lut_moms['W'].shape)
        # Reset momentum accumulators for conv and full layers, if desired
        if clear_moms:
            for moms in self.conv_moms:
                moms['W'] = np.zeros(moms['W'].shape)
                moms['b'] = np.zeros(moms['b'].shape)
            for moms in self.full_moms:
                moms['W'] = np.zeros(moms['W'].shape)
                moms['b'] = np.zeros(moms['b'].shape)
        return

    def feedforward(self, X, use_dropout=False):
        """Feedforward.
        """
        # Set kmaxes dynamically
        km_steps = len(self.kmax_layers)
        for i in range(km_steps):
            km_layer = self.kmax_layers[i]
            km_layer.kmax = []
            a = float(i+1) / float(km_steps)
            for x in X:
                k = int(((1.0 - a) * len(x)) + (a * self.k_max))
                k = max(k, self.k_max)
                km_layer.kmax.append(k)
        # Set dropout rate
        for drop_layer in self.drop_layers:
            if use_dropout:
                drop_layer.set_drop_rate(0.0)
            else:
                drop_layer.set_drop_rate(0.5)
        # Feedforward
        self.all_layers[0].feedforward(X, True)
        return self.all_layers[-1].Y

    def backprop(self, dLdY):
        """Backprop.
        """
        # Backproooooop
        self.all_layers[-1].backprop(dLdY, True)
        return

    def safe_softmax(self, Y):
        """Compute a relatively (numerically) safe softmax."""
        Y_exp = np.exp(Y - np.max(Y, axis=1, keepdims=True))
        Y_sm = Y_exp / np.sum(Y_exp, axis=1, keepdims=True)
        return Y_sm

    def cross_entropy(self, Yh, Y):
        """Cross-entropy loss/grad for predictions Yh and true classes Y."""
        Y_i = np.zeros(Yh.shape)
        Yh_sm = self.safe_softmax(Yh)
        # Compute loss and gradient due to cross-entropy
        L = 0.0
        for (i, y) in enumerate(Y):
            Y_i[i,y] = 1.0
            L = L - np.log(Yh_sm[i,y])
        dLdYh = Yh_sm - Y_i
        # Add a bit of loss and gradient for squared outputs
        L = L + (1e-2 * 0.5 * np.sum(Yh**2.0))
        dLdYh = dLdYh + (1e-2 * Yh)
        return [L, dLdYh]

    def mc_l2_hinge(self, Yh, Y):
        """Multi-class L2 hinge loss (as 1-vs-all)."""
        Yi = -1.0 * np.ones(Yh.shape)
        # Compute loss and gradient due to multi-class L2 hinge loss
        for (i, y) in enumerate(Y):
            Yi[i,y] = 1.0
        margin = 1.0 - (Yi * Yh)
        margin = margin * (margin > 0.0)
        L = 0.5 * np.sum(margin**2.0)
        dLdYh = -(Yi * margin)
        # Add a bit of loss and gradient for squared outputs
        L = L + (1e-3 * 0.5 * np.sum(Yh**2.0))
        dLdYh = dLdYh + (1e-2 * Yh)
        return [L, dLdYh]

    def process_training_batch(self, X, Y, learn_rate, use_dropout=False):
        """Process a batch of phrases Xb with labels Yb."""
        # Run feedforward for the batch
        batch_size = float(len(X))
        Yh = self.feedforward(X, use_dropout)
        # Compute loss and gradient for the network predictions
        loss_info = self.cross_entropy(Yh, Y)
        L = loss_info[0] / batch_size
        dLdYh = loss_info[1] / batch_size
        # Run backprop for the given loss gradients
        self.backprop(dLdYh)
        # LUT layer uses adagrad updates
        p_mom = self.lut_moms['W']
        p_grad = self.lut_layer.param_grads['W']
        p_mom = p_mom + p_grad**2.0
        self.lut_layer.params['W'] -= \
                (learn_rate * (p_grad / (np.sqrt(p_mom) + 1e-2)))
        # Conv layers use standard momentum
        for (layer, layer_moms) in zip(self.conv_layers, self.conv_moms):
            layer_moms['W'] = (0.9 * layer_moms['W']) + layer.param_grads['W']
            layer_moms['b'] = (0.9 * layer_moms['b']) + layer.param_grads['b']
            layer.params['W'] -= learn_rate * layer_moms['W']
            layer.params['b'] -= learn_rate * layer_moms['b']
        # Full layers use standard momentum
        for (layer, layer_moms) in zip(self.full_layers, self.full_moms):
            layer_moms['W'] = (0.9 * layer_moms['W']) + layer.param_grads['W']
            layer_moms['b'] = (0.9 * layer_moms['b']) + layer.param_grads['b']
            layer.params['W'] -= learn_rate * layer_moms['W']
            layer.params['b'] -= learn_rate * layer_moms['b']
        # Reset gradient accumulators and apply norm bounds (via clipping)
        for layer in self.all_layers:
            if layer.has_params:
                layer.clip_params()
                layer.reset_grads(shrink=0.0)
        return L

    def dev_loss(self, X, Y, M, Ws=[]):
        """Compute DEV-regularized loss for inputs X with target outputs Y.
        """
        return 1

    def train(self, X, Y, opts={}):
        """Train this network using observations X/Y and options 'opts'.

        This does SGD.
        """
        print("Training the KMaxNet")
        return 1

##########################
# HAPPY FUN HELPER FUNCS #
##########################

def rand_idx_list(max_idx, samples):
    """Sample "samples" random ints between 0 and "max_idx"."""
    idx_list = [npr.randint(0, high=max_idx) for i in range(samples)]
    return idx_list

if __name__ == '__main__':
    from time import clock as clock
    print("Bonjour, monde!")







##############
# EYE BUFFER #
##############
