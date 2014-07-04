from time import clock
import numpy as np
import numpy.random as npr
import gnumpy as gp


#########################
# FULLY-CONNECTED LAYER #
#########################

class GPUFullLayer:
    def __init__(self, in_layer=False, in_dim=0, out_dim=0):
	# Set stuff for managing this type of layer
	self.dim_input = in_dim
	self.dim_output = out_dim
	self.params = {}
	self.params['W'] = gp.randn((in_dim, out_dim))
	self.params['b'] = gp.zeros((1, out_dim))
	self.param_grads = {}
	self.param_grads['W'] = gp.zeros((in_dim, out_dim))
	self.param_grads['b'] = gp.zeros((1, out_dim))
	self.max_norm = 10.0
	self.comp_time = 0.0
	# Set common stuff for all types layers
	self.has_params = True
	self.X = []
	self.Y = []
	self.dLdX = []
	self.dLdY = []
	# Set the input source for this layer, and inform the input source of
	# its output destination
	self.input_layer = in_layer
	if self.input_layer:
            self.input_layer.output_layer = self
	# Don't set self.output_layer, as it will be set by the layer that
	# receives this layer's output as input (see above).
	self.output_layer = False
	return

    def init_params(self, w_scale=0.01, b_scale=0.0):
	"""Randomly initialize the weights in this layer."""
	self.params['W'] = w_scale * gp.randn((self.dim_input, self.dim_output))
	self.param_grads['W'] = gp.zeros((self.dim_input, self.dim_output))
	self.params['b'] = gp.zeros((1, self.dim_output))
	self.param_grads['b'] = gp.zeros((1, self.dim_output))
	return

    def clip_params(self):
	"""Bound L2 (column-wise) norm of self.params['W'] by wt_bnd."""
	EPS = 1e-5
	W = self.params['W']
	# Compute L2 norm of weights inbound to each node in this layer
	w_norms = gp.sqrt(gp.sum(W**2.0,axis=0) + EPS)
	# Compute scales based on norms and the upperbound set by wt_bnd
	w_scales = self.max_norm / w_norms
	mask = (w_scales < 1.0)
	w_scales = (w_scales * mask) + (1.0 - mask)
	w_scales = w_scales[gp.newaxis,:]
	# Rescale weights to meet the bound set by wt_bnd
	W = W * w_scales
	return

    def feedforward(self, input, auto_prop=False):
	"""Run feedforward for this layer.
	"""
        t1 = clock()
	# Cleanup detritus from any previous feedforward
	self._cleanup()
	# Do new feedforward...
	self.X = gp.garray(input)
	self.Y = gp.dot(self.X, self.params['W']) + self.params['b']
	t2 = clock()
        #self.comp_time = self.comp_time + (t2 - t1)
	if auto_prop and self.output_layer:
            self.output_layer.feedforward(self.Y, True)
	return self.Y

    def backprop(self, dLdY_bp, auto_prop=False):
	"""Backprop through this layer.
	"""
        t1 = clock()
	self.dLdY = gp.garray(dLdY_bp)
	# Compute gradient with respect to layer parameters
	dLdW = gp.dot(self.X.T, self.dLdY)
	dLdb = gp.sum(self.dLdY, axis=0)
	dLdb = dLdb[gp.newaxis,:]
	self.param_grads['W'] = self.param_grads['W'] + dLdW
	self.param_grads['b'] = self.param_grads['b'] + dLdb
	# Compute gradient with respect to layer input
	self.dLdX = gp.dot(self.dLdY, self.params['W'].T)
	t2 = clock()
        #self.comp_time = self.comp_time + (t2 - t1)
	if auto_prop and self.input_layer:
            self.input_layer.backprop(self.dLdX, True)
	return self.dLdX

    def safe_softmax(self, Y):
	"""Compute a relatively (numerically) safe softmax."""
	Y_max = gp.max(Y, axis=1)
	Y_max = Y_max[:,gp.newaxis]
	Y_exp = gp.exp(Y - Y_max)
	Y_sum = gp.sum(Y_exp, axis=1)
	Y_sum = Y_sum[:,gp.newaxis]
	Y_sm = Y_exp / Y_sum
	return Y_sm

    def cross_entropy(self, Yh, Y_ind):
	"""Cross-entropy loss/grad for predictions Yh and true classes Y."""
        t1 = clock()
	Y_ind = gp.garray(Y_ind)
	Yh_sm = self.safe_softmax(Yh)
	dLdYh = Yh_sm - Y_ind
        t2 = clock()
	self.comp_time = self.comp_time + (t2 - t1)
	return dLdYh

    def reset_grads(self, shrink=0.0):
	"""Reset the gradient accumulators for this layer."""
	self.param_grads['W'] = shrink * self.param_grads['W']
	self.param_grads['b'] = shrink * self.param_grads['b']
	return

    def _cleanup(self, auto_prop=False):
	"""Cleanup temporary feedforward/backprop stuff."""
	self.X = []
	self.Y = []
	self.dLdX = []
	self.dLdY = []
	return

class FullLayer:
    def __init__(self, in_layer=False, in_dim=0, out_dim=0):
        # Set stuff for managing this type of layer
        self.dim_input = in_dim
        self.dim_output = out_dim
        self.params = {}
        self.params['W'] = npr.randn(in_dim, out_dim)
        self.params['b'] = np.zeros((1, out_dim))
        self.param_grads = {}
        self.param_grads['W'] = np.zeros((in_dim, out_dim))
        self.param_grads['b'] = np.zeros((1, out_dim))
        self.max_norm = 10.0
        self.comp_time = 0.0
        # Set common stuff for all types layers
        self.has_params = True
        self.X = []
        self.Y = []
        self.dLdX = []
        self.dLdY = []
        # Set the input source for this layer, and inform the input source of
        # its output destination
        self.input_layer = in_layer
        if self.input_layer:
            self.input_layer.output_layer = self
        # Don't set self.output_layer, as it will be set by the layer that
        # receives this layer's output as input (see above).
        self.output_layer = False
        return

    def init_params(self, w_scale=0.01, b_scale=0.0):
        """Randomly initialize the weights in this layer."""
        self.params['W'] = w_scale * npr.randn(self.dim_input, self.dim_output)
        self.param_grads['W'] = np.zeros((self.dim_input, self.dim_output))
        self.params['b'] = np.zeros((1, self.dim_output))
        self.param_grads['b'] = np.zeros((1, self.dim_output))
        return

    def clip_params(self):
        """Bound L2 (column-wise) norm of self.params['W'] by wt_bnd."""
        EPS = 1e-5
        W = self.params['W']
        # Compute L2 norm of weights inbound to each node in this layer
        w_norms = np.sqrt(np.sum(W**2.0,axis=0) + EPS)
        # Compute scales based on norms and the upperbound set by wt_bnd
        w_scales = self.max_norm / w_norms
        mask = (w_scales < 1.0)
        w_scales = (w_scales * mask) + (1.0 - mask)
        w_scales = w_scales[np.newaxis,:]
        # Rescale weights to meet the bound set by wt_bnd
        W = W * w_scales
        return

    def feedforward(self, input, auto_prop=False):
        """Run feedforward for this layer.
        """
        t1 = clock()
        # Cleanup detritus from any previous feedforward
        self._cleanup()
        # Do new feedforward...
        self.X = input
        self.Y = np.dot(self.X, self.params['W']) + self.params['b']
        self.dLdY = np.zeros(self.Y.shape)
        t2 = clock()
        self.comp_time = self.comp_time + (t2 - t1)
        if auto_prop and self.output_layer:
            self.output_layer.feedforward(self.Y, True)
        return self.Y

    def backprop(self, dLdY_bp, auto_prop=False):
        """Backprop through this layer.
        """
        t1 = clock()
        self.dLdY = self.dLdY + dLdY_bp
        # Compute gradient with respect to layer parameters
        dLdW = np.dot(self.X.T, self.dLdY)
        dLdb = np.sum(self.dLdY, axis=0, keepdims=True)
        self.param_grads['W'] = self.param_grads['W'] + dLdW
        self.param_grads['b'] = self.param_grads['b'] + dLdb
        # Compute gradient with respect to layer input
        self.dLdX = np.dot(self.dLdY, self.params['W'].T)
        t2 = clock()
        self.comp_time = self.comp_time + (t2 - t1)
        if auto_prop and self.input_layer:
            self.input_layer.backprop(self.dLdX, True)
        return self.dLdX

    def safe_softmax(self, Y):
        """Compute a relatively (numerically) safe softmax."""
        Y_exp = np.exp(Y - np.max(Y, axis=1, keepdims=True))
        Y_sm = Y_exp / np.sum(Y_exp, axis=1, keepdims=True)
        return Y_sm

    def cross_entropy(self, Yh, Y_ind):
        """Cross-entropy loss/grad for predictions Yh and true classes Y."""
        t1 = clock()
        Yh_sm = self.safe_softmax(Yh)
        dLdYh = Yh_sm - Y_ind
        t2 = clock()
        self.comp_time = self.comp_time + (t2 - t1)
        return dLdYh

    def reset_grads(self, shrink=0.0):
        """Reset the gradient accumulators for this layer."""
        self.param_grads['W'] = shrink * self.param_grads['W']
        self.param_grads['b'] = shrink * self.param_grads['b']
        return

    def _cleanup(self, auto_prop=False):
        """Cleanup temporary feedforward/backprop stuff."""
        self.X = []
        self.Y = []
        self.dLdX = []
        self.dLdY = []
        return

#######################
# LOOK-UP TABLE LAYER #
#######################

class LUTLayer:
    def __init__(self, key_count, embed_dim):
        # Set stuff for managing this type of layer
        self.comp_time = 0.0
        self.params = {}
        self.params['W'] = npr.randn(key_count, embed_dim)
        self.param_grads = {}
        self.param_grads['W'] = np.zeros(self.params['W'].shape)
        self.key_count = key_count
        self.embed_dim = embed_dim
        self.max_norm = 10.0
        # Set common stuff for all types layers
        self.has_params = True
        self.X = []
        self.Y = []
        self.dLdX = []
        self.dLdY = []
        # Set the input source for this layer to False
        self.input_layer = False
        # Don't set self.output_layer, as it will be set by the layer that
        # receives this layer's output as input (see above).
        self.output_layer = False
        return

    def init_params(self, w_scale=0.01):
        """Randomly initialize the weights in this layer."""
        self.params['W'] = w_scale * npr.randn(self.key_count, self.embed_dim)
        self.param_grads['W'] = np.zeros((self.key_count, self.embed_dim))
        return

    def clip_params(self):
        """Bound L2 (row-wise) norm of self.params['W'] by wt_bnd."""
        EPS = 1e-5
        W = self.params['W']
        # Compute L2 norm of weights inbound to each node in this layer
        w_norms = np.sqrt(np.sum(W**2.0,axis=1) + EPS)
        # Compute scales based on norms and the upperbound set by wt_bnd
        w_scales = self.max_norm / w_norms
        mask = (w_scales < 1.0)
        w_scales = (w_scales * mask) + (1.0 - mask)
        w_scales = w_scales[:,np.newaxis]
        # Rescale weights to meet the bound set by wt_bnd
        W = W * w_scales
        # Store clipped parameters
        self.params['W'] = W
        return

    def feedforward(self, input, auto_prop=False):
        """Run feedforward for this layer.

        The input passed to feedforward here should be either a single list
        of integer indices into the look-up table or a list of lut index lists.
        """
        # Cleanup detritus from any previous feedforward
        t1 = clock()
        self._cleanup()
        #
        self.X = input
        # Use look-up table to generate the desired sequences
        self.Y = self.params['W'][self.X,:]
        #
        t2 = clock()
        self.comp_time = self.comp_time + (t2 - t1)
        if auto_prop and self.output_layer:
            self.output_layer.feedforward(self.Y, True)
        return self.Y

    def backprop(self, dLdY_bp, auto_prop=False):
        """Backprop through this layer.
        """
        t1 = clock()
        self.dLdY = dLdY_bp
        # Add the gradients to the gradient accumulator
        for i in range(self.dLdY.shape[0]):
            self.param_grads['W'][self.X[i],:] += self.dLdY[i,:]
        t2 = clock()
        self.comp_time = self.comp_time + (t2 - t1)
        return self.param_grads['W']

    def reset_grads(self, shrink=0.0):
        """Reset the gradient accumulators for this layer."""
        self.param_grads['W'] = shrink * self.param_grads['W']
        return

    def _cleanup(self):
        """Cleanup temporary feedforward/backprop stuff."""
        self.X = []
        self.Y = []
        self.dLdX = []
        self.dLdY = []
        return

#################
# DROPOUT LAYER #
#################

class DropLayer:
    def __init__(self, drop_rate=0.0, in_layer=False):
        # Set stufff required for managing this type of layer
        self.dYdX = []
        self.drop_rate = drop_rate
        self.drop_scale = 1.0 / (1.0 - drop_rate)
        # Set stuff common to all layer types
        self.has_params = False
        self.X = []
        self.Y = []
        self.dLdX = []
        self.dLdY = []
        # Set the input source for this layer, and inform the input source of
        # its output destination
        self.input_layer = in_layer
        if self.input_layer:
            self.input_layer.output_layer = self
        # Don't set self.output_layer, as it will by the layer that receives
        # this layer's output as input (see above).
        self.output_layer = []
        return

    def set_drop_rate(self, drop_rate):
        """Set the drop rate for this drop layer."""
        self.drop_rate = drop_rate
        self.drop_scale = 1.0 / (1.0 - drop_rate)
        return

    def feedforward(self, input, auto_prop=False):
        """Perform feedforward through this layer.
        """
        # Cleanup detritus from any previous feedforward
        self._cleanup()
        # Do new feedforward
        self.X = input
        if (type(self.X) is list):
            # Respond to a list list of gparrays
            self.dYdX = []
            self.Y = []
            self.dLdX = []
            for x in self.X:
                drop_mask = self.drop_scale * \
                        (npr.rand(x.shape[0], x.shape[1]) > self.drop_rate)
                self.dYdX.append(drop_mask)
                self.Y.append(drop_mask * x)
        else:
            # Respond to a single gparray
            drop_mask = self.drop_scale * \
                    (npr.rand(self.X.shape[0], self.X.shape[1]) > self.drop_rate)
            self.dYdX = drop_mask
            self.Y = drop_mask * self.X
        if auto_prop and self.output_layer:
            self.output_layer.feedforward(self.Y, True)
        return self.Y

    def backprop(self, dLdY_bp, auto_prop=False):
        """Perform backprop through this layer.
        """
        assert (type(dLdY_bp) is type(self.Y))
        if (type(dLdY_bp) is list):
            # Respond to a list of gparrays
            self.dLdX = []
            for (i, dldy_bp) in enumerate(dLdY_bp):
                self.dLdX.append(dldy_bp * self.dYdX[i])
        else:
            # Respond to a single gparray
            self.dLdX = dLdY_bp * self.dYdX
        if auto_prop and self.input_layer:
            self.input_layer.backprop(self.dLdX, True)
        return self.dLdX

    def _cleanup(self):
        """Clear all temp variables for this layer."""
        self.X = []
        self.Y = []
        self.dYdX = []
        self.dLdX = []
        return



#######################
# RANDOM KNICK-KNACKS #
#######################

def rand_idx_list(max_idx, samples):
    """Sample "samples" random ints between 0 and "max_idx"."""
    idx_list = npr.randint(0, high=max_idx, size=(samples,))
    return idx_list


if __name__ == '__main__':
    batch_count = 20
    batch_size = 1000
    word_count = 50000
    embed_dim = 200

    # Create a lookup table for word representations
    wordLUT = LUTLayer(word_count, embed_dim)
    # Create a full/softmax layer for classification
    classLayer = GPUFullLayer(in_layer=False, in_dim=embed_dim, out_dim=word_count)

    # Initialize params for the LUT and softmax classifier
    wordLUT.init_params(0.05)
    classLayer.init_params(0.05)

    print("Processing batches:")
    t1 = clock()
    for b in range(batch_count):
        anchor_idx = rand_idx_list(word_count, batch_size)
        context_idx = rand_idx_list(word_count, batch_size)
        Xb = wordLUT.feedforward(anchor_idx)
        Yb = classLayer.feedforward(Xb)
        Y_ind = np.zeros(Yb.shape)
        Y_ind[np.arange(Y_ind.shape[0]),context_idx] = 1.0
        dLdYb = classLayer.cross_entropy(Yb, Y_ind)
        dLdXb = gp.as_numpy_array(classLayer.backprop(dLdYb))
        wordLUT.backprop(dLdXb)
        print(".")
    t2 = clock()
    e_time = t2 - t1
    print("Word count: {0:d}, word dim: {1:d}".format(word_count, embed_dim))
    print("elapsed time: {0:.4f}".format(e_time))
    print("lut time: {0:.4f}".format(wordLUT.comp_time))
    print("softmax time: {0:.4f}".format(classLayer.comp_time))
    print("Words per second: {0:.4f}".format((1.0*batch_count*batch_size /
        e_time)))


##############
# EYE BUFFER #
##############
