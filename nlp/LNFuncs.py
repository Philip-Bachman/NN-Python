import numpy as np
import numpy.random as npr
import gnumpy as gp

##############################################################################
# The functions below (i.e. ----_trans(X, mode)) provide a collection of     #
# non-linearities ready for use by LNLayer objects. When used in feedforward #
# mode (i.e. mode=='ff'), X will be transformed by the given non-linearity.  #
# When used in backprop mode (i.e. mode=='bp'), X is assumed to be a dict    #
# containing the values needed for backpropping a gradient dLdA on the       #
# post-transform activation A produced by feedforward. For all transforms    #
# except norm_trans, this requires X['A'] = A and X['dLdA'] = dLdA. For      #
# norm_trans, we also need X['X'] = X, where X is the pre-activation matrix  #
# that was feedforwarded to produce A (i.e. A = norm_trans(X, 'ff')).        #
#                                                                            #
# For use with gnumpy: these functions all return, and operate fully in      #
# terms of garrays (i.e. they stay on GPU), if inputs are garray-based.      #
##############################################################################

def line_trans(X, mode='ff'):
    """Compute feedforward and backprop for line yeslinearity."""
    if (mode == 'ff'):
        F = X
    if (mode == 'bp'):
        F = X['dLdA']
    return gp.garray(F)

def relu_trans(X, mode='ff'):
    """Compute feedforward and backprop for ReLu nonlinearity."""
    if (mode == 'ff'):
        F = X * (X > 0.0)
    if (mode == 'bp'):
        F = (X['A'] > 0.0) * X['dLdA']
    return F

def rehu_trans(X, mode='ff'):
    """Compute feedforward and backprop for ReHu nonlinearity."""
    if (mode == 'ff'):
        M_quad = (X > 0.0)
        M_line = (X > 0.5)
        M_quad = M_quad - M_line
        F = (M_line * (X - 0.25)) + (M_quad * X**2.0)
    if (mode == 'bp'):
        M_quad = (X['A'] < 0.25)
        M_line = 1.0 - M_quad
        F = (2.0 * M_quad * gp.sqrt(X['A'])) + M_line
        F = F * X['dLdA']
    return F

def tanh_trans(X, mode='ff'):
    """Compute feedforward and backprop for tanh nonlinearity."""
    if (mode == 'ff'):
        F = gp.tanh(X)
    if (mode == 'bp'):
        F = (1.0 - X['A']**2.0) * X['dLdA']
    return F

def norm_trans(X, mode='ff'):
    """Compute feedforward and backprop for unit-normalization."""
    EPS = 0.00000001
    if (mode == 'ff'):
        N = gp.sqrt(gp.sum(X**2.0, axis=1) + EPS)
        N = N[:,gp.newaxis]
        F = X / N
    if (mode == 'bp'):
        N = gp.sqrt(gp.sum(X['X']**2.0, axis=1) + EPS)
        N = N[:,gp.newaxis]
        V = X['dLdA'] * X['X']
        V = gp.sum(V, axis=1)
        V = V[:,gp.newaxis]
        F = (X['dLdA'] / N) - (X['A'] * (V / (N**2.0)))
    return F

def kspr_trans(X, mode='ff', k=25):
    """Compute feedforward and backprop for k-sparse activation."""
    if (mode == 'ff'):
        X_abs = abs(X)
        X_abs.sort(axis=1)
        thresh = X_abs[:,-k].reshape((X.shape[0],1))
        F = X * (abs(X) >= thresh)
    if (mode == 'bp'):
        F = (abs(X['A']) > 0) * X['dLdA']
    return F

###############################################################################
# The function below computes the Dropout Ensemble Variance loss and its      #
# gradients. The DEV regularizer penalizes variance in the activations of the #
# dropout ensemble, after applying some transformation to the activations.    #
#                                                                             #
# For use with gnumpy: if all arrays A[i] passed into dev_loss() are garrays, #
# then all computations will stay on GPU, and returned dict will contain a    #
# scalar for 'L' and a garray for 'dLdA'.                                     #
###############################################################################

def dev_loss(A, dev_type=1, use_shepherd=0):
    """DEV regularizer, cool stuff."""
    b_reps = len(A)
    b_obs = A[0].shape[0]
    At = []
    for i in range(b_reps):
        if (dev_type == 1):
            At.append(norm_trans(A[i],'ff'))
        elif (dev_type == 2):
            At.append(tanh_trans(A[i],'ff'))
        elif (dev_type == 3):
            At.append(line_trans(A[i],'ff'))
        else:
            raise Exception('Unknown DEV types.')
    # Compute the mean activations for this ensemble sample
    N = float(A[0].shape[1])
    n = float(b_reps)
    m = float(b_obs * b_reps * N)
    Am = gp.zeros(At[0].shape)
    if (use_shepherd != 1):
        for i in range(b_reps):
            Am = Am + At[i]
        Am = Am / float(b_reps)
    else:
        Am = At[0]
    # Compute difference from mean of each set of droppy activations
    Ad = [(At[i] - Am) for i in range(b_reps)]
    L = sum([gp.sum(ad**2.0) for ad in Ad]) / m
    dLdA = []
    if (use_shepherd != 1):
        Add = gp.zeros(At[0].shape)
        for i in range(b_reps):
            Add = Add + Ad[i]
        for i in range(b_reps):
            dLdA.append(-(2.0/m) * ((((1.0/n) - 1.0) * Ad[i]) + \
                    ((1.0/n) * (Add - Ad[i]))))
    else:
        for i in range(b_reps):
            if (i == 0):
                dLdA.append(gp.zeros(Ad[0].shape))
            else:
                dLdA.append((2.0 / m) * Ad[i])
        for i in range(1,b_reps):
            dLdA[0] = dLdA[0] - dLdA[i]
    # Backpropagate gradient on variance through the desired transform
    for i in range(b_reps):
        BP = {'X': A[i], 'A': At[i], 'dLdA': dLdA[i]}
        if (dev_type == 1):
            dLdA[i] = norm_trans(BP, 'bp')
        elif (dev_type == 2):
            dLdA[i] = tanh_trans(BP, 'bp')
        elif (dev_type == 3):
            dLdA[i] = line_trans(BP, 'bp')
    return {'L': L, 'dLdA': dLdA}


#############################################################################
# The functions below are classification/regression losses to be applied to #
# activations output by a network's final layer (most typically).           #
#                                                                           #
# For use with gnumpy: when the Yh and Y passed to these fns are both       #
# garrays, then all computations and returned values remain on GPU.         #
#############################################################################

def loss_mclr(Yh, Y):
    """Compute mutinomial logistic regression loss for Yh, w.r.t. Y.

    Values in Yh should probably be network outputs, and each row in Y must
    be a +1/-1 indicator vector for the target class of a row in Yh.
    """
    obs_count = float(Y.shape[0])
    # Get boolean mask for each observation's target class
    cl_mask = (Y > 0.0)
    # Compute softmax distribution tranform of Yh
    sm_sum = gp.sum(gp.exp(Yh), axis=1)
    P = gp.exp(Yh) / sm_sum[:,gp.newaxis]
    dL = (P - cl_mask) / obs_count
    logP = gp.log(P) * cl_mask
    L = -gp.sum(logP) / obs_count
    return {'L': L, 'dL': dL}

def loss_mcl2h(Yh, Y):
    """Compute one-vs-all L2 hinge loss for Yh, w.r.t. Y.

    Values in Yh should probably be network outputs, and each row in Y must
    be a +1/-1 indicator vector for the target class of a row in Yh.
    """
    obs_count = float(Y.shape[0])
    # margin_lapse gives [1 - f(x)*y(x)]_+ (i.e. hinge loss, for y(x) = +/-1)
    # note: margin_lapse is strictly non-negative
    margin_lapse = 1.0 - (Y * Yh)
    margin_lapse = (margin_lapse * (margin_lapse > 0.0))
    L = gp.sum(margin_lapse**2.0) / obs_count
    dL = ((-2.0 / obs_count) * (Y * margin_lapse))
    return {'L': L, 'dL': dL}

def loss_lsq(Yh, Y):
    """Compute least-squares (i.e. typical regression) loss for Yh w.r.t. Y.

    Values in Yh should probably be network outputs, and each row in Y must
    give the real-valued target outputs for each observation. Vector-valued
    target outputs are handled just fine.
    """
    obs_count = float(Y.shape[0])
    R = Yh - Y
    L = gp.sum(R**2.0) / obs_count
    dL = (2.0 / obs_count) * R
    return {'L': L, 'dL': dL}

def loss_hsq(Yh, Y, delta=0.5):
    """Compute Huberized least-squares loss for Yh w.r.t. Y.

    Values in Yh should probably be network outputs, and each row in Y must
    give the real-valued target outputs for each observation. Vector-valued
    target outputs are handled just fine.
    """
    obs_count = float(Y.shape[0])
    R = Yh - Y
    mask =(gp.abs(R) < delta)
    L = (mask * R**2.0) + ((1 - mask) * ((mask * R) - delta**2.0))
    L = gp.sum(L) / obs_count
    dL = ((2.0*delta) / obs_count) * ((mask * R) + ((1 - mask) * gp.sign(R)))
    return {'L': L, 'dL': dL}

######################################################################
# Basic function for appending a (scaled) bias column to some inputs #
######################################################################

def bias(X, bias_val=1.0):
    """Append a bias columns of magnitude bias_val to X."""
    Xb = gp.concatenate((X, gp.ones((X.shape[0],1))), axis=1)
    return Xb

def unbias(X):
    """Remove bias column from X."""
    Xnb = X[:,:-1]
    return Xnb

############################################################################## 
# The functions below are for changing between indicator/categorical classes #
#                                                                            #
# Note: My neural network implementations expect to be given +1/-1 indicator #
#       matrices when training, so the converters below should probably just #
#       be used once, on the joint collection of all class labels for train  #
#       and test data. Indicator-form class labels can then just be saved    #
#       alongside the observations, for easy reuse.                          #
#                                                                            #
# For gnumpy: these are not expected to be called often by speed-critical    #
# code. class_cats returns a garray, as does class_inds, for all inputs.     #
############################################################################## 

def class_cats(Yi):
    """Change +1/-1 class indicator matrix to categorical vector."""
    if not gp.is_garray(Yi):
        Yi = gp.garray(Yi)
    Yc = Yi.argmax(axis=1)
    return Yc

def class_inds(Yc, class_idx=np.array([0])):
    """Change class categorical vector to +1/-1 indicator matrix."""
    Yc = np.int32(Yc)
    if (class_idx.size == 0):
        class_idx = np.unique(Yc)
    class_idx = np.int32(class_idx)
    class_count = class_idx.size
    Yi = -np.ones((Yc.size, class_count))
    # Simple elementwise-scan; this doesn't need to be fast.
    for o in range(Yi.shape[0]):
        for c in range(Yi.shape[1]):
            if (Yc[o] == class_idx[c]):
                Yi[o,c] = 1
    return gp.garray(Yi)

##########################################################
# The functions below are basic convenience functions... #
#                                                        #
# trte_split returns values on the GPU, and sample_obs   #
# is written to keep values on the GPU, and to avoid     #
# allocating new memory (it requires the receiving       #
# garrays to be passed as arguments).                    #
##########################################################

def trte_split(X, Y, tr_frac):
    """Split the data in X/Y into training and testing portions."""
    if gp.is_garray(X):
        X = X.as_numpy_array()
    else:
        X = np.array(X)
    if gp.is_garray(Y):
        Y = Y.as_numpy_array()
    else:
        Y = np.array(Y)
    obs_count = X.shape[0]
    obs_dim = X.shape[1]
    tr_count = round(tr_frac * obs_count)
    te_count = obs_count - tr_count
    Xtr = np.zeros((tr_count, X.shape[1]))
    Ytr = np.zeros((tr_count, Y.shape[1]))
    Xte = np.zeros((te_count, X.shape[1]))
    Yte = np.zeros((te_count, Y.shape[1]))
    idx = npr.permutation(range(obs_count))
    # Basic manual iteration
    for i in range(obs_count):
        if (i < tr_count):
            Xtr[i,:] = X[idx[i],:]
            Ytr[i,:] = Y[idx[i],:]
        else:
            Xte[(i - tr_count),:] = X[idx[i],:]
            Yte[(i - tr_count),:] = Y[idx[i],:]
    return [gp.garray(Xtr), gp.garray(Ytr), gp.garray(Xte), gp.garray(Yte)]

def sample_obs(X, Y, Xs, Ys):
    """Sample a random subset of the observations in X/Y.

    This requires receiving arrays Xs and Ys, with matching row counts and
    column dimensions suited to X/Y. Sampled observations are written into
    Xs/Ys, to allow reuse of GPU-resident garrays.
    """
    if (Xs.shape[0] != Ys.shape[0]):
        raise Exception('Mismatched sample receiving arrays.')
    if ((Xs.shape[1] != X.shape[1]) or (Ys.shape[1] != Y.shape[1])):
        raise Exception('Wismatched sample receiving arrays.')
    samp_count = Xs.shape[0]
    obs_count = X.shape[0]
    idx = npr.permutation(range(obs_count))
    idx = [int(i) for i in idx]
    for i in range(samp_count):
        Xs[i,:] = X[idx[i%obs_count],:]
        Ys[i,:] = Y[idx[i%obs_count],:]
    return


##############################################################################
# The function below is for checking and setting default training parameters #
##############################################################################

def check_opts(opts={}):
    """Check the SGD training options in opts, and set some defaults."""
    if not opts.has_key('rounds'):
        opts['rounds'] = 50000
    if not opts.has_key('start_rate'):
        opts['start_rate'] = 0.1
    if not opts.has_key('decay_rate'):
        opts['decay_rate'] = 0.1**(1.0 / opts['rounds'])
    if not opts.has_key('momentum'):
        opts['momentum'] = 0.8
    if not opts.has_key('batch_size'):
        opts['batch_size'] = 100
    if not opts.has_key('dev_reps'):
        opts['dev_reps'] = 4
    if not opts.has_key('do_validate'):
        opts['do_validate'] = 0
    if (opts['do_validate'] == 1):
        if not (opts.has_key('Xv') and opts.has_key('Yv')):
            raise Exception('Validation requires validation set.')
    opts['momentum'] = min(1, max(opts['momentum'], 0))
    return opts


###############################################################
# Basic testing, to see the functions aren't _totally_ broken #
###############################################################

if __name__ == '__main__':
    from time import clock
    obs_count = 1000
    class_count = 100
    Y = gp.sign(gp.randn((obs_count, class_count)))
    Yh = gp.randn((obs_count, class_count))
    # Check that loss functions won't crash
    t1 = clock()
    print "Computing all losses 10 times:",
    for i in range(10):
        loss_info = loss_mclr(Yh, Y)
        loss_info = loss_mcl2h(Yh, Y)
        loss_info = loss_lsq(Yh, Y)
        loss_info = loss_hsq(Yh, Y)
        print ".",
    print " "
    t2 = clock()
    print "Total time: " + str(t2 - t1)
    #  Check that class representation converters won't crash
    obs_count = 20
    class_count = 4
    class_idx = np.array(range(class_count))
    Yc = npr.randint(0,class_count,(obs_count, 1))
    Ym = class_inds(Yc, class_idx)
    print "Class vector: "
    print str(Yc)
    print "Class indicator matrix: "
    print str(Ym)




##############
# EYE BUFFER #
##############
