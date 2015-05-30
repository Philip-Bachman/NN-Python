import numpy as np
import theano
import theano.tensor as T

# library with theano PDF functions
PI = np.pi
C = -0.5 * np.log(2*PI)

def normal(x, mean, sd):
	return C - T.log(T.abs_(sd)) - ((x - mean)**2 / (2 * sd**2))

def normal2(x, mean, logvar):
	return C - logvar/2 - (x - mean)**2 / (2 * T.exp(logvar))

def laplace(x, mean, logvar):
    sd = T.exp(0.5 * logvar)
    return -(abs(x - mean) / sd) - (0.5 * logvar) - np.log(2)
    
def standard_normal(x):
	return C - (x**2 / 2)

# Centered laplace with unit scale (b=1)
def standard_laplace(x):
	return np.log(0.5) - T.abs_(x)

# Centered student-t distribution
# v>0 is degrees of freedom
# See: http://en.wikipedia.org/wiki/Student's_t-distribution
def studentt(x, v):
	gamma1 = log_gamma_lanczos((v+1)/2.)
	gamma2 = log_gamma_lanczos(0.5*v)
	return gamma1 - 0.5 * T.log(v * PI) - gamma2 - (v+1)/2. * T.log(1 + (x*x)/v)

################################################################
# Funcs for temporary backwards compatibilit while refactoring #
################################################################

def log_prob_bernoulli(p_true, p_approx, mask=None):
    """
    Compute log probability of some binary variables with probabilities
    given by p_true, for probability estimates given by p_approx. We'll
    compute joint log probabilities over row-wise groups. (Theano version).
    """
    if mask is None:
        mask = T.ones((1, p_approx.shape[1]))
    log_prob_1 = p_true * T.log(p_approx+1e-6)
    log_prob_0 = (1.0 - p_true) * T.log((1.0 - p_approx)+1e-6)
    log_prob_01 = log_prob_1 + log_prob_0
    row_log_probs = T.sum((log_prob_01 * mask), axis=1, keepdims=True)
    return row_log_probs

def log_prob_bernoulli_np(p_true, p_approx, mask=None):
    """
    Compute log probability of some binary variables with probabilities
    given by p_true, for probability estimates given by p_approx. We'll
    compute joint log probabilities over row-wise groups. (Numpy version).
    """
    if mask is None:
        mask = np.ones((1, p_approx.shape[1]))
    log_prob_1 = p_true * np.log(p_approx+1e-6)
    log_prob_0 = (1.0 - p_true) * np.log((1.0 - p_approx)+1e-6)
    log_prob_01 = log_prob_1 + log_prob_0
    row_log_probs = np.sum((log_prob_01 * mask), axis=1)
    return row_log_probs

def log_prob_gaussian(mu_true, mu_approx, les_sigmas=1.0, mask=None):
    """
    Compute log probability of some continuous variables with values given
    by mu_true, w.r.t. gaussian distributions with means given by mu_approx
    and standard deviations given by les_sigmas.
    """
    if mask is None:
        mask = T.ones((1, mu_approx.shape[1]))
    ind_log_probs = C - T.log(T.abs_(les_sigmas)) - \
            ((mu_true - mu_approx)**2.0 / (2.0 * les_sigmas**2.0))
    row_log_probs = T.sum((ind_log_probs * mask), axis=1, keepdims=True)
    return row_log_probs

def log_prob_gaussian2(mu_true, mu_approx, log_vars=1.0, mask=None):
    """
    Compute log probability of some continuous variables with values given
    by mu_true, w.r.t. gaussian distributions with means given by mu_approx
    and log variances given by les_logvars.
    """
    if mask is None:
        mask = T.ones((1, mu_approx.shape[1]))
    ind_log_probs = C - (0.5 * log_vars)  - \
            ((mu_true - mu_approx)**2.0 / (2.0 * T.exp(log_vars)))
    row_log_probs = T.sum((ind_log_probs * mask), axis=1, keepdims=True)
    return T.cast(row_log_probs, 'floatX')

def gaussian_kld(mu_left, logvar_left, mu_right, logvar_right):
    """
    Compute KL divergence between a bunch of univariate Gaussian distributions
    with the given means and log-variances.
    We do KL(N(mu_left, logvar_left) || N(mu_right, logvar_right)).
    """
    gauss_klds = 0.5 * (logvar_right - logvar_left + \
            (T.exp(logvar_left) / T.exp(logvar_right)) + \
            ((mu_left - mu_right)**2.0 / T.exp(logvar_right)) - 1.0)
    return gauss_klds

def gaussian_kld_BN(logvar_left, logvar_right):
    """
    Compute KL divergence between a bunch of univariate Gaussian distributions
    with the given means and log-variances.
    We do KL(N(mu_left, logvar_left) || N(mu_right, logvar_right)).
    """
    gauss_klds = 0.5 * (logvar_right - logvar_left + \
            (T.exp(logvar_left) / T.exp(logvar_right)) + \
            (1.0 / T.exp(logvar_right)) - 1.0)
    return gauss_klds

#################################
# Log-gamma function for theano #
#################################
LOG_PI = np.log(PI)
LOG_SQRT_2PI = np.log(np.sqrt(2*PI))
def log_gamma_lanczos(z):
    # reflection formula. Normally only used for negative arguments,
    # but here it's also used for 0 < z < 0.5 to improve accuracy in this region.
    flip_z = 1 - z
    # because both paths are always executed (reflected and non-reflected),
    # the reflection formula causes trouble when the input argument is larger than one.
    # Note that for any z > 1, flip_z < 0.
    # To prevent these problems, we simply set all flip_z < 0 to a 'dummy' value.
    # This is not a problem, since these computations are useless anyway and
    # are discarded by the T.switch at the end of the function.
    flip_z = T.switch(flip_z < 0, 1, flip_z)
    small = LOG_PI - T.log(T.sin(PI * z)) - log_gamma_lanczos_sub(flip_z)
    big = log_gamma_lanczos_sub(z)
    return T.switch(z < 0.5, small, big)
   
## version that isn't vectorised, since g is small anyway
def log_gamma_lanczos_sub(z): #expanded version
    # Coefficients used by the GNU Scientific Library
    g = 7
    p = np.array([0.99999999999980993, 676.5203681218851, -1259.1392167224028,
                  771.32342877765313, -176.61502916214059, 12.507343278686905,
                  -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7])                    
    z = z - 1
    x = p[0]
    for i in range(1, g+2):
        x += p[i]/(z+i)
    t = z + g + 0.5
    return LOG_SQRT_2PI + (z + 0.5) * T.log(t) - t + T.log(x)

############################
# PARZEN DENSITY ESTIMATOR #
############################
import time
import gc

def get_nll(x, parzen, batch_size=100):
    """
    Credit: Yann N. Dauphin
    """

    inds = range(x.shape[0])
    n_batches = int(np.ceil(float(len(inds)) / batch_size))

    times = []
    nlls = []
    for i in range(n_batches):
        begin = time.time()
        nll = parzen(x[inds[i::n_batches]])
        end = time.time()
        times.append(end-begin)
        nlls.extend(nll)
        if i % 10 == 0:
            print i, np.mean(times), np.mean(nlls)
    return np.array(nlls)


def log_mean_exp(a):
    """
    Credit: Yann N. Dauphin
    """
    max_ = a.max(1)
    result = max_ + T.log(T.exp(a - max_.dimshuffle(0, 'x')).mean(1))
    return result 

def theano_parzen(mu, sigma):
    """
    Credit: Yann N. Dauphin
    """
    x = T.matrix()
    mu = theano.shared(mu)
    a = ( x.dimshuffle(0, 'x', 1) - mu.dimshuffle('x', 0, 1) ) / sigma
    E = log_mean_exp(-0.5*(a**2).sum(2))
    Z = mu.shape[1] * T.log(sigma * np.sqrt(np.pi * 2))
    parzen_func = theano.function([x], E - Z)
    return parzen_func

def cross_validate_sigma(samples, data, sigmas, batch_size):
    """
    Find which sigma is best for the Parzen estimator bound.
    """
    lls = []
    best_ll = -1e6
    best_lls = None
    best_sigma = None
    for sigma in sigmas:
        print sigma
        parzen = theano_parzen(samples, sigma)
        tmp = get_nll(data, parzen, batch_size=batch_size)
        sigma_lls = np.asarray(tmp)
        mean_ll = sigma_lls.mean()
        lls.append(mean_ll)
        if (mean_ll > best_ll):
            best_ll = mean_ll
            best_lls = sigma_lls
            best_sigma = sigma
        del parzen
        gc.collect()
    return [best_sigma, best_ll, best_lls]