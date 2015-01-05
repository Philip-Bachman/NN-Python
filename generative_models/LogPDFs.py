import numpy as np
import theano
import theano.tensor as T
from NetLayers import safe_log

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
    compute joint log probabilities over row-wise groups.
    """
    if mask is None:
        mask = T.ones((1, mu_approx.shape[1]))
    log_prob_1 = p_true * safe_log(p_approx)
    log_prob_0 = (1.0 - p_true) * safe_log(1.0 - p_approx)
    log_prob_01 = log_prob_1 + log_prob_0
    row_log_probs = T.sum((log_prob_01 * mask), axis=1, keepdims=True)
    return row_log_probs

#logpxz = -0.5*np.log(2 * np.pi) - log_sigma_decoder - (0.5 * ((x - mu_decoder) / T.exp(log_sigma_decoder))**2)

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

def log_prob_gaussian2(mu_true, mu_approx, les_logvars=1.0, mask=None):
    """
    Compute log probability of some continuous variables with values given
    by mu_true, w.r.t. gaussian distributions with means given by mu_approx
    and log variances given by les_logvars.
    """
    if mask is None:
        mask = T.ones((1, mu_approx.shape[1]))
    log_sigmas = les_logvars / 2.0
    ind_log_probs = C - log_sigmas  - \
            (0.5 * ((mu_true - mu_approx) / T.exp(log_sigmas))**2.0)
    row_log_probs = T.sum((ind_log_probs * mask), axis=1, keepdims=True)
    return row_log_probs

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
