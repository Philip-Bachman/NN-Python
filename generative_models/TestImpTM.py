##################################################################
# Code for testing the variational Multi-Stage Generative Model. #
##################################################################

# basic python
import numpy as np
import numpy.random as npr
import cPickle

# theano business
import theano
import theano.tensor as T

# phil's sweetness
import utils
from GPSImputer import TemplateMatchImputer
from load_data import load_udm, load_mnist, load_tfd, load_svhn_gray
from HelperFuncs import construct_masked_data, shift_and_scale_into_01, \
                        row_shuffle, to_fX

RESULT_PATH = "IMP_MNIST_TM/"

###############################
###############################
## TEST GPS IMPUTER ON MNIST ##
###############################
###############################

def test_mnist(occ_dim=15, drop_prob=0.0):
    #########################################
    # Format the result tag more thoroughly #
    #########################################
    dp_int = int(100.0 * drop_prob)
    result_tag = RESULT_PATH + "TM_OD{}_DP{}".format(occ_dim, dp_int)

    ##########################
    # Get some training data #
    ##########################
    rng = np.random.RandomState(1234)
    dataset = 'data/mnist.pkl.gz'
    datasets = load_udm(dataset, as_shared=False, zero_mean=False)
    Xtr = datasets[0][0]
    Xva = datasets[1][0]
    Xtr = to_fX(shift_and_scale_into_01(Xtr))
    Xva = to_fX(shift_and_scale_into_01(Xva))
    tr_samples = Xtr.shape[0]
    va_samples = Xva.shape[0]
    batch_size = 200
    batch_reps = 1
    all_pix_mean = np.mean(np.mean(Xtr, axis=1))
    data_mean = to_fX(all_pix_mean * np.ones((Xtr.shape[1],)))

    TM = TemplateMatchImputer(x_train=Xtr, x_type='bernoulli')

    log_name = "{}_RESULTS.txt".format(result_tag)
    out_file = open(log_name, 'wb')

    Xva = row_shuffle(Xva)
    # record an estimate of performance on the test set
    xi, xo, xm = construct_masked_data(Xva, drop_prob=drop_prob, \
                                       occ_dim=occ_dim, data_mean=data_mean)
    result = TM.best_match_nll(xo, xm)
    match_on_known = np.mean(result[0])
    match_on_unknown = np.mean(result[1])
    str0 = "Test 1:"
    str1 = "    match on known   : {}".format(match_on_known)
    str2 = "    match on unknown : {}".format(match_on_unknown)
    joint_str = "\n".join([str0, str1, str2])
    print(joint_str)
    out_file.write(joint_str+"\n")
    out_file.flush()
    out_file.close()
    return


if __name__=="__main__":
    #########
    # MNIST #
    #########
    test_mnist(occ_dim=0, drop_prob=0.6)
    test_mnist(occ_dim=0, drop_prob=0.7)
    test_mnist(occ_dim=0, drop_prob=0.8)
    test_mnist(occ_dim=0, drop_prob=0.9)
    test_mnist(occ_dim=14, drop_prob=0.0)
    test_mnist(occ_dim=16, drop_prob=0.0)
