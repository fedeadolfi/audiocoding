"""! This module enables to compute a faster convolutional matching pursuit,
keeping track of the cross-correlations """
import numpy as np
from matchingpursuit import reconstruct
import matplotlib.pyplot as plt

np.seterr(invalid="raise")

def gram_correlations(ker_dic):
    """ computes al the cross correlations between the kernels """
    gram = {}
    for i, phii in enumerate(ker_dic):
        for j, phij in enumerate(ker_dic):
            gram[i, j] = np.correlate(phii, phij, "full")
    return gram

def sig_correlations(sig, ker_dic):
    """ computes the cross correlation between the input signal and all the kernels"""
    corr = []
    for ker in ker_dic:
        corr.append(np.correlate(sig, ker, "full"))
    return corr

def best_fit(corr):
    """ returns the best matching kernel index,
    and relatively to the full cross-correlation.
    returns also the spike """
    maximum = -1
    for i, local_c in enumerate(corr):
        abs_c = np.abs(local_c)
        argmax_c = np.argmax(abs_c)
        max_c = abs_c[argmax_c]
        if max_c > maximum:
            maximum = max_c
            best_ker = i
            best_tau = argmax_c
            best_spike = local_c[argmax_c]
    #print best_spike
    return best_ker, best_tau, best_spike

def substract_kernel(rest, kernel, spike, onset):
    """! Substract a kernel from rest at a certain onset """
    kernel = np.array(kernel)
    if onset < 0:
        kernel = kernel[-onset:]
        onset = 0
    if onset + kernel.size >= rest.size:
        #print "this is imposibru"
        kernel = kernel[:rest.size - onset]
    #rest[onset:onset + kernel.size] -= spike * kernel / np.sum(kernel**2)
    rest[onset:onset + kernel.size] -= spike * kernel

def update_correlations(corr, gram, spike, ker_ind, translation, kernel, rest):
    """update the correlation of the rest with the kernels """
    for i, cor in enumerate(corr):
        if translation < 0 or translation + gram[ker_ind, i].size >= cor.size:
            corr[i] = np.correlate(rest, kernel[i], "full")
        else:
            substract_kernel(cor, gram[ker_ind, i], spike, translation)
        #substract_kernel(cor, gram[i, ker_ind], spike, translation)

def fast_conv_mp(ker_dic, signal, threshold=0.1, criterion="spike"):
    """ matching pursuit, but keeping track of correlations so that it's faster """
    for ker in ker_dic:
        ker /= np.sqrt(np.sum(ker**2))
    rest = np.array(signal)
    current_spike = threshold + 1
    spikes = {}
    tau = {}
    corr = sig_correlations(signal, ker_dic)
    gram = gram_correlations(ker_dic)
    counter = 0
    while (criterion == "spike" and abs(current_spike) > threshold) or (criterion == 'sparse'
                                                                        and counter < threshold):
        current_ker_ind, current_rel_trans, current_spike = best_fit(corr)
        if np.isinf(current_spike):
            print "INFF"
        current_abs_trans = current_rel_trans - ker_dic[current_ker_ind].size + 1
        if current_ker_ind not in spikes:
            spikes[current_ker_ind] = []
            tau[current_ker_ind] = []
        spikes[current_ker_ind].append(current_spike)
        tau[current_ker_ind].append(current_abs_trans)
        substract_kernel(rest, ker_dic[current_ker_ind], current_spike, current_abs_trans)
        update_correlations(corr, gram, current_spike, current_ker_ind, current_abs_trans,
                            ker_dic, rest)
        counter += 1
        if criterion == "sparse" and counter % 10 == 0:
            print "{}/{}".format(counter, threshold)
    return spikes, tau, rest

if __name__ == "__main__":
    #X_SIG = np.random.randn(100)
    X_SIG = np.sin(np.arange(100)/5.)
    KER = []
    for k in range(10):
        KER.append(np.random.randn(10))
        #KER.append(np.sin(np.arange(20)*k/15.))
    SPIKES, TAU, REST = fast_conv_mp(KER, X_SIG)
    X_REC = reconstruct(KER, TAU, SPIKES, X_SIG.size)
    plt.plot(X_SIG)
    plt.plot(X_REC)
    plt.plot(REST)
    plt.show()
