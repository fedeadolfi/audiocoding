"""! This module provides functions to perform a matching pursuit
of kernels on a 1-d signal """
import numpy as np
import matplotlib.pyplot as plt

def max_onset(kernel, signal):
    """ find for which translation kernel match signal the best"""
    maximum = -1.
    for onset in range(-kernel.size + 1, signal.size):
        ker = np.array(kernel)
        ons = onset
        if onset < 0:
            ker = ker[-onset:]
            ons = 0
        elif onset + ker.size >= signal.size:
            ker = ker[:(signal.size - onset)]
        inner = np.dot(signal[ons:ons + ker.size], ker)
        if np.abs(inner) > maximum:
            maximum = np.abs(inner)
            out_onset = onset
            spike = inner
    return out_onset, spike

def matching_pursuit(ker_dic, signal, threshold=0.1):
    """! approximate signal with kernels dictionary
    @param ker_dic a list of numpy array, each element being a kernel
    @param signal a numpy array, the signal to be approximated
    @param threshold the minimum spike value
    @return spikes, a list of list containing the spikes of each kernels,
            tau, the times were corresponding spiker are fired
            rest, the approximation - the input signal"""
    rest = np.array(signal)
    current_spike = threshold + 1
    spikes = {}
    tau = {}
    spikes_counter = 0
    while np.abs(current_spike) > threshold and spikes_counter < signal.size:
        maximum = -1.
        for i, kernel in enumerate(ker_dic):
            index, spike = max_onset(kernel, rest)
            if np.abs(spike) > maximum:
                current_kernel_ind = i
                current_spike = spike
                current_tau = index
                maximum = np.abs(spike)
        if current_kernel_ind not in spikes:
            spikes[current_kernel_ind] = []
            tau[current_kernel_ind] = []
        spikes[current_kernel_ind].append(current_spike)
        tau[current_kernel_ind].append(current_tau)
        substract_kernel(rest, ker_dic[current_kernel_ind], current_spike, current_tau)
        spikes_counter += 1
        #print "kernel {} - pos {} ; rest energy: {}".format(current_kernel_ind, current_tau,
        #                                                    np.sum(rest**2))
    return spikes, tau, rest

def substract_kernel(rest, kernel, spike, onset):
    """! Substract a kernel from rest at a certain onset """
    kernel = np.array(kernel)
    if onset < 0:
        kernel = kernel[-onset:]
        onset = 0
    if onset + kernel.size >= rest.size:
        kernel = kernel[:rest.size - onset]
    rest[onset:onset + kernel.size] -= spike * kernel / np.sum(kernel**2)

def reconstruct(ker_dic, tau, spikes, length):
    """! reconstruct an encoded signal """
    out = np.zeros(length)
    for ker_ind in spikes:
        ker_spikes = spikes[ker_ind]
        for s_index, spike in enumerate(ker_spikes):
            onset = tau[ker_ind][s_index]
            kernel = np.array(ker_dic[ker_ind])
            if onset < 0:
                kernel = kernel[-onset:]
                onset = 0
            if onset + kernel.size >= length:
                kernel = kernel[:length - onset]
            out[onset:onset + kernel.size] += spike*kernel / np.sum(kernel**2)
    return out

if __name__ == "__main__":
    #X_SIG = np.random.randn(100)
    X_SIG = np.sin(np.arange(100)/5.)
    KER = []
    for k in range(10):
        KER.append(np.random.randn(10))
        #KER.append(np.sin(np.arange(20)*k/15.))
    SPIKES, TAU, REST = matching_pursuit(KER, X_SIG)
    X_REC = reconstruct(KER, TAU, SPIKES, X_SIG.size)
    plt.plot(X_SIG)
    plt.plot(X_REC)
    plt.plot(REST)
    plt.show()
    #kernel_size = 10
    #KERNEL_POS = 20
    #x1 = np.ones(kernel_size)
    #x2 = 0.8*np.concatenate([np.zeros(KERNEL_POS), np.ones(kernel_size), np.zeros(kernel_size)])
    #conv = np.convolve(x1, x2, "same")
    #print conv
    #print np.argmax(conv)
    #DETECTED = np.argmax(conv) - kernel_size/2
    #x2[DETECTED:DETECTED+kernel_size] -= x1
    #print x2
