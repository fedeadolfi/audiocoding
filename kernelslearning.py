"""! The modules which provides functions to learn the kernels """
import numpy as np
import matplotlib.pyplot as plt
from matchingpursuit import matching_pursuit, reconstruct

def initialize_kernels(length, number):
    """! Initialize random kernels of a certain length, zero-padded on both ends
    by 1/10 its total length """
    out = []
    for _ in range(number):
        out.append(np.random.randn(length))
        out[-1][:length/10] = np.zeros(length/10)
        out[-1][-length/10:] = np.zeros(length/10)
    return out

def partial_likelyhood(ker_spikes, ker_onsets, ker_length, residual):
    """! Computes the partial derivative of the log-likelyhood w.r.t a particular
    kernel """
    out = np.zeros(ker_length)
    for s_index, spike in enumerate(ker_spikes):
        onset = ker_onsets[s_index]
        if onset < 0:
            pass
        elif onset + ker_length > residual.size:
            pass
        else:
            local_rest = residual[onset:onset+ker_length]
            out += spike*local_rest
    return out / (residual.std()**2)

def update_dictionnary(ker_dic, spikes, tau, residual, learning_rate=0.5):
    """! Update the kernel dictionnary via gradient ascent """
    for index, ker in enumerate(ker_dic):
        if index in spikes:
            partial = partial_likelyhood(spikes[index], tau[index], ker.size, residual)
            ker += learning_rate*partial

def learn_kernels(ker_dic, audio_waveform):
    """ Update the kernels to fit a particular audio waveform """
    print "* Matching pursuit"
    spikes, tau, rest = matching_pursuit(ker_dic, audio_waveform)
    plt.plot(audio_waveform)
    plt.plot(reconstruct(ker_dic, tau, spikes, audio_waveform.size))
    plt.plot(rest)
    plt.show()
    print "* Dictionnary update"
    update_dictionnary(ker_dic, spikes, tau, rest)

def main():
    """! main """
    ker_dic = initialize_kernels(30, 10)
    waveform = np.sin(np.arange(500)/4.)
    #plt.plot(waveform)
    #plt.show()
    for k in range(2):
        print "iteration: {}".format(k)
        learn_kernels(ker_dic, waveform)
    for ker in ker_dic:
        plt.figure()
        plt.plot(ker)
    plt.show()
if __name__ == "__main__":
    main()
