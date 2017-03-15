"""! The modules which provides functions to learn the kernels """
import numpy as np
import matplotlib.pyplot as plt
from matchingpursuit import matching_pursuit, reconstruct

PLOT = False

def initialize_kernels(length, number):
    """! Initialize random kernels of a certain length, zero-padded on both ends
    by 1/10 its total length """
    out = []
    for _ in range(number):
        out.append(np.random.randn(length))
        out[-1][:length/10] = np.zeros(length/10)
        out[-1][-length/10:] = np.zeros(length/10)
    return out

def partial_likelyhood(ker_spikes, ker_onsets, ker_length, ker_energy, residual):
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
            out += spike*local_rest / ker_energy
    return out # / (residual.std()**2)

def find_best_step(residual, partial, ker_spikes, ker_onsets, ker_energy):
    """! Computes the best step for the gradient ascent of one kernel """
    partial_sum = np.zeros(residual.shape)
    for s_index, spike in enumerate(ker_spikes):
        onset = ker_onsets[s_index]
        part = np.array(partial)
        if onset < 0:
            part = part[-onset:]
            onset = 0
        elif onset + part.size >= residual.size:
            part = part[:(residual.size - onset)]
        partial_sum[onset:onset + part.size] += spike * part / ker_energy
    if PLOT:
        plt.figure()
        plt.plot(partial)
        plt.plot(partial_sum)
        plt.title("partial")
        plt.scatter(ker_onsets, ker_spikes)
        plt.show()
    return np.dot(partial_sum, residual) / np.sum(residual**2)


def update_dictionnary(ker_dic, spikes, tau, residual): #, learning_rate=0.5):
    """! Update the kernel dictionnary via gradient ascent """
    for index, ker in enumerate(ker_dic):
        if index in spikes:
            partial = partial_likelyhood(spikes[index], tau[index], ker.size,
                                         np.sum(ker**2), residual)
            step = find_best_step(residual, partial, spikes[index], tau[index], np.sum(ker**2))
            #print step
            #ker += learning_rate*partial
            ker += step*partial
            #ker /= np.sum(ker**2)

def learn_kernels(ker_dic, audio_waveform):
    """ Update the kernels to fit a particular audio waveform """
    print "* Matching pursuit"
    spikes, tau, rest = matching_pursuit(ker_dic, audio_waveform)
    if PLOT:
        plt.plot(audio_waveform)
        plt.plot(reconstruct(ker_dic, tau, spikes, audio_waveform.size))
        plt.plot(ker_dic[0])
        plt.plot(rest)
    print "* Dictionnary update"
    update_dictionnary(ker_dic, spikes, tau, rest)
    if PLOT:
        #plt.figure()
        #plt.plot(ker_dic[0])
        plt.show()

def main():
    """! main """
    ker_dic = initialize_kernels(30, 2)
    waveform = np.sin(np.arange(200)/4.)
    #waveform = np.sin(np.arange(10)/3.)
    #ker_dic = [np.sin(np.arange(25)/4.)]
    old_kers = []
    for ker in ker_dic:
        old_kers.append(np.array(ker))
    for k in range(100):
        print "iteration: {}".format(k)
        learn_kernels(ker_dic, waveform)
    for i, ker in enumerate(ker_dic):
        plt.figure()
        plt.plot(old_kers[i])
        plt.plot(ker)
    plt.show()
if __name__ == "__main__":
    main()
