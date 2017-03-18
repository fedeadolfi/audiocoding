"""! The modules which provides functions to learn the kernels """
import numpy as np
import matplotlib.pyplot as plt
from matchingpursuit import matching_pursuit


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
            out[-onset:] += spike*residual[:ker_length+onset]
        elif onset + ker_length > residual.size:
            out[:residual.size-onset] += spike * residual[onset:]
        else:
            local_rest = residual[onset:onset+ker_length]
            out += spike*local_rest / ker_energy
    return out

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
    return np.dot(partial_sum, residual) / np.sum(partial_sum**2)


def update_dictionnary(ker_dic, spikes, tau, residual, max_size=-1):
    """! Update the kernel dictionnary via gradient ascent """
    for index, ker in enumerate(ker_dic):
        if index in spikes:
            partial = partial_likelyhood(spikes[index], tau[index], ker.size,
                                         np.sum(ker**2), residual)
            step = find_best_step(residual, partial, spikes[index], tau[index], np.sum(ker**2))
            ker += step*partial
            if max_size > 0:
                ker_resized = resize_kernel(ker)
                if ker_resized.size < max_size:
                    if ker_resized.size < residual.size:
                        ker_dic[index] = ker_resized

def resize_kernel(kernel):
    """! If the kernel as a significant value on one of its etremity it is streched
    """
    threshold = np.max(np.abs(kernel))/5.
    if np.max(np.abs(kernel[:kernel.size/10])) > threshold:
        kernel = np.concatenate([np.zeros(kernel.size/10), kernel])
    if np.max(np.abs(kernel[-kernel.size/10:])) > threshold:
        kernel = np.concatenate([kernel, np.zeros(kernel.size/10)])
    return kernel

def learn_kernels(ker_dic, audio_waveform, max_ker_size=-1):
    """ Update the kernels to fit a particular audio waveform """
    #print "* Matching pursuit"
    spikes, tau, rest = matching_pursuit(ker_dic, audio_waveform)
    #print "* Dictionnary update"
    update_dictionnary(ker_dic, spikes, tau, rest, max_ker_size)
    for ker in ker_dic:
        ker /= np.sqrt(np.sum(ker**2))
        plt.show()

def main():
    """! main """
    ker_dic = initialize_kernels(30, 3)
    waveform = np.concatenate([np.sin(np.arange(200)/5.), np.sin(np.arange(200)/2.)])
    #waveform = np.sin(np.arange(200)/3.)
    #ker_dic = [np.sin(np.arange(25)/4.)]
    old_kers = []
    for ker in ker_dic:
        old_kers.append(np.array(ker))
    for _ in range(1000):
        #print "iteration: {}".format(k)
        learn_kernels(ker_dic, waveform, 60)
    for _, ker in enumerate(ker_dic):
        plt.figure()
        plt.plot(ker)
    plt.figure()
    plt.plot(waveform)
    plt.show()
if __name__ == "__main__":
    main()
