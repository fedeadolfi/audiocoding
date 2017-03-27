"""! In this module we actually learn kernels with Lewicki's method,
using samples of speech """
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from kernelslearning import learn_kernels, initialize_kernels
#from matchingpursuit import max_onset
from pydub import AudioSegment

FS = 16384

def save_dic(ker_dic, name):
    """! write kernels in separate files """
    for i, ker in enumerate(ker_dic):
        np.save(name + "_ker{}".format(i), ker)

def read_kernels(name):
    """! read kernels in saved in files begining by name """
    kernels = [np.load(filename) for filename in glob(name + "*.npy")]
    return kernels


def import_database():
    """! read all mp3 files in directory "samples" """
    speechs = [AudioSegment.from_mp3(mp3file) for mp3file in glob("samples/radiotv/*.mp3")]
    arrays = []
    for speech in speechs:
        speech = speech.set_frame_rate(FS).set_channels(1)
        print "sampling frequency: {}, {} channel(s)".format(speech.frame_rate,
                                                             speech.channels)
        array = np.array(speech.get_array_of_samples(), dtype=float)
        array /= array.std()
        slice_ = 300000
        for i in range(array.size/slice_):
            arrays.append(array[i*slice_:(i+1)*slice_])
        #arrays.append(np.array(speech.get_array_of_samples(), dtype=float))
        arrays[-1] /= arrays[-1].std()
    return arrays

def learn_from_database(database, n_kernels, ker_size, iterations):
    """! initialize random kernels and learn them with EM algorithm, on a list of
    waveforms """
    ker_dic = initialize_kernels(ker_size, n_kernels)
    #ker_dic = read_kernels("results/radiotv")
    #n_waveforms = len(database)
    l_train = 5000
    for ite in range(iterations):
        if ite % 1 == 0:
            print "iteration #{}".format(ite)
        #learn_kernels(ker_dic, database[ite % n_waveforms], max_ker_size=3*ker_size)
        wavform = database[np.random.randint(len(database))]
        start = np.random.randint(wavform.size - l_train)
        wavform = wavform[start:start+l_train]
        learn_kernels(ker_dic, wavform, max_ker_size=2*ker_size)
        save_dic(ker_dic, "results/radiotv")
    return ker_dic

def main():
    """! main """
    waveforms = import_database()
    kernels = learn_from_database(waveforms, 32, 100, 3000)
    for ker in kernels:
        plt.figure()
        plt.plot(ker)
    plt.show()
if __name__ == "__main__":
    main()
