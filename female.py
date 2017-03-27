"""! test with cleaner source """
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from kernelslearning import learn_kernels, initialize_kernels
#from matchingpursuit import max_onset
from pydub import AudioSegment
from learn import save_dic

# pylint: disable=C0103
FS = 16384

vocals = [AudioSegment.from_wav(wavfile) for wavfile in glob("samples/*.wav")]
sounds = []
for sample in vocals:
    sample = sample.set_frame_rate(FS).set_channels(1)
    s_array = np.array(sample.get_array_of_samples(), dtype=float)
    s_array /= s_array.std()
    sounds.append(s_array)

n_iter = 3000
ker_size = 100
n_kernels = 32
ker_dic = initialize_kernels(ker_size, n_kernels)
def read_kernels(name):
    """! read kernels in saved in files begining by name """
    kernels = [np.load(filename) for filename in glob(name + "*.npy")]
    return kernels
#ker_dic = read_kernels("female2")

for ite in range(n_iter):
    if ite % 1 == 0:
        print "iteration #{}".format(ite)
    learn_kernels(ker_dic, sounds[np.random.randint(len(sounds))], max_ker_size=350)
    save_dic(ker_dic, "manyspeakers")

for ker in ker_dic:
    plt.figure()
    plt.plot(ker)
plt.show()
