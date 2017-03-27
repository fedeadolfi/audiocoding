"""! try to learn with 1 file only """
#from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from kernelslearning import learn_kernels, initialize_kernels
#from matchingpursuit import max_onset
from pydub import AudioSegment
from learn import save_dic

radio = AudioSegment.from_mp3("samples/radiotv/Arielle Dombasle est une femme tout terrain.mp3")
waveform = np.array(radio.get_array_of_samples(), dtype=float)
waveform /= waveform.std()
length = 10*16384
waveform = waveform[28*16384:38*16384]

ker_dic = initialize_kernels(100, 32)
for it in range(30):
    print "!!!! iteration {} !!!!".format(it)
    #learn_kernels(ker_dic, waveform, 300, threshold=waveform.size/10, criterion="sparse")
    learn_kernels(ker_dic, waveform, 300, threshold=0.1, criterion="spike")
    save_dic(ker_dic, "results/10secs2")

#for ker in ker_dic:
#    plt.figure()
#    plt.plot(ker)
#plt.show()
    
