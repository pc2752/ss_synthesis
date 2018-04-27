# from __future__ import division
import os,re
import collections
import soundfile as sf
import numpy as np
from scipy.stats import norm
import pyworld as pw
import matplotlib.pyplot as plt
import sys
import h5py
import stempeg

import config
import utils


def main():

    # maximus=np.zeros(66)
    # minimus=np.ones(66)*1000
    wav_files=[x for x in os.listdir(config.wav_dir_mus) if x.endswith('.stem.mp4') and not x.startswith(".")]
    
    count = 0

    for lf in wav_files:
        
        # print(lf)
        audio,fs = stempeg.read_stems(os.path.join(config.wav_dir_mus,lf), stem_id=[0,1,2,3,4])

        mixture = audio[0]

        drums = audio[1]

        bass = audio[2]

        acc = audio[3]

        vocals = audio[4]

        backing = np.clip(drums+bass+acc, 0.0,1.0)

        if len(backing.shape) == 2:
            backing = (backing[:,0]+backing[:,1])/2

        # import pdb;pdb.set_trace()

        back_stft = abs(utils.stft(backing))

        hdf5_file = h5py.File(config.backing_dir+'mus_'+lf[:-9]+'.hdf5', mode='w')

        hdf5_file.create_dataset("back_stft", back_stft.shape, np.float32)

        hdf5_file["back_stft"][:,:] = back_stft

        hdf5_file.close()

        count+=1

        utils.progress(count,len(wav_files))

if __name__ == '__main__':
    main()