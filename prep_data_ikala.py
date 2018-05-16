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

import config
import utils


def main():

    # maximus=np.zeros(66)
    # minimus=np.ones(66)*1000
    wav_files=[x for x in os.listdir(config.wav_dir) if x.endswith('.wav')]
    count=0


    for lf in wav_files:
        # print(lf)
        audio,fs = sf.read(os.path.join(config.wav_dir,lf))

        vocals = np.array(audio[:,1])

        mixture = (audio[:,0]+audio[:,1])*0.7

        backing = np.array(audio[:,0])

        voc_stft = abs(utils.stft(vocals))
        mix_stft = abs(utils.stft(mixture))
        back_stft = abs(utils.stft(backing))

        assert voc_stft.shape==mix_stft.shape

        out_feats = utils.stft_to_feats(vocals,fs)

        if not out_feats.shape[0]==voc_stft.shape[0] :
            if out_feats.shape[0]<voc_stft.shape[0]:
                while out_feats.shape[0]<voc_stft.shape[0]:
                    out_feats = np.concatenate(((out_feats,np.zeros((1,out_feats.shape[1])))))
            elif out_feats.shape[0]<voc_stft.shape[0]:
                print("You are an idiot")

        assert out_feats.shape[0]==voc_stft.shape[0]

        hdf5_file = h5py.File(config.voice_dir+'ikala_'+lf[:-4]+'.hdf5', mode='w')

        hdf5_file.create_dataset("voc_stft", voc_stft.shape, np.float32)

        hdf5_file.create_dataset("feats", out_feats.shape, np.float32)

        hdf5_file["voc_stft"][:,:] = voc_stft

        hdf5_file["feats"][:,:] = out_feats

        hdf5_file.close()

        hdf5_file = h5py.File(config.backing_dir+'ikala_'+lf[:-4]+'.hdf5', mode='w')

        hdf5_file.create_dataset("back_stft", back_stft.shape, np.float32)

        hdf5_file.create_dataset("mix_stft", mix_stft.shape, np.float32)

        hdf5_file["back_stft"][:,:] = back_stft

        hdf5_file["mix_stft"][:,:] = mix_stft

        hdf5_file.close()

        count+=1

        utils.progress(count,len(wav_files))





if __name__ == '__main__':
    main()