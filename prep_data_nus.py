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
    singers = next(os.walk(config.wav_dir_nus))[1]
    

    for singer in singers:
        sing_dir = config.wav_dir_nus+singer+'/sing/'
        read_dir = config.wav_dir_nus+singer+'/read/'
        sing_wav_files=[x for x in os.listdir(sing_dir) if x.endswith('.wav') and not x.startswith('.')]

        count = 0

        print ("Processing singer %s" % singer)
        for lf in sing_wav_files:
        # print(lf)
            audio,fs = sf.read(os.path.join(sing_dir,lf))

            if len(audio.shape) == 2:

                vocals = np.array((audio[:,1]+audio[:,0])/2)

            else: 
                vocals = np.array(audio)

            voc_stft = abs(utils.stft(vocals))

            out_feats = utils.stft_to_feats(vocals,fs)

            import pdb;pdb.set_trace()

            if not out_feats.shape[0]==voc_stft.shape[0] :
                if out_feats.shape[0]<voc_stft.shape[0]:
                    while out_feats.shape[0]<voc_stft.shape[0]:
                        out_feats = np.concatenate(((out_feats,np.zeros((1,out_feats.shape[1])))))
                elif out_feats.shape[0]<voc_stft.shape[0]:
                    print("You are an idiot")

            assert out_feats.shape[0]==voc_stft.shape[0]

            hdf5_file = h5py.File(config.voice_dir+'nus_'+singer+'_sing_'+lf[:-4]+'.hdf5', mode='w')

            hdf5_file.create_dataset("voc_stft", voc_stft.shape, np.float32)

            hdf5_file.create_dataset("feats", out_feats.shape, np.float32)

            hdf5_file["voc_stft"][:,:] = voc_stft

            hdf5_file["feats"][:,:] = out_feats

            hdf5_file.close()

            count+=1

            utils.progress(count,len(sing_wav_files))


        read_wav_files=[x for x in os.listdir(read_dir) if x.endswith('.wav') and not x.startswith('.')]
        print ("Processing reader %s" % singer)
        count = 0
        for lf in sing_wav_files:
        # print(lf)
            audio,fs = sf.read(os.path.join(read_dir,lf))

            if len(audio.shape) == 2:

                vocals = np.array((audio[:,1]+audio[:,0])/2)

            else: 
                vocals = np.array(audio)

            voc_stft = abs(utils.stft(vocals))

            out_feats = utils.stft_to_feats(vocals,fs)

            

            if not out_feats.shape[0]==voc_stft.shape[0] :
                if out_feats.shape[0]<voc_stft.shape[0]:
                    while out_feats.shape[0]<voc_stft.shape[0]:
                        out_feats = np.concatenate(((out_feats,np.zeros((1,out_feats.shape[1])))))
                elif out_feats.shape[0]<voc_stft.shape[0]:
                    print("You are an idiot")

            assert out_feats.shape[0]==voc_stft.shape[0] 

            hdf5_file = h5py.File(config.voice_dir+'nus_'+singer+'_read_'+lf[:-4]+'.hdf5', mode='w')

            hdf5_file.create_dataset("voc_stft", voc_stft.shape, np.float32)

            hdf5_file.create_dataset("feats", out_feats.shape, np.float32)

            hdf5_file["voc_stft"][:,:] = voc_stft

            hdf5_file["feats"][:,:] = out_feats

            hdf5_file.close()

            count+=1

            utils.progress(count,len(read_wav_files))


if __name__ == '__main__':
    main()