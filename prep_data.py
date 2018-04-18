# from __future__ import division
import os,re
import collections
import soundfile as sf
import numpy as np
from scipy.stats import norm
import pyworld as pw
import matplotlib.pyplot as plt
from reduce import sp_to_mfsc, mfsc_to_sp, ap_to_wbap,wbap_to_ap, get_warped_freqs, sp_to_mgc, mgc_to_sp
from vocoder import extract_sp_world, extract_ap_world, gen_wave_world
import sys

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

        mixture = np.clip(audio[:,0]+audio[:,1],0.0,1.0)

        voc_stft = abs(utils.stft(vocals))
        mix_stft = abs(utils.stft(mixture))

        assert voc_stft.shape==mix_stft.shape

        out_feats = utils.input_to_feats(os.path.join(config.wav_dir,lf))

        out_feats = np.concatenate(((out_feats,np.zeros((1,out_feats.shape[1])))))


        assert out_feats.shape[0]==voc_stft.shape[0]

        np.save(config.dir_npy+lf[:-4]+'_voc_stft',voc_stft)
        np.save(config.dir_npy+lf[:-4]+'_mix_stft',mix_stft)
        np.save(config.dir_npy+lf[:-4]+'_synth_feats',out_feats)



        count+=1
        utils.progress(count,len(wav_files))
    import pdb;pdb.set_trace()




if __name__ == '__main__':
    main()