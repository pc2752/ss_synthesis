import os,re
import collections
import csv
import soundfile as sf
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt 
import config

singer_dir = './ayeha_samples/'

def get_notes():
    lab_files = [x for x in os.listdir(singer_dir) if x.endswith('.lab')]
    for lf in lab_files[0:1]:
        lab_f=open(singer_dir+lf)
        note_f=open(singer_dir+lf[:-4]+'.notes')
        phos=lab_f.readlines()
        notes=note_f.readlines()
        lab_f.close()
        note_f.close()

        phonemes = []
        noters = []

        for pho in phos:
            st,en,phonote=pho.split()
            st = int(int(st)/50000)
            en = int(int(en)/50000)
            if phonote == 'pau' or phonote == 'br':
                phonote = 'sil'
            phonemes.append([st,en,phonote])

        for note in notes:
            st,en,phonote = note.split()
            st=int(int(st)/5)
            en=int(int(en)/5)
            if phonote == 'sil':
                phonote = 0
            noters.append([st,en,phonote])


        strings_p = np.zeros(phonemes[-1][1])

        for i in range(len(phonemes)):
            pho=phonemes[i]
            value = config.phonemas.index(pho[2])
            strings_p[pho[0]:pho[1]+1] = value

        strings_n = np.zeros(noters[-1][1])

        for i in range(len(noters)):
            pho = noters[i]
            value = pho[2]
            strings_n[pho[0]:pho[1]+1] = value

        strings_n = strings_n[:len(strings_p)]

        return strings_n, strings_p


        # import pdb;pdb.set_trace()



# main()