import mir_eval

import matplotlib.pyplot as plt

import os
import sys
import time
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import h5py

import config
from data_pipeline import data_gen
import modules_tf as modules
import utils


file_list = [x for x in os.listdir(config.ikala_gt_fo_dir) if not x.startswith('.') and x.endswith('.pv')]
count = 0
for file_name in file_list:
    inter_list = []
    initial_f0_values = open(config.ikala_gt_fo_dir+file_name).readlines()
    for i,f0_value in enumerate(initial_f0_values):
        inter_list.append(str(i*0.032*10000000)+' '+str(f0_value[:-2]))
    utils.list_to_file(inter_list,'./ikala_eval/ikala_gt/'+file_name)
    count+=1

    utils.progress(count, len(file_list))


    # old_times = np.linspace(0.0,30.0,937)
    # new_times = np.linspace(0.0,30.0,5169)
    # import pdb;pdb.set_trace()
    # f_list = mir_eval.melody.resample_melody_series(old_times,mir_eval.melody.freq_to_voicing(np.array(inter_list)),new_times)    

    # import pdb;pdb.set_trace()

