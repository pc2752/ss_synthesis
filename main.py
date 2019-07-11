import models
import tensorflow as tf
import argparse
import os, sys
import config
import utils
import numpy as np
import mir_eval

def train(_):
    model = models.SSSynth()
    model.train()

def eval_hdf5_file(file_name):
    model = models.SSSynth()
    model.test_file_hdf5(file_name)

if __name__ == '__main__':
    if len(sys.argv)<2 or sys.argv[1] == '-help' or sys.argv[1] == '--help' or sys.argv[1] == '--h' or sys.argv[1] == '-h':
        print("%s --help or -h or --h or -help to see this menu" % sys.argv[0])
        print("%s --train or -t or --t or -train to train the model" % sys.argv[0])
        print("%s -e or --e or -eval or --eval  <filename> to evaluate an hdf5 file" % sys.argv[0])
        print("%s -v or --v or -val or --val <filename> to evaluate a wav file" % sys.argv[0])

    else:
        if sys.argv[1] == '-train' or sys.argv[1] == '--train' or sys.argv[1] == '--t' or sys.argv[1] == '-t':
            print("Training")
            tf.app.run(main=train)

        elif sys.argv[1] == '-e' or sys.argv[1] == '--e' or sys.argv[1] == '--eval' or sys.argv[1] == '-eval':
            if len(sys.argv)<3:
                print("Please give a hdf5 file to evaluate")
            else:
                file_name = sys.argv[2]
                if not file_name.endswith('.hdf5'):
                    file_name = file_name+'.hdf5'
                if not file_name in [x for x in os.listdir(config.voice_dir)]:
                    print("Currently only supporting hdf5 files which are in the dataset, will be expanded later.")
                    print([x for x in os.listdir(config.voice_dir) if x.startswith('nus' )or x.startswith('casas') or x.startswith('med')])
                else:
                    eval_hdf5_file(file_name)
