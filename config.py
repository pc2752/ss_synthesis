import numpy as np
import tensorflow as tf

wav_dir = '../datasets/iKala/Wavfile/'
wav_dir_nus = '../datasets/nus-smc-corpus_48/'
wav_dir_mus = '../datasets/musdb18/train/'


voice_dir = './voice/'
backing_dir = './backing/'
log_dir = './log/'
data_log = './log/data_log.log'


dir_npy = './data_npy/'
stat_dir = './stats/'
h5py_file_train = './data_h5py/train.hdf5'
h5py_file_val = './data_h5py/val.hdf5'
val_dir = './val_dir/'

in_mode = 'mix'
norm_mode_out = "max_min"
norm_mode_in = "max_min"

voc_ext = '_voc_stft.npy'
feats_ext = '_synth_feats.npy'


split = 0.9

augment = True
aug_prob = 0.4

# Hyperparameters
num_epochs = 2000
batches_per_epoch_train = 500
batches_per_epoch_val = 100
batch_size = 60 
samples_per_file = 15
max_phr_len = 50
input_features = 513
lstm_size = 128
output_features = 66
highway_layers = 4
highway_units = 128
init_lr = 0.0001
num_conv_layers = 8
conv_filters = 128
conv_activation = tf.nn.relu
dropout_rate = 0.0
projection_size = 3
fs = 44100
comp_mode = 'mfsc'

print_every = 1
save_every = 5



dtype = tf.float32