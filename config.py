import numpy as np
import tensorflow as tf

ikala_gt_fo_dir = '../datasets/iKala/PitchLabel/'
wav_dir = '../datasets/iKala/Wavfile/'
wav_dir_nus = '../datasets/nus-smc-corpus_48/'
wav_dir_mus = '../datasets/musdb18/train/'
wav_dir_mir = '../datasets/MIR1k/'
wav_dir_med = '../datasets/medleydB/'
wav_dir_timit = '../datasets/TIMIT/TIMIT/'


voice_dir = './voice/'
backing_dir = './backing/'
log_dir = './log/'
log_dir_m1 = './log_m1/'
# log_dir = './log_mfsc_6_best_so_far/'
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

f0_weight = 10
max_models_to_keep = 100
f0_threshold = 1

def get_teacher_prob(epoch):
    if epoch < 500:
        return 0.95
    elif epoch < 1000:
        return 0.75
    else:
        return 0.55



phonemas = ['t', 'y', 'l', 'k', 'aa', 'jh', 'ae', 'ng', 'ah', 'hh', 'z', 'ey', 'f', 'uw', 'iy', 'ay', 'b', 's', 'd', 'sil', 'p', 'n', 'sh', 'ao', 'g', 'ch', 'ih', 'eh', 'aw', 'sp', 'oy', 'th', 'w', 'ow', 'v', 'uh', 'm', 'er', 'zh', 'r', 'dh']

# phonemas_weights = [1.91694048e-03, 3.13983774e-03, 2.37052131e-03, 3.88045684e-03,
#        1.41986299e-03, 1.12648565e-02, 3.30023014e-03, 5.00321922e-03,
#        5.87243483e-04, 4.37742526e-03, 1.97692391e-02, 9.70398460e-04,
#        3.21655616e-03, 1.35928733e-03, 5.93524695e-04, 5.65175305e-04,
#        6.80717094e-03, 1.10015365e-03, 4.38444037e-03, 1.70260315e-04,
#        8.75424154e-03, 1.16470447e-03, 8.02211731e-03, 1.75907101e-03,
#        8.74937266e-03, 1.27897334e-02, 1.20364751e-03, 8.12214268e-04,
#        3.27038554e-03, 2.33057364e-01, 1.74212315e-02, 2.22823967e-02,
#        2.25256804e-03, 8.29516836e-04, 6.36704322e-03, 1.80612767e-02,
#        2.42758721e-03, 1.96789743e-03, 5.61834716e-01, 2.38381211e-03,
#        8.39230304e-03]

phonemas_weights = np.ones(41)
phonemas_weights[19] = 0.5
phonemas_weights[15] = 0.75
phonemas_weights[8] = 0.75
phonemas_weights[14] = 0.75
phonemas_weights[27] = 0.75
val_files = 30

singers = ['ADIZ', 'JLEE', 'JTAN', 'KENN', 'MCUR', 'MPOL', 'MPUR', 'NJAT', 'PMAR', 'SAMF', 'VKOW' ,'ZHIY']

split = 0.9

augment = True
aug_prob = 1.0
noise_threshold = 0.1
pred_mode = 'all'

# Hyperparameters
num_epochs = 1000
batches_per_epoch_train = 1000
batches_per_epoch_val = 300
batches_per_epoch_val_m1 = 300
batch_size = 30
samples_per_file = 5
max_phr_len = 32
input_features = 513

first_embed = 256


lstm_size = 128

output_features = 66
highway_layers = 4
highway_units = 128
init_lr = 0.001
num_conv_layers = 8
conv_filters = 128
conv_activation = tf.nn.relu
dropout_rate = 0.0
projection_size = 3
fs = 44100
comp_mode = 'mfsc'
hoptime = 5.80498866

noise = 0.05

wavenet_layers = 5
rec_field = 2**wavenet_layers
wavenet_filters = 128

print_every = 1
save_every = 10

use_gan = False
gan_lr = 0.001

dtype = tf.float32
