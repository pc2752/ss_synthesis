import numpy as np
import tensorflow as tf

ikala_gt_fo_dir = '../datasets/iKala/PitchLabel/'
wav_dir = '../datasets/iKala/Wavfile/'
wav_dir_nus = '../datasets/nus-smc-corpus_48/'
wav_dir_mus = '../datasets/musdb18/train/'
wav_dir_mir = '../datasets/MIR1k/'
wav_dir_med = '../datasets/medleydB/'
wav_dir_vctk = '../datasets/VCTK/VCTK_files/VCTK-Corpus/wav48/'
wav_dir_vctk_lab = '../datasets/VCTK/VCTK_files/VCTK-Corpus/forPritish/'
wav_dir_timit = '../datasets/TIMIT/TIMIT/TRAIN/'


voice_dir = './voice/'
voice_dir_timit = './voice_timit/'
backing_dir = './backing/'
log_dir = './log_feat_to_feat_speak/'
log_dir_m1 = './log_m1_old/'
# log_dir = './log_mfsc_6_best_so_far/'
data_log = './log/data_log.log'


dir_npy = './data_npy/'
stat_dir = './stats/'
stat_dir_timit = './stats_timit/'
h5py_file_train = './data_h5py/train.hdf5'
h5py_file_val = './data_h5py/val.hdf5'
val_dir = './val_dir_synth/'

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



phonemas = ['t', 'y', 'l', 'k', 'aa', 'jh', 'ae', 'ng', 'ah', 'hh', 'z', 'ey', 'f', 'uw', 'iy', 'ay', 'b', 's', 'd', 'sil', 'p', 'n', 'sh', 'ao', 'g', 'ch', 'ih', 'eh', 'aw', 'sp', 'oy', 'th', 'w', 'ow', 'v', 'uh', 'm', 'er', 'zh', 'r', 'dh', 'ax']

phonemas_timit = ['ih', 'ax', 'bcl', 'ux', 'sh', 'tcl', 'aw', 'v', 'y', 'w', 'uw', 'ah', 'th', 'ow', 'b', 'd', 'g', 'en', 's', 'ae', 'k', 'pcl', 'p', 'jh', 'kcl', 'hh', 'ch', 'gcl', 'eng', 'm', 'n', 'hv', 'ay', 'sil', 'aa', 'q', 'uh', 'eh', 'ix', 'f', 't', 'nx', 'oy', 'ng', 'el', 'epi', 'l', 'h#', 'ey', 'axr', 'dx', 'er', 'ao', 'zh', 'ax-h', 'dcl', 'iy', 'dh', 'r', 'z', 'em']

speakers_timit = ['FMBG0', 'MMVP0', 'MJMA0', 'FSKP0', 'FSLS0', 'MRMS0', 'MHRM0', 'MKDT0', 'MPSW0', 'MCEW0', 'MPGH0', 'MMDM0', 'MRTJ0', 'MWDK0', 'MRGS0', 'MJDG0', 'MREM0', 'FSDC0', 'FPAD0', 'MKAJ0', 'FVKB0', 'FMJF0', 'FJHK0', 'MDBP0', 'FAPB0', 'FSJW0', 'MLNS0', 'MJFR0', 'FCDR1', 'MRGM0', 'FCLT0', 'FSBK0', 'FALK0', 'MDWM0', 'MTLB0', 'FKLC0', 'FKDE0', 'MSRG0', 'MSMS0', 'FDNC0', 'MTPF0', 'MJRK0', 'MJXA0', 'MRLJ1', 'MRRE0', 'MSDB0', 'MAKR0', 'MRDS0', 'FMKC0', 'MTDB0', 'MTJS0', 'MTWH1', 'MRAB1', 'FPMY0', 'FNKL0', 'MTJU0', 'MESJ0', 'MMDB0', 'MMEB0', 'MTPR0', 'FCKE0', 'FBLV0', 'FKDW0', 'FTMG0', 'FJKL0', 'MCDD0', 'MRAM0', 'FDMY0', 'MDPB0', 'FGDP0', 'MJWS0', 'MJLB0', 'MEWM0', 'MGAF0', 'MHBS0', 'MHMR0', 'MRWA0', 'MTBC0', 'MARW0', 'FVFB0', 'MJJB0', 'MDWH0', 'MTMT0', 'FJSP0', 'MTRR0', 'MRDD0', 'MDLM0', 'MSAT1', 'MRJH0', 'MCPM0', 'MRMH0', 'FLMC0', 'MMAG0', 'MDWD0', 'MRJM1', 'MJEB1', 'MDLC1', 'MREW1', 'MSFV0', 'MDEM0', 'MJPG0', 'MDSJ0', 'MTAT1', 'MPAR0', 'FJRB0', 'FPAC0', 'MJDA0', 'FGCS0', 'FLMK0', 'MGXP0', 'FDKN0', 'FEAC0', 'MPRB0', 'FAJW0', 'FBMH0', 'FCMG0', 'MJPM1', 'MREE0', 'MDKS0', 'MGSH0', 'FKAA0', 'FAEM0', 'FEME0', 'MFXS0', 'FTBW0', 'MBBR0', 'FKFB0', 'FCYL0', 'MDJM0', 'MARC0', 'MSVS0', 'MJEE0', 'MSDH0', 'FSSB0', 'MKAH0', 'FDXW0', 'MJWT0', 'MRWS0', 'MAKB0', 'FLMA0', 'MJMM0', 'MDPK0', 'FSRH0', 'FCAG0', 'MSMR0', 'FDAW0', 'FLAC0', 'MWEM0', 'MPMB0', 'MRKM0', 'MPEB0', 'MMDM1', 'MFMC0', 'MRLK0', 'MHMG0', 'MDPS0', 'MDMT0', 'MKLS0', 'MDLB0', 'MGSL0', 'MBML0', 'MKLW0', 'FJEN0', 'MWRE0', 'MWAD0', 'MMGK0', 'FBMJ0', 'MJLG1', 'MJRH0', 'MBWP0', 'MKLR0', 'MWAC0', 'FLJA0', 'MDCM0', 'FSMM0', 'MRFL0', 'MKJO0', 'FSGF0', 'MSRR0', 'FSAG0', 'MSEM1', 'FREH0', 'FCAJ0', 'MKDB0', 'FGRW0', 'MPPC0', 'FSCN0', 'MRAI0', 'MMCC0', 'MTPP0', 'MCAL0', 'FMAH1', 'MFER0', 'FSAH0', 'MTXS0', 'MKAG0', 'FJXM0', 'MEJS0', 'MBGT0', 'MKLS1', 'MBSB0', 'MESG0', 'FTBR0', 'FSMA0', 'FPAB1', 'MGAR0', 'FBJL0', 'MCDR0', 'MBAR0', 'MSDS0', 'MVLO0', 'MJAC0', 'MTMN0', 'MHIT0', 'FLTM0', 'MCSS0', 'MJMD0', 'MRML0', 'FBCH0', 'MSFH0', 'MFXV0', 'MJFH0', 'MSAT0', 'MRXB0', 'FLJD0', 'MKXL0', 'MBCG0', 'MRFK0', 'FJLR0', 'MSMC0', 'MDNS0', 'MMLM0', 'FJXP0', 'MPGR1', 'MDDC0', 'FECD0', 'MBTH0', 'FJWB1', 'MVJH0', 'FETB0', 'FPAF0', 'FLHD0', 'MTAT0', 'MDLR0', 'FLOD0', 'FSJS0', 'MCEF0', 'MDEF0', 'MRHL0', 'MTKP0', 'FDJH0', 'MILB0', 'MRLD0', 'FKLC1', 'MJDE0', 'MPRT0', 'MRBC0', 'FMJU0', 'MDHS0', 'FLET0', 'FBCG1', 'MTJG0', 'MJSR0', 'MAEB0', 'MDLC2', 'MRMG0', 'MKAM0', 'MPRK0', 'MWAR0', 'MTKD0', 'MKLN0', 'MFRM0', 'MDLC0', 'FGMB0', 'MMDS0', 'MRMB0', 'MAJP0', 'MHJB0', 'MJAE0', 'MKDD0', 'MMWS1', 'MDAS0', 'MRLJ0', 'FMJB0', 'MCXM0', 'MCLM0', 'MDLH0', 'FALR0', 'FDAS1', 'MDCD0', 'MZMB0', 'MJRP0', 'FDML0', 'MREH1', 'MJPM0', 'MNTW0', 'FBAS0', 'MWRP0', 'MAEO0', 'FNTB0', 'MDED0', 'FLAG0', 'MRCG0', 'MMAM0', 'MJRG0', 'FSKL0', 'MTJM0', 'FSKC0', 'FSJK1', 'MJRH1', 'MMJB1', 'MMAR0', 'MDBB1', 'MJAI0', 'FLEH0', 'MDSS1', 'MPGR0', 'MJBG0', 'FLKM0', 'FMPG0', 'MJRA0', 'MRJT0', 'FEAR0', 'MJJM0', 'MRVG0', 'FRLL0', 'MJXL0', 'MLBC0', 'FMMH0', 'MRJB1', 'MLEL0', 'MCDC0', 'MSJK0', 'MEJL0', 'FJDM2', 'FVMH0', 'MTER0', 'MGAK0', 'FEEH0', 'MVRW0', 'MRAB0', 'FSPM0', 'MRSO0', 'MEGJ0', 'FJLG0', 'MMGC0', 'MDAC0', 'MMPM0', 'FDTD0', 'FJRP1', 'MMAA0', 'FPLS0', 'MRAV0', 'FKLH0', 'MTML0', 'MCAE0', 'FJSK0', 'MFWK0', 'MTCS0', 'MMWS0', 'MJJJ0', 'MPFU0', 'MDSS0', 'MWCH0', 'MDHL0', 'FRJB0', 'MMXS0', 'MMGG0', 'MBMA1', 'MGAW0', 'MGRL0', 'MJKR0', 'MLJH0', 'MHXL0', 'FHLM0', 'FTLG0', 'FSMS1', 'FHXS0', 'MBMA0', 'FKSR0', 'MJDM0', 'MKRG0', 'FCMM0', 'FCEG0', 'MCTM0', 'MJWG0', 'FPJF0', 'MWSH0', 'MGRP0', 'MNET0', 'MEFG0', 'FLJG0', 'MKES0', 'MGJC0', 'FSDJ0', 'MADC0', 'FMKF0', 'MBEF0', 'MEDR0', 'MADD0', 'MTPG0', 'MDRD0', 'FTAJ0', 'MAFM0', 'MAPV0', 'MEAL0', 'FDFB0', 'MDLR1', 'MMRP0', 'MMSM0', 'MGAG0', 'FMEM0', 'MCTH0', 'MTRC0', 'FKKH0', 'MDTB0', 'MTLC0', 'MJEB0', 'MJHI0', 'MABC0', 'MMEA0', 'MPRD0', 'MRTC0', 'MSTF0', 'FCJF0', 'MMAB1', 'MRJM0', 'MTAB0', 'MLSH0', 'MJDC0', 'MSAH1', 'MJLS0', 'FSAK0', 'FSJG0', 'MSAS0', 'MBOM0', 'MWGR0', 'FPAZ0', 'MMDG0', 'MSES0', 'FCJS0', 'MRLR0', 'FCRZ0', 'MRCW0', 'MRPC1', 'MTDP0', 'MWSB0', 'MDMA0', 'MBJV0', 'MTQC0', 'FEXM0', 'MTAS0', 'MCRE0', 'MMWB0', 'MRDM0', 'MCLK0', 'MCHL0', 'MGES0']

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

phonemas_weights = np.ones(61)
# phonemas_weights[19] = 0.5
# phonemas_weights[15] = 0.75
# phonemas_weights[8] = 0.75
# phonemas_weights[14] = 0.75
# phonemas_weights[27] = 0.8
# phonemas_weights[33] = 0.8
# phonemas_weights[11] = 0.8
# phonemas_weights[17] = 0.85
# phonemas_weights[21] = 0.85
# phonemas_weights[13] = 0.85
# phonemas_weights[4] = 0.85
# phonemas_weights[34] = 0.95
# phonemas_weights[16] = 0.95
# phonemas_weights[22] = 0.95
# phonemas_weights[40] = 0.95
# phonemas_weights[24] = 0.95
# phonemas_weights[20] = 0.95
# phonemas_weights[5] = 0.95
# phonemas_weights[25] = 0.95
# phonemas_weights[30] = 0.95
# phonemas_weights[35] = 0.95
# phonemas_weights[10] = 0.95
# phonemas_weights[31] = 0.95
# phonemas_weights[29] = 1.0
# phonemas_weights[38] = 1.0

val_files = 30

singers = ['ADIZ', 'JLEE', 'JTAN', 'KENN', 'MCUR', 'MPOL', 'MPUR', 'NJAT', 'PMAR', 'SAMF', 'VKOW' ,'ZHIY', 'p255', 'p285', 'p260', 'p247', 'p266', 'p364', 'p265', 'p233', 'p341', 'p347', 'p243', 'p300', 'p284', 'p283', 'p239', 'p269', 'p236', 'p281', 'p293', 'p241', 'p240', 'p259', 'p244', 'p271', 'p294', 'p287', 'p263', 'p261', 'p334', 'p323', 'p227', 'p282', 'p313', 'p248', 'p277', 'p297', 'p314', 'p250', 'p335', 'p374', 'p315', 'p304', 'p298', 'p288', 'p234', 'p310', 'p262', 'p329', 'p251', 'p330', 'p339', 'p312', 'p256', 'p258', 'p231', 'p249', 'p317', 'p301', 'p292', 'p306', 'p360', 'p272', 'p316', 'p311', 'p308', 'p318', 'p229', 'p245', 'p361', 'p232', 'p257', 'p264', 'p237', 'p226', 'p246', 'p351', 'p270', 'p228', 'p286', 'p267', 'p376', 'p333', 'p252', 'p253', 'p345', 'p254', 'p278', 'p336', 'p268', 'p363', 'p326', 'p303', 'p362', 'p295', 'p274', 'p273', 'p305', 'p343', 'p276', 'p275', 'p225', 'p238', 'p302', 'p279', 'p307', 'p299', 'p340', 'p280', 'p230']


vctk_speakers = ['p255', 'p285', 'p260', 'p247', 'p266', 'p364', 'p265', 'p233', 'p341', 'p347', 'p243', 'p300', 'p284', 'p283', 'p239', 'p269', 'p236', 'p281', 'p293', 'p241', 'p240', 'p259', 'p244', 'p271', 'p294', 'p287', 'p263', 'p261', 'p334', 'p323', 'p227', 'p282', 'p313', 'p248', 'p277', 'p297', 'p314', 'p250', 'p335', 'p374', 'p315', 'p304', 'p298', 'p288', 'p234', 'p310', 'p262', 'p329', 'p251', 'p330', 'p339', 'p312', 'p256', 'p258', 'p231', 'p249', 'p317', 'p301', 'p292', 'p306', 'p360', 'p272', 'p316', 'p311', 'p308', 'p318', 'p229', 'p245', 'p361', 'p232', 'p257', 'p264', 'p237', 'p226', 'p246', 'p351', 'p270', 'p228', 'p286', 'p267', 'p376', 'p333', 'p252', 'p253', 'p345', 'p254', 'p278', 'p336', 'p268', 'p363', 'p326', 'p303', 'p362', 'p295', 'p274', 'p273', 'p305', 'p343', 'p276', 'p275', 'p225', 'p238', 'p302', 'p279', 'p307', 'p299', 'p340', 'p280', 'p230']

split = 0.95

augment = True
aug_prob = 0.5

noise_threshold = 0.005 #0.7 for the unnormalized features
pred_mode = 'all'

# Hyperparameters
num_epochs = 500
num_epochs_m1 = 2000
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
wavenet_filters = 256

print_every = 1
save_every = 10

use_gan = False
gan_lr = 0.001

dtype = tf.float32


timit_val_files = ['timit_DR2_MTJG0_+SX440.hdf5', 'timit_DR2_MTJG0_+SX80.hdf5', 'timit_DR2_MTJG0_+SI1520.hdf5', 'timit_DR2_MTJG0_+SX260.hdf5', 'timit_DR2_MTJG0_+SI2157.hdf5', 'timit_DR2_MTJG0_+SI890.hdf5', 'timit_DR2_MTJG0_+SX170.hdf5', 'timit_DR2_MTJG0_+SA1.hdf5', 'timit_DR2_MTJG0_+SA2.hdf5', 'timit_DR2_MTJG0_+SX350.hdf5']