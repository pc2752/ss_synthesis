import numpy as np
import random
import os
import matplotlib.pyplot as plt
import h5py
import logging

import config
import utils
from utils import normalize, denormalize


logger = logging.getLogger('myapp')
hdlr = logging.FileHandler(config.data_log)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.setLevel(logging.INFO)

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def numpy_to_h5py(in_dir=config.dir_npy, split = config.split):
    
    """Creats h5py datasets for training and validation from the numpy featers

    Input:
        in_dir: Directory with numpy data.
        split: split to use for training and validation

    """

    in_files=[x[:-13] for x in os.listdir(in_dir) if x.endswith('_voc_stft.npy') and not x.startswith('._')]

    random.shuffle(in_files)


    num_files = len(in_files)

    split_idx = int(num_files*split)

    trn_files = in_files[:split_idx]

    val_files = in_files[split_idx:]

    num_val_files = len(val_files)

    print('Processing %d training files' % split_idx)
    logger.info('Processing %d training files' % split_idx)

    logger.info('Training file: %s' % config.h5py_file_train)

    voc_shape_trn = [split_idx, 5170,config.input_features]

    mix_shape_trn = [split_idx, 5170,config.input_features]

    feats_shape_trn = [split_idx, 5170,config.output_features]

    hdf5_file = h5py.File(config.h5py_file_train, mode='w')

    hdf5_file.create_dataset("voc_stft", voc_shape_trn, np.float32)

    hdf5_file.create_dataset("back_stft", voc_shape_trn, np.float32)

    hdf5_file.create_dataset("mix_stft", mix_shape_trn, np.float32)

    hdf5_file.create_dataset("feats", feats_shape_trn, np.float32)


    i = 0

    for f in trn_files:

        voc_stft = np.load(in_dir+f+'_voc_stft.npy')

        voc_stft = voc_stft.astype('float32')

        mix_stft = np.load(in_dir+f+'_mix_stft.npy')

        mix_stft = mix_stft.astype('float32')

        back_stft = np.load(in_dir+f+'_back_stft.npy')

        back_stft = back_stft.astype('float32')

        synth_feats = np.load(in_dir+f+'_synth_feats.npy')

        synth_feats = synth_feats.astype('float32')

        hdf5_file["voc_stft"][i,...] = voc_stft

        hdf5_file["mix_stft"][i,...] = mix_stft

        hdf5_file["back_stft"][i,...] = back_stft

        hdf5_file["feats"][i,...] = synth_feats

        i+=1
        utils.progress(i, split_idx)

        logger.info('Processed training file: %s' % f)

    hdf5_file.close()

    print('Processing %d validation files' % num_val_files)
    logger.info('Processing %d validation files' % num_val_files)

    logger.info('Validation file: %s' % config.h5py_file_val)

    voc_shape_trn = [num_val_files, 5170,config.input_features]

    mix_shape_trn = [num_val_files, 5170,config.input_features]

    feats_shape_trn = [num_val_files, 5170,config.output_features]

    hdf5_file = h5py.File(config.h5py_file_val, mode='w')

    hdf5_file.create_dataset("voc_stft", voc_shape_trn, np.float32)

    hdf5_file.create_dataset("mix_stft", mix_shape_trn, np.float32)

    hdf5_file.create_dataset("back_stft", voc_shape_trn, np.float32)

    hdf5_file.create_dataset("feats", feats_shape_trn, np.float32)


    i = 0

    for f in val_files:

        voc_stft = np.load(in_dir+f+'_voc_stft.npy')

        voc_stft = voc_stft.astype('float32')

        mix_stft = np.load(in_dir+f+'_mix_stft.npy')

        mix_stft = mix_stft.astype('float32')

        synth_feats = np.load(in_dir+f+'_synth_feats.npy')

        synth_feats = synth_feats.astype('float32')

        back_stft = np.load(in_dir+f+'_back_stft.npy')

        back_stft = back_stft.astype('float32')

        hdf5_file["voc_stft"][i,...] = voc_stft

        hdf5_file["mix_stft"][i,...] = mix_stft

        hdf5_file["back_stft"][i,...] = back_stft

        hdf5_file["feats"][i,...] = synth_feats

        i+=1
        utils.progress(i, num_val_files)

        logger.info('Processed validation file: %s' % f)

    hdf5_file.close()
    # return original_ffts

# import pdb;pdb.set_trace()
def test_h5py(original_ffts,train_filename=config.h5py_file_train):
    recon_ffts = []

    hdf5_file = h5py.File(train_filename, "r")

    for i in range(3):
        recon_ffts.append([hdf5_file["voc_stft"][i], hdf5_file["mix_stft"][i], hdf5_file["feats"][i]])


    plt.subplot(611)
    plt.imshow(np.log(original_ffts[2][0].T),aspect='auto',origin='lower')
    plt.subplot(612)
    plt.imshow(np.log(recon_ffts[2][0].T),aspect='auto',origin='lower')

    plt.subplot(613)
    plt.imshow(np.log(original_ffts[2][1].T),aspect='auto',origin='lower')
    plt.subplot(614)
    plt.imshow(np.log(recon_ffts[2][1].T),aspect='auto',origin='lower')

    plt.subplot(615)
    plt.imshow(original_ffts[2][2].T,aspect='auto',origin='lower')
    plt.subplot(616)
    plt.imshow(recon_ffts[2][2].T,aspect='auto',origin='lower')
    plt.show()


def get_stats(file_list=[config.h5py_file_train, config.h5py_file_val], feat = 'feats'):
    hdf5_file = h5py.File(file_list[0], "r")
    feats = hdf5_file[feat]
    for filename in file_list[1:]:
        hdf5_file = h5py.File(filename,"r")
        feats = np.concatenate((feats,hdf5_file[feat]))
    feats = feats.reshape(-1,feats.shape[-1])
    means = feats.mean(axis=0)
    stds = feats.std(axis=0)
    # import pdb;pdb.set_trace()
    maximus = feats.max(axis=0)
    minimus = feats.min(axis=0)
    import pdb;pdb.set_trace()
    np.save(config.stat_dir+feat+'_means',means)
    np.save(config.stat_dir+feat+'_stds',stds)
    np.save(config.stat_dir+feat+'_maximus',maximus)
    np.save(config.stat_dir+feat+'_minimus',minimus)
    

def val_generator(train_filename=config.h5py_file_val, in_mode=config.in_mode):
    hdf5_file = h5py.File(train_filename, "r")
    if in_mode == 'voc':
        inps = hdf5_file["voc_stft"]
        feat = "voc_stft"
    elif in_mode == 'mix':
        inps = hdf5_file["mix_stft"]
        feat = "mix_stft"
    targ = hdf5_file["feats"]
    num_files = inps.shape[0]
    for i in range(num_files):
        in_batch, nchunks_in = utils.generate_overlapadd(inps[i])

        in_batch = normalize(in_batch, feat)

        targ_batch, nchunks_targ = utils.generate_overlapadd(targ[i])

        targ_batch = normalize(targ_batch, "feats")
        yield in_batch, nchunks_in, targ_batch, nchunks_targ




def get_batches(train_filename=config.h5py_file_train, in_mode=config.in_mode, batches_per_epoch=config.batches_per_epoch_train):
    # data_path = 'train.tfrecords'
    hdf5_file = h5py.File(train_filename, "r")

    num_files = hdf5_file["voc_stft"].shape[0]
    max_files_to_process = int(config.batch_size/config.samples_per_file)
    for k in range(batches_per_epoch):

        voc_stfts = []
        mix_stfts = []
        featss = []

        if config.augment:
            
            for i in range(max_files_to_process):
                file_index_voc = np.random.randint(0,num_files)
                voc = hdf5_file["voc_stft"][file_index_voc]
                fea = hdf5_file["feats"][file_index_voc]
                mix = hdf5_file["mix_stft"][file_index_voc]

                for j in range(config.samples_per_file):
                    augment = np.random.rand(1)<config.aug_prob
                    index=np.random.randint(0,5170-config.max_phr_len)
                    if augment:
                        file_index_back = np.random.randint(0,num_files)
                        back = hdf5_file["back_stft"][file_index_back][index:index+config.max_phr_len]
                        mix_stfts.append(np.clip(voc[index:index+config.max_phr_len]+back, 0.0,1.0))
                    else:
                        mixer = normalize(mix[index:index+config.max_phr_len], feat='mix_stft', mode=config.norm_mode_in)
                        mix_stfts.append(mixer)
                    featss.append(fea[index:index+config.max_phr_len])
            featss = normalize(featss, feat='feats', mode=config.norm_mode_out)
            # Trying without vuv part
            # featss[:,:,-2] = featss[:,:,-2]*(1-featss[:,:,-1])
            yield np.array(mix_stfts), np.array(featss)
                

        else:

            for i in range(max_files_to_process):
                file_index = np.random.randint(0,num_files)
                if in_mode == 'voc':
                    voc = hdf5_file["voc_stft"][file_index]
                elif in_mode == 'mix':
                    mix = hdf5_file["mix_stft"][file_index]
                fea = hdf5_file["feats"][file_index]
                for j in range(config.samples_per_file):
                    index=np.random.randint(0,5170-config.max_phr_len)
                    if in_mode == 'voc':
                        voc_stfts.append(voc[index:index+config.max_phr_len])
                    elif in_mode == 'mix':    
                        mix_stfts.append(mix[index:index+config.max_phr_len])
                    featss.append(fea[index:index+config.max_phr_len])
            featss = normalize(featss, feat='feats', mode=config.norm_mode_out)
            if in_mode == 'voc':
                voc_stfts = normalize(voc_stfts, feat='voc_stft', mode=config.norm_mode_in)
                yield np.array(voc_stfts), np.array(featss)
            elif in_mode == 'mix':
                mix_stfts = normalize(mix_stfts, feat='mix_stft', mode=config.norm_mode_in)
                yield np.array(mix_stfts), np.array(featss)


def main():
    get_stats(feat='feats')
    # numpy_to_h5py()
    # vg = val_generator()
    gen = get_batches()

    while True:
        mix, feats = next(gen)
        plt.figure(1)
        plt.subplot(211)
        plt.imshow(np.log(mix.reshape(-1,513)).T, aspect='auto', origin='lower')
        plt.subplot(212)
        plt.imshow(feats.reshape(-1,66).T, aspect='auto', origin='lower')
        plt.show()
        import pdb;pdb.set_trace()


if __name__ == '__main__':
    main()



get_batches()