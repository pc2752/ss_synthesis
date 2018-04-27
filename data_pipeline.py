import numpy as np
import os
import time
import h5py

import config

def data_gen(mode = 'Train'):



    voc_list = [x for x in os.listdir(config.voice_dir) if x.endswith('.hdf5') and not x.startswith('._')]

    back_list = [x for x in os.listdir(config.backing_dir) if x.endswith('.hdf5') and not x.startswith('._')]

    mix_list = [x for x in os.listdir(config.backing_dir) if x.endswith('.hdf5') and x.startswith('ikala')]

    train_list = mix_list[:int(len(mix_list)*config.split)]

    val_list = mix_list[int(len(mix_list)*config.split):]



    max_files_to_process = int(config.batch_size/config.samples_per_file)

    if mode == "Train":

        for k in range(config.batches_per_epoch_train):

            inputs = []
            targets = []

            # start_time = time.time()

            for i in range(max_files_to_process):
                augment = np.random.rand(1)<config.aug_prob

                # augment = True

                if augment:

                    # print("Augmenting")

                    voc_index = np.random.randint(0,len(voc_list))
                    voc_to_open = voc_list[voc_index]

                    voc_file = h5py.File(config.voice_dir+voc_to_open, "r")

                    voc_stft = voc_file['voc_stft']

                    feats = voc_file['feats']

                    back_index = np.random.randint(0,len(back_list))

                    back_to_open = back_list[back_index]

                    back_file = h5py.File(config.backing_dir+back_to_open, "r")

                    back_stft = back_file['back_stft']


                    for j in range(config.samples_per_file):
                            voc_idx = np.random.randint(0,len(voc_stft)-config.max_phr_len)
                            bac_idx = np.random.randint(0,len(back_stft)-config.max_phr_len)
                            mix_stft = voc_stft[voc_idx:voc_idx+config.max_phr_len,:] + back_stft[bac_idx:bac_idx+config.max_phr_len,:]
                            targets.append(feats[voc_idx:voc_idx+config.max_phr_len,:])
                            inputs.append(mix_stft)

        
                else:
                    # print("not augmenting")

                    file_index = np.random.randint(0,len(train_list))

                    tr_file = train_list[file_index]

                    voc_file = h5py.File(config.voice_dir+tr_file, "r")


                    feats = voc_file['feats'] 

                    mix_file = h5py.File(config.backing_dir+tr_file, "r")

                    mix_stft = mix_file["mix_stft"]

                    for j in range(config.samples_per_file):
                        voc_idx = np.random.randint(0,len(mix_stft)-config.max_phr_len)

                        inputs.append(mix_stft[voc_idx:voc_idx+config.max_phr_len,:])

                        targets.append(feats[voc_idx:voc_idx+config.max_phr_len,:])

            targets = np.array(targets)
            inputs = np.array(inputs)

            yield inputs, targets

    else:
        # print('val')
        for k in range(config.batches_per_epoch_val):

            inputs = []
            targets = []

            # start_time = time.time()

            for i in range(max_files_to_process):

                    file_index = np.random.randint(0,len(val_list))

                    tr_file = val_list[file_index]

                    voc_file = h5py.File(config.voice_dir+tr_file, "r")


                    feats = voc_file['feats'] 

                    mix_file = h5py.File(config.backing_dir+tr_file, "r")

                    mix_stft = mix_file["mix_stft"]

                    for j in range(config.samples_per_file):
                        voc_idx = np.random.randint(0,len(mix_stft)-config.max_phr_len)

                        inputs.append(mix_stft[voc_idx:voc_idx+config.max_phr_len,:])

                        targets.append(feats[voc_idx:voc_idx+config.max_phr_len,:])

            targets = np.array(targets)
            inputs = np.array(inputs)
            yield inputs, targets

        # import pdb;pdb.set_trace()



def main():
    # get_stats(feat='feats')
    gen = data_gen(mode='val')
    # vg = val_generator()
    # gen = get_batches()


    import pdb;pdb.set_trace()


if __name__ == '__main__':
    main()