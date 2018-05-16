import numpy as np
import os
import time
import h5py

import matplotlib.pyplot as plt

import config
import utils


def gen_train_val():
    mix_list = [x for x in os.listdir(config.backing_dir) if x.endswith('.hdf5') and x.startswith('ikala')]

    train_list = mix_list[:int(len(mix_list)*config.split)]

    val_list = mix_list[int(len(mix_list)*config.split):]

    utils.list_to_file(val_list,config.log_dir+'train_val.txt')




def data_gen(mode = 'Train'):



    voc_list = [x for x in os.listdir(config.voice_dir) if x.endswith('.hdf5') and not x.startswith('._') and not x.startswith('mir')]

    back_list = [x for x in os.listdir(config.backing_dir) if x.endswith('.hdf5') and not x.startswith('._') and not x.startswith('mir')]

    mix_list = [x for x in os.listdir(config.backing_dir) if x.endswith('.hdf5') and x.startswith('ikala')]

    train_list = mix_list[:int(len(mix_list)*config.split)]

    val_list = mix_list[int(len(mix_list)*config.split):]

    stat_file = h5py.File(config.stat_dir+'stats.hdf5', mode='r')

    max_feat = np.array(stat_file["feats_maximus"])
    min_feat = np.array(stat_file["feats_minimus"])
    max_voc = np.array(stat_file["voc_stft_maximus"])
    min_voc = np.array(stat_file["voc_stft_minimus"])
    max_back = np.array(stat_file["back_stft_maximus"])
    min_back = np.array(stat_file["back_stft_minimus"])
    max_mix = np.array(max_voc)+np.array(max_back)
    stat_file.close()
    # min_mix = 


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

                    # print("Vocal file: %s" % voc_file)

                    voc_stft = voc_file['voc_stft']

                    feats = voc_file['feats']

                    back_index = np.random.randint(0,len(back_list))

                    back_to_open = back_list[back_index]

                    back_file = h5py.File(config.backing_dir+back_to_open, "r")

                    # print("Backing file: %s" % back_file)

                    back_stft = back_file['back_stft']


                    for j in range(config.samples_per_file):
                            voc_idx = np.random.randint(0,len(voc_stft)-config.max_phr_len)
                            bac_idx = np.random.randint(0,len(back_stft)-config.max_phr_len)
                            mix_stft = voc_stft[voc_idx:voc_idx+config.max_phr_len,:] + back_stft[bac_idx:bac_idx+config.max_phr_len,:]*np.clip(np.random.rand(1),0.0,0.9)
                            targets.append(feats[voc_idx:voc_idx+config.max_phr_len,:])
                            inputs.append(mix_stft)

        
                else:
                    # print("not augmenting")

                    file_index = np.random.randint(0,len(train_list))

                    tr_file = train_list[file_index]

                    voc_file = h5py.File(config.voice_dir+tr_file, "r")

                    # print("Vocal file: %s" % voc_file)


                    feats = voc_file['feats'] 

                    mix_file = h5py.File(config.backing_dir+tr_file, "r")

                    mix_stft = mix_file["mix_stft"]

                    for j in range(config.samples_per_file):
                        voc_idx = np.random.randint(0,len(mix_stft)-config.max_phr_len)

                        inputs.append(mix_stft[voc_idx:voc_idx+config.max_phr_len,:])

                        targets.append(feats[voc_idx:voc_idx+config.max_phr_len,:])

            targets = np.array(targets)
            inputs = np.array(inputs)

            targets = (targets-min_feat)/(max_feat-min_feat)
            inputs = inputs/max_mix

            # inputs = np.clip(inputs,0.0,1.0)

            

            # assert inputs.max() <= 1.0

            # assert targets.max() <= 1.0

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

            # inputs = np.clip(inputs,0.0,1.0)

            # targets = utils.normalize(targets, 'feats', mode=config.norm_mode_out)
            targets = (targets-min_feat)/(max_feat-min_feat)
            inputs = inputs/max_mix

            # assert inputs.max() <= 1.0

            # assert targets.max() <= 1.0

            yield inputs, targets

        # import pdb;pdb.set_trace()

def get_stats():
    voc_list = [x for x in os.listdir(config.voice_dir) if x.endswith('.hdf5') and not x.startswith('._')]

    back_list = [x for x in os.listdir(config.backing_dir) if x.endswith('.hdf5') and not x.startswith('._')]

    max_feat = np.zeros(66)
    min_feat = np.ones(66)*1000

    max_voc = np.zeros(513)
    min_voc = np.ones(513)*1000

    max_mix = np.zeros(513)
    min_mix = np.ones(513)*1000    

    for voc_to_open in voc_list:

        voc_file = h5py.File(config.voice_dir+voc_to_open, "r")

        voc_stft = voc_file['voc_stft']

        feats = voc_file['feats']

        maxi_voc_stft = np.array(voc_stft).max(axis=0)

        # if np.array(feats).min()<0:
        #     import pdb;pdb.set_trace()

        for i in range(len(maxi_voc_stft)):
            if maxi_voc_stft[i]>max_voc[i]:
                max_voc[i] = maxi_voc_stft[i]

        mini_voc_stft = np.array(voc_stft).min(axis=0)

        for i in range(len(mini_voc_stft)):
            if mini_voc_stft[i]<min_voc[i]:
                min_voc[i] = mini_voc_stft[i]

        maxi_voc_feat = np.array(feats).max(axis=0)

        for i in range(len(maxi_voc_feat)):
            if maxi_voc_feat[i]>max_feat[i]:
                max_feat[i] = maxi_voc_feat[i]

        mini_voc_feat = np.array(feats).min(axis=0)

        for i in range(len(mini_voc_feat)):
            if mini_voc_feat[i]<min_feat[i]:
                min_feat[i] = mini_voc_feat[i]   

    for voc_to_open in back_list:

        voc_file = h5py.File(config.backing_dir+voc_to_open, "r")

        voc_stft = voc_file["back_stft"]

        maxi_voc_stft = np.array(voc_stft).max(axis=0)

        # if np.array(feats).min()<0:
        #     import pdb;pdb.set_trace()

        for i in range(len(maxi_voc_stft)):
            if maxi_voc_stft[i]>max_mix[i]:
                max_mix[i] = maxi_voc_stft[i]

        mini_voc_stft = np.array(voc_stft).min(axis=0)

        for i in range(len(mini_voc_stft)):
            if mini_voc_stft[i]<min_mix[i]:
                min_mix[i] = mini_voc_stft[i]

    hdf5_file = h5py.File(config.stat_dir+'stats.hdf5', mode='w')

    hdf5_file.create_dataset("feats_maximus", [66], np.float32) 
    hdf5_file.create_dataset("feats_minimus", [66], np.float32)   
    hdf5_file.create_dataset("voc_stft_maximus", [513], np.float32) 
    hdf5_file.create_dataset("voc_stft_minimus", [513], np.float32)   
    hdf5_file.create_dataset("back_stft_maximus", [513], np.float32) 
    hdf5_file.create_dataset("back_stft_minimus", [513], np.float32)   

    hdf5_file["feats_maximus"][:] = max_feat
    hdf5_file["feats_minimus"][:] = min_feat
    hdf5_file["voc_stft_maximus"][:] = max_voc
    hdf5_file["voc_stft_minimus"][:] = min_voc
    hdf5_file["back_stft_maximus"][:] = max_mix
    hdf5_file["back_stft_minimus"][:] = min_mix

    # import pdb;pdb.set_trace()

    hdf5_file.close()


def main():
    gen_train_val()
    # get_stats()
    # gen = data_gen()
    # while True :
    #     inputs, targets = next(gen)

    #     plt.subplot(411)
    #     plt.imshow(np.log(1+inputs.reshape(-1,513).T),aspect='auto',origin='lower')
    #     plt.subplot(412)
    #     plt.imshow(targets.reshape(-1,66)[:,:64].T,aspect='auto',origin='lower')
    #     plt.subplot(413)
    #     plt.plot(targets.reshape(-1,66)[:,-2])
    #     plt.subplot(414)
    #     plt.plot(targets.reshape(-1,66)[:,-1])

    #     plt.show()
    #     # vg = val_generator()
    #     # gen = get_batches()


    #     import pdb;pdb.set_trace()


if __name__ == '__main__':
    main()