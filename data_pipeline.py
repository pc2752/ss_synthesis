import numpy as np
import os
import time
import h5py

import matplotlib.pyplot as plt
import collections
import config
import utils


def gen_train_val():
    mix_list = [x for x in os.listdir(config.backing_dir) if x.endswith('.hdf5') and x.startswith('med') ]

    train_list = mix_list[:int(len(mix_list)*config.split)]

    val_list = mix_list[int(len(mix_list)*config.split):]

    utils.list_to_file(val_list,config.log_dir+'val_files.txt')

    utils.list_to_file(train_list,config.log_dir+'train_files.txt')




def data_gen(mode = 'Train'):



    # voc_list = [x for x in os.listdir(config.voice_dir) if x.endswith('.hdf5') and not x.startswith('._') and not x.startswith('mir') and not x.startswith('ikala') and not x.startswith('nus_KENN') and not x == 'nus_MCUR_read_17.hdf5']

    voc_list = [x for x in os.listdir(config.voice_dir) if x.endswith('.hdf5') and x.startswith('nus') and not x.startswith('nus_KENN') and not x == 'nus_MCUR_read_17.hdf5']

    back_list = [x for x in os.listdir(config.backing_dir) if x.endswith('.hdf5') and not x.startswith('._') and not x.startswith('mir') and not x.startswith('med')]

    mix_list = [x for x in os.listdir(config.backing_dir) if x.endswith('.hdf5') and x.startswith('med') ]

    # train_list = mix_list[:int(len(mix_list)*config.split)]

    # val_list = mix_list[int(len(mix_list)*config.split):]

    # import pdb;pdb.set_trace()

    train_list = mix_list

    val_list = [x for x in os.listdir(config.backing_dir) if x.endswith('.hdf5') and x.startswith('ikala') ]  

    # import pdb;pdb.set_trace()

    stat_file = h5py.File(config.stat_dir+'stats.hdf5', mode='r')

    max_feat = np.array(stat_file["feats_maximus"])
    min_feat = np.array(stat_file["feats_minimus"])
    max_voc = np.array(stat_file["voc_stft_maximus"])
    min_voc = np.array(stat_file["voc_stft_minimus"])
    max_back = np.array(stat_file["back_stft_maximus"])
    min_back = np.array(stat_file["back_stft_minimus"])
    max_mix = np.array(max_voc)+np.array(max_back)
    stat_file.close()

    # import pdb;pdb.set_trace()
    # min_mix = 


    max_files_to_process = int(config.batch_size/config.samples_per_file)

    if mode == "Train":

        for k in range(config.batches_per_epoch_train):

            inputs = []
            targets_f0_1 = []
            targets_f0_2 = []

            # start_time = time.time()

            for i in range(max_files_to_process):
                augment = np.random.rand(1)<config.aug_prob

                Flag = False

                # augment = True

                if augment:

                    # print("Augmenting")

                    voc_index = np.random.randint(0,len(voc_list))
                    voc_to_open = voc_list[voc_index]



                    voc_file = h5py.File(config.voice_dir+voc_to_open, "r")



                    # print("Vocal file: %s" % voc_file)

                    voc_stft = np.array(voc_file['voc_stft'])

                    # plt.imshow(np.log(voc_stft.T), aspect = 'auto', origin = 'lower')
                    # plt.show()

                    feats = np.array(voc_file['feats'])

                    f0 = feats[:,-2]

                    med = np.median(f0[f0 > 0])

                    f0[f0==0] = med

                    f0_nor = (f0 - min_feat[-2])/(max_feat[-2]-min_feat[-2])

                    f0_quant = np.rint(f0_nor*255) + 1

                    f0_quant = f0_quant * (1-feats[:,-1]) 


                    back_index = np.random.randint(0,len(back_list))

                    back_to_open = back_list[back_index]

                    back_file = h5py.File(config.backing_dir+back_to_open, "r")
                    if voc_to_open.startswith('nus'):
                        Flag = True
                        pho_target = np.array(voc_file["phonemes"])
                        pho_targs = []
                    else:
                        Flag = False

                    # print("Backing file: %s" % back_file)

                    back_stft = back_file['back_stft']


                    for j in range(config.samples_per_file):
                            voc_idx = np.random.randint(0,len(voc_stft)-config.max_phr_len)
                            bac_idx = np.random.randint(0,len(back_stft)-config.max_phr_len)
                            mix_stft = voc_stft[voc_idx:voc_idx+config.max_phr_len,:]
                            # *np.clip(np.random.rand(1),0.5,0.9) + back_stft[bac_idx:bac_idx+config.max_phr_len,:]*np.clip(np.random.rand(1),0.0,0.9)+ np.random.rand(config.max_phr_len,config.input_features)*np.clip(np.random.rand(1),0.0,config.noise_threshold)
                            targets_f0_1.append(f0_quant[voc_idx:voc_idx+config.max_phr_len])
                            targets_f0_2.append(f0_quant[voc_idx:voc_idx+config.max_phr_len])
                            if Flag:
                                pho_targs.append(pho_target[voc_idx:voc_idx+config.max_phr_len])
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

                    # import pdb;pdb.set_trace()

                    for j in range(config.samples_per_file):
                        voc_idx = np.random.randint(0,len(mix_stft)-config.max_phr_len)

                        inputs.append(mix_stft[voc_idx:voc_idx+config.max_phr_len,:] + np.random.rand(config.max_phr_len,config.input_features)*config.noise_threshold)

                        targets.append(feats[voc_idx:voc_idx+config.max_phr_len,:])

            targets_f0_1 = np.array(targets_f0_1)

            targets_f0_2 = np.array(targets_f0_2)
            
            inputs = np.array(inputs)

            # f0_tens = (np.floor(targets[:,:,-2:-1]/10)*10 - 30)/50


            # targets = (targets-min_feat)/(max_feat-min_feat)
            inputs_norm = inputs/(inputs.max(axis = 1).max(axis = 0))

            # inputs_norm = inputs/max_voc

            # plt.figure(1)
            # plt.subplot(211)
            # plt.imshow(np.log(inputs.reshape(-1,513).T),aspect = 'auto', origin ='lower' )
            # plt.subplot(212)
            # plt.imshow(np.log(inputs_norm.reshape(-1,513).T),aspect = 'auto', origin ='lower' )



            # import pdb;pdb.set_trace()

            if Flag:
                yield inputs_norm, targets_f0_1, targets_f0_2, np.array(pho_targs)

            else:
                yield inputs, targets, None, Flag

    else:

        val_file_index = np.random.randint(0,len(val_list)-config.val_files)

        for file_name in [x for x in os.listdir(config.voice_dir) if x.startswith('nus_KENN_sing') or x == 'nus_MCUR_read_17.hdf5']:

            if not file_name.startswith('nus'):
                Flag = False

                voc_file = h5py.File(config.voice_dir+file_name, "r")


                feats = voc_file['feats'] 

                mix_file = h5py.File(config.backing_dir+file_name, "r")

                mix_stft = mix_file["mix_stft"]

                # import pdb;pdb.set_trace()

                lent = len(mix_stft)

                in_batches, nchunks_in = utils.generate_overlapadd(mix_stft)

                targ_batches, nchunks_in = utils.generate_overlapadd(feats)

                count = 0

                for inputs, targets in zip(in_batches,targ_batches):
                    # f0_ones = (np.floor(targets[:,:,-2:-1]/10) - 30)/50

                    targets = (targets-min_feat)/(max_feat-min_feat)
                    inputs = inputs/max_mix

                    count+=1


                    if config.pred_mode == 'f0':

                        yield inputs, np.concatenate((f0_ones,targets[:,:,-2:]),axis = -1), nchunks_in, lent, count, in_batches.shape[0]
                    else:
                        yield inputs, targets, None, nchunks_in, lent, count, in_batches.shape[0], Flag
            else:
                count = 0
                Flag = True


                voc_file = h5py.File(config.voice_dir+file_name, "r")
                feats = voc_file['feats'] 
                voc_stft = voc_file['voc_stft']

                lent = len(voc_stft)

                pho_target = np.array(voc_file["phonemes"]).reshape(-1,1)
                in_batches, nchunks_in = utils.generate_overlapadd(voc_stft)

                targ_batches, nchunks_in = utils.generate_overlapadd(feats)
                pho_batches, nchunks_in = utils.generate_overlapadd(pho_target)

                for inputs, targets, phos in zip(in_batches,targ_batches,pho_batches):
                    # f0_ones = (np.floor(targets[:,:,-2:-1]/10) - 30)/50

                    targets = (targets-min_feat)/(max_feat-min_feat)
                    inputs = inputs/max_mix


                    count+=1


                    if config.pred_mode == 'f0':

                        yield inputs, np.concatenate((f0_ones,targets[:,:,-2:]),axis = -1), nchunks_in, lent, count, in_batches.shape[0]
                    else:
                        yield inputs, targets, phos.reshape(config.batch_size,-1), nchunks_in, lent, count, in_batches.shape[0], Flag



def get_stats():
    voc_list = [x for x in os.listdir(config.voice_dir) if x.endswith('.hdf5') and not x.startswith('._') and not x.startswith('mir')]

    back_list = [x for x in os.listdir(config.backing_dir) if x.endswith('.hdf5') and not x.startswith('._') and not x.startswith('mir') and not x.startswith('med')]

    max_feat = np.zeros(66)
    min_feat = np.ones(66)*1000

    max_voc = np.zeros(513)
    min_voc = np.ones(513)*1000

    max_mix = np.zeros(513)
    min_mix = np.ones(513)*1000    

    for voc_to_open in voc_list:

        voc_file = h5py.File(config.voice_dir+voc_to_open, "r")

        voc_stft = voc_file['voc_stft']

        feats = np.array(voc_file['feats'])

        f0 = feats[:,-2]

        med = np.median(f0[f0 > 0])

        f0[f0==0] = med

        feats[:,-2] = f0

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


def get_stats_phonems():

    phon=collections.Counter([])

    voc_list = [x for x in os.listdir(config.voice_dir) if x.endswith('.hdf5') and x.startswith('nus') and not x.startswith('nus_KENN') and not x == 'nus_MCUR_read_17.hdf5']

    for voc_to_open in voc_list:

        voc_file = h5py.File(config.voice_dir+voc_to_open, "r")
        pho_target = np.array(voc_file["phonemes"])
        phon += collections.Counter(pho_target)
    phonemas_weights = np.zeros(41)
    for pho in phon:
        phonemas_weights[pho] = phon[pho]

    phonemas_above_threshold = [config.phonemas[x[0]] for x in np.argwhere(phonemas_weights>70000)]

    pho_order = phonemas_weights.argsort()

    # phonemas_weights = 1.0/phonemas_weights
    # phonemas_weights = phonemas_weights/sum(phonemas_weights)
    import pdb;pdb.set_trace()


def main():
    # gen_train_val()
    # get_stats_phonems()
    gen = data_gen('Train')
    while True :
        start_time = time.time()
        inputs, targets_f0_1, targets_f0_2, pho_targs = next(gen)
        print(time.time()-start_time)

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


        import pdb;pdb.set_trace()


if __name__ == '__main__':
    main()