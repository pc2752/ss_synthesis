import tensorflow as tf

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
from reduce import mgc_to_mfsc




def eval_file():
    file_path=config.wav_dir

    # log_dir = './log_ikala_notrain/'
    log_dir = config.log_dir


    mode =0 

    stat_file = h5py.File(config.stat_dir+'stats.hdf5', mode='r')

    max_feat = np.array(stat_file["feats_maximus"])
    min_feat = np.array(stat_file["feats_minimus"])
    max_voc = np.array(stat_file["voc_stft_maximus"])
    min_voc = np.array(stat_file["voc_stft_minimus"])
    max_back = np.array(stat_file["back_stft_maximus"])
    min_back = np.array(stat_file["back_stft_minimus"])
    max_mix = np.array(max_voc)+np.array(max_back)

    with tf.Graph().as_default():
        
        input_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len,config.input_features),name='input_placeholder')


        with tf.variable_scope('First_Model') as scope:
            harm, ap, f0, vuv = modules.nr_wavenet(input_placeholder)


        saver = tf.train.Saver(max_to_keep= config.max_models_to_keep)


        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess = tf.Session()

        sess.run(init_op)

        ckpt = tf.train.get_checkpoint_state(log_dir)

        if ckpt and ckpt.model_checkpoint_path:
            print("Using the model in %s"%ckpt.model_checkpoint_path)
            # saver.restore(sess, ckpt.model_checkpoint_path)
            saver.restore(sess, './log/model.ckpt-59')
            

        # import pdb;pdb.set_trace()

        files = [x for x in os.listdir(config.wav_dir) if x.endswith('.wav') and not x.startswith('.')]
        diffs = []
        count = 0
        for file_name in files:

            count+=1

            mix_stft = utils.file_to_stft(os.path.join(file_path,file_name), mode = mode)

            targs = utils.input_to_feats(os.path.join(file_path,file_name), mode = mode)

            # f0_sac = utils.file_to_sac(os.path.join(file_path,file_name))
            # f0_sac = (f0_sac-min_feat[-2])/(max_feat[-2]-min_feat[-2])

            in_batches, nchunks_in = utils.generate_overlapadd(mix_stft)
            in_batches = in_batches/max_mix
            # in_batches = utils.normalize(in_batches, 'mix_stft', mode=config.norm_mode_in)
            val_outer = []

            first_pred = []

            cleaner = []

            gan_op =[]

            for in_batch in in_batches:
                val_harm, val_ap, val_f0, val_vuv = sess.run([harm, ap, f0, vuv], feed_dict={input_placeholder: in_batch})
                if config.use_gan:
                    val_op = sess.run(gen_op, feed_dict={input_placeholder: in_batch})
                    
                    gan_op.append(val_op)

                # first_pred.append(harm1)
                # cleaner.append(val_harm)
                val_harm = val_harm
                val_outs = np.concatenate((val_harm, val_ap, val_f0, val_vuv), axis=-1)
                val_outer.append(val_outs)

            val_outer = np.array(val_outer)
            val_outer = utils.overlapadd(val_outer, nchunks_in)    
            val_outer[:,-1] = np.round(val_outer[:,-1])
            val_outer = val_outer[:targs.shape[0],:]
            val_outer = np.clip(val_outer,0.0,1.0)

            #Test purposes only
            # first_pred = np.array(first_pred)
            # first_pred = utils.overlapadd(first_pred, nchunks_in) 

            # cleaner = np.array(cleaner)
            # cleaner = utils.overlapadd(cleaner, nchunks_in) 

            f0_output = val_outer[:,-2]*((max_feat[-2]-min_feat[-2])+min_feat[-2])
            f0_output = f0_output*(1-targs[:,-1])
            f0_output = utils.new_base_to_hertz(f0_output)
            f0_gt = targs[:,-2]
            f0_gt = f0_gt*(1-targs[:,-1])
            f0_gt = utils.new_base_to_hertz(f0_gt)
            f0_outputs = []
            gt_outputs = []
            for i,f0_o in enumerate(f0_output):
                f0_outputs.append(str(i*0.00580498866*10000000)+' '+str(f0_o))

            for i,f0_o in enumerate(f0_gt):
                gt_outputs.append(str(i*0.00580498866*10000000)+' '+str(f0_o))


            utils.list_to_file(f0_outputs,'./ikala_eval/net_out/'+file_name[:-4]+'.pv')
            utils.list_to_file(gt_outputs,'./ikala_eval/sac_gt/'+file_name[:-4]+'.pv')
        #     f0_difference = np.nan_to_num(abs(f0_gt-f0_output))
        #     f0_greater = np.where(f0_difference>config.f0_threshold)

        #     diff_per = f0_greater[0].shape[0]/len(f0_output)
        #     diffs.append(str(1-diff_per))
            utils.progress(count, len(files))
        #     # import pdb;pdb.set_trace()
        # utils.list_to_file(diffs,'./ikala_eval/correct_percentage.txt')
        # utils.list_to_file(files,'./ikala_eval/ikala_files.txt')

def cross_comp():
    ikala_gt_dir = './ikala_eval/ikala_gt/'
    net_out_dir = './ikala_eval/net_out/'
    sac_gt_dir = './ikala_eval/sac_gt/'

    file_list = [x for x in os.listdir(net_out_dir) if x.endswith('.pv') and not x.startswith('.')]

    output = []

    for file_name in file_list:
        out_time, out_freq = mir_eval.io.load_time_series(net_out_dir+file_name)



        for i, freq in enumerate(out_freq):
            if float(freq) == 0.0 :
                out_freq[i] = 0
            else:
                out_freq[i] = utils.f0_to_hertz(float(freq))





        out_freq, out_vuv = mir_eval.melody.freq_to_voicing(out_freq)




        ref_time_o, ref_freq_o = mir_eval.io.load_time_series(ikala_gt_dir+file_name)

        for i, freq in enumerate(ref_freq_o):
            if float(freq) == 0.0 :
                ref_freq_o[i] = 0
            else:
                ref_freq_o[i] = utils.f0_to_hertz(float(freq))      

        plt.figure(1)
        plt.plot(out_freq)
        plt.plot(ref_freq_o)
        plt.show()

        # import pdb;pdb.set_trace()  

        haha = mir_eval.melody.evaluate(ref_time_o,ref_freq_o,out_time,out_freq)

        out_string = file_name

        for key in haha.keys():
            out_string = out_string+';'+str(haha[key])

        # import pdb;pdb.set_trace()

        # ref_freq_o, ref_vuv_o = mir_eval.melody.freq_to_voicing(ref_freq_o)
        # ref_freq,ref_vuv = mir_eval.melody.resample_melody_series(ref_time_o,ref_freq_o, ref_vuv_o,out_time) 

        # out_freq_o, out_vuv_o = mir_eval.melody.resample_melody_series(out_time,out_freq, out_vuv,ref_time_o) 

        # raw_pitch_accuracy_10_o = mir_eval.melody.raw_pitch_accuracy(ref_vuv_o,ref_freq_o,out_vuv_o,out_freq_o, cent_tolerance = 10)  
        # raw_pitch_accuracy_25_o = mir_eval.melody.raw_pitch_accuracy(ref_vuv_o,ref_freq_o,out_vuv_o,out_freq_o, cent_tolerance = 25)  
        # raw_pitch_accuracy_50_o = mir_eval.melody.raw_pitch_accuracy(ref_vuv_o,ref_freq_o,out_vuv_o,out_freq_o, cent_tolerance = 50)  
        # raw_chroma_accuracy_o = mir_eval.melody.raw_chroma_accuracy(ref_vuv_o,ref_freq_o,out_vuv_o,out_freq_o)  


        # raw_pitch_accuracy_10 = mir_eval.melody.raw_pitch_accuracy(ref_vuv,ref_freq,out_vuv,out_freq, cent_tolerance = 10)  
        # raw_pitch_accuracy_25 = mir_eval.melody.raw_pitch_accuracy(ref_vuv,ref_freq,out_vuv,out_freq, cent_tolerance = 25)  
        # raw_pitch_accuracy_50 = mir_eval.melody.raw_pitch_accuracy(ref_vuv,ref_freq,out_vuv,out_freq, cent_tolerance = 50)  
        # raw_chroma_accuracy = mir_eval.melody.raw_chroma_accuracy(ref_vuv,ref_freq,out_vuv,out_freq)  

        # import pdb;pdb.set_trace()
        output.append(out_string)

        # output.append(file_name+';'+str(raw_pitch_accuracy_10)+';'+str(raw_pitch_accuracy_25)+';'+str(raw_pitch_accuracy_50)+';'+str(raw_chroma_accuracy)+';'+str(raw_pitch_accuracy_10_o)+';'+str(raw_pitch_accuracy_25_o)+';'+str(raw_pitch_accuracy_50_o)+';'+str(raw_chroma_accuracy_o))

    utils.list_to_file(output,'./ikala_eval/mir_eval_results.txt')







if __name__ == '__main__':
    # eval_file()
    cross_comp()