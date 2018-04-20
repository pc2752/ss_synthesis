import tensorflow as tf


import matplotlib.pyplot as plt

import os
import sys
import time
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

import config
from data_to_h5py import get_batches, val_generator
import modules_tf as modules
import utils
from reduce import mgc_to_mfsc

def binary_cross(p,q):
    return -(p * tf.log(q + 1e-12) + (1 - p) * tf.log( 1 - q + 1e-12))

def train(_):
    with tf.Graph().as_default():
        
        input_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len,config.input_features),name='input_placeholder')
        tf.summary.histogram('inputs', input_placeholder)
        target_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len,config.output_features),name='target_placeholder')
        tf.summary.histogram('targets', target_placeholder)


        harm, ap, f0, vuv = modules.bi_static_stacked_RNN(input_placeholder)

        tf.summary.histogram('harm', harm)

        tf.summary.histogram('ap', ap)

        tf.summary.histogram('f0', f0)

        tf.summary.histogram('vuv', vuv)

        harm_loss = tf.reduce_sum(tf.abs(harm - target_placeholder[:,:,:60])*np.linspace(1.0,0.7,60))

        ap_loss = tf.reduce_sum(tf.abs(ap - target_placeholder[:,:,60:-2]))

        f0_loss = tf.reduce_sum(tf.abs(f0 - target_placeholder[:,:,-2:-1])*10.0) 

        # vuv_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=, logits=vuv))

        vuv_loss = tf.reduce_mean(tf.reduce_sum(binary_cross(target_placeholder[:,:,-1:],vuv)))

        loss = harm_loss + ap_loss + f0_loss + vuv_loss

        harm_summary = tf.summary.scalar('harm_loss', harm_loss)

        ap_summary = tf.summary.scalar('ap_loss', ap_loss)

        f0_summary = tf.summary.scalar('f0_loss', f0_loss)

        vuv_summary = tf.summary.scalar('vuv_loss', vuv_loss)

        loss_summary = tf.summary.scalar('total_loss', loss)

        global_step = tf.Variable(0, name='global_step', trainable=False)

        optimizer = tf.train.AdamOptimizer()

        train_function = optimizer.minimize(loss, global_step= global_step)

        # train_harm = optimizer.minimize(harm_loss, global_step= global_step)

        # train_ap = optimizer.minimize(ap_loss, global_step= global_step)

        # train_f0 = optimizer.minimize(f0_loss, global_step= global_step)

        # train_vuv = optimizer.minimize(vuv_loss, global_step= global_step)

        summary = tf.summary.merge_all()

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        saver = tf.train.Saver()
        sess = tf.Session()

        sess.run(init_op)

        ckpt = tf.train.get_checkpoint_state(config.log_dir)

        if ckpt and ckpt.model_checkpoint_path:
            print("Using the model in %s"%ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)


        train_summary_writer = tf.summary.FileWriter(config.log_dir+'train/', sess.graph)
        val_summary_writer = tf.summary.FileWriter(config.log_dir+'val/', sess.graph)

        
        start_epoch = int(sess.run(tf.train.get_global_step())/(config.batches_per_epoch_train))

        print("Start from: %d" % start_epoch)
        for epoch in xrange(start_epoch, config.num_epochs):
            data_generator = get_batches()
            start_time = time.time()

            epoch_loss_harm = 0
            epoch_loss_ap = 0
            epoch_loss_f0 = 0
            epoch_loss_vuv = 0
            epoch_total_loss = 0

            epoch_loss_harm_val = 0
            epoch_loss_ap_val = 0
            epoch_loss_f0_val = 0
            epoch_loss_vuv_val = 0
            epoch_total_loss_val = 0

            batch_num = 0
            batch_num_val = 0
            val_generator = get_batches(train_filename=config.h5py_file_val, batches_per_epoch=config.batches_per_epoch_val)

            with tf.variable_scope('Training'):

                for voc, feat in data_generator:

                    _, step_loss_harm, step_loss_ap, step_loss_f0, step_loss_vuv, step_total_loss = sess.run([train_function, harm_loss, ap_loss, f0_loss, vuv_loss, loss], feed_dict={input_placeholder: voc,target_placeholder: feat})

                    # _, step_loss_harm = sess.run([train_harm, harm_loss], feed_dict={input_placeholder: voc,target_placeholder: feat})
                    # _, step_loss_ap = sess.run([train_ap, ap_loss], feed_dict={input_placeholder: voc,target_placeholder: feat})
                    # _, step_loss_f0 = sess.run([train_f0, f0_loss], feed_dict={input_placeholder: voc,target_placeholder: feat})
                    # _, step_loss_vuv = sess.run([train_vuv, vuv_loss], feed_dict={input_placeholder: voc,target_placeholder: feat})

                    epoch_loss_harm+=step_loss_harm
                    epoch_loss_ap+=step_loss_ap
                    epoch_loss_f0+=step_loss_f0
                    epoch_loss_vuv+=step_loss_vuv
                    epoch_total_loss+=step_total_loss
                    utils.progress(batch_num,config.batches_per_epoch_train, suffix = 'training done')
                    batch_num+=1


                
                epoch_loss_harm = epoch_loss_harm/(config.batches_per_epoch_train *config.batch_size)
                epoch_loss_ap = epoch_loss_ap/(config.batches_per_epoch_train *config.batch_size)
                epoch_loss_f0 = epoch_loss_f0/(config.batches_per_epoch_train *config.batch_size)
                epoch_loss_vuv = epoch_loss_vuv/(config.batches_per_epoch_train *config.batch_size)
                epoch_total_loss = epoch_total_loss/(config.batches_per_epoch_train *config.batch_size)

                summary_str = sess.run(summary, feed_dict={input_placeholder: voc,target_placeholder: feat})
                train_summary_writer.add_summary(summary_str, epoch)
                # summary_writer.add_summary(summary_str_val, epoch)
                train_summary_writer.flush()

            with tf.variable_scope('Validation'):

                for voc, feat in val_generator:
                    step_loss_harm_val = sess.run(harm_loss, feed_dict={input_placeholder: voc,target_placeholder: feat})
                    step_loss_ap_val = sess.run(ap_loss, feed_dict={input_placeholder: voc,target_placeholder: feat})
                    step_loss_f0_val = sess.run(f0_loss, feed_dict={input_placeholder: voc,target_placeholder: feat})
                    step_loss_vuv_val = sess.run(vuv_loss, feed_dict={input_placeholder: voc,target_placeholder: feat})
                    step_total_loss_val = sess.run(loss, feed_dict={input_placeholder: voc,target_placeholder: feat})

                    epoch_loss_harm_val+=step_loss_harm_val
                    epoch_loss_ap_val+=step_loss_ap_val
                    epoch_loss_f0_val+=step_loss_f0_val
                    epoch_loss_vuv_val+=step_loss_vuv_val
                    epoch_total_loss_val+=step_total_loss_val

                    utils.progress(batch_num_val,config.batches_per_epoch_val, suffix = 'validiation done')
                    batch_num_val+=1

                epoch_loss_harm_val = epoch_loss_harm_val/(config.batches_per_epoch_train *config.batch_size)
                epoch_loss_ap_val = epoch_loss_ap_val/(config.batches_per_epoch_train *config.batch_size)
                epoch_loss_f0_val = epoch_loss_f0_val/(config.batches_per_epoch_train *config.batch_size)
                epoch_loss_vuv_val = epoch_loss_vuv_val/(config.batches_per_epoch_train *config.batch_size)
                epoch_total_loss_val = epoch_total_loss_val/(config.batches_per_epoch_train *config.batch_size)

                summary_str = sess.run(summary, feed_dict={input_placeholder: voc,target_placeholder: feat})
                val_summary_writer.add_summary(summary_str, epoch)
                # summary_writer.add_summary(summary_str_val, epoch)
                val_summary_writer.flush()

            duration = time.time() - start_time

            if (epoch+1) % config.print_every == 0:
                # Print status to stdout.
                print('epoch %d: Harm Training Loss = %.10f (%.3f sec)' % (epoch+1, epoch_loss_harm, duration))
                print('epoch %d: Ap Training Loss = %.10f (%.3f sec)' % (epoch+1, epoch_loss_ap, duration))
                print('epoch %d: F0 Training Loss = %.10f (%.3f sec)' % (epoch+1, epoch_loss_f0, duration))
                print('epoch %d: VUV Training Loss = %.10f (%.3f sec)' % (epoch+1, epoch_loss_vuv, duration))

                print('epoch %d: Harm Validation Loss = %.10f (%.3f sec)' % (epoch+1, epoch_loss_harm_val, duration))
                print('epoch %d: Ap Validation Loss = %.10f (%.3f sec)' % (epoch+1, epoch_loss_ap_val, duration))
                print('epoch %d: F0 Validation Loss = %.10f (%.3f sec)' % (epoch+1, epoch_loss_f0_val, duration))
                print('epoch %d: VUV Validation Loss = %.10f (%.3f sec)' % (epoch+1, epoch_loss_vuv_val, duration))

                # Update the events file.



            if (epoch + 1) % config.save_every == 0 or (epoch + 1) == config.num_epochs:
                checkpoint_file = os.path.join(config.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=epoch)


def synth_file(file_name, file_path=config.wav_dir, show_plots=True, save_file=True):
    with tf.Graph().as_default():
        
        input_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len,config.input_features),name='input_placeholder')

        harm, ap, f0, vuv = modules.bi_static_stacked_RNN(input_placeholder)

        saver = tf.train.Saver()


        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess = tf.Session()

        sess.run(init_op)

        ckpt = tf.train.get_checkpoint_state(config.log_dir)

        if ckpt and ckpt.model_checkpoint_path:
            print("Using the model in %s"%ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)


        # summary_writer = tf.summary.FileWriter(config.log_dir, sess.graph)

        mix_stft = utils.file_to_stft(os.path.join(file_path,file_name))

        targs = utils.input_to_feats(os.path.join(file_path,file_name))

        in_batches, nchunks_in = utils.generate_overlapadd(mix_stft)
        in_batches = utils.normalize(in_batches, 'mix_stft', mode=config.norm_mode_in)
        val_outer = []

        for in_batch in in_batches:
            val_harm, val_ap, val_f0, val_vuv = sess.run([harm, ap, f0, vuv], feed_dict={input_placeholder: in_batch})
            val_outs = np.concatenate((val_harm, val_ap, val_f0, val_vuv), axis=-1)
            val_outer.append(val_outs)

        val_outer = np.array(val_outer)
        val_outer = utils.overlapadd(val_outer, nchunks_in)    
        val_outer[:,-1] = np.round(val_outer[:,-1])
        val_outer = val_outer[:targs.shape[0],:]
        val_outer = np.clip(val_outer,0.0,1.0)

        #Test purposes only
        

        targs = utils.normalize(targs, 'feats', mode=config.norm_mode_out)

        

        if show_plots:

            # import pdb;pdb.set_trace()
        
            ins = val_outer[:,:60]
            outs = targs[:,:60]
            plt.figure(1)
            plt.subplot(211)
            plt.imshow(ins.T, origin='lower', aspect='auto')
            plt.subplot(212)
            plt.imshow(outs.T, origin='lower', aspect='auto')
            plt.figure(2)
            plt.subplot(211)
            plt.imshow(val_outer[:,60:-2].T, origin='lower', aspect='auto')
            plt.subplot(212)
            plt.imshow(targs[:,60:-2].T, origin='lower', aspect='auto')
            plt.figure(3)
            plt.plot(val_outer[:,-2], label = "Predicted Value")
            plt.plot(targs[:,-2], label="Ground Truth")
            plt.legend()
            plt.figure(4)
            plt.subplot(211)
            plt.plot(val_outer[:,-1])
            plt.subplot(212)
            plt.plot(targs[:,-1])
            plt.show()
        if save_file:
            # val_outer[:,-2:] = targs[:,-2:]

            val_outer = np.ascontiguousarray(utils.denormalize(val_outer,'feats', mode=config.norm_mode_out))
            utils.feats_to_audio(val_outer,file_name[:-4]+'_synth')
            print("File saved to %s" % config.val_dir+file_name[:-4]+'_synth.wav')

                




if __name__ == '__main__':
    if sys.argv[1] == '-train' or sys.argv[1] == '--train' or sys.argv[1] == '--t' or sys.argv[1] == '-t':
        print("Training")
        tf.app.run(main=train)
    elif sys.argv[1] == '-synth' or sys.argv[1] == '--synth' or sys.argv[1] == '--s' or sys.argv[1] == '-s':
        if len(sys.argv)<3:
            print("Please give a file to synthesize")
        else:
            file_name = sys.argv[2]
            if not file_name.endswith('.wav'):
                file_name = file_name+'.wav'
            print("Synthesizing File %s"% file_name)
            if '-p' in sys.argv or '--p' in sys.argv or '-plot' in sys.argv or '--plot' in sys.argv:
                
                if '-ns' in sys.argv or '--ns' in sys.argv or '-nosave' in sys.argv or '--nosave' in sys.argv:
                    print("Just showing plots for File %s"% sys.argv[2])
                    synth_file(file_name,show_plots=True, save_file=False)
                else:
                    print("Synthesizing File %s And Showing Plots"% sys.argv[2])
                    synth_file(file_name,show_plots=True, save_file=True)
            else:
                print("Synthesizing File %s, Not Showing Plots"% sys.argv[2])

    elif sys.argv[1] == '-help' or sys.argv[1] == '--help' or sys.argv[1] == '--h' or sys.argv[1] == '-h':
        print("%s --train to train the model"%sys.argv[0])
        print("%s --synth <filename> to synthesize file"%sys.argv[0])
        print("%s --synth <filename> -- plot to synthesize file and show plots"%sys.argv[0])
        print("%s --synth <filename> -- plot --ns to just show plots"%sys.argv[0])
  


