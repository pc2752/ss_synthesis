import tensorflow as tf


import matplotlib.pyplot as plt

import os
import sys
import time
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import h5py

import config
from data_pipeline_m1 import data_gen
import modules_tf as modules
import utils
from reduce import mgc_to_mfsc


def binary_cross(p,q):
    return -(p * tf.log(q + 1e-12) + (1 - p) * tf.log( 1 - q + 1e-12))

def train(_):
    stat_file = h5py.File(config.stat_dir+'stats.hdf5', mode='r')
    max_feat = np.array(stat_file["feats_maximus"])
    min_feat = np.array(stat_file["feats_minimus"])
    with tf.Graph().as_default():
        
        input_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len,config.input_features),name='input_placeholder')
        tf.summary.histogram('inputs', input_placeholder)
        target_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len,config.output_features),name='target_placeholder')
        tf.summary.histogram('targets', target_placeholder)

        with tf.variable_scope('First_Model') as scope:
            harm, ap, f0, vuv = modules.nr_wavenet(input_placeholder)

            # tf.summary.histogram('initial_output', op)

            tf.summary.histogram('harm', harm)

            tf.summary.histogram('ap', ap)

            tf.summary.histogram('f0', f0)

            tf.summary.histogram('vuv', vuv)

        if config.use_gan:

            with tf.variable_scope('Generator') as scope: 
                gen_op = modules.GAN_generator(harm)
            with tf.variable_scope('Discriminator') as scope: 
                D_real = modules.GAN_discriminator(target_placeholder[:,:,:60],input_placeholder)
                scope.reuse_variables()
                D_fake = modules.GAN_discriminator(gen_op+harmy,input_placeholder)

            # Comment out these lines to train without GAN

            D_loss_real = -tf.reduce_mean(tf.log(D_real + 1e-12))
            D_loss_fake = -tf.reduce_mean(tf.log(1. - (D_fake + 1e-12)))

            D_loss = D_loss_real+D_loss_fake

            D_summary_real = tf.summary.scalar('Discriminator_Loss_Real', D_loss_real)
            D_summary_fake = tf.summary.scalar('Discriminator_Loss_Fake', D_loss_fake)



            G_loss_GAN = -tf.reduce_mean(tf.log(D_fake + 1e-12)) 
            G_loss_diff = tf.reduce_sum(tf.abs(gen_op+harmy - target_placeholder[:,:,:60])*(1-target_placeholder[:,:,-1:]))*0.5
            G_loss = G_loss_GAN+G_loss_diff

            G_summary_GAN = tf.summary.scalar('Generator_Loss_GAN', G_loss_GAN)
            G_summary_diff = tf.summary.scalar('Generator_Loss_diff', G_loss_diff)


            vars = tf.trainable_variables()

            # import pdb;pdb.set_trace()
            
            d_params = [v for v in vars if v.name.startswith('Discriminator/D')]
            g_params = [v for v in vars if v.name.startswith('Generator/G')]

            # import pdb;pdb.set_trace()

            # d_optimizer_grad = tf.train.GradientDescentOptimizer(learning_rate=config.gan_lr).minimize(D_loss, var_list=d_params)
            # g_optimizer = tf.train.GradientDescentOptimizer(learning_rate=config.gan_lr).minimize(G_loss, var_list=g_params)

            d_optimizer = tf.train.GradientDescentOptimizer(learning_rate=config.gan_lr).minimize(D_loss, var_list=d_params)
            # g_optimizer_diff = tf.train.AdamOptimizer(learning_rate=config.gan_lr).minimize(G_loss_diff, var_list=g_params)
            g_optimizer = tf.train.AdamOptimizer(learning_rate=config.gan_lr).minimize(G_loss, var_list=g_params)

        # initial_loss = tf.reduce_sum(tf.abs(op - target_placeholder[:,:,:60])*np.linspace(1.0,0.7,60)*(1-target_placeholder[:,:,-1:]))

        harm_loss = tf.reduce_sum(tf.abs(harm - target_placeholder[:,:,:60])*np.linspace(1.0,0.7,60)*(1-target_placeholder[:,:,-1:]))

        ap_loss = tf.reduce_sum(tf.abs(ap - target_placeholder[:,:,60:-2])*(1-target_placeholder[:,:,-1:]))

        f0_loss = tf.reduce_sum(tf.abs(f0 - target_placeholder[:,:,-2:-1])*(1-target_placeholder[:,:,-1:])) 

        # vuv_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=, logits=vuv))

        vuv_loss = tf.reduce_mean(tf.reduce_sum(binary_cross(target_placeholder[:,:,-1:],vuv)))

        loss = harm_loss + ap_loss + vuv_loss + f0_loss * config.f0_weight

        # initial_summary = tf.summary.scalar('initial_loss', initial_loss)

        harm_summary = tf.summary.scalar('harm_loss', harm_loss)

        ap_summary = tf.summary.scalar('ap_loss', ap_loss)

        f0_summary = tf.summary.scalar('f0_loss', f0_loss)

        vuv_summary = tf.summary.scalar('vuv_loss', vuv_loss)

        loss_summary = tf.summary.scalar('total_loss', loss)

        global_step = tf.Variable(0, name='global_step', trainable=False)

        optimizer = tf.train.AdamOptimizer(learning_rate = config.init_lr)

        # optimizer_f0 = tf.train.AdamOptimizer(learning_rate = config.init_lr)

        train_function = optimizer.minimize(loss, global_step= global_step)

        # train_f0 = optimizer.minimize(f0_loss, global_step= global_step)

        # train_harm = optimizer.minimize(harm_loss, global_step= global_step)

        # train_ap = optimizer.minimize(ap_loss, global_step= global_step)

        # train_f0 = optimizer.minimize(f0_loss, global_step= global_step)

        # train_vuv = optimizer.minimize(vuv_loss, global_step= global_step)

        summary = tf.summary.merge_all()

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        saver = tf.train.Saver(max_to_keep= config.max_models_to_keep)
        sess = tf.Session()

        sess.run(init_op)

        ckpt = tf.train.get_checkpoint_state(config.log_dir_m1)

        if ckpt and ckpt.model_checkpoint_path:
            print("Using the model in %s"%ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)


        train_summary_writer = tf.summary.FileWriter(config.log_dir_m1+'train/', sess.graph)
        val_summary_writer = tf.summary.FileWriter(config.log_dir_m1+'val/', sess.graph)

        
        start_epoch = int(sess.run(tf.train.get_global_step())/(config.batches_per_epoch_train))

        print("Start from: %d" % start_epoch)
        f0_accs = []
        for epoch in xrange(start_epoch, config.num_epochs):
            val_f0_accs = []


            data_generator = data_gen()
            start_time = time.time()

            epoch_loss_harm = 0
            epoch_loss_ap = 0
            epoch_loss_f0 = 0
            epoch_loss_vuv = 0
            epoch_total_loss = 0
            # epoch_initial_loss = 0

            epoch_loss_harm_val = 0
            epoch_loss_ap_val = 0
            epoch_loss_f0_val = 0
            epoch_loss_vuv_val = 0
            epoch_total_loss_val = 0
            # epoch_initial_loss_val = 0

            if config.use_gan:
                epoch_loss_generator_GAN = 0
                epoch_loss_generator_diff = 0
                epoch_loss_discriminator_real = 0
                epoch_loss_discriminator_fake = 0

                val_epoch_loss_generator_GAN = 0
                val_epoch_loss_generator_diff = 0
                val_epoch_loss_discriminator_real = 0
                val_epoch_loss_discriminator_fake = 0

            batch_num = 0
            batch_num_val = 0
            val_generator = data_gen(mode='val')

            # val_generator = get_batches(train_filename=config.h5py_file_val, batches_per_epoch=config.batches_per_epoch_val_m1)

            with tf.variable_scope('Training'):

                for voc, feat in data_generator:

                    voc = np.clip(voc + np.random.rand(config.batch_size, config.max_phr_len,config.input_features)*np.clip(np.random.rand(1),0.0,0.2), 0.0, 1.0)

                    _, step_loss_harm, step_loss_ap,  step_loss_f0, step_loss_vuv, step_total_loss = sess.run([train_function, 
                        harm_loss, ap_loss, f0_loss, vuv_loss, loss], feed_dict={input_placeholder: voc,target_placeholder: feat})
                    # _, step_loss_f0 = sess.run([train_f0, f0_loss], feed_dict={input_placeholder: voc,target_placeholder: feat})
                    
                    if config.use_gan:
                        _, step_dis_loss_real, step_dis_loss_fake = sess.run([d_optimizer, D_loss_real,D_loss_fake], feed_dict={input_placeholder: voc,target_placeholder: feat})
                        _, step_gen_loss_GAN, step_gen_loss_diff = sess.run([g_optimizer, G_loss_GAN, G_loss_diff], feed_dict={input_placeholder: voc,target_placeholder: feat})
                    # else :
                    #     _, step_dis_loss_real, step_dis_loss_fake = sess.run([d_optimizer_grad, D_loss_real,D_loss_fake], feed_dict={input_placeholder: voc,target_placeholder: feat})
                    #     _, step_gen_loss_diff = sess.run([g_optimizer_diff, G_loss_diff], feed_dict={input_placeholder: voc,target_placeholder: feat})
                    #     step_gen_loss_GAN = 0




                    # _, step_loss_harm = sess.run([train_harm, harm_loss], feed_dict={input_placeholder: voc,target_placeholder: feat})
                    # _, step_loss_ap = sess.run([train_ap, ap_loss], feed_dict={input_placeholder: voc,target_placeholder: feat})
                    # _, step_loss_f0 = sess.run([train_f0, f0_loss], feed_dict={input_placeholder: voc,target_placeholder: feat})
                    # _, step_loss_vuv = sess.run([train_vuv, vuv_loss], feed_dict={input_placeholder: voc,target_placeholder: feat})

                    # epoch_initial_loss+=step_initial_loss
                    epoch_loss_harm+=step_loss_harm
                    epoch_loss_ap+=step_loss_ap
                    epoch_loss_f0+=step_loss_f0
                    epoch_loss_vuv+=step_loss_vuv
                    epoch_total_loss+=step_total_loss

                    if config.use_gan:

                        epoch_loss_generator_GAN+=step_gen_loss_GAN
                        epoch_loss_generator_diff+=step_gen_loss_diff
                        epoch_loss_discriminator_real+=step_dis_loss_real
                        epoch_loss_discriminator_fake+=step_dis_loss_fake



                    utils.progress(batch_num,config.batches_per_epoch_train, suffix = 'training done')
                    batch_num+=1


                # epoch_initial_loss = epoch_initial_loss/(config.batches_per_epoch_train *config.batch_size*config.max_phr_len*60)
                epoch_loss_harm = epoch_loss_harm/(config.batches_per_epoch_train *config.batch_size*config.max_phr_len*60)
                epoch_loss_ap = epoch_loss_ap/(config.batches_per_epoch_train *config.batch_size*config.max_phr_len*4)
                epoch_loss_f0 = epoch_loss_f0/(config.batches_per_epoch_train *config.batch_size*config.max_phr_len)
                epoch_loss_vuv = epoch_loss_vuv/(config.batches_per_epoch_train *config.batch_size*config.max_phr_len)
                epoch_total_loss = epoch_total_loss/(config.batches_per_epoch_train *config.batch_size*config.max_phr_len*66)

                if config.use_gan:

                    epoch_loss_generator_GAN = epoch_loss_generator_GAN/(config.batches_per_epoch_train *config.batch_size)
                    epoch_loss_generator_diff = epoch_loss_generator_diff/(config.batches_per_epoch_train *config.batch_size*config.max_phr_len*60)
                    epoch_loss_discriminator_real = epoch_loss_discriminator_real/(config.batches_per_epoch_train *config.batch_size)
                    epoch_loss_discriminator_fake = epoch_loss_discriminator_fake/(config.batches_per_epoch_train *config.batch_size)
                

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

                    if config.use_gan:

                        val_epoch_loss_generator_GAN += step_gen_loss_GAN
                        val_epoch_loss_generator_diff += step_gen_loss_diff
                        val_epoch_loss_discriminator_real += step_dis_loss_real
                        val_epoch_loss_discriminator_fake += step_dis_loss_fake

                    utils.progress(batch_num_val,config.batches_per_epoch_val_m1, suffix = 'validiation done')
                    batch_num_val+=1
                    
                # f0_accs.append(np.mean(val_f0_accs))

                # epoch_initial_loss_val = epoch_initial_loss_val/(config.batches_per_epoch_val_m1 *config.batch_size*config.max_phr_len*60)
                epoch_loss_harm_val = epoch_loss_harm_val/(batch_num_val *config.batch_size*config.max_phr_len*60)
                epoch_loss_ap_val = epoch_loss_ap_val/(batch_num_val *config.batch_size*config.max_phr_len*4)
                epoch_loss_f0_val = epoch_loss_f0_val/(batch_num_val *config.batch_size*config.max_phr_len)
                epoch_loss_vuv_val = epoch_loss_vuv_val/(batch_num_val *config.batch_size*config.max_phr_len)
                epoch_total_loss_val = epoch_total_loss_val/(batch_num_val *config.batch_size*config.max_phr_len*66)

                if config.use_gan:

                    val_epoch_loss_generator_GAN = val_epoch_loss_generator_GAN/(config.batches_per_epoch_val_m1 *config.batch_size)
                    val_epoch_loss_generator_diff = val_epoch_loss_generator_diff/(config.batches_per_epoch_val_m1 *config.batch_size*config.max_phr_len*60)
                    val_epoch_loss_discriminator_real = val_epoch_loss_discriminator_real/(config.batches_per_epoch_val_m1 *config.batch_size)
                    val_epoch_loss_discriminator_fake = val_epoch_loss_discriminator_fake/(config.batches_per_epoch_val_m1 *config.batch_size)

                summary_str = sess.run(summary, feed_dict={input_placeholder: voc,target_placeholder: feat})
                val_summary_writer.add_summary(summary_str, epoch)
                # summary_writer.add_summary(summary_str_val, epoch)
                val_summary_writer.flush()

            duration = time.time() - start_time

            # np.save('./ikala_eval/accuracies', f0_accs)

            if (epoch+1) % config.print_every == 0:
                print('epoch %d: Harm Training Loss = %.10f (%.3f sec)' % (epoch+1, epoch_loss_harm, duration))
                print('        : Ap Training Loss = %.10f ' % (epoch_loss_ap))
                print('        : F0 Training Loss = %.10f ' % (epoch_loss_f0))
                print('        : VUV Training Loss = %.10f ' % (epoch_loss_vuv))
                # print('        : Initial Training Loss = %.10f ' % (epoch_initial_loss))

                if config.use_gan:

                    print('        : Gen GAN Training Loss = %.10f ' % (epoch_loss_generator_GAN))
                    print('        : Gen diff Training Loss = %.10f ' % (epoch_loss_generator_diff))
                    print('        : Discriminator Training Loss Real = %.10f ' % (epoch_loss_discriminator_real))
                    print('        : Discriminator Training Loss Fake = %.10f ' % (epoch_loss_discriminator_fake))

                print('        : Harm Validation Loss = %.10f ' % (epoch_loss_harm_val))
                print('        : Ap Validation Loss = %.10f ' % (epoch_loss_ap_val))
                print('        : F0 Validation Loss = %.10f ' % (epoch_loss_f0_val))
                print('        : VUV Validation Loss = %.10f ' % (epoch_loss_vuv_val))
                
                # if (epoch + 1) % config.save_every == 0 or (epoch + 1) == config.num_epochs:
                # print('        : Mean F0 IKala Accuracy  = %.10f ' % (np.mean(val_f0_accs)))

                # print('        : Mean F0 IKala Accuracy = '+'%{1:.{0}f}%'.format(np.mean(val_f0_accs)))
                # print('        : Initial Validation Loss = %.10f ' % (epoch_initial_loss_val))

                if config.use_gan:

                    print('        : Gen GAN Validation Loss = %.10f ' % (val_epoch_loss_generator_GAN))
                    print('        : Gen diff Validation Loss = %.10f ' % (val_epoch_loss_generator_diff))
                    print('        : Discriminator Validation Loss Real = %.10f ' % (val_epoch_loss_discriminator_real))
                    print('        : Discriminator Validation Loss Fake = %.10f ' % (val_epoch_loss_discriminator_fake))


            if (epoch + 1) % config.save_every == 0 or (epoch + 1) == config.num_epochs:
                utils.list_to_file(val_f0_accs,'./ikala_eval/accuracies_'+str(epoch+1)+'.txt')
                checkpoint_file = os.path.join(config.log_dir_m1, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=epoch)


def synth_file(file_name, file_path=config.wav_dir, show_plots=True, save_file=True):
    if file_name.startswith('ikala'):
        file_name = file_name[6:]
        file_path = config.wav_dir
        utils.write_ori_ikala(os.path.join(file_path,file_name),file_name)
        mode =0
    elif file_name.startswith('mir'):
        file_name = file_name[4:]
        file_path = config.wav_dir_mir
        utils.write_ori_ikala(os.path.join(file_path,file_name),file_name)
        mode =0
    elif file_name.startswith('med'):
        file_name = file_name[4:]
        file_path = config.wav_dir_med
        utils.write_ori_med(os.path.join(file_path,file_name),file_name)
        mode =2

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

            # harmy = harm_1+harm

        if config.use_gan:
            with tf.variable_scope('Generator') as scope: 
                gen_op = modules.GAN_generator(harm)
        # with tf.variable_scope('Discriminator') as scope: 
        #     D_real = modules.GAN_discriminator(target_placeholder[:,:,:60],input_placeholder)
        #     scope.reuse_variables()
        #     D_fake = modules.GAN_discriminator(gen_op,input_placeholder)


        saver = tf.train.Saver(max_to_keep= config.max_models_to_keep)


        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess = tf.Session()

        sess.run(init_op)

        ckpt = tf.train.get_checkpoint_state(config.log_dir_m1)

        if ckpt and ckpt.model_checkpoint_path:
            print("Using the model in %s"%ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)

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

        if config.use_gan:
            gan_op = np.array(gan_op)
            gan_op = utils.overlapadd(gan_op, nchunks_in) 


        targs = (targs-min_feat)/(max_feat-min_feat)

        # first_pred = (first_pred-min_feat[:60])/(max_feat[:60]-min_feat[:60])
        # cleaner = (cleaner-min_feat[:60])/(max_feat[:60]-min_feat[:60])


        

        if show_plots:

            # import pdb;pdb.set_trace()
        
            ins = val_outer[:,:60]
            outs = targs[:,:60]
            plt.figure(1)
            ax1 = plt.subplot(211)
            plt.imshow(ins.T, origin='lower', aspect='auto')
            ax1.set_title("Predicted Harm ", fontsize = 10)
            ax2 = plt.subplot(212)
            plt.imshow(outs.T, origin='lower', aspect='auto')
            ax2.set_title("Ground Truth Harm ", fontsize = 10)
            # ax1 = plt.subplot(413)
            # plt.imshow(first_pred.T, origin='lower', aspect='auto')
            # ax1.set_title("Initial Prediction ", fontsize = 10)
            # ax2 = plt.subplot(412)
            # plt.imshow(cleaner.T, origin='lower', aspect='auto')
            # ax2.set_title("Residual Added ", fontsize = 10)

            if config.use_gan:
                plt.figure(5)
                ax1 = plt.subplot(411)
                plt.imshow(ins.T, origin='lower', aspect='auto')
                ax1.set_title("Predicted Harm ", fontsize = 10)
                ax2 = plt.subplot(414)
                plt.imshow(outs.T, origin='lower', aspect='auto')
                ax2.set_title("Ground Truth Harm ", fontsize = 10)
                ax1 = plt.subplot(412)
                plt.imshow(gan_op.T, origin='lower', aspect='auto')
                ax1.set_title("GAN output ", fontsize = 10)
                ax1 = plt.subplot(413)
                plt.imshow((gan_op[:ins.shape[0],:]+ins).T, origin='lower', aspect='auto')
                ax1.set_title("GAN output ", fontsize = 10)



            plt.figure(2)
            ax1 = plt.subplot(211)
            plt.imshow(val_outer[:,60:-2].T, origin='lower', aspect='auto')
            ax1.set_title("Predicted Aperiodic Part", fontsize = 10)
            ax2 = plt.subplot(212)
            plt.imshow(targs[:,60:-2].T, origin='lower', aspect='auto')
            ax2.set_title("Ground Truth Aperiodic Part", fontsize = 10)
            

            plt.figure(3)

            f0_output = val_outer[:,-2]*((max_feat[-2]-min_feat[-2])+min_feat[-2])
            f0_output = f0_output*(1-targs[:,-1])
            f0_output[f0_output == 0] = np.nan
            plt.plot(f0_output, label = "Predicted Value")
            f0_gt = targs[:,-2]*((max_feat[-2]-min_feat[-2])+min_feat[-2])
            f0_gt = f0_gt*(1-targs[:,-1])
            f0_gt[f0_gt == 0] = np.nan
            plt.plot(f0_gt, label="Ground Truth")
            f0_difference = np.nan_to_num(abs(f0_gt-f0_output))
            f0_greater = np.where(f0_difference>config.f0_threshold)
            diff_per = f0_greater[0].shape[0]/len(f0_output)
            plt.suptitle("Percentage correct = "+'{:.3%}'.format(1-diff_per))
            # import pdb;pdb.set_trace()


            # import pdb;pdb.set_trace()
            # uu = f0_sac[:,0]*(1-f0_sac[:,1])
            # uu[uu == 0] = np.nan
            # plt.plot(uu, label="Sac f0")
            plt.legend()
            plt.figure(4)
            ax1 = plt.subplot(211)
            plt.plot(val_outer[:,-1])
            ax1.set_title("Predicted Voiced/Unvoiced", fontsize = 10)
            ax2 = plt.subplot(212)
            plt.plot(targs[:,-1])
            ax2.set_title("Ground Truth Voiced/Unvoiced", fontsize = 10)
            plt.show()
        if save_file:
            
            val_outer = np.ascontiguousarray(val_outer*(max_feat-min_feat)+min_feat)
            targs = np.ascontiguousarray(targs*(max_feat-min_feat)+min_feat)

            # import pdb;pdb.set_trace()

            # val_outer = np.ascontiguousarray(utils.denormalize(val_outer,'feats', mode=config.norm_mode_out))
            try:
                utils.feats_to_audio(val_outer,file_name[:-4]+'_synth_pred_f0')
                print("File saved to %s" % config.val_dir+file_name[:-4]+'_synth_pred_f0.wav')
            except:
                print("Couldn't synthesize with predicted f0")
            try:
                val_outer[:,-2:] = targs[:,-2:]
                utils.feats_to_audio(val_outer,file_name[:-4]+'_synth_ori_f0')
                print("File saved to %s" % config.val_dir+file_name[:-4]+'_synth_ori_f0.wav')
            except:
                print("Couldn't synthesize with original f0")
                




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
                synth_file(file_name,show_plots=False, save_file=True)

    elif sys.argv[1] == '-help' or sys.argv[1] == '--help' or sys.argv[1] == '--h' or sys.argv[1] == '-h':
        print("%s --train to train the model"%sys.argv[0])
        print("%s --synth <filename> to synthesize file"%sys.argv[0])
        print("%s --synth <filename> -- plot to synthesize file and show plots"%sys.argv[0])
        print("%s --synth <filename> -- plot --ns to just show plots"%sys.argv[0])
    else:
        print("Unable to decipher inputs please use %s --help for help on how to use this function"%sys.argv[0])