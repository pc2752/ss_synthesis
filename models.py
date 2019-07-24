import tensorflow as tf
import modules_tf as modules
import config
from data_pipeline import data_gen
import time, os
import utils
import h5py
import numpy as np
import mir_eval
import pandas as pd
from random import randint
import librosa
# import sig_process

import soundfile as sf

import matplotlib.pyplot as plt
from scipy.ndimage import filters

def binary_cross(p,q):
    return -(p * tf.log(q + 1e-12) + (1 - p) * tf.log( 1 - q + 1e-12))

class Model(object):
    def __init__(self):
        self.get_placeholders()
        self.model()


    def test_file_all(self, file_name, sess):
        """
        Function to extract multi pitch from file. Currently supports only HDF5 files.
        """
        scores = self.extract_f0_file(file_name, sess)
        return scores

    def validate_file(self, file_name, sess):
        """
        Function to extract multi pitch from file, for validation. Currently supports only HDF5 files.
        """
        scores = self.extract_f0_file(file_name, sess)
        pre = scores['Precision']
        acc = scores['Accuracy']
        rec = scores['Recall']
        return pre, acc, rec




    def load_model(self, sess, log_dir):
        """
        Load model parameters, for synthesis or re-starting training. 
        """
        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep= config.max_models_to_keep)


        sess.run(self.init_op)

        ckpt = tf.train.get_checkpoint_state(log_dir)

        if ckpt and ckpt.model_checkpoint_path:
            print("Using the model in %s"%ckpt.model_checkpoint_path)
            self.saver.restore(sess, ckpt.model_checkpoint_path)


    def save_model(self, sess, epoch, log_dir):
        """
        Save the model.
        """
        checkpoint_file = os.path.join(log_dir, 'model.ckpt')
        self.saver.save(sess, checkpoint_file, global_step=epoch)

    def print_summary(self, print_dict, epoch, duration):
        """
        Print training summary to console, every N epochs.
        Summary will depend on model_mode.
        """

        print('epoch %d took (%.3f sec)' % (epoch + 1, duration))
        for key, value in print_dict.items():
            print('{} : {}'.format(key, value))
            

class SSSynth(Model):

    def get_optimizers(self):
        """
        Returns the optimizers for the model, based on the loss functions and the mode. 
        """
        self.harm_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'Harm_Model')
        self.ap_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'Ap_Model')
        self.f0_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'F0_Model')
        self.vuv_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'Vuv_Model')

        self.harm_optimizer = tf.train.AdamOptimizer(learning_rate = config.init_lr)
        self.ap_optimizer = tf.train.AdamOptimizer(learning_rate = config.init_lr)
        self.f0_optimizer = tf.train.AdamOptimizer(learning_rate = config.init_lr)
        self.vuv_optimizer = tf.train.AdamOptimizer(learning_rate = config.init_lr)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.global_step_ap = tf.Variable(0, name='global_step_ap', trainable=False)
        self.global_step_f0 = tf.Variable(0, name='global_step_f0', trainable=False)
        self.global_step_vuv = tf.Variable(0, name='global_step_vuv', trainable=False)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.harm_train_function = self.harm_optimizer.minimize(self.harm_loss, global_step = self.global_step, var_list = self.harm_params)
            self.ap_train_function = self.ap_optimizer.minimize(self.ap_loss, global_step = self.global_step, var_list = self.ap_params)
            self.f0_train_function = self.harm_optimizer.minimize(self.f0_loss, global_step = self.global_step, var_list = self.f0_params)
            self.vuv_train_function = self.ap_optimizer.minimize(self.vuv_loss, global_step = self.global_step, var_list = self.vuv_params)

    def loss_function(self):
        """
        returns the loss function for the model, based on the mode. 
        """

        self.harm_loss = tf.reduce_sum(tf.abs(self.harm - self.harm_placeholder)*np.linspace(1.0,0.7,60))

        self.ap_loss = tf.reduce_sum(tf.abs(self.ap - self.ap_placeholder))

        self.f0_loss = tf.reduce_sum(tf.abs(self.f0 - self.f0_placeholder)*(1-self.vuv_placeholder)) 

        self.vuv_loss = tf.reduce_sum(tf.reduce_sum(binary_cross(self.vuv_placeholder, self.vuv)))

        # self.loss = self.harm_loss + self.ap_loss + self.vuv_loss + self.f0_loss *config.f0_weight

    def get_summary(self, sess, log_dir):
        """
        Gets the summaries and summary writers for the losses.
        """

        self.harm_summary = tf.summary.scalar('harm_loss', self.harm_loss)

        self.ap_summary = tf.summary.scalar('ap_loss', self.ap_loss)

        self.f0_summary = tf.summary.scalar('f0_loss', self.f0_loss)

        self.vuv_summary = tf.summary.scalar('vuv_loss', self.vuv_loss)

        self.train_summary_writer = tf.summary.FileWriter(log_dir+'train/', sess.graph)
        self.val_summary_writer = tf.summary.FileWriter(log_dir+'val/', sess.graph)
        self.summary = tf.summary.merge_all()

    def get_placeholders(self):
        """
        Returns the placeholders for the model. 
        Depending on the mode, can return placeholders for either just the generator or both the generator and discriminator.
        """

        self.input_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len,config.input_features),name='input_placeholder')
        self.harm_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len,60),name='harm_placeholder')
        self.ap_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len,4),name='ap_placeholder')
        self.f0_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len,1),name='f0_placeholder')
        self.vuv_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len,1),name='vuv_placeholder')
        self.is_train = tf.placeholder(tf.bool, name="is_train")

    def train(self):
        """
        Function to train the model, and save Tensorboard summary, for N epochs. 
        """
        sess = tf.Session()

        self.loss_function()
        self.get_optimizers()
        self.load_model(sess, config.log_dir)
        self.get_summary(sess, config.log_dir)
        start_epoch = int(sess.run(tf.train.get_global_step()) / (config.batches_per_epoch_train))


        print("Start from: %d" % start_epoch)


        for epoch in range(start_epoch, config.num_epochs):
            data_generator = data_gen()
            val_generator = data_gen(mode = 'Val')
            start_time = time.time()


            batch_num = 0
            epoch_final_loss = 0
            epoch_harm_loss = 0
            epoch_ap_loss = 0
            epoch_vuv_loss = 0
            epoch_f0_loss = 0

            val_final_loss = 0
            val_harm_loss = 0
            val_ap_loss = 0
            val_vuv_loss = 0
            val_f0_loss = 0

            with tf.variable_scope('Training'):
                for voc, feat in data_generator:

                    harm_loss, ap_loss, f0_loss, vuv_loss, summary_str = self.train_model(voc, feat, sess)

                    epoch_harm_loss+=harm_loss
                    epoch_ap_loss+=ap_loss
                    epoch_f0_loss+=f0_loss
                    epoch_vuv_loss+=vuv_loss

                    self.train_summary_writer.add_summary(summary_str, epoch)
                    self.train_summary_writer.flush()

                    utils.progress(batch_num,config.batches_per_epoch_train, suffix = 'training done')

                    batch_num+=1

                epoch_harm_loss = epoch_harm_loss/batch_num
                epoch_ap_loss = epoch_ap_loss/batch_num
                epoch_f0_loss = epoch_f0_loss/batch_num
                epoch_vuv_loss = epoch_vuv_loss/batch_num

                print_dict = {"Harm Loss": epoch_harm_loss}
                print_dict["Ap Loss"] =  epoch_ap_loss
                print_dict["F0 Loss"] =  epoch_f0_loss
                print_dict["Vuv Loss"] =  epoch_vuv_loss

            if (epoch + 1) % config.validate_every == 0:
                batch_num = 0
                with tf.variable_scope('Validation'):
                    for voc, feat in val_generator:

                        harm_loss, ap_loss, f0_loss, vuv_loss, summary_str = self.validate_model(voc, feat, sess)
                        val_harm_loss+=harm_loss
                        val_ap_loss+=ap_loss
                        val_f0_loss+=f0_loss
                        val_vuv_loss+=vuv_loss

                        self.val_summary_writer.add_summary(summary_str, epoch)
                        self.val_summary_writer.flush()
                        batch_num+=1

                        utils.progress(batch_num, config.batches_per_epoch_val, suffix='validation done')

                    val_harm_loss = val_harm_loss/batch_num
                    val_ap_loss = val_ap_loss/batch_num
                    val_f0_loss = val_f0_loss/batch_num
                    val_vuv_loss = val_vuv_loss/batch_num

                    print_dict["Val Harm Loss"] =  val_harm_loss
                    print_dict["Val Ap Loss"] =  val_ap_loss
                    print_dict["Val F0 Loss"] =  val_f0_loss
                    print_dict["Val Vuv Loss"] =  val_vuv_loss

            end_time = time.time()
            if (epoch + 1) % config.print_every == 0:
                self.print_summary(print_dict, epoch, end_time-start_time)
            if (epoch + 1) % config.save_every == 0 or (epoch + 1) == config.num_epochs:
                self.save_model(sess, epoch+1, config.log_dir)

    def train_model(self, voc, feat, sess):
        """
        Function to train the model for each epoch
        """
        voc = np.clip(voc + np.random.normal(0,.5,(voc.shape)) * 0.4, 0.0, 1.0)

        # teacher_train = np.random.rand(1)<0.5

           #  if epoch<1000 or not teacher_train:

        feed_dict = {self.input_placeholder: voc, self.harm_placeholder: feat[:,:,:60], self.ap_placeholder: feat[:,:,60:64], \
        self.f0_placeholder: feat[:,:,-2:-1], self.vuv_placeholder: feat[:,:,-1:], self.is_train: True}
        _,_,_,_, harm_loss, ap_loss, f0_loss, vuv_loss = sess.run([self.harm_train_function, self.ap_train_function, self.f0_train_function,self.vuv_train_function,
            self.harm_loss, self.ap_loss, self.f0_loss, self.vuv_loss ], feed_dict=feed_dict)


        summary_str = sess.run(self.summary, feed_dict=feed_dict)

        return harm_loss, ap_loss, f0_loss, vuv_loss, summary_str

    def validate_model(self, voc, feat, sess):
        """
        Function to train the model for each epoch
        """
        feed_dict = {self.input_placeholder: voc, self.harm_placeholder: feat[:,:,:60], self.ap_placeholder: feat[:,:,60:64], \
        self.f0_placeholder: feat[:,:,-2:-1], self.vuv_placeholder: feat[:,:,-1:], self.is_train: True}
        harm_loss, ap_loss, f0_loss, vuv_loss = sess.run([self.harm_loss, self.ap_loss, self.f0_loss, self.vuv_loss ], feed_dict=feed_dict)


        summary_str = sess.run(self.summary, feed_dict=feed_dict)

        return harm_loss, ap_loss, f0_loss, vuv_loss, summary_str



    def read_hdf5_file(self, file_name):
        """
        Function to read and process input file, given name and the synth_mode.
        Returns features for the file based on mode (0 for hdf5 file, 1 for wav file).
        Currently, only the HDF5 version is implemented.
        """

        stat_file = h5py.File(config.stat_dir+'stats.hdf5', mode='r')
        max_feat = np.array(stat_file["feats_maximus"])
        min_feat = np.array(stat_file["feats_minimus"])
        max_voc = np.array(stat_file["voc_stft_maximus"])
        min_voc = np.array(stat_file["voc_stft_minimus"])
        max_mix = np.array(stat_file["back_stft_maximus"])
        min_mix = np.array(stat_file["back_stft_minimus"])
        stat_file.close()

        with h5py.File(config.backing_dir + file_name) as mix_file:

            assert 'mix_stft' in mix_file, "This HDF5 file does not have the mixture sepctrogram, please use the wav file if available"

            mix_stft = np.array(mix_file['mix_stft'])[()]

        with h5py.File(config.voice_dir + file_name) as feat_file:

            feats = np.array(feat_file['feats'])


        return mix_stft, feats

    def test_file_hdf5(self, file_name):
        """
        Function to extract vocals from hdf5 file.
        """
        sess = tf.Session()
        self.load_model(sess, log_dir = config.log_dir)
        mix_stft, feats = self.read_hdf5_file(file_name)
        out_feats = self.process_file(mix_stft,  sess)
        self.plot_features(feats, out_feats)

    def test_file_wav(self, file_name):
        """
        Function to extract vocals from wav file.
        """
        sess = tf.Session()
        self.load_model(sess, log_dir = config.log_dir)
        mix_stft = utils.file_to_stft(file_name)
        out_feats = self.process_file(mix_stft, sess)

    def plot_features(self, feats, out_feats):
        """
        Function to plot output and ground truth features
        """
        plt.figure(1)
        
        ax1 = plt.subplot(211)

        plt.imshow(feats[:,:-2].T,aspect='auto',origin='lower')

        ax1.set_title("Ground Truth Vocoder Features", fontsize=10)

        ax3 =plt.subplot(212, sharex = ax1, sharey = ax1)

        ax3.set_title("Output STFT", fontsize=10)

        plt.imshow(out_feats[:,:-2].T,aspect='auto',origin='lower')
        
        plt.figure(2)

        f0_output = out_feats[:feats.shape[0],-2]
        f0_output = f0_output*(1-feats[:,-1])
        f0_output[f0_output == 0] = np.nan
        plt.plot(f0_output, label = "Predicted Value")
        f0_gt = feats[:,-2]
        f0_gt = f0_gt*(1-feats[:,-1])
        f0_gt[f0_gt == 0] = np.nan
        plt.plot(f0_gt, label="Ground Truth")
        f0_difference = np.nan_to_num(abs(f0_gt-f0_output))
        f0_greater = np.where(f0_difference>config.f0_threshold)
        diff_per = f0_greater[0].shape[0]/len(f0_output)
        plt.suptitle("Percentage correct = "+'{:.3%}'.format(1-diff_per))

        plt.show()


    def process_file(self, mix_stft, sess):

        stat_file = h5py.File(config.stat_dir+'stats.hdf5', mode='r')

        max_voc = np.array(stat_file["voc_stft_maximus"])
        min_voc = np.array(stat_file["voc_stft_minimus"])

        max_feat = np.array(stat_file["feats_maximus"])
        min_feat = np.array(stat_file["feats_minimus"])
        stat_file.close()

        in_batches_stft, nchunks_in = utils.generate_overlapadd(mix_stft)

        in_batches_stft = in_batches_stft/max_voc 

        out_batches_feats = []

        for in_batch_stft in in_batches_stft :
            feed_dict = {self.input_placeholder: in_batch_stft, self.is_train: False}
            harm = sess.run(self.harm, feed_dict=feed_dict)
            feed_dict = {self.input_placeholder: in_batch_stft,self.harm_placeholder:harm, self.is_train: False}
            ap = sess.run(self.ap, feed_dict=feed_dict)
            feed_dict = {self.input_placeholder: in_batch_stft,self.harm_placeholder:harm,self.ap_placeholder:ap, self.is_train: False}
            f0 = sess.run(self.f0, feed_dict=feed_dict)
            feed_dict = {self.input_placeholder: in_batch_stft,self.harm_placeholder:harm,self.ap_placeholder:ap,self.f0_placeholder:f0, self.is_train: False}
            vuv = sess.run(self.vuv, feed_dict=feed_dict)
            val_feats = np.concatenate((harm, ap, f0, vuv), axis=-1)
            out_batches_feats.append(val_feats)

        out_batches_feats = np.array(out_batches_feats)

        out_feats = utils.overlapadd(out_batches_feats,nchunks_in)

        out_feats[:,-1] = np.round(out_feats[:,-1])

        out_feats = out_feats*(max_feat-min_feat)+min_feat

        return out_feats

    def model(self):
        """
        The main model function, takes and returns tensors.
        Defined in modules.

        """

        with tf.variable_scope('Harm_Model') as scope:
            self.harm = modules.harm_network(self.input_placeholder, self.is_train)
        with tf.variable_scope('Ap_Model') as scope:
            self.ap = modules.ap_network(self.input_placeholder, self.harm_placeholder, self.is_train)
        with tf.variable_scope('F0_Model') as scope:
            self.f0 = modules.f0_network(self.input_placeholder, tf.concat([self.harm_placeholder, self.ap_placeholder], axis = -1), self.is_train)
        with tf.variable_scope('Vuv_Model') as scope:
            self.vuv = modules.vuv_network(self.input_placeholder, tf.concat([self.harm_placeholder, self.ap_placeholder, self.f0_placeholder], axis = -1), self.is_train)






