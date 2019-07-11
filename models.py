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

		self.optimizer = tf.train.AdamOptimizer(learning_rate = config.init_lr)

		self.global_step = tf.Variable(0, name='global_step', trainable=False)

		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			self.final_train_function = self.final_optimizer.minimize(self.loss, global_step = self.global_step)


	def loss_function(self):
		"""
		returns the loss function for the model, based on the mode. 
		"""

		self.harm_loss = tf.reduce_sum(tf.abs(self.harm - self.target_placeholder[:,:,:60])*np.linspace(1.0,0.7,60)*(1-self.target_placeholder[:,:,-1:]))

		self.ap_loss = tf.reduce_sum(tf.abs(self.ap - self.target_placeholder[:,:,60:-2])*(1-self.target_placeholder[:,:,-1:]))

		self.f0_loss = tf.reduce_sum(tf.abs(self.f0 - self.target_placeholder[:,:,-2:-1])*(1-self.target_placeholder[:,:,-1:])) 

		self.vuv_loss = tf.reduce_mean(tf.reduce_sum(binary_cross(self.target_placeholder[:,:,-1:],self.vuv)))

		self.loss = self.harm_loss + self.ap_loss + self.vuv_loss + self.f0_loss * config.f0_weight

	def get_summary(self, sess, log_dir):
		"""
		Gets the summaries and summary writers for the losses.
		"""

		self.harm_summary = tf.summary.scalar('harm_loss', self.harm_loss)

		self.ap_summary = tf.summary.scalar('ap_loss', self.ap_loss)

		self.f0_summary = tf.summary.scalar('f0_loss', self.f0_loss)

		self.vuv_summary = tf.summary.scalar('vuv_loss', self.vuv_loss)

		self.loss_summary = tf.summary.scalar('total_loss', self.loss)

		self.train_summary_writer = tf.summary.FileWriter(log_dir+'train/', sess.graph)
		self.val_summary_writer = tf.summary.FileWriter(log_dir+'val/', sess.graph)
		self.summary = tf.summary.merge_all()

	def get_placeholders(self):
		"""
		Returns the placeholders for the model. 
		Depending on the mode, can return placeholders for either just the generator or both the generator and discriminator.
		"""

		self.input_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len,config.input_features),name='input_placeholder')
		self.target_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len,config.output_features),name='target_placeholder')

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

					final_loss, summary_str = self.train_model(voc, feat, sess)

					epoch_final_loss+=final_loss

					self.train_summary_writer.add_summary(summary_str, epoch)
					self.train_summary_writer.flush()

					utils.progress(batch_num,config.batches_per_epoch_train, suffix = 'training done')

					batch_num+=1

				epoch_final_loss = epoch_final_loss/batch_num

				print_dict = {"Final Loss": epoch_final_loss}

			if (epoch + 1) % config.validate_every == 0:
				batch_num = 0
				with tf.variable_scope('Validation'):
					for voc, feat in val_generator:

						final_loss, summary_str= self.validate_model(voc, feat, sess)
						val_final_loss+=final_loss

						self.val_summary_writer.add_summary(summary_str, epoch)
						self.val_summary_writer.flush()
						batch_num+=1

						utils.progress(batch_num, config.batches_per_epoch_val, suffix='validation done')

					val_final_loss = val_final_loss/batch_num

					print_dict["Val Final Loss"] =  val_final_loss

			end_time = time.time()
			if (epoch + 1) % config.print_every == 0:
				self.print_summary(print_dict, epoch, end_time-start_time)
			if (epoch + 1) % config.save_every == 0 or (epoch + 1) == config.num_epochs:
				self.save_model(sess, epoch+1, config.log_dir)

	def train_model(self, voc, feat, sess):
		"""
		Function to train the model for each epoch
		"""
		feed_dict = {self.input_placeholder: voc, self.output_placeholder: feat}


		_, final_loss = sess.run([self.train_function, self.final_loss], feed_dict=feed_dict)

		summary_str = sess.run(self.summary, feed_dict=feed_dict)

		return final_loss, summary_str

	def validate_model(self,mix_in, singer_targs, voc_out, f0_out,pho_targs, sess):
		"""
		Function to train the model for each epoch
		"""
		feed_dict = {self.input_placeholder: voc, self.output_placeholder: feat}

		final_loss = sess.run(self.final_loss, feed_dict=feed_dict)

		summary_str = sess.run(self.summary, feed_dict=feed_dict)

		return final_loss, summary_str



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
			feed_dict = {self.input_placeholder: in_batch_stft}
			harm, ap, f0, vuv = sess.run([self.harm, self.ap, self.f0, self.vuv], feed_dict=feed_dict)

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

		with tf.variable_scope('First_Model') as scope:
			self.harm, self.ap, self.f0, self.vuv = modules.nr_wavenet(self.input_placeholder)






