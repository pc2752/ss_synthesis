from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import tensorflow as tf
import numpy as np
from tensorflow.python import debug as tf_debug
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib import rnn
import config


tf.logging.set_verbosity(tf.logging.INFO)


def bi_dynamic_stacked_RNN(x, input_lengths, scope='RNN'):
    with tf.variable_scope(scope):
    # x = tf.layers.dense(x, 128)

        cell = tf.nn.rnn_cell.LSTMCell(num_units=config.lstm_size, state_is_tuple=True)
        cell2 = tf.nn.rnn_cell.LSTMCell(num_units=config.lstm_size, state_is_tuple=True)

        outputs, _state1, state2  = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            cells_fw=[cell,cell2],
            cells_bw=[cell,cell2],
            dtype=config.dtype,
            sequence_length=input_lengths,
            inputs=x)

    return outputs

def bi_static_stacked_RNN(x, scope='RNN'):
    """
    Input and output in batch major format
    """
    with tf.variable_scope(scope):
        x = tf.unstack(x, config.max_phr_len, 1)

        output = x
        num_layer = 2
        # for n in range(num_layer):
        lstm_fw = tf.nn.rnn_cell.LSTMCell(config.lstm_size, state_is_tuple=True)
        lstm_bw = tf.nn.rnn_cell.LSTMCell(config.lstm_size, state_is_tuple=True)

        _initial_state_fw = lstm_fw.zero_state(config.batch_size, tf.float32)
        _initial_state_bw = lstm_bw.zero_state(config.batch_size, tf.float32)

        output, _state1, _state2 = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw, lstm_bw, output, 
                                                  initial_state_fw=_initial_state_fw,
                                                  initial_state_bw=_initial_state_bw, 
                                                  scope='BLSTM_')
        output = tf.stack(output)
        output_fw = output[0]
        output_bw = output[1]
        output = tf.transpose(output, [1,0,2])


        # output = tf.layers.dense(output, config.output_features, activation=tf.nn.relu) # Remove this to use cbhg

        return output




def bi_dynamic_RNN(x, input_lengths, scope='RNN'):
    """
    Stacked dynamic RNN, does not need unpacking, but needs input_lengths to be specified
    """

    with tf.variable_scope(scope):

        cell = tf.nn.rnn_cell.LSTMCell(num_units=config.lstm_size, state_is_tuple=True)

        outputs, states  = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell,
            cell_bw=cell,
            dtype=config.dtype,
            sequence_length=input_lengths,
            inputs=x)

        outputs = tf.concat(outputs, axis=2)

    return outputs


def RNN(x, scope='RNN'):
    with tf.variable_scope(scope):
        x = tf.unstack(x, config.max_phr_len, 1)

        lstm_cell = rnn.BasicLSTMCell(num_units=config.lstm_size)

        # Get lstm cell output
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=config.dtype)
        outputs=tf.stack(outputs)
        outputs = tf.transpose(outputs, [1,0,2])

    return outputs


def highwaynet(inputs, scope='highway', units=config.highway_units):
    with tf.variable_scope(scope):
        H = tf.layers.dense(
        inputs,
        units=units,
        activation=tf.nn.relu,
        name='H')
        T = tf.layers.dense(
        inputs,
        units=units,
        activation=tf.nn.sigmoid,
        name='T',
        bias_initializer=tf.constant_initializer(-1.0))
    return H * T + inputs * (1.0 - T)


def conv(inputs, kernel_size, filters=config.conv_filters, activation=config.conv_activation, training=True, scope='conv'):
  with tf.variable_scope(scope):
    x = tf.layers.conv1d(inputs,filters=filters,kernel_size=kernel_size,activation=activation,padding='same')
    return tf.layers.batch_normalization(x, training=training)


# def build_encoder(inputs):
#     embedding_encoder = variable_scope.get_variable("embedding_encoder", [config.vocab_size, config.inp_embedding_size], dtype=config.dtype)

def conv_bank(inputs, scope='conv_bank', num_layers=config.num_conv_layers, training=True):
    with tf.variable_scope(scope):
        outputs = [conv(inputs, k, training=training, scope='conv_%d' % k) for k in range(1, num_layers+1)]
        outputs = tf.concat(outputs, axis=-1)
        outputs = tf.layers.max_pooling1d(outputs,pool_size=2,strides=1,padding='same')
    return outputs




def cbhg(inputs, scope='cbhg', training=True):
    with tf.variable_scope(scope):
        # Prenet
        if training:
            dropout = config.dropout_rate
        else:
            dropout = 0.0
        # tf.summary.histogram('inputs', inputs)
        prenet_out = tf.layers.dropout(tf.layers.dense(inputs, config.lstm_size*2), rate=dropout)
        prenet_out = tf.layers.dropout(tf.layers.dense(prenet_out, config.lstm_size), rate=dropout)
        # tf.summary.histogram('prenet_output', prenet_out)

        # Conv Bank
        x = conv_bank(prenet_out, training=training)


        # Projections
        x = conv(x, config.projection_size, config.conv_filters, training=training, scope='proj_1')
        x = conv(x, config.projection_size, config.conv_filters,activation=None, training=training, scope='proj_2')

        assert x.shape[-1]==config.highway_units

        x = x+prenet_out

        for i in range(config.highway_layers):
            x = highwaynet(x, scope='highway_%d' % (i+1))
        x = bi_static_stacked_RNN(x)
        x = tf.layers.dense(x, config.output_features)

        output = tf.layers.dense(x, 128, activation=tf.nn.relu) # Remove this to use cbhg
        harm = tf.layers.dense(output, 60, activation=tf.nn.relu)
        ap = tf.layers.dense(output, 4, activation=tf.nn.relu)
        f0 = tf.layers.dense(output, 64, activation=tf.nn.relu) 
        f0 = tf.layers.dense(f0, 1, activation=tf.nn.relu)
        vuv = tf.layers.dense(ap, 1, activation=tf.nn.sigmoid)
        phonemes = tf.layers.dense(output, 41, activation=tf.nn.relu)
    return harm, ap, f0, vuv, phonemes
        


def nr_wavenet_block(conditioning, dilation_rate = 2):
    # inputs = tf.reshape(inputs, [config.batch_size, config.max_phr_len, config.input_features])

    con_pad_forward = tf.pad(conditioning, [[0,0],[dilation_rate,0],[0,0]],"CONSTANT")
    con_pad_backward = tf.pad(conditioning, [[0,0],[0,dilation_rate],[0,0]],"CONSTANT")
    con_sig_forward = tf.layers.conv1d(con_pad_forward, config.wavenet_filters, 2, dilation_rate = dilation_rate, padding = 'valid')
    con_sig_backward = tf.layers.conv1d(con_pad_backward, config.wavenet_filters, 2, dilation_rate = dilation_rate, padding = 'valid')
    # con_sig = tf.layers.conv1d(conditioning,config.wavenet_filters,1)

    sig = tf.sigmoid(con_sig_forward+con_sig_backward)


    con_tanh_forward = tf.layers.conv1d(con_pad_forward, config.wavenet_filters, 2, dilation_rate = dilation_rate, padding = 'valid')
    con_tanh_backward = tf.layers.conv1d(con_pad_backward, config.wavenet_filters, 2, dilation_rate = dilation_rate, padding = 'valid')    
    # con_tanh = tf.layers.conv1d(conditioning,config.wavenet_filters,1)

    tanh = tf.tanh(con_tanh_forward+con_tanh_backward)


    outputs = tf.multiply(sig,tanh)

    skip = tf.layers.conv1d(outputs,config.wavenet_filters,1)

    residual = skip + conditioning

    return skip, residual


def nr_wavenet(inputs, num_block = config.wavenet_layers):


    receptive_field = 2**num_block

    first_conv = tf.layers.conv1d(inputs, config.wavenet_filters, 1)
    skips = []
    skip, residual = nr_wavenet_block(first_conv, dilation_rate=1)
    output = skip
    for i in range(num_block):
        skip, residual = nr_wavenet_block(residual, dilation_rate=2**(i+1))
        skips.append(skip)
    for skip in skips:
        output+=skip
    output = output+first_conv

    output = tf.nn.relu(output)

    harm = tf.layers.conv1d(output,config.wavenet_filters,1)

    ap = tf.layers.conv1d(output,config.wavenet_filters,1)

    f0 = tf.layers.conv1d(output,config.wavenet_filters,1)

    vuv = tf.layers.conv1d(output,config.wavenet_filters,1)

    harm = tf.layers.dense(harm, 60, activation=tf.nn.relu)
    ap = tf.layers.dense(ap, 4, activation=tf.nn.relu)
    f0 = tf.layers.dense(f0, 1, activation=tf.nn.relu)
    vuv = tf.layers.dense(vuv, 1, activation=tf.nn.sigmoid)

    return harm, ap, f0, vuv

def encoder_conv_block(inputs, layer_num, is_train, num_filters = config.filters):

    output = tf.layers.batch_normalization(tf.nn.relu(tf.layers.conv2d(inputs, num_filters * 2**int(layer_num/2), (config.filter_len,1)
        , strides=(2,1),  padding = 'same', name = "G_"+str(layer_num), kernel_initializer=tf.random_normal_initializer(stddev=0.02))), training = is_train, name = "GBN_"+str(layer_num))
    return output

def decoder_conv_block(inputs, layer, layer_num, is_train, num_filters = config.filters):

    deconv = tf.image.resize_images(inputs, size=(int(config.max_phr_len/2**(config.encoder_layers - 1 - layer_num)),1), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # embedding = tf.tile(embedding,[1,int(config.max_phr_len/2**(config.encoder_layers - 1 - layer_num)),1,1])

    deconv = tf.layers.batch_normalization( tf.nn.relu(tf.layers.conv2d(deconv, layer.shape[-1]
        , (config.filter_len,1), strides=(1,1),  padding = 'same', name =  "D_"+str(layer_num), kernel_initializer=tf.random_normal_initializer(stddev=0.02))), training = is_train, name =  "DBN_"+str(layer_num))

    # embedding =tf.nn.relu(tf.layers.conv2d(embedding, layer.shape[-1]
    #     , (config.filter_len,1), strides=(1,1),  padding = 'same', name =  "DEnc_"+str(layer_num)))

    deconv =  tf.concat([deconv, layer], axis = -1)

    return deconv

def encoder_decoder_archi(inputs, is_train):
    """
    Input is assumed to be a 4-D Tensor, with [batch_size, phrase_len, 1, features]
    """

    encoder_layers = []

    encoded = inputs

    encoder_layers.append(encoded)

    for i in range(config.encoder_layers):
        encoded = encoder_conv_block(encoded, i, is_train)
        encoder_layers.append(encoded)
    
    encoder_layers.reverse()



    decoded = encoder_layers[0]

    for i in range(config.encoder_layers):
        decoded = decoder_conv_block(decoded, encoder_layers[i+1], i, is_train)

    return decoded

def harm_network(inputs, is_train):

    inputs = tf.reshape(inputs, [config.batch_size, config.max_phr_len, 1, -1])

    inputs = tf.layers.batch_normalization(tf.layers.dense(inputs, config.wavenet_filters
        , name = "P_in"), training = is_train)
 
    output = encoder_decoder_archi(inputs, is_train)


    output = tf.layers.batch_normalization(tf.layers.dense(output, config.filters, name = "P_F"), training = is_train)

    output = tf.squeeze(output)

    harm = tf.layers.conv1d(output,config.wavenet_filters,1)

    harm = tf.layers.dense(harm, 60, activation=tf.nn.relu)

    return harm

def ap_network(inputs, conditioning, is_train):

    inputs = tf.concat([inputs, conditioning], axis = -1)

    inputs = tf.reshape(inputs, [config.batch_size, config.max_phr_len, 1, -1])

    inputs = tf.layers.batch_normalization(tf.layers.dense(inputs, config.wavenet_filters
        , name = "P_in"), training = is_train)
 
    output = encoder_decoder_archi(inputs, is_train)


    output = tf.layers.batch_normalization(tf.layers.dense(output, config.filters, name = "P_F"), training = is_train)

    output = tf.squeeze(output)

    ap = tf.layers.conv1d(output,config.wavenet_filters,1)

    ap = tf.layers.dense(ap, 4, activation=tf.nn.relu)

    return ap

def f0_network(inputs, conditioning, is_train):

    inputs = tf.concat([inputs, conditioning], axis = -1)

    inputs = tf.reshape(inputs, [config.batch_size, config.max_phr_len, 1, -1])

    inputs = tf.layers.batch_normalization(tf.layers.dense(inputs, config.wavenet_filters
        , name = "P_in"), training = is_train)
 
    output = encoder_decoder_archi(inputs, is_train)


    output = tf.layers.batch_normalization(tf.layers.dense(output, config.filters, name = "P_F"), training = is_train)

    output = tf.squeeze(output)

    f0 = tf.layers.conv1d(output,config.wavenet_filters,1)

    f0 = tf.layers.dense(f0, 1, activation=tf.nn.relu)

    return f0

def vuv_network(inputs, conditioning, is_train):

    inputs = tf.concat([inputs, conditioning], axis = -1)

    inputs = tf.reshape(inputs, [config.batch_size, config.max_phr_len, 1, -1])

    inputs = tf.layers.batch_normalization(tf.layers.dense(inputs, config.wavenet_filters
        , name = "P_in"), training = is_train)
 
    output = encoder_decoder_archi(inputs, is_train)


    output = tf.layers.batch_normalization(tf.layers.dense(output, config.filters, name = "P_F"), training = is_train)

    output = tf.squeeze(output)

    vuv = tf.layers.conv1d(output,config.wavenet_filters,1)

    vuv = tf.layers.dense(vuv, 1, activation=tf.nn.sigmoid)

    return vuv


def main():    
    vec = tf.placeholder("float", [config.batch_size, config.max_phr_len, config.input_features])
    tec = np.random.rand(config.batch_size, config.max_phr_len, config.input_features) #  batch_size, time_steps, features
    seqlen = tf.placeholder("float", [config.batch_size, 256])
    outs = f0_network_2(seqlen, vec, vec)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    f0_1= sess.run(outs, feed_dict={vec: tec, seqlen: np.random.rand(config.batch_size, 256)})
    # writer = tf.summary.FileWriter('.')
    # writer.add_graph(tf.get_default_graph())
    # writer.add_summary(summary, global_step=1)
    import pdb;pdb.set_trace()


if __name__ == '__main__':
  main()