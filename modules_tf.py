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
    return harm, ap, f0, vuv
        


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

    prenet_out = tf.layers.dense(inputs, config.lstm_size*2)
    prenet_out = tf.layers.dense(prenet_out, config.lstm_size)

    receptive_field = 2**num_block

    first_conv = tf.layers.conv1d(prenet_out, config.wavenet_filters, 1)
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

    output = tf.layers.conv1d(output,config.wavenet_filters,1)

    output = tf.nn.relu(output)

    output = tf.layers.conv1d(output,config.wavenet_filters,1)

    output = tf.nn.relu(output)

    harm = tf.layers.dense(output, 60, activation=tf.nn.relu)
    ap = tf.layers.dense(output, 4, activation=tf.nn.relu)
    f0 = tf.layers.dense(output, 64, activation=tf.nn.relu) 
    f0 = tf.layers.dense(f0, 1, activation=tf.nn.relu)
    vuv = tf.layers.dense(ap, 1, activation=tf.nn.sigmoid)

    return harm, ap, f0, vuv

def psuedo_r_wavenet(inputs, num_block = config.wavenet_layers):

    prenet_out = tf.layers.dense(inputs, config.lstm_size*2)
    prenet_out = tf.layers.dense(prenet_out, config.lstm_size)

    receptive_field = 2**num_block

    first_conv = tf.layers.conv1d(prenet_out, config.wavenet_filters, 1)
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

    output = tf.layers.conv1d(output,config.wavenet_filters,1)

    output = tf.nn.relu(output)

    output = tf.layers.conv1d(output,config.wavenet_filters,1)

    output = tf.nn.relu(output)

    harm_1 = tf.layers.dense(output, 60, activation=tf.nn.relu)
    ap = tf.layers.dense(output, 4, activation=tf.nn.relu)
    f0 = tf.layers.dense(output, 64, activation=tf.nn.relu) 
    f0 = tf.layers.dense(f0, 1, activation=tf.nn.relu)
    vuv = tf.layers.dense(ap, 1, activation=tf.nn.sigmoid)

    first_conv_2 = tf.layers.conv1d(harm_1, config.wavenet_filters, 1)

    skips_2 = []
    skip, residual = nr_wavenet_block(first_conv_2, dilation_rate=1)
    output_2 = skip
    for i in range(num_block):
        skip, residual = nr_wavenet_block(residual, dilation_rate=2**(i+1))
        skips_2.append(skip)
    for skip in skips_2:
        output_2+=skip 
    output_2 = output_2+first_conv_2

    output_2 = tf.nn.relu(output_2)

    harm = tf.layers.dense(output_2, 60)

    return harm_1,harm, ap, f0, vuv
    # return x

def wavenet_block(inputs, conditioning, dilation_rate = 2):
    # inputs = tf.reshape(inputs, [config.batch_size, config.max_phr_len, config.input_features])
    in_padded = tf.pad(inputs, [[0,0],[dilation_rate,0],[0,0]],"CONSTANT")
    con_pad_forward = tf.pad(conditioning, [[0,0],[dilation_rate,0],[0,0]],"CONSTANT")
    con_pad_backward = tf.pad(conditioning, [[0,0],[0,dilation_rate],[0,0]],"CONSTANT")
    in_sig = tf.layers.conv1d(in_padded, config.wavenet_filters, 2, dilation_rate = dilation_rate, padding = 'valid')
    con_sig_forward = tf.layers.conv1d(con_pad_forward, config.wavenet_filters, 2, dilation_rate = dilation_rate, padding = 'valid')
    con_sig_backward = tf.layers.conv1d(con_pad_backward, config.wavenet_filters, 2, dilation_rate = dilation_rate, padding = 'valid')
    # con_sig = tf.layers.conv1d(conditioning,config.wavenet_filters,1)

    sig = tf.sigmoid(in_sig+con_sig_forward[:,:in_sig.shape[1],:]+con_sig_backward[:,:in_sig.shape[1],:])

    in_tanh = tf.layers.conv1d(in_padded, config.wavenet_filters, 2, dilation_rate = dilation_rate, padding = 'valid')
    con_tanh_forward = tf.layers.conv1d(con_pad_forward, config.wavenet_filters, 2, dilation_rate = dilation_rate, padding = 'valid')
    con_tanh_backward = tf.layers.conv1d(con_pad_backward, config.wavenet_filters, 2, dilation_rate = dilation_rate, padding = 'valid')    
    # con_tanh = tf.layers.conv1d(conditioning,config.wavenet_filters,1)

    tanh = tf.tanh(in_tanh+con_tanh_forward[:,:in_tanh.shape[1],:]+con_tanh_backward[:,:in_tanh.shape[1],:])


    outputs = tf.multiply(sig,tanh)

    skip = tf.layers.conv1d(outputs,1,1)

    residual = skip + inputs

    return skip, residual


def wavenet(inputs, conditioning, num_block = config.wavenet_layers):
    receptive_field = 2**num_block

    first_conv = tf.layers.conv1d(inputs, 66, 1)
    skips = []
    skip, residual = wavenet_block(inputs, conditioning, dilation_rate=1)
    output = skip
    for i in range(num_block):
        skip, residual = wavenet_block(residual, conditioning, dilation_rate=2**(i+1))
        skips.append(skip)
    for skip in skips:
        output+=skip
    output = output+first_conv

    output = tf.nn.relu(output)

    output = tf.layers.conv1d(output,66,1)

    output = tf.nn.relu(output)

    output = tf.layers.conv1d(output,66,1)

    output = tf.nn.relu(output)

    vuv = tf.sigmoid(output[:,:,-1:])
    return output[:,:,:-1],vuv


def GAN_generator(inputs, num_block = config.wavenet_layers):
    # inputs = tf.reshape(inputs, [config.batch_size, config.max_phr_len, config.input_features,1])
    # inputs = tf.layers.conv2d(inputs, config.wavenet_filters, 5,  padding = 'same', name = "G_1")
    # # inputs = tf.layers.conv2d(inputs, config.wavenet_filters, 5,  padding = 'same', name = "G_2")
    # # inputs = tf.layers.conv2d(inputs, config.wavenet_filters, 5,  padding = 'same', name = "G_3")
    # # inputs = tf.layers.conv2d(inputs, config.wavenet_filters, 5,  padding = 'same', name = "G_4")
    # # inputs = tf.layers.conv2d(inputs, config.wavenet_filters, 5,  padding = 'same', name = "G_5")

    # inputs = tf.layers.conv2d(inputs, 1, 5,  padding = 'same', name = "G_6")

    inputs = tf.layers.dense(inputs, config.lstm_size, name = "G_1")
    inputs = tf.layers.dense(inputs, 60, name = "G_2")
    first_conv_2 = tf.layers.conv1d(inputs, config.wavenet_filters, 1)

    skips_2 = []
    skip, residual = nr_wavenet_block(first_conv_2, dilation_rate=1)
    output_2 = skip
    for i in range(num_block):
        skip, residual = nr_wavenet_block(residual, dilation_rate=2**(i+1))
        skips_2.append(skip)
    for skip in skips_2:
        output_2+=skip 
    output_2 = output_2+first_conv_2

    output_2 = tf.nn.relu(output_2)

    harm = tf.nn.relu(tf.layers.dense(output_2, 60))
    # import pdb;pdb.set_trace()
    # inputs = tf.reshape(inputs,[config.batch_size, config.max_phr_len, config.input_features] )
    return harm
    # import pdb;pdb.set_trace()


def GAN_discriminator(inputs, conditioning):
    ops = tf.concat([inputs, conditioning], 2)
    ops = tf.layers.dense(ops, config.lstm_size, name = "D_1")
    ops = tf.layers.dense(ops, 30, name = "D_2")
    ops = tf.layers.dense(ops, 1, name = "D_3")
    ops = tf.reshape(ops, [config.batch_size,-1])
    ops = tf.layers.dense(ops, 1, name = "D_4")
    ops = tf.nn.sigmoid(ops)
    return ops



def main():    
    vec = tf.placeholder("float", [config.batch_size, config.max_phr_len, config.input_features])
    tec = np.random.rand(config.batch_size, config.max_phr_len, config.input_features) #  batch_size, time_steps, features
    seqlen = tf.placeholder(tf.int32, [config.batch_size])
    outs = GAN_discriminator(vec,vec)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    output = sess.run(outs, feed_dict={vec: tec, seqlen: np.random.rand(config.batch_size)})
    # writer = tf.summary.FileWriter('.')
    # writer.add_graph(tf.get_default_graph())
    # writer.add_summary(summary, global_step=1)
    import pdb;pdb.set_trace()


if __name__ == '__main__':
  main()