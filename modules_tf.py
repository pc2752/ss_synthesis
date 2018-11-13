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

def deconv2d(input_, output_shape,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="deconv2d"):
  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.random_normal_initializer(stddev=stddev))
    
    # try:
    deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1], nmae = name)

    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    if with_w:
      return deconv, w, biases
    else:
      return deconv


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
        


def nr_wavenet_block(conditioning, dilation_rate = 2, scope = 'nr_wavenet_block', name = "name"):

    with tf.variable_scope(scope):
    # inputs = tf.reshape(inputs, [config.batch_size, config.max_phr_len, config.input_features])

        con_pad_forward = tf.pad(conditioning, [[0,0],[dilation_rate,0],[0,0]],"CONSTANT")
        con_pad_backward = tf.pad(conditioning, [[0,0],[0,dilation_rate],[0,0]],"CONSTANT")
        con_sig_forward = tf.layers.conv1d(con_pad_forward, config.wavenet_filters, 2, dilation_rate = dilation_rate, padding = 'valid', name = name+"1")
        con_sig_backward = tf.layers.conv1d(con_pad_backward, config.wavenet_filters, 2, dilation_rate = dilation_rate, padding = 'valid', name = name+"2")
        # con_sig = tf.layers.conv1d(conditioning,config.wavenet_filters,1)

        sig = tf.sigmoid(con_sig_forward+con_sig_backward)


        con_tanh_forward = tf.layers.conv1d(con_pad_forward, config.wavenet_filters, 2, dilation_rate = dilation_rate, padding = 'valid', name = name+"3")
        con_tanh_backward = tf.layers.conv1d(con_pad_backward, config.wavenet_filters, 2, dilation_rate = dilation_rate, padding = 'valid', name = name+"4")    
        # con_tanh = tf.layers.conv1d(conditioning,config.wavenet_filters,1)

        tanh = tf.tanh(con_tanh_forward+con_tanh_backward)


        outputs = tf.multiply(sig,tanh)

        skip = tf.layers.conv1d(outputs,config.wavenet_filters,1, name = name+"5")

        residual = skip + conditioning

    return skip, residual

def nr_wavenet_block_d(conditioning,filters=2, scope = 'nr_wavenet_block', name = "name"):

    with tf.variable_scope(scope):
    # inputs = tf.reshape(inputs, [config.batch_size, config.max_phr_len, config.input_features])

        # con_pad_forward = tf.pad(conditioning, [[0,0],[dilation_rate,0],[0,0]],"CONSTANT")
        # con_pad_backward = tf.pad(conditioning, [[0,0],[0,dilation_rate],[0,0]],"CONSTANT")
        con_sig_forward = tf.layers.conv1d(conditioning, config.wavenet_filters, filters, padding = 'valid', name = name+"1")
        # con_sig_backward = tf.layers.conv1d(con_pad_backward, config.wavenet_filters, 2, dilation_rate = dilation_rate, padding = 'valid', name = name+"2")
        # con_sig = tf.layers.conv1d(conditioning,config.wavenet_filters,1)

        sig = tf.sigmoid(con_sig_forward)


        con_tanh_forward = tf.layers.conv1d(conditioning, config.wavenet_filters, filters, padding = 'valid', name = name+"2")

        tanh = tf.tanh(con_tanh_forward)


        outputs = tf.multiply(sig,tanh)

        skip = tf.layers.conv1d(outputs,config.wavenet_filters,1, name = name+"5")

        residual = skip

    return skip, residual


def nr_wavenet(inputs, num_block = config.wavenet_layers):
    prenet_out = tf.layers.dense(inputs, config.lstm_size*2)
    prenet_out = tf.layers.dense(prenet_out, config.lstm_size)

    receptive_field = 2**num_block

    first_conv = tf.layers.conv1d(prenet_out, config.wavenet_filters, 1)
    skips = []
    skip, residual = nr_wavenet_block(first_conv, dilation_rate=1, scope = "nr_wavenet_block_0")
    output = skip
    for i in range(num_block):
        skip, residual = nr_wavenet_block(residual, dilation_rate=2**(i+1), scope = "nr_wavenet_block_"+str(i+1))
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

def nr_wavenet_block_2(conditioning,inputs, dilation_rate = 2, scope = 'nr_wavenet_block', name = "name"):

    with tf.variable_scope(scope):
    # inputs = tf.reshape(inputs, [config.batch_size, config.max_phr_len, config.input_features])

        con_pad_forward = tf.pad(conditioning, [[0,0],[dilation_rate,0],[0,0]],"CONSTANT")
        con_pad_backward = tf.pad(conditioning, [[0,0],[0,dilation_rate],[0,0]],"CONSTANT")

        in_pad_forward = tf.pad(inputs, [[0,0],[dilation_rate,0],[0,0]],"CONSTANT")
        in_pad_backward = tf.pad(inputs, [[0,0],[0,dilation_rate],[0,0]],"CONSTANT")


        con_sig_forward = tf.layers.conv1d(con_pad_forward, config.wavenet_filters, 2, dilation_rate = dilation_rate, padding = 'valid', name = name+"1")
        con_sig_backward = tf.layers.conv1d(con_pad_backward, config.wavenet_filters, 2, dilation_rate = dilation_rate, padding = 'valid', name = name+"2")

        in_sig_forward = tf.layers.conv1d(in_pad_forward, config.wavenet_filters, 2, dilation_rate = dilation_rate, padding = 'valid', name = name+"3")
        in_sig_backward = tf.layers.conv1d(in_pad_backward, config.wavenet_filters, 2, dilation_rate = dilation_rate, padding = 'valid', name = name+"4")
        # con_sig = tf.layers.conv1d(conditioning,config.wavenet_filters,1)

        sig = tf.sigmoid(con_sig_forward+con_sig_backward+in_sig_backward+in_sig_forward)


        con_tanh_forward = tf.layers.conv1d(con_pad_forward, config.wavenet_filters, 2, dilation_rate = dilation_rate, padding = 'valid', name = name+"5")
        con_tanh_backward = tf.layers.conv1d(con_pad_backward, config.wavenet_filters, 2, dilation_rate = dilation_rate, padding = 'valid', name = name+"6")   

        in_tanh_forward = tf.layers.conv1d(in_pad_forward, config.wavenet_filters, 2, dilation_rate = dilation_rate, padding = 'valid', name = name+"7")
        in_tanh_backward = tf.layers.conv1d(in_pad_backward, config.wavenet_filters, 2, dilation_rate = dilation_rate, padding = 'valid', name = name+"8")    
        # con_tanh = tf.layers.conv1d(conditioning,config.wavenet_filters,1)

        tanh = tf.tanh(con_tanh_forward+con_tanh_backward+in_tanh_backward+in_tanh_forward)


        outputs = tf.multiply(sig,tanh)

        skip = tf.layers.conv1d(outputs,config.wavenet_filters,1, name = name+"9")

        residual = skip + conditioning

    return skip, residual


def final_net(singer_label, f0_notation, phones):

    singer_label = tf.reshape(singer_label, [config.batch_size,1, -1], name = "f_r_1")

    singer_label = tf.tile(singer_label, [1, config.max_phr_len, 1], name = "f_r_2")

    conditioning = tf.concat([phones, singer_label, f0_notation], axis = -1)

    num_block = config.wavenet_layers

    receptive_field = 2**num_block

    first_conv = tf.layers.conv1d(conditioning, config.wavenet_filters, 1, name = "f_c_1")
    skips = []
    skip, residual = nr_wavenet_block(first_conv,dilation_rate=1, scope = "f_nr_wavenet_block_0", name ="f_nr_wavenet_block_0" )
    output = skip
    for i in range(num_block):
        skip, residual = nr_wavenet_block(residual, dilation_rate=2**(i+1), scope = "f_nr_wavenet_block_"+str(i+1), name = "f_nr_wavenet_block_"+str(i+1))
        skips.append(skip)
    for skip in skips:
        output+=skip
    
    output = output+first_conv

    output = tf.layers.conv1d(output,config.wavenet_filters,1, name = "f_c_2")

    output = tf.layers.dense(output, 64, name = "f_f_1")

    return output



def phone_network(inputs):

    prenet_out = tf.layers.dense(inputs, config.lstm_size*2, name = "dense_p_1")
    prenet_out = tf.layers.dense(prenet_out, config.lstm_size, name = "dense_p_2")

    num_block = config.wavenet_layers

    receptive_field = 2**num_block

    first_conv = tf.layers.conv1d(prenet_out, config.wavenet_filters, 1, name = "conv_p_1")
    skips = []
    skip, residual = nr_wavenet_block(first_conv, dilation_rate=1, scope = "p_nr_wavenet_block_0", name = "p_nr_wavenet_block_0")
    output = skip
    for i in range(num_block):
        skip, residual = nr_wavenet_block(residual, dilation_rate=2**(i+1), scope = "p_nr_wavenet_block_"+str(i+1), name = "p_nr_wavenet_block_"+str(i+1) )
        skips.append(skip)
    for skip in skips:
        output+=skip
    output = output+first_conv

    output = tf.layers.conv1d(output,config.wavenet_filters,1, name = "conv_p_2")

    output = tf.layers.dense(output, 42, name = "dense_p_3")

    return output

def GAN_discriminator(inputs, singer_label):
    # ops = tf.concat([inputs,  f0_input_placeholder_midi, pho_input_placeholder], axis = -1)
    # ops = inputs
    singer_label = tf.reshape(singer_label, [config.batch_size,1, -1], name = "d_r_1")

    singer_label = tf.tile(singer_label, [1, config.max_phr_len, 1], name = "d_r_2")

    inputs = tf.concat([inputs,singer_label], axis = -1)

    num_block = config.wavenet_layers

    receptive_field = 2**num_block
    num_block = config.wavenet_layers

    first_conv = tf.layers.conv1d(inputs, config.wavenet_filters, 1, name = "d_c_1")
    skips = []
    skip, residual = nr_wavenet_block_d(first_conv, 1, scope = "d_nr_wavenet_block_0", name ="d_nr_wavenet_block_0" )
    output = skip
    for i in range(num_block):
        skip, residual = nr_wavenet_block_d(residual, 2**(i+1), scope = "d_nr_wavenet_block_"+str(i+1), name = "d_nr_wavenet_block_"+str(i+1))

    output = skip

    output = tf.layers.conv1d(output,config.wavenet_filters,1, name = "conv_d_1")



    # import pdb;pdb.set_trace()

    # output = tf.layers.conv1d(output,config.wavenet_filters,1, name = "conv_d_2")

    output = tf.reshape(output, [config.batch_size, -1], name = "r_d_1")
    
    # output = tf.layers.dense(output, 128, name = "s9", activation =tf.nn.relu)

    output = tf.layers.dense(output, 1, name = "s10")

    return output


def singer_network(inputs, prob):



    embed_1 = tf.layers.dense(inputs, 32, name = "s1")


    conv1 = tf.layers.conv1d(inputs=embed_1, filters=128, kernel_size=2, padding='same', activation=tf.nn.relu, name = "s2")

    # maxpool1 = tf.layers.max_pooling1d(conv1, pool_size=2, strides=2, padding='same', name = "s3")

    conv2 = tf.layers.conv1d(inputs=conv1, filters=64, kernel_size=4, padding='same', activation=tf.nn.relu, name = "s4")

    # maxpool2 = tf.layers.max_pooling1d(conv2, pool_size=2, strides=2, padding='same', name = "s5")

    conv3 = tf.layers.conv1d(inputs=conv2, filters=32, kernel_size=4, padding='same', activation=tf.nn.relu, name = "s6")

    # encoded = tf.layers.max_pooling1d(conv3, pool_size=2, strides=2, padding='same', name = "s7")

    encoded = tf.reshape(conv3, [config.batch_size, -1], name = "s8")

    encoded_1= tf.layers.dense(encoded, 256, name = "s9")

    singer = tf.layers.dense(encoded_1, 12, name = "s10")

    return encoded_1, singer






    # return x

def wavenet_block(inputs, conditioning, dilation_rate = 2, scope = 'wavenet_block'):
    # inputs = tf.reshape(inputs, [config.batch_size, config.max_phr_len, config.input_features])
    in_padded = tf.pad(inputs, [[0,0],[dilation_rate,0],[0,0]],"CONSTANT")
    con_pad_forward = tf.pad(conditioning, [[0,0],[dilation_rate,0],[0,0]],"CONSTANT")
    con_pad_backward = tf.pad(conditioning, [[0,0],[0,dilation_rate],[0,0]],"CONSTANT")
    in_sig = tf.layers.conv1d(in_padded, config.wavenet_filters, 2, dilation_rate = dilation_rate, padding = 'valid')
    con_sig_forward = tf.layers.conv1d(con_pad_forward, config.wavenet_filters, 2, dilation_rate = dilation_rate, padding = 'valid')
    con_sig_backward = tf.layers.conv1d(con_pad_backward, config.wavenet_filters, 2, dilation_rate = dilation_rate, padding = 'valid')
    # con_sig = tf.layers.conv1d(conditioning,config.wavenet_filters,1)

    sig = tf.sigmoid(in_sig+con_sig_forward+con_sig_backward)

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
    skip, residual = wavenet_block(first_conv, conditioning, dilation_rate=1)
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

    harm_1 = tf.layers.dense(output, 60, activation=tf.nn.relu)
    ap = tf.layers.dense(output, 4, activation=tf.nn.relu)
    f0 = tf.layers.dense(output, 64, activation=tf.nn.relu) 
    f0 = tf.layers.dense(f0, 1, activation=tf.nn.relu)
    vuv = tf.layers.dense(ap, 1, activation=tf.nn.sigmoid)
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
    inputs = tf.layers.conv1d(inputs, config.wavenet_filters, 1)

    inputs = tf.layers.conv1d(inputs, config.wavenet_filters, 2, padding = 'same', name = "G_c1")
    inputs = tf.layers.conv1d(inputs, config.wavenet_filters, 4, padding = 'same', name = "G_c2")
    inputs = tf.layers.conv1d(inputs, config.wavenet_filters, 8, padding = 'same', name = "G_c3")
    inputs = tf.layers.conv1d(inputs, config.wavenet_filters, 16, padding = 'same', name = "G_c4")
    inputs = tf.layers.conv1d(inputs, config.wavenet_filters, 32, padding = 'same', name = "G_c5")

    harm = tf.nn.tanh(tf.layers.dense(inputs, 60, name = "G_3"))
    # import pdb;pdb.set_trace()
    # inputs = tf.reshape(inputs,[config.batch_size, config.max_phr_len, config.input_features] )
    return harm
    # import pdb;pdb.set_trace()









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