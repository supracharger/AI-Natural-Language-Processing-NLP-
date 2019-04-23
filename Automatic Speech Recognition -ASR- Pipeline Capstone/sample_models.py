from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, MaxPooling1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # batch normalization 
    bn_rnn = BatchNormalization(momentum=0.07)(simp_rnn)
    # TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='elu',
                     name='conv1d')(input_data)
    # Batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Recurrent layer
    simp_rnn = GRU(units, activation='elu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # Batch normalization
    bn_rnn = BatchNormalization(momentum=0.07)(simp_rnn)
    # TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    assert recur_layers > 0
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Recurrent layers, each with batch normalization
    recurLayer = input_data
    # Loop number of Recurrent Layers
    for r in range(recur_layers):
        # GRU Recurrent Layer elu
        simp_rnn = GRU(units, activation='elu',
                        return_sequences=True, implementation=2, name='rnn'+str(r+1))(recurLayer)
        # Batch Norm. for the GRU Layer
        recurLayer = BatchNormalization(momentum=0.07)(simp_rnn)
    # TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(recurLayer)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Bidirectional recurrent layer
    bidir_rnn = Bidirectional(GRU(units, activation='elu',
                    return_sequences=True, implementation=2, name='rnn'))(input_data)
    # Batch Norm.
    bn = BatchNormalization(momentum=0.07)(bidir_rnn)
    # TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn)
    # Softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def final_model(input_dim, units, output_dim=29, conv_border_mode='same'):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Specify the layers in network
    # Convolutional layer
    conv1 = Conv1D(256, 16, strides=2, padding=conv_border_mode,
                     activation='elu', name='conv1')(input_data)
    # Batch Norm: Conv Layer
    bn_cnn = BatchNormalization(momentum=0.1, name='bn_conv_1d')(conv1)
    # Bidirectional GRU Recurrent Layer elu
    bidir_rnn = Bidirectional(GRU(units, activation='elu',
                    return_sequences=True, implementation=2, name='birnn'))(bn_cnn)
    # Batch Norm: Bidirectional GRU
    bn = BatchNormalization(momentum=0.1)(bidir_rnn)
    # 2nd GRU Recurrent Layer
    gru = GRU(units, activation='elu',
                return_sequences=True, implementation=2, name='rnn')(bn)
    # Batch Norm: GRU Layer
    bn_gru = BatchNormalization(momentum=0.1)(gru)
    # Time Distributed Dense Layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_gru)
    # Softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # Specify model.output_length
    model.output_length = lambda x: cnn_output_length(x, 16, conv_border_mode, stride=2)
    print(model.summary())
    return model