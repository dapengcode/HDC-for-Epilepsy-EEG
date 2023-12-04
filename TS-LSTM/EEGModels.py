

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, LSTM,MaxPooling1D,GlobalMaxPooling1D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K



def EEGLSTM(nb_classes, Chans = 64, Samples = 128, dropoutType = 'Dropout'):
    
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1   = Input(shape = (Samples, Chans))

    ##################################################################
    block1       = LSTM(64, return_sequences=True)(input1)
    block1       = BatchNormalization()(block1)
    block1       = LSTM(32, return_sequences=True)(block1)
    block1       = BatchNormalization()(block1)
    block1       = LSTM(16, return_sequences=False)(block1)
    #block1       = GlobalMaxPooling1D()(block1)
    dense        = Dense(nb_classes, name = 'dense2')(block1)
    softmax      = Activation('softmax', name = 'softmax')(dense)

    
    return Model(inputs=input1, outputs=softmax)

