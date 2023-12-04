

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling2D, AveragePooling2D, LSTM,MaxPooling1D,GlobalMaxPooling1D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K

def LBPCNN(nb_classes, Chans = 64, Samples = 128, dropoutType = 'Dropout'):
    
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1   = Input(shape = (Samples, Chans))

    ##################################################################
    block1       = Conv1D(10, 5, input_shape=(Samples, Chans))(input1)
    block1       = Conv1D(10, 5)(block1)
    block1       = MaxPooling1D(20)(block1)
    block1       = Conv1D(20, 5)(block1)
    block1       = Dense(150, name = 'dense1')(block1)
    block1       = GlobalMaxPooling1D()(block1)
    dense        = Dense(nb_classes, name = 'dense2')(block1)
    softmax      = Activation('softmax', name = 'softmax')(dense)

    return Model(inputs=input1, outputs=softmax)



def LBPLSTM(nb_classes, Chans = 64, Samples = 128, dropoutType = 'Dropout'):
    
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1   = Input(shape = (Chans, Samples))

    ##################################################################
    block1       = LSTM(200, return_sequences=True)(input1)
    block1       = LSTM(150, return_sequences=True)(block1)
    block1       = LSTM(100, return_sequences=True)(block1)
    block1       = MaxPooling1D(20)(block1)
    block1       = GlobalMaxPooling1D()(block1)
    dense        = Dense(nb_classes, name = 'dense2')(block1)
    softmax      = Activation('softmax', name = 'softmax')(dense)

    
    return Model(inputs=input1, outputs=softmax)

