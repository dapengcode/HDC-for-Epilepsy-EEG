
import tensorflow as tf
import numpy as np
import time
import keras
#from utils.utils import save_logs
#from utils.utils import calculate_metrics
#from utils.utils import save_test_duration
#from sklearn.metrics import roc_auc_score


class Classifier_INCEPTION:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True, batch_size=64, lr=0.001,
                 nb_filters=32, use_residual=True, use_bottleneck=True, depth=6, kernel_size=41, nb_epochs=1500):

        self.output_directory = output_directory

        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size - 1
        self.callbacks = None
        self.batch_size = batch_size
        self.bottleneck_size = 32
        self.nb_epochs = nb_epochs
        self.lr = lr
        self.verbose = verbose

        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            self.model.summary()
            self.model.save_weights(self.output_directory + 'model_init.hdf5')

    def _inception_module(self, input_tensor, stride=1, activation='linear'):

        if self.use_bottleneck and int(input_tensor.shape[-1]) > self.bottleneck_size:
            input_inception = keras.layers.Conv1D(filters=self.bottleneck_size, kernel_size=1,
                                                  padding='same', activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        #kernel_size_s = [1,3,5,10,20,40,80,160]
        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]#40 20 10

        conv_list = []

        for i in range(len(kernel_size_s)):
            input_tensor2 = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=1,
                            strides=stride, padding='same', activation=activation, use_bias=False)(input_inception)          
            
            input_tensor3 = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=kernel_size_s[i]//2,
                            strides=stride, padding='same', activation=activation, use_bias=False)(input_tensor2)            
            
            conv_list.append(keras.layers.Conv1D(filters=self.nb_filters, kernel_size=kernel_size_s[i]//2,
                                                 strides=stride, padding='same', activation=activation, use_bias=False)(
                input_tensor3))

        max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_6 = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=1,
                                     padding='same', activation=activation, use_bias=False)(max_pool_1)

        conv_list.append(conv_6)
        
        conv_7 = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=1,
                                     padding='same', activation=activation, use_bias=False)(input_tensor)

        conv_list.append(conv_7)

        x = keras.layers.Concatenate(axis=2)(conv_list)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)
        return x

    def se_block(self, x, f, rate=16):
        #print(rate)
        m = keras.layers.GlobalAveragePooling1D()(x)
        m = keras.layers.Dense(f//rate, activation='relu')(m)
        m = keras.layers.Dense(f, activation='sigmoid')(m)
        return keras.layers.multiply([x, m])


    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                         padding='same', use_bias=False)(input_tensor)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        x = keras.layers.Add()([shortcut_y, out_tensor])
        x = keras.layers.Activation('relu')(x)
        return x

    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape)

        x = input_layer
        input_res = input_layer

        for d in range(self.depth):

            x = self._inception_module(x)

            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x
            
            #x = self.se_block(x, int(np.shape(x)[-1]),10)
            

        gap_layer = keras.layers.GlobalAveragePooling1D()(x)
        
        #gap_layer = keras.layers.Dropout(0.5)(gap_layer)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        #model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(self.lr),
        #              metrics=['acc'])

        #reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=30,
        #                                              min_lr=0.0001)

        #file_path = self.output_directory + 'best_model.hdf5'

        #model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss',
        #                                                   save_best_only=True)

        #self.callbacks = [reduce_lr, model_checkpoint]

        return model
