import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, \
    Activation, Dropout
from model_base import BaseModel
import numpy as np


class ConvolutionBlock(tf.keras.Model):
    def __init__(self, pooling_type, kernel_size, batch_norm, residual):
        super(ConvolutionBlock, self).__init__()
        self.batch_norm = batch_norm
        self.residual = residual

        self.conv = Conv2D(kernel_size=kernel_size, strides=(1, 1), padding='same', filters=64)
        self.batch_norm = BatchNormalization()
        self.activation = Activation('relu')
        self.pooling = self.create_pooling_layer(pooling_type)

    @staticmethod
    def create_pooling_layer(pooling_type):
        if pooling_type == 'max':
            return MaxPooling2D(pool_size=(3, 3), strides=(2, 2))
        else:
            if pooling_type == 'average':
                return AveragePooling2D(pool_size=(3, 3), strides=(2, 2))
            else:
                return None

    def call(self, inputs, training=False, **kwargs):
        x = self.conv(inputs)
        if self.batch_norm:
            x = self.batch_norm(x)

        if self.residual:
            x = x + inputs
        x = self.activation(x)
        if self.pooling is not None:
            x = self.pooling(x)
        return x


class SimpleCNNModel(BaseModel):

    def __init__(self, pooling_type, kernel_size, first_dense_size, dropout, batch_norm, residual):
        super(SimpleCNNModel, self).__init__()

        self.dropout = dropout

        self.conv_block1 = ConvolutionBlock(pooling_type, kernel_size, batch_norm, residual=False)
        self.conv_block2 = ConvolutionBlock(pooling_type, kernel_size, batch_norm, residual)

        self.flatten = Flatten()

        self.d1 = Dense(first_dense_size, activation='relu')
        if self.dropout is not None:
            self.dropout1 = Dropout(dropout)
        self.d2 = Dense(384, activation='relu')
        if self.dropout is not None:
            self.dropout2 = Dropout(dropout)
        self.d3 = Dense(192, activation='relu')
        if self.dropout is not None:
            self.dropout3 = Dropout(dropout)
        self.d4 = Dense(6, activation='softmax')

        # STATISTICS
        self.train_learning_accuracy = []
        self.test_learning_accuracy = []
        self.train_learning_losses = []
        self.test_learning_losses = []
        return

    def call(self, inputs, training=False, **kwargs):
        """Makes forward pass of the network."""
        x = self.conv_block1(inputs)
        x = self.conv_block2(x)

        x = self.flatten(x)
        x = self.d1(x)
        if self.dropout is not None:
            x = self.dropout1(x)
        x = self.d2(x)
        if self.dropout is not None:
            x = self.dropout2(x)
        x = self.d3(x)
        if self.dropout is not None:
            x = self.dropout3(x)
        return self.d4(x)


def recreate_cnn_model_from_weights(weights_filename) -> SimpleCNNModel:
    # TODO: here we should extract these parameters to additional file
    model_recreated = SimpleCNNModel(
        pooling_type='max',
        kernel_size=(3, 3),
        first_dense_size=1600,
        dropout=0,
        batch_norm=True,
        residual=True
    )
    model_recreated.build(input_shape=(1, 54, 54, 3))
    read_weights = np.load(weights_filename, allow_pickle=True).tolist()
    model_recreated.set_weights(read_weights)
    return model_recreated
