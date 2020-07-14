import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from model_base import BaseModel
import numpy as np


class VGGModel(BaseModel):

    def __init__(self):
        super(VGGModel, self).__init__()

        self.vgg = tf.keras.applications.VGG16(
            input_shape=(54, 54, 3),
            include_top=False,
            weights='imagenet'
        )
        self.vgg.trainable = False
        self.global_pooling = GlobalAveragePooling2D()
        self.dense = Dense(6, activation='softmax')
        return

    def call(self, inputs, training=False, **kwargs):
        """Makes forward pass of the network."""
        x = self.vgg(inputs)
        x = self.global_pooling(x)
        return self.dense(x)


def recreate_vgg_model_from_weights(model_filename) -> VGGModel:
    model_recreated = VGGModel()
    model_recreated.build(input_shape=(1, 54, 54, 3))
    read_weights = np.load(model_filename, allow_pickle=True).tolist()
    model_recreated.set_weights(read_weights)
    return model_recreated
