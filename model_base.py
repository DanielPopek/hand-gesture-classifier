import tensorflow as tf
import numpy as np
from tqdm import tqdm_notebook as tqdm


class BaseModel(tf.keras.Model):

    def __init__(self):
        super(BaseModel, self).__init__()

        # STATISTICS
        self.train_learning_accuracy = []
        self.test_learning_accuracy = []
        self.train_learning_losses = []
        self.test_learning_losses = []
        self.test_f1 = []
        return

    def predict(self, x, **kwargs):
        res = self.call(x)
        print('call result', res)
        print('predict result', np.argmax(res, axis=-1))
        return np.argmax(res, axis=-1)

    @tf.function
    def train_step(self, images, labels, optimizer, loss_function, train_loss_metric, train_accuracy):
        with tf.GradientTape() as tape:
            predictions = self(images)
            loss = loss_function(labels, predictions)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        train_loss_metric(loss)
        train_accuracy(labels, predictions)
        return

    @tf.function
    def test_step(self, images, labels, loss_function, test_loss_metric, test_accuracy):
        predictions = self(images)
        t_loss = loss_function(labels, predictions)
        test_loss_metric(t_loss)
        test_accuracy(labels, predictions)
        return

    def fit(
            self,
            loss_function=tf.keras.losses.SparseCategoricalCrossentropy,
            epochs=10,
            optimizer=tf.keras.optimizers.Adam,
            learning_rate=0.001,
            batch_size=32,
            verbose=False,
            augmented=False,
            train_dataset_path='',
            validation_dataset_path='',
            image_height=0,
            image_width=0,
            class_names=None,
            **kwargs
    ):

        if class_names is None:
            class_names = []
        loss_function = loss_function()

        self.train_learning_accuracy = []
        self.test_learning_accuracy = []
        self.train_learning_losses = []
        self.test_learning_losses = []

        train_loss_metric = tf.keras.metrics.Mean(name='train_loss_metric')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        test_loss_metric = tf.keras.metrics.Mean(name='test_loss_metric')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        for epoch in tqdm(range(epochs)):

            image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

            train_data_gen = image_generator.flow_from_directory(
                directory=train_dataset_path,
                batch_size=batch_size,
                shuffle=True,
                target_size=(image_height, image_width),
                classes=list(class_names)
            )

            test_data_gen = image_generator.flow_from_directory(
                directory=validation_dataset_path,
                batch_size=batch_size,
                shuffle=True,
                target_size=(image_height, image_width),
                classes=list(class_names)
            )

            steps_per_epoch_train = len(train_data_gen) - 1
            steps_per_epoch_test = len(test_data_gen) - 1

            for n in range(steps_per_epoch_train):
                batch_images, batch_labels = next(train_data_gen)
                batch_images = tf.image.resize(
                    batch_images,
                    size=(image_width, image_height),
                    method=tf.image.ResizeMethod.BILINEAR,
                    preserve_aspect_ratio=False,
                    antialias=False,
                    name=None
                )
                batch_labels = np.reshape(np.argmax(batch_labels, axis=-1), newshape=(batch_size, 1))
                self.train_step(
                    batch_images,
                    batch_labels,
                    optimizer,
                    loss_function,
                    train_loss_metric,
                    train_accuracy
                )

            for t in range(steps_per_epoch_test):
                test_images, test_labels = next(test_data_gen)
                test_images = tf.image.resize(
                    test_images,
                    size=(image_width, image_height),
                    method=tf.image.ResizeMethod.BILINEAR,
                    preserve_aspect_ratio=False,
                    antialias=False,
                    name=None
                )
                test_labels = np.reshape(np.argmax(test_labels, axis=-1), newshape=(batch_size, 1))
                self.test_step(test_images, test_labels, loss_function, test_loss_metric, test_accuracy)

            if verbose:
                template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
                print(template.format(
                    epoch + 1,
                    train_loss_metric.result(),
                    train_accuracy.result() * 100,
                    test_loss_metric.result(),
                    test_accuracy.result() * 100)
                )

            self.train_learning_accuracy.append(train_accuracy.result().numpy() * 100)
            self.test_learning_accuracy.append(test_accuracy.result().numpy() * 100)
            self.train_learning_losses.append(train_loss_metric.result().numpy())
            self.test_learning_losses.append(test_loss_metric.result().numpy())

            # Reset the metrics for the next epoch
            train_loss_metric.reset_states()
            train_accuracy.reset_states()
            test_loss_metric.reset_states()
            test_accuracy.reset_states()
        return
