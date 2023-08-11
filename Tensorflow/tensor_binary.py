# TensorFlow and tf.keras
import datetime

import tensorflow as tf
import kerastuner as kt
from tensorflow.keras import losses
from kerastuner import HyperModel
from tensorflow.keras import layers
import numpy as np
from tensorflow.python.keras import regularizers


class BinaryHyperModel(HyperModel):
    def __init__(self, input_shape_converted, input_shape, configuration_layer_tuner, flatten):
        super().__init__()
        self.flatten = flatten
        self.input_shape_converted = input_shape_converted
        self.input_shape = input_shape
        self.configuration_layer_tuner = configuration_layer_tuner

    def build(self, hp):
        model = tf.keras.Sequential()

        if self.flatten:
            model.add(tf.keras.layers.Flatten(input_shape=self.input_shape))

        # configuration is a dictionary --> {layer 1: [min, max, step], layer 2: ..., etc.}
        for k in self.configuration_layer_tuner.keys():
            hp_units = hp.Int('units_{}'.format(k),
                              min_value=self.configuration_layer_tuner[k][0],
                              max_value=self.configuration_layer_tuner[k][1],
                              step=self.configuration_layer_tuner[k][2])

            model.add(tf.keras.layers.Dense(
                units=hp_units, activation=hp.Choice(
                    'dense_activation',
                    values=['relu', 'tanh', 'sigmoid'],
                    default='relu'),
                input_shape=self.input_shape_converted,
                kernel_regularizer=regularizers.l2(l2=hp.Choice('l2_{}'.format(k), values=[1e-3, 1e-4, 1e-5]))))

            model.add(tf.keras.layers.Dropout(rate=hp.Float('rate_{}'.format(k),
                                                            min_value=0.0,
                                                            max_value=0.95,
                                                            step=0.25
                                                            )))

        model.add(layers.Dense(2))

        model.compile(
            loss=losses.BinaryCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
            metrics=['accuracy']
        )

        return model


def build_model(input_shape, input_train, target_train, configuration_layer_tuner, flatten=False):
    input_shape_converted = (input_train.shape[1],)
    hypermodel = BinaryHyperModel(input_shape_converted, input_shape, configuration_layer_tuner, flatten=flatten)
    tuner = kt.Hyperband(hypermodel, objective='val_accuracy', max_epochs=10,
                         directory='model_dir', project_name='kt_proj')
    tuner.search(input_train, target_train, epochs=10, validation_split=0.2, verbose=0)
    best_model = tuner.get_best_models(num_models=1)[0]
    return best_model


def fit_model(model, input_data, target_data, tensorboard_callback, epochs=10):
    model.fit(input_data, target_data, epochs=epochs, callbacks=[tensorboard_callback])
    return model


def evaluate_model(model, input_data, target_data, verbose=2):
    test_loss, test_acc = model.evaluate(input_data, target_data, verbose=verbose)
    return test_loss, test_acc


def predict_model(model, input_data):
    predictions = model.predict(input_data)
    return predictions[0]


def convert_to_probability_model(model):
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    return probability_model


def predict_probability_model(model, input_data):
    predictions = model.predict(input_data)
    return np.argmax(predictions[0])


def save_model(model, filename):
    model.save(filename)


def load_model(filename):
    tf.keras.models.load_model(filename)


def setup_tensorboard():
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    return tensorboard_callback
