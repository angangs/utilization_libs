# TensorFlow and tf.keras
import datetime

import tensorflow as tf
import kerastuner as kt
from kerastuner import HyperModel
from tensorflow.keras import layers
from tensorflow.python.keras import regularizers


class RegressionHyperModel(HyperModel):
    def __init__(self, input_shape, configuration_layer_tuner):
        super().__init__()
        self.input_shape = input_shape
        self.configuration_layer_tuner = configuration_layer_tuner

    def build(self, hp):
        model = tf.keras.Sequential()

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
                input_shape=self.input_shape,
                kernel_regularizer=regularizers.l2(l2=hp.Choice('l2_{}'.format(k), values=[1e-3, 1e-4, 1e-5]))))

            model.add(tf.keras.layers.Dropout(rate=hp.Float('rate_{}'.format(k),
                                                            min_value=0.0,
                                                            max_value=0.95,
                                                            step=0.25
                                                            )))

        model.add(layers.Dense(1))

        model.compile(
            optimizer='rmsprop', loss='mse', metrics=['mse']
        )

        return model


def build_model(input_train, target_train, configuration_layer_tuner):
    input_shape = (input_train.shape[1],)
    hypermodel = RegressionHyperModel(input_shape, configuration_layer_tuner)
    tuner = kt.Hyperband(hypermodel, objective='mse', max_epochs=10,
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


def save_model(model, filename):
    model.save(filename)


def load_model(filename):
    tf.keras.models.load_model(filename)


def setup_tensorboard():
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    return tensorboard_callback


fashion_mnist = tf.keras.datasets.boston_housing
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
model = build_model(input_train=train_images, target_train=train_labels,
                    configuration_layer_tuner={
                        1: [16, 64, 16],
                        2: [128, 512, 128]
                    })
tc = setup_tensorboard()
model = fit_model(model, train_images, train_labels, tensorboard_callback=tc)
loss, mse = evaluate_model(model, test_images, test_labels)
