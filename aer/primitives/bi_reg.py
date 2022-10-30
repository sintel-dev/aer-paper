# -*- coding: utf-8 -*-
"""Implementation of Bidirectional with Regressor."""
import logging
import tempfile

import numpy as np
import tensorflow as tf
from mlprimitives.utils import import_object
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from orion.primitives.timeseries_errors import regression_errors

LOGGER = logging.getLogger(__name__)


def build_layer(layer: dict, hyperparameters: dict):
    layer_class = import_object(layer['class'])
    layer_kwargs = layer['parameters'].copy()
    # TODO: Upgrade to using tf.keras.layers.Wrapper in mlprimitives.
    if issubclass(layer_class, tf.keras.layers.Wrapper):
        layer_kwargs['layer'] = build_layer(layer_kwargs['layer'], hyperparameters)
    for key, value in layer_kwargs.items():
        if isinstance(value, str):
            layer_kwargs[key] = hyperparameters.get(value, value)
    return layer_class(**layer_kwargs)


class BiReg(object):
    """BidirectionalReg for time series regression.

    Args:
        layers_encoder (list):
            List containing layers of encoder.
        layers_generator (list):
            List containing layers of generator.
        optimizer (str):
            String denoting the keras optimizer.
        input_shape (tuple):
            Optional. Tuple denoting the shape of an input sample.
        learning_rate (float):
            Optional. Float denoting the learning rate of the optimizer. Default 0.005.
        epochs (int):
            Optional. Integer denoting the number of epochs. Default 2000.
        latent_dim (int):
            Optional. Integer denoting dimension of latent space. Default 20.
        batch_size (int):
            Integer denoting the batch size. Default 64.
        hyperparameters (dictionary):
            Optional. Dictionary containing any additional inputs.
    """

    def __getstate__(self):
        networks = ['forward_reg_model', 'reverse_reg_model']
        modules = ['forward_optimizer', 'reverse_optimizer',
                   'forward_fit_history', 'reverse_fit_history']

        state = self.__dict__.copy()

        for module in modules:
            del state[module]

        for network in networks:
            with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as fd:
                tf.keras.models.save_model(state.pop(network), fd.name, overwrite=True)

                state[network + '_str'] = fd.read()

        return state

    def __setstate__(self, state):
        networks = ['forward_reg_model', 'reverse_reg_model']
        for network in networks:
            with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as fd:
                fd.write(state.pop(network + '_str'))
                fd.flush()

                state[network] = tf.keras.models.load_model(fd.name)

        self.__dict__ = state

    def _build_model(self, hyperparameters, layers, input_shape):
        x = Input(shape=input_shape)
        model = tf.keras.models.Sequential()

        for layer in layers:
            built_layer = build_layer(layer, hyperparameters)
            model.add(built_layer)

        return Model(x, model(x))

    def _setdefault(self, kwargs, key, value):
        if key in kwargs:
            return

        if key in self.hyperparameters and self.hyperparameters[key] is None:
            kwargs[key] = value

    def __init__(self, layers_forward_reg: list, layers_reverse_reg: list,
                 forward_optimizer: str, reverse_optimizer: str,
                 input_shape: tuple = None, learning_rate: float = 0.001,
                 epochs: int = 35, batch_size: int = 64, shuffle: bool = True,
                 verbose: bool = True, callbacks: tuple = tuple(),
                 validation_split: float = 0.0, **hyperparameters):

        self.input_shape = input_shape
        self.layers_forward_reg = layers_forward_reg
        self.layers_reverse_reg = layers_reverse_reg
        self.epochs = epochs
        self.batch_size = batch_size
        self.forward_optimizer = import_object(forward_optimizer)(learning_rate)
        self.reverse_optimizer = import_object(reverse_optimizer)(learning_rate)

        self.shuffle = shuffle
        self.verbose = verbose
        self.hyperparameters = hyperparameters
        self.validation_split = validation_split
        for callback in callbacks:
            callback['class'] = import_object(callback['class'])
        self.callbacks = callbacks

        self._fitted = False
        self.forward_fit_history = None
        self.reverse_fit_history = None

    def _augment_hyperparameters(self, X, y, kwargs):
        input_shape = np.asarray(X)[0].shape
        self.input_shape = self.input_shape or input_shape
        return kwargs

    def _build_bidirectional_reg(self, **kwargs):
        hyperparameters = self.hyperparameters.copy()
        hyperparameters.update(kwargs)
        self.forward_reg_model = self._build_model(
            hyperparameters, self.layers_forward_reg, self.input_shape)
        self.forward_reg_model.compile(optimizer=self.forward_optimizer,
                                       loss='mean_squared_error')

        self.reverse_reg_model = self._build_model(
            hyperparameters, self.layers_reverse_reg, self.input_shape)
        self.reverse_reg_model.compile(optimizer=self.reverse_optimizer,
                                       loss='mean_squared_error')

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Fit the model.

        Args:
            X (ndarray):
                N-dimensional array containing the input sequences for the model.
            y (ndarray):
                N-dimensional array containing the output sequences we want to reconstruct.
        """
        forward_X = X[:, :-1, :]
        forward_y = y[:, -1]
        reverse_X = np.flip(X, 1)[:, :-1, :]
        reverse_y = np.flip(y, 1)[:, -1]

        if not self._fitted:
            self._augment_hyperparameters(forward_X, forward_y, kwargs)
            self._build_bidirectional_reg(**kwargs)

        callbacks = [
            callback['class'](**callback.get('args', dict()))
            for callback in self.callbacks
        ]

        self.forward_fit_history = self.forward_reg_model.fit(
            forward_X, forward_y, batch_size=self.batch_size, epochs=self.epochs,
            shuffle=self.shuffle, verbose=self.verbose, callbacks=callbacks,
            validation_split=self.validation_split)

        self.reverse_fit_history = self.reverse_reg_model.fit(
            reverse_X, reverse_y, batch_size=self.batch_size, epochs=self.epochs,
            shuffle=self.shuffle, verbose=self.verbose, callbacks=callbacks,
            validation_split=self.validation_split)

        self._fitted = True

    def predict(self, X: np.ndarray) -> tuple:
        """Predict values using the initialized object.

        Args:
            X (ndarray):
                N-dimensional array containing the input sequences for the model.

        Returns:
            ndarray:
                N-dimensional array containing the reconstructions for each input sequence.
            ndarray:
                N-dimensional array containing the regression for each input sequence.
        """
        forward_X = X[:, :-1, :]
        forward_y_hat = self.forward_reg_model.predict(forward_X)

        reverse_X = np.flip(X, 1)[:, :-1, :]
        reverse_y_hat = self.reverse_reg_model.predict(reverse_X)

        output = (forward_y_hat, reverse_y_hat,
                  self.forward_fit_history.history, self.reverse_fit_history.history)
        return output


def score_anomalies(y, forward_y_hat, reverse_y_hat, index, window_size: int = 100,
                    hide_start: int = None):
    """Compute an array of anomaly scores.

    Anomaly scores are calculated using a combination of bi-directional regression score.

    Args:
        y (ndarray):
            Ground truth.
        forward_y_hat (ndarray):
            Predicted values. Each timestamp has multiple predictions.
        reverse_y_hat (ndarray):
            Predicted values. Each timestamp has multiple predictions.
        index (ndarray):
            Time index for each y (start position of the window).
        window_size (int):
            Length of time series sequence.
        hide_start (int):
            Optional. Number of steps to hide from the regression errors.
    Returns:
        ndarray:
            Array of anomaly scores.
        ndarray:
            Array of true index.
    """
    time_steps = window_size - 1
    hide_start = hide_start if hide_start else int(0.01 * len(forward_y_hat))

    forward_y = y[:, -1]
    reverse_y = np.flip(y, 1)[:, -1]

    f_scores = regression_errors(forward_y, forward_y_hat, smoothing_window=0.01, smooth=True)
    f_scores[:hide_start] = 0
    f_scores = np.concatenate([np.zeros(time_steps), f_scores])

    r_scores = regression_errors(reverse_y, reverse_y_hat, smoothing_window=0.01, smooth=True)
    r_scores[:hide_start] = min(r_scores)
    r_scores = np.concatenate([r_scores, np.zeros(time_steps)])

    total_scores = f_scores + r_scores
    total_scores[time_steps + hide_start:-time_steps] /= 2
    true_index = index[:-1]

    return total_scores, true_index