"""
Time Series error calculation functions.
"""

import math

import numpy as np
import pandas as pd
from pyts.metrics import dtw
from scipy import integrate

from orion.primitives.timeseries_errors import _point_wise_error, _area_error, _dtw_error


def reconstruction_errors(y, y_hat, step_size=1, score_window=10, smoothing_window=0.01,
                          smooth=True, rec_error_type='point', mask=False):
    """Compute an array of reconstruction errors.

    Compute the discrepancies between the expected and the
    predicted values according to the reconstruction error type.

    Args:
        y (ndarray):
            Ground truth.
        y_hat (ndarray):
            Predicted values. Each timestamp has multiple predictions.
        step_size (int):
            Optional. Indicating the number of steps between windows in the predicted values.
            If not given, 1 is used.
        score_window (int):
            Optional. Size of the window over which the scores are calculated.
            If not given, 10 is used.
        smoothing_window (float or int):
            Optional. Size of the smoothing window, when float it is expressed as a proportion
            of the total length of y. If not given, 0.01 is used.
        smooth (bool):
            Optional. Indicates whether the returned errors should be smoothed.
            If not given, `True` is used.
        rec_error_type (str):
            Optional. Reconstruction error types ``["point", "area", "dtw"]``.
            If not given, "point" is used.
        mask (bool):
            Optional. Mask the start of anomaly scores.
            If not given, `True` is used.

    Returns:
        ndarray:
            Array of reconstruction errors.
    """
    if isinstance(smoothing_window, float):
        smoothing_window = min(math.trunc(len(y) * smoothing_window), 200)

    true = [item[0] for item in y.reshape((y.shape[0], -1))]
    for item in y[-1][1:]:
        true.extend(item)

    predictions = []
    predictions_vs = []

    pred_length = y_hat.shape[1]
    num_errors = y_hat.shape[1] + step_size * (y_hat.shape[0] - 1)

    for i in range(num_errors):
        intermediate = []
        for j in range(max(0, i - num_errors + pred_length), min(i + 1, pred_length)):
            intermediate.append(y_hat[i - j, j])
        if intermediate:
            predictions.append(np.median(np.asarray(intermediate)))

            predictions_vs.append([[
                np.min(np.asarray(intermediate)),
                np.percentile(np.asarray(intermediate), 25),
                np.percentile(np.asarray(intermediate), 50),
                np.percentile(np.asarray(intermediate), 75),
                np.max(np.asarray(intermediate))
            ]])

    true = np.asarray(true)
    predictions = np.asarray(predictions)
    predictions_vs = np.asarray(predictions_vs)

    # Compute reconstruction errors
    if rec_error_type.lower() == "point":
        errors = _point_wise_error(true, predictions)

    elif rec_error_type.lower() == "area":
        errors = _area_error(true, predictions, score_window)

    elif rec_error_type.lower() == "dtw":
        errors = _dtw_error(true, predictions, score_window)

    # Apply smoothing
    if smooth:
        errors = pd.Series(errors).rolling(
            smoothing_window, center=True, min_periods=smoothing_window // 2).mean().values

    if mask:
        mask_length = int(0.01 * len(errors))
        errors[:mask_length] = min(errors)

    return errors, predictions_vs
