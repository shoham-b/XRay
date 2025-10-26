import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def find_peaks_naive(df: pd.DataFrame) -> pd.DataFrame:
    """
    Finds peaks in the data by taking the maximum intensity.
    Assumes 'Angle' and 'Intensity' columns.
    """
    # This is a very naive approach, just returns the max intensity point
    peak_index = df["Intensity"].idxmax()
    return df.loc[[peak_index]]


def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-(((x - mean) / stddev) ** 2) / 2)


def find_peaks_fitting(df: pd.DataFrame, guess: list | None = None) -> tuple:
    """
    Finds peaks by fitting a Gaussian to the data.
    Assumes 'Angle' and 'Intensity' columns.
    Returns the parameters of the fitted Gaussian (amplitude, mean, stddev).
    """
    x_data = df["Angle"]
    y_data = df["Intensity"]

    if guess is None:
        # Make a reasonable guess for the parameters
        mean_guess = x_data[y_data.idxmax()]
        amplitude_guess = y_data.max()
        stddev_guess = (x_data.max() - x_data.min()) / 10
        guess = [amplitude_guess, mean_guess, stddev_guess]

    popt, pcov = curve_fit(gaussian, x_data, y_data, p0=guess)
    return popt
