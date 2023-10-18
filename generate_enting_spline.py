from wafo_interpolate import SmoothSpline
import numpy as np


def enting(y_d, cut_off_period, x_d=None):
    """
    Generate a smoothing 'Enting' spline through a 1d array y_d with a given cut-off period.

    Parameters:
    -----------
    y_d : array-like
        The 1d array containing the data points to be smoothed using the 'Enting' spline.

    cut_off_period : float
        The cut-off period is a parameter that controls the level of smoothing, being the
        period at which 50% of the singal is attenuated. Smaller periods (i.e., higher
        frequencies) are filtered out. As such, a smaller cut-off period results in weaker
        smoothing, while a larger value results in stronger smoothing.

    x_d : array-like, optional
        The time points corresponding to the data in y_d. If provided, this array should
        have the same length as y_d. If not provided, it is assumed that the data points
        are evenly spaced in time, and x_d is generated accordingly. The cut_off_period
        needs to be provided in the same units as x_d.

    Returns:
    --------
    data_sp : array-like
        The 'Enting' spline curve that represents a smoothed version of the input data y_d.
        It provides a smooth fit to the data, with the level of smoothness controlled by
        the cut-off period.

    References:
    -----------
    - Enting, I. G. (1987). On the use of smoothing splines to filter CO2 data.
      Journal of Geophysical Research: Atmospheres, 92(D9), 10977-10984.

    - Carl de Boor. A Practical Guide to Splines. Springer New York, NY, 1978.

    - The code in wafo_interpolate is based on wafo 0.11 code for Python 2.6
      (https://pypi.org/project/wafo/), adapted to be compatible with Python 3,
      and tested against a Fortran implementation of the smoothing spline.
    """

    if x_d is not None:
        x_d = x_d.astype("float")
        # Mean data spacing
        dx = abs(max(x_d) - min(x_d)) / (len(x_d) - 1)
    else:
        x_d = np.arange(len(y_d)).astype("float")
        dx = 1
    # Calculate lambda
    la = (float(cut_off_period) / (2. * np.pi))**4 / dx
    # Calulate smoothing factor p
    p = 1. / (1 + la)
    # Spline through data
    pp = SmoothSpline(x_d, y_d, p)
    data_sp = pp(x_d)
    return data_sp
