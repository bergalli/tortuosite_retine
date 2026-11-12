"""
smoothers.py

Implements the Strategy pattern for curve smoothing algorithms.

Each Smoother subclass encapsulates one smoothing algorithm and its
parameters. New algorithms can be added by subclassing `Smoother`
without modifying any existing code.

Usage:
    smoother = SplineSmoother(smooth=0.05)
    x_new, y_new = smoother.apply(t, x, y, num_points=200)

    or via the registry / Curve.smooth(method="spline", smooth=0.05)
"""

from abc import ABC, abstractmethod
from csaps import csaps

import numpy as np
from tortuosite_score.external_models.curve_toolkit.utils import geometry
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d


class Smoother(ABC):
    """
    Abstract base class for all curve smoothing strategies.

    Subclasses must implement `apply`, which takes the arc-length
    parametrization `t` and the coordinate arrays `x`, `y`, and returns
    the smoothed (and possibly resampled) coordinate arrays.
    """

    @abstractmethod
    def apply(self, t, x, y, num_points=None):
        """
        Applies the smoothing algorithm.

        Parameters:
            t : np.ndarray
                Arc-length parametrization of the input points.
            x, y : np.ndarray
                Coordinates of the input curve.
            num_points : int, optional
                Number of points in the output. If None, defaults to
                len(x). Some smoothers (e.g. moving average) may ignore
                this and always return the same number of points as input.

        Returns:
            (x_smooth, y_smooth) : tuple of np.ndarray
        """
        raise NotImplementedError
    


class SplineSmoother(Smoother):
    """
    Smooths via a cubic smoothing spline fit independently to x(t) and
    y(t), where t is arc length. Equivalent in spirit to MATLAB's csaps.

    Parameters:
        smooth : float
            Smoothing factor passed to UnivariateSpline's `s` parameter.
            0 = exact interpolation; larger values = smoother fit.
        k : int
            Degree of the smoothing spline (default cubic, k=3).
    """

    def __init__(self, smooth=0.0, k=3):
        self.smooth = smooth
        self.k = k

    def apply(self, t, x, y, num_points=None):
        n_out = num_points if num_points is not None else len(x)
        t2 = np.linspace(t.min(), t.max(), n_out)

        spline_x = UnivariateSpline(t, x, s=self.smooth, k=self.k)
        spline_y = UnivariateSpline(t, y, s=self.smooth, k=self.k)

        return spline_x(t2), spline_y(t2)



class CubicSplineSmoother(Smoother):
    """
    Smooths via a cubic spline interpolation fit independently to x(t) and
    y(t), where t is arc length. This is an exact interpolation (no smoothing).

    Parameters:
        None
    """

    def __init__(self, smooth=0.0):
        self.smooth = smooth

    def apply(self, t, x, y, num_points=None):
        n_out = num_points if num_points is not None else len(x)
        t2 = np.linspace(t.min(), t.max(), n_out)

        Bsx = csaps(t, x, smooth=self.smooth)
        Bsy = csaps(t, y, smooth=self.smooth)
        
        t2 = np.linspace(t.min(), t.max(), num_points)
                         
        xfit = Bsx(t2)
        yfit = Bsy(t2)

        return xfit, yfit
        


class Savitzky_Golay_Smoother(Smoother):
    """
    Smooths using a Savitzky-Golay filter, which fits successive local
    polynomials. Good at preserving peak shapes while reducing noise.

    Parameters:
        window : int
            Length of the filter window (must be odd and >= poly_order + 2).
        poly_order : int
            Order of the polynomial used to fit each window.
    """

    def __init__(self, window=7, poly_order=3):
        self.window = window
        self.poly_order = poly_order

    def apply(self, t, x, y, num_points=None):
        x_smooth = savgol_filter(x, self.window, self.poly_order)
        y_smooth = savgol_filter(y, self.window, self.poly_order)
        return x_smooth, y_smooth



class GaussianSmoother(Smoother):
    """
    Smooths by convolving x and y independently with a Gaussian kernel.

    Parameters:
        sigma : float
            Standard deviation of the Gaussian kernel, in samples.
    """

    def __init__(self, sigma=2.0):
        self.sigma = sigma

    def apply(self, t, x, y, num_points=None):
        x_smooth = gaussian_filter1d(x, sigma=self.sigma)
        y_smooth = gaussian_filter1d(y, sigma=self.sigma)
        return x_smooth, y_smooth



class ScaleCompensatedSmoother(Smoother):
    DEFAULTS = {2.0: 1.5e-3, 2.5: 1.2e-4, 3.0: 1.07e-5}

    def __init__(self, gamma=2.5, c=None, eval_frac=0.25):
        self.gamma = gamma
        self.c = c
        self.eval_frac = eval_frac
    
    def strip_duplicates(self, x, y):
        t = geometry.cumulative_arclength(x, y)
        keep = np.concatenate([[True], np.diff(t) > 1e-10])
        return x[keep], y[keep], t[keep]

    def apply(self, t, x, y, num_points=200):
        if self.c is None:
            self.c = self.DEFAULTS.get(self.gamma, 1.2e-4)

        x, y, t = self.strip_duplicates(x, y)
        L = t[-1]
        lam_eff = self.c * (L ** self.gamma)
        p = 1.0 / (1.0 + lam_eff)          # csaps parameter, p->1 interpolates

        t_eval = np.linspace(t.min(), t.max(), num_points)
        x_smooth = csaps(t, x, t_eval, smooth=p)
        y_smooth = csaps(t, y, t_eval, smooth=p)
        return x_smooth, y_smooth


    
# ----------------------------------------------------------------------
# Registry: maps a string method name -> Smoother class
# ----------------------------------------------------------------------

SMOOTHER_REGISTRY = {
    "spline": SplineSmoother,
    "savgol": Savitzky_Golay_Smoother,
    "gaussian": GaussianSmoother,
    "cubic_spline": CubicSplineSmoother, 
    "scale_cubic_spline": ScaleCompensatedSmoother,
}


def get_smoother(method, **kwargs):
    """
    Factory function: builds a Smoother instance from a registry key.

    Parameters:
        method : str
            One of the keys in SMOOTHER_REGISTRY (e.g. 'spline', 'savgol').
        **kwargs :
            Passed to the smoother's constructor.

    Returns:
        Smoother instance
    """
    if method not in SMOOTHER_REGISTRY:
        available = ", ".join(SMOOTHER_REGISTRY.keys())
        raise ValueError(f"Unknown smoothing method '{method}'. Available: {available}")
    return SMOOTHER_REGISTRY[method](**kwargs)
