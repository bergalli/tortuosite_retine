
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from tortuosite_score.external_models.curve_toolkit.utils import geometry
from tortuosite_score.external_models.curve_toolkit.utils.Smoothers import Smoother, get_smoother
from tortuosite_score.external_models.curve_toolkit.utils.Parametrization_Strategy import Parametrization, get_parametrization
from tortuosite_score.external_models.curve_toolkit.Tortuosity import TortuosityMeasure, get_tortuosity_measure




class Curve:
    """
    Represents an open 2D curve defined by a sequence of (x, y) points.
 
    Attributes:
        x (np.ndarray): x-coordinates of the curve points
        y (np.ndarray): y-coordinates of the curve points
    """

    def __init__(self, x, y):
        """
        Initializes a Curve instance.

        Parameters:
            x (np.ndarray): x-coordinates of the curve points
            y (np.ndarray): y-coordinates of the curve points
        """
        x = np.asarray(x, dtype=float).flatten()
        y = np.asarray(y, dtype=float).flatten()
 
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same length")
        if x.shape[0] < 2:
            raise ValueError("A curve requires at least 2 points")
 
        self.x = x
        self.y = y

 
    def __len__(self):
        return self.x.shape[0]


    def __repr__(self):
        return f"Curve(n_points={len(self)}, length={self.arclength():.4f})"

    @property
    def points(self):
        """Returns an (N, 2) array of [x, y] coordinate pairs."""
        return np.column_stack((self.x, self.y))
    

    def update(self, x, y):
        """
        Update X and Y coordinates

        Parameters:
            x (np.ndarray): x-coordinates of the curve points
            y (np.ndarray): y-coordinates of the curve points
        """
        x = np.asarray(x, dtype=float).flatten()
        y = np.asarray(y, dtype=float).flatten()
 
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same length")
        if x.shape[0] < 2:
            raise ValueError("A curve requires at least 2 points")
 
        self.x = x
        self.y = y


    # ------------------------------------------------------------------
    # Arc length
    # ------------------------------------------------------------------
 
    def segment_lengths(self):
        """Returns the Euclidean distance between each pair of consecutive points."""
        return geometry.segment_lengths(self.x, self.y)
 

    def cumulative_arclength(self):
        """
        Returns the cumulative arc length at each point along the curve,
        starting at 0 for the first point.
        """
        return geometry.cumulative_arclength(self.x, self.y)
 


    def arclength(self):
        """Returns the total arc length of the curve."""
        return geometry.arclength(self.x, self.y)
    
    
    
    def resample(self, num_points, kind="linear"):
        """
        Resamples the curve to a given number of points, evenly spaced
        in arc length, without smoothing (interpolation only).
 
        Parameters:
            num_points : int
                Number of points in the resampled curve.
            kind : str
                Interpolation type passed to scipy.interpolate.interp1d
                (e.g. 'linear', 'cubic').
 
        Returns:
            Curve
                A new resampled Curve instance.
        """
        t = self.cumulative_arclength()
 
        interp_x = interp1d(t, self.x, kind=kind)
        interp_y = interp1d(t, self.y, kind=kind)
 
        t2 = np.linspace(t.min(), t.max(), num_points)
        return Curve(interp_x(t2), interp_y(t2))
    

    # ------------------------------------------------------------------
    # Smoothing 
    # ------------------------------------------------------------------

    def smooth(self, method="spline", num_points=None, **kwargs):
        """
        Smooths the curve using the given strategy and returns a new Curve.
 
        This method dispatches to a `Smoother` implementation (see
        smoothers.py), following the Strategy pattern: new smoothing
        algorithms can be added there without touching this method.
 
        Parameters:
            method : str or Smoother instance
                Either a registry key ('spline', 'moving_average',
                'savgol', 'gaussian') or a custom Smoother instance.
            num_points : int, optional
                Number of points in the output curve. Ignored by
                smoothers that must preserve the input sampling (e.g.
                moving_average).
            **kwargs :
                Passed to the smoother's constructor when `method` is a
                string (e.g. smooth=0.05, k=3 for 'spline').
 
        Returns:
            Curve
                A new, smoothed Curve instance.
 
        Examples:
            curve.smooth("spline", smooth=0.05, num_points=200)
            curve.smooth("savgol", window=9, poly_order=3)
            curve.smooth(MyCustomSmoother())
        """
        if isinstance(method, Smoother):
            smoother = method
        else:
            smoother = get_smoother(method, **kwargs)
 
        t = np.linspace(0, self.arclength(), len(self))
        x_smooth, y_smooth = smoother.apply(t, self.x, self.y, num_points=num_points)
 
        return Curve(x_smooth, y_smooth)
    

    # ------------------------------------------------------------------
    # Parametrization
    # ------------------------------------------------------------------

    def parametrize(self, method="arclength", **kwargs):
        """
        Parametrize the curve using the given strategy and return the parameter values and coordinates.

        This method dispatches to a `Parametrization` implementation, following the 
        Strategy pattern (see Parametrization_Strategy.py): new parametrization 
        algorithms can be added there without touching this method.

        Parameters:
            method : str or Parametrization instance Either a registry key 
            **kwargs :
                Passed to the parametrization's constructor when `method`
                required extended parameters (e.g. num_points=200, kind='cubic').

        Returns:
            Curve
                A new, re-parametrized Curve instance.

        Examples:
            t, x, y = curve.parametrize("arclength")
            t, x, y = curve.parametrize("uniform", num_points=100)
            t, x, y = curve.parametrize("adaptive", num_points=150, curvature_weight=2.0)
        """
        
        
        if isinstance(method, Parametrization):
            param = method
        else:
            param = get_parametrization(method, **kwargs)
        
        t, x, y = param.apply(self)
        return Curve(x, y)


    # ------------------------------------------------------------------
    # Tortuosity
    # ------------------------------------------------------------------
 
    def tortuosity(self, method="arc_chord", **kwargs):
        """
        Computes a tortuosity measure for this curve.
 
        This method dispatches to a `TortuosityMeasure` implementation.
 
        Parameters:
            method : str or TortuosityMeasure instance
                Either a registry key ('arc_chord', 'soam',
                'mean_curvature', 'icm') or a custom TortuosityMeasure
                instance.
            **kwargs :
                Passed to the measure's constructor when `method` is a
                string.
 
        Returns:
            float
                The computed tortuosity value. Interpretation depends
                on the chosen measure — see tortuosity.py docstrings.
 
        Examples:
            curve.tortuosity()                    # arc-chord ratio (default)
            curve.tortuosity("scc")
        """
        if isinstance(method, TortuosityMeasure):
            measure = method
        else:
            measure = get_tortuosity_measure(method, **kwargs)
 
        return measure.compute(self.x, self.y)
    


    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    
    def plot(self, *other_curves, ax=None, show_points=False, labels=None, **plot_kwargs):
        """
        Plots this curve using matplotlib. Additional Curve instances
        can be passed as positional arguments to plot them together on
        the same axes.
 
        Parameters:
            *other_curves : Curve
                Zero or more additional Curve instances to plot
                alongside this one, e.g. curve1.plot(curve2, curve3).
            ax : matplotlib Axes, optional
                Axes to plot on. If None, a new figure/axes is created.
            show_points : bool
                If True, marks each individual point on every curve.
            labels : list of str, optional
                Legend labels, one per curve (this curve first, then
                other_curves in order). If given, ax.legend() is called
                automatically.
            **plot_kwargs :
                Additional keyword arguments passed to ax.plot() for
                THIS curve only (e.g. color, linewidth, linestyle).
                Other curves use matplotlib's default color cycling so
                they remain visually distinct.
 
        Returns:
            matplotlib Axes
                The axes the curve(s) were plotted on.
 
        Examples:
            curve1.plot(curve2)
            curve1.plot(curve2, curve3, show_points=True)
            curve1.plot(curve2, labels=["original", "smoothed"])
        """
 
        if ax is None:
            _, ax = plt.subplots()
 
        marker = "." if show_points else None
 
        all_curves = (self,) + other_curves
        for i, curve in enumerate(all_curves):
            label = labels[i] if labels is not None else None
            if i == 0:
                ax.plot(curve.x, curve.y, marker=marker, alpha=0.5, label=label, **plot_kwargs)
            else:
                # Other curves use default styling/color cycling so they
                # don't collide with THIS curve's explicit plot_kwargs.
                ax.plot(curve.x, curve.y, marker=marker, alpha=0.5, label=label)
 
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)

        if labels is not None:
            ax.legend()
 
        
        plt.show()
        return ax