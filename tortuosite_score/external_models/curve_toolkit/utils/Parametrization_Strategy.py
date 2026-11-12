import numpy as np

from abc import ABC, abstractmethod
from scipy.interpolate import interp1d, splprep, splev
from scipy.optimize import minimize_scalar

from tortuosite_score.external_models.curve_toolkit.utils import geometry


"""
Parametrization_Strategy.py

Implements the Strategy pattern for curve parametrization methods.
Each Parametrization subclass encapsulates one parametrization algorithm.
"""

class Parametrization(ABC):
    """
    Abstract base class for all curve parametrization strategies.

    Subclasses must implement `apply`, which takes a Curve instance and
    returns a parametrization of its points. The parametrization should
    return parameter values `t` and the coordinates `x`, `y` at those
    parameter values.

    The resulting parametrization can be used for resampling, smoothing,
    or other operations that require a parameterized curve.
    """

    @abstractmethod
    def apply(self, curve):
        """
        Applies the parametrization strategy to a curve.

        Parameters:
            curve : Curve
                The curve to parametrize.

        Returns:
            (t, x, y) : tuple of np.ndarray
                t : Parameter values at each point
                x, y : Coordinates corresponding to each parameter value
        """
        raise NotImplementedError
    


class UniformParametrization(Parametrization):
    """
    Uniform parametrization where t is equally spaced from 0 to 1.

    This is the simplest parametrization, using uniform sampling of the
    parameter space regardless of the geometric properties of the curve.
    Points are not evenly spaced in arc length, only in parameter space.

    Parameters:
        num_points : int, optional
            Number of points to use for parametrization. If None, uses
            the original number of points.
    """

    def __init__(self, num_points=None):
        self.num_points = num_points

    def apply(self, curve):
        n = self.num_points if self.num_points is not None else len(curve)
        t = np.linspace(0, 1, n)
        
        # Interpolate to get points at uniform parameter values
        interp_x = interp1d(np.linspace(0, 1, len(curve)), curve.x, kind='linear')
        interp_y = interp1d(np.linspace(0, 1, len(curve)), curve.y, kind='linear')
        
        return t, interp_x(t), interp_y(t)



class ArcLengthParametrization(Parametrization):
    """
    Arc-length parametrization where t is the cumulative arc length.

    This is the standard geometric parametrization where t represents
    the distance along the curve from the start. Points are evenly
    spaced in arc length.

    Parameters:
        num_points : int, optional
            Number of points to use for parametrization. If None, uses
            the original number of points.
        kind : str
            Interpolation type for resampling ('linear' or 'cubic').
    """

    def __init__(self, num_points=None, kind='linear'):
        self.num_points = num_points
        self.kind = kind

    def apply(self, curve):
        n = self.num_points if self.num_points is not None else len(curve)
        
        # Get cumulative arc length
        s = curve.cumulative_arclength()
        total_length = s[-1]
        
        # Create interpolators for x and y as functions of arc length
        interp_x = interp1d(s, curve.x, kind=self.kind)
        interp_y = interp1d(s, curve.y, kind=self.kind)
        
        # Generate evenly spaced arc length values
        t = np.linspace(0, total_length, n)
        
        return t, interp_x(t), interp_y(t)



class SCC_Parametrization(Parametrization):
    """
    Slope Chain parametrization where t is the cumulative arc length.

    This parametrization is based on the computation of the slope chain code of the curve 

    Parameters:
        num_points : int, optional
            Number of points to use for parametrization. If None, uses
            the original number of points.
        kind : str
            Interpolation type for resampling ('linear' or 'cubic').
    """

    def __init__(self, num_points=None):
        self.num_points = num_points
        
    def apply(self, curve):
        if self.num_points is None:
            self.num_points = len(curve)

        scc_x, scc_y, arclen = geometry.SCC_parametrization(curve.x, curve.y, n_points=self.num_points)
        
        return None, scc_x, scc_y



class ChordLengthParametrization(Parametrization):
    """
    Chord-length parametrization where t is the cumulative chord length.

    Similar to arc-length, but uses the chord length between consecutive
    points instead of the actual curve length. This is often used as an
    approximation to arc-length parametrization.

    Parameters:
        num_points : int, optional
            Number of points to use for parametrization. If None, uses
            the original number of points.
    """

    def __init__(self, num_points=None):
        self.num_points = num_points

    def apply(self, curve):
        n = self.num_points if self.num_points is not None else len(curve)
        
        # Compute cumulative chord lengths
        dx = np.diff(curve.x)
        dy = np.diff(curve.y)
        chord_lengths = np.sqrt(dx**2 + dy**2)
        cumulative_chords = np.concatenate(([0], np.cumsum(chord_lengths)))
        total_chord = cumulative_chords[-1]
        
        # Interpolate using chord length as parameter
        interp_x = interp1d(cumulative_chords, curve.x, kind='linear')
        interp_y = interp1d(cumulative_chords, curve.y, kind='linear')
        
        t = np.linspace(0, total_chord, n)
        
        return t, interp_x(t), interp_y(t)



class CentripetalParametrization(Parametrization):
    """
    Centripetal parametrization often used in NURBS and spline fitting.

    This parametrization uses the square root of the chord length, which
    often gives better results than uniform or chord-length parametrization
    for curves with varying curvature.

    Parameters:
        num_points : int, optional
            Number of points to use for parametrization. If None, uses
            the original number of points.
        power : float
            The power to raise the chord length to. Default is 0.5
            (centripetal). Common values are 0.5 (centripetal) and 1.0
            (chord-length).
    """

    def __init__(self, num_points=None, power=0.5):
        self.num_points = num_points
        self.power = power

    def apply(self, curve):
        n = self.num_points if self.num_points is not None else len(curve)
        
        # Compute chord lengths and raise to power
        dx = np.diff(curve.x)
        dy = np.diff(curve.y)
        chord_lengths = np.sqrt(dx**2 + dy**2)
        powered_lengths = chord_lengths ** self.power
        cumulative_powered = np.concatenate(([0], np.cumsum(powered_lengths)))
        total_powered = cumulative_powered[-1]
        
        # Interpolate using powered chord length as parameter
        interp_x = interp1d(cumulative_powered, curve.x, kind='linear')
        interp_y = interp1d(cumulative_powered, curve.y, kind='linear')
        
        t = np.linspace(0, total_powered, n)
        
        return t, interp_x(t), interp_y(t)



class NaturalParametrization(Parametrization):
    """
    Natural parametrization using B-spline fitting.

    This fits a B-spline to the curve and then parametrizes it naturally
    using the spline's internal parametrization. This can give smoother
    parametrizations than simple interpolation methods.

    Parameters:
        num_points : int, optional
            Number of points to use for parametrization. If None, uses
            the original number of points.
        s : float, optional
            Smoothing factor for the B-spline. Larger values give
            smoother fits.
        k : int
            Degree of the B-spline (3 for cubic).
    """

    def __init__(self, num_points=None, s=0.0, k=3):
        self.num_points = num_points
        self.s = s
        self.k = k

    def apply(self, curve):
        n = self.num_points if self.num_points is not None else len(curve)
        
        # Fit B-spline to the curve points
        points = np.column_stack((curve.x, curve.y))
        tck, u = splprep(points.T, s=self.s, k=self.k)
        
        # Evaluate the spline at equally spaced parameter values
        t = np.linspace(0, 1, n)
        x_smooth, y_smooth = splev(t, tck)
        
        # Estimate arc length parameter using the spline
        # (use the B-spline's internal parameterization)
        return t, x_smooth, y_smooth



class AdaptiveParametrization(Parametrization):
    """
    Adaptive parametrization that allocates more points to regions
    of high curvature.

    This parametrization attempts to place more points where the curve
    has high curvature, and fewer points where the curve is straight,
    preserving detail while keeping the total number of points fixed.

    Parameters:
        num_points : int
            Number of points to use for parametrization.
        curvature_weight : float
            Weight for curvature vs. arc length. Higher values allocate
            more points to high-curvature regions.
    """

    def __init__(self, num_points, curvature_weight=1.0):
        self.num_points = num_points
        self.curvature_weight = curvature_weight

    def apply(self, curve):
        n = self.num_points
        
        # First get arc-length parametrization
        s = curve.cumulative_arclength()
        total_length = s[-1]
        
        # Estimate curvature at each point
        # Using the angle between consecutive segments
        dx = np.gradient(curve.x)
        dy = np.gradient(curve.y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        
        # Curvature: (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2)
        curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-10)**(1.5)
        
        # Normalize curvature
        curvature = curvature / (np.sum(curvature) + 1e-10)
        
        # Create parameter density function (weighted combination of arc length and curvature)
        arc_density = np.ones_like(curve.x) / len(curve.x)
        density = (arc_density + self.curvature_weight * curvature) / (1 + self.curvature_weight)
        
        # Compute cumulative density for adaptive sampling
        cumulative_density = np.concatenate(([0], np.cumsum(density[:-1])))
        cumulative_density = cumulative_density / cumulative_density[-1]
        
        # Interpolate to get points at adaptive parameter values
        interp_x = interp1d(s, curve.x, kind='linear')
        interp_y = interp1d(s, curve.y, kind='linear')
        
        # Sample more densely where curvature is high
        t_values = np.interp(np.linspace(0, 1, n), cumulative_density, s)
        
        return t_values, interp_x(t_values), interp_y(t_values)



class BiharmonicParametrization(Parametrization):
    """
    Biharmonic parametrization for generating smooth parameterization.

    This uses a biharmonic equation to generate a smooth parametrization
    that preserves the shape while ensuring smooth parameter values.

    Parameters:
        num_points : int, optional
            Number of points to use for parametrization. If None, uses
            the original number of points.
        iterations : int
            Number of smoothing iterations.
    """

    def __init__(self, num_points=None, iterations=10):
        self.num_points = num_points
        self.iterations = iterations

    def apply(self, curve):
        n = self.num_points if self.num_points is not None else len(curve)
        
        # Start with arc-length parametrization
        s = curve.cumulative_arclength()
        total_length = s[-1]
        
        # Initial parameter values
        t = np.linspace(0, total_length, len(curve))
        
        # Smooth the parameter values using biharmonic smoothing
        # This is a simple implementation using finite differences
        for _ in range(self.iterations):
            t_new = t.copy()
            for i in range(1, len(t) - 1):
                # Biharmonic smoothing: t_i = (-t_{i-2} + 4*t_{i-1} + 4*t_{i+1} - t_{i+2}) / 6
                if i == 1:
                    t_new[i] = (2*t[0] + 4*t[1] + 4*t[2] - t[3]) / 9
                elif i == len(t) - 2:
                    t_new[i] = (-t[-4] + 4*t[-3] + 4*t[-2] + 2*t[-1]) / 9
                else:
                    t_new[i] = (-t[i-2] + 4*t[i-1] + 4*t[i+1] - t[i+2]) / 6
            t = t_new
        
        # Interpolate to get desired number of points
        interp_x = interp1d(t, curve.x, kind='linear')
        interp_y = interp1d(t, curve.y, kind='linear')
        
        t_param = np.linspace(t.min(), t.max(), n)
        
        return t_param, interp_x(t_param), interp_y(t_param)


# ----------------------------------------------------------------------
# Registry: maps a string method name -> Parametrization class
# ----------------------------------------------------------------------

PARAMETRIZATION_REGISTRY = {
    "uniform": UniformParametrization,
    "arclength": ArcLengthParametrization,
    "chord": ChordLengthParametrization,
    "centripetal": CentripetalParametrization,
    "natural": NaturalParametrization,
    "adaptive": AdaptiveParametrization,
    "biharmonic": BiharmonicParametrization,
    "scc": SCC_Parametrization,
}


def get_parametrization(method, **kwargs):
    """
    Factory function: builds a Parametrization instance from a registry key.

    Parameters:
        method : str
            One of the keys in PARAMETRIZATION_REGISTRY
            (e.g. 'arclength', 'uniform', 'centripetal').
        **kwargs :
            Passed to the parametrization's constructor.

    Returns:
        Parametrization instance

    Examples:
        param = get_parametrization("arclength", num_points=200)
        param = get_parametrization("adaptive", num_points=150, curvature_weight=2.0)
    """
    if method not in PARAMETRIZATION_REGISTRY:
        available = ", ".join(PARAMETRIZATION_REGISTRY.keys())
        raise ValueError(f"Unknown parametrization method '{method}'. Available: {available}")
    return PARAMETRIZATION_REGISTRY[method](**kwargs)