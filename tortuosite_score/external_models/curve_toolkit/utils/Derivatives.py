"""
derivatives.py

Implements the Strategy pattern for computing derivatives of a curve's
x(t), y(t) coordinates with respect to arc length t.

Each DerivativeMethod subclass encapsulates one numerical approach

Usage:
    from curve import Curve
    from derivatives import CurveDerivatives

    curve = Curve(x, y)
    deriv = CurveDerivatives(curve)

    t, dxdt, dydt = deriv.compute(method="finite_difference", order=1)
    t, d2xdt2, d2ydt2 = deriv.compute(method="csaps", order=2, smooth=0.9)
    curvature = deriv.curvature(method="spline", smooth=0.05)
"""

from abc import ABC, abstractmethod

import numpy as np
from csaps import csaps
from scipy.interpolate import UnivariateSpline

from tortuosite_score.external_models.curve_toolkit.utils import geometry



class DerivativeMethod(ABC):
    """
    Abstract base class for all derivative computation strategies.

    Subclasses must implement `compute`, which takes a parametrization
    `t` and a value array `values` (e.g. x(t) or y(t)), and returns the
    derivative of the requested order evaluated at `t_eval`.
    """

    @abstractmethod
    def compute(self, t, values, order=1, t_eval=None):
        """
        Computes the derivative of `values` with respect to `t`.

        Parameters:
            t : np.ndarray
                Parametrization (e.g. arc length) of the input samples.
            values : np.ndarray
                Sampled values to differentiate (e.g. x or y coordinates).
            order : int
                Order of the derivative (1 = first derivative,
                2 = second derivative, etc.).
            t_eval : np.ndarray, optional
                Points at which to evaluate the derivative. If None,
                defaults to `t` itself.

        Returns:
            np.ndarray
                The derivative values evaluated at `t_eval`.
        """
        raise NotImplementedError



class FiniteDifferenceDerivative(DerivativeMethod):
    """
    Computes derivatives using finite differences (numpy.gradient),
    applied repeatedly for higher orders.

    Finite differences are only defined at the original sample points,
    so `t_eval` (if given) must match `t` exactly — this method cannot
    resample onto a different grid. 
    """

    def compute(self, t, values, order=1, t_eval=None):
        if t_eval is not None and (len(t_eval) != len(t) or not np.allclose(t_eval, t)):
            raise ValueError(
                "FiniteDifferenceDerivative can only evaluate at the original "
                "sample points t (finite differences are undefined off-grid). "
                "Use method='spline' or method='csaps' to evaluate at arbitrary points."
            )

        result = values
        for _ in range(order):
            result = np.gradient(result, t, edge_order=2)
        return result



class SplineDerivative(DerivativeMethod):
    """
    Computes derivatives by fitting a scipy UnivariateSpline to
    values(t), then analytically differentiating the fitted spline.

    Parameters:
        smooth : float
            Smoothing factor passed to UnivariateSpline's `s` parameter.
            0 = exact interpolation through every point.
        k : int
            Degree of the spline (must satisfy k > order for the
            requested derivative order; default k=3, cubic).
    """

    def __init__(self, smooth=0.0, k=3):
        self.smooth = smooth
        self.k = k

    def compute(self, t, values, order=1, t_eval=None):
        if order >= self.k:
            raise ValueError(
                f"Spline degree k={self.k} must be greater than the requested "
                f"derivative order={order}. Increase k when constructing "
                f"SplineDerivative to support higher-order derivatives."
            )

        spline = UnivariateSpline(t, values, s=self.smooth, k=self.k)
        deriv_spline = spline.derivative(n=order)

        eval_points = t_eval if t_eval is not None else t
        return deriv_spline(eval_points)



class CsapsDerivative(DerivativeMethod):
    """
    Computes derivatives by fitting a csaps cubic smoothing spline to
    values(t), then evaluating its analytical derivative.

    Parameters:
        smooth : float
            Smoothing parameter in [0, 1], same convention as MATLAB's
            csaps: 0 = least-squares line fit, 1 = exact interpolation.

    Requires the `csaps` package: pip install csaps
    """

    def __init__(self, smooth=0.9):
        self.smooth = smooth

    def compute(self, t, values, order=1, t_eval=None):

        if order > 2:
            raise ValueError(
                "CsapsDerivative supports order=1 or order=2 "
                "(cubic smoothing splines have a well-defined 2nd derivative "
                "and a discontinuous/zero 3rd+ derivative)."
            )

        spline = csaps(t, values, smooth=self.smooth)

        eval_points = t_eval if t_eval is not None else t
        return spline(eval_points, nu=order)


# ----------------------------------------------------------------------
# Registry: maps a string method name -> DerivativeMethod class
# ----------------------------------------------------------------------

DERIVATIVE_REGISTRY = {
    "finite_difference": FiniteDifferenceDerivative,
    "spline": SplineDerivative,
    "csaps": CsapsDerivative,
}


def get_derivative_method(method, **kwargs):
    """
    Factory function: builds a DerivativeMethod instance from a
    registry key.

    Parameters:
        method : str
            One of the keys in DERIVATIVE_REGISTRY
            ('finite_difference', 'spline', 'csaps').
        **kwargs :
            Passed to the method's constructor.

    Returns:
        DerivativeMethod instance
    """
    if method not in DERIVATIVE_REGISTRY:
        available = ", ".join(DERIVATIVE_REGISTRY.keys())
        raise ValueError(
            f"Unknown derivative method '{method}'. Available: {available}"
        )
    return DERIVATIVE_REGISTRY[method](**kwargs)


# ----------------------------------------------------------------------
# CurveDerivatives: operates on a Curve instance
# ----------------------------------------------------------------------

class CurveDerivatives:
    """
    Computes derivatives of a Curve's x(t), y(t) coordinates with
    respect to arc length t, using pluggable DerivativeMethod strategies.

    Parameters:
        curve : Curve
            The curve to differentiate. The curve's own arc-length
            parametrization (curve.arclength()) is used as t.

    Examples:
        deriv = CurveDerivatives(curve)
        t, dxdt, dydt = deriv.compute("finite_difference", order=1)
        t, d2xdt2, d2ydt2 = deriv.compute("csaps", order=2, smooth=0.9)
        curvature = deriv.curvature("spline", smooth=0.05)
    """

    def __init__(self, curve, t=None):
        self.curve = curve
        if t is not None:
            self.t = t
        else:        
            self.t = np.linspace(0, curve.arclength(), len(curve))
        

    def compute(self, method="finite_difference", order=1, num_points=None, **kwargs):
        """
        Computes dx/dt and dy/dt (or higher-order derivatives) of the
        curve's coordinates.

        Parameters:
            method : str or DerivativeMethod instance
                Either a registry key ('finite_difference', 'spline',
                'csaps') or a custom DerivativeMethod instance.
            order : int
                Derivative order (1 = first derivative, 2 = second, ...).
            num_points : int, optional
                Number of points at which to evaluate the derivative.
                Ignored (must be None) when method='finite_difference',
                since finite differences are only defined on the
                original sample grid.
            **kwargs :
                Passed to the method's constructor when `method` is a
                string (e.g. smooth=0.05, k=3 for 'spline').

        Returns:
            (t_eval, dxdt, dydt) : tuple of np.ndarray
                The evaluation points and the corresponding derivatives
                of x and y with respect to t.
        """
        if isinstance(method, DerivativeMethod):
            strategy = method
        else:
            strategy = get_derivative_method(method, **kwargs)

        t_eval = None
        if num_points is not None:
            t_eval = np.linspace(self.t.min(), self.t.max(), num_points)
        

        dxdt = strategy.compute(self.t, self.curve.x, order=order, t_eval=t_eval)
        dydt = strategy.compute(self.t, self.curve.y, order=order, t_eval=t_eval)

        return (t_eval if t_eval is not None else self.t), dxdt, dydt


    def first_order(self, method="finite_difference", num_points=None, **kwargs):
        """Convenience wrapper: compute(order=1, ...) — the tangent vector components."""
        return self.compute(method=method, order=1, num_points=num_points, **kwargs)


    def second_order(self, method="finite_difference", num_points=None, **kwargs):
        """Convenience wrapper: compute(order=2, ...) — the curvature vector components."""
        return self.compute(method=method, order=2, num_points=num_points, **kwargs)


    def curvature(self, method="finite_difference", num_points=None, **kwargs):
        """
        Computes the signed curvature of the curve:

            kappa(t) = (x' * y'' - y' * x'') / (x'^2 + y'^2)^(3/2)

        using the first and second derivatives from the chosen method.

        Parameters:
            method : str or DerivativeMethod instance
                Passed through to compute() for both derivative orders.
            num_points : int, optional
                Number of points at which to evaluate curvature.
            **kwargs :
                Passed to the method's constructor.

        Returns:
            (t_eval, kappa) : tuple of np.ndarray
        """
        t_eval, dxdt, dydt = self.compute(method=method, order=1, num_points=num_points, **kwargs)
        _, d2xdt2, d2ydt2 = self.compute(method=method, order=2, num_points=num_points, **kwargs)

        denom = (dxdt**2 + dydt**2) ** 1.5
        denom[np.isclose(denom, 0)] = np.finfo(float).eps

        kappa = (dxdt * d2ydt2 - dydt * d2xdt2) / denom
        kappa[np.isclose(kappa, 0, atol=1e-12), ] = 0.0
        return t_eval, kappa
