"""
tortuosity.py

Implements the Strategy pattern for curve tortuosity measurements.

Each TortuosityMeasure subclass encapsulates one way of quantifying tortuosity. 
New measures can be added by subclassing `TortuosityMeasure` without modifying any existing code.

Usage:
    measure = ArcChordTortuosity()
    value = measure.compute(x, y)

    or via the registry / Curve.tortuosity(method="arc_chord")
"""

import numpy as np
from abc import ABC, abstractmethod

from tortuosite_score.external_models.curve_toolkit.utils import geometry
from tortuosite_score.external_models.curve_toolkit.utils.Derivatives import CurveDerivatives


class _ArrayCurve:
    """
    Minimal duck-typed stand-in for a Curve, exposing just enough
    (.x, .y, .arclength(), len()) for CurveDerivatives to operate on
    raw x, y arrays without constructing a full Curve instance.
    """
 
    def __init__(self, x, y):
        self.x = x
        self.y = y
 
    def arclength(self):
        return geometry.arclength(self.x, self.y)

    def __len__(self):
        return len(self.x)
    


class TortuosityMeasure(ABC):
    """
    Abstract base class for all tortuosity measurement strategies.

    Subclasses must implement `compute`, which takes the curve's x, y
    coordinate arrays and returns a single scalar tortuosity value.
    """

    @abstractmethod
    def compute(self, x, y):
        """
        Computes a tortuosity value for the given curve.

        Parameters:
            x, y : np.ndarray
                Coordinates of the curve.

        Returns:
            float
                The computed tortuosity value. Interpretation depends
                on the specific measure (see each subclass docstring).
        """
        raise NotImplementedError
    



class Distance_Metric_Tortuosity(TortuosityMeasure):
    """
    Classic arc-chord ratio (a.k.a. Distance Metric, DM):

        tortuosity = arc_length / chord_length

    where chord_length is the straight-line distance between the
    curve's start and end points.

    Returns 1.0 for a perfectly straight curve; values increase as the
    curve becomes more convoluted. Undefined (returns np.inf) if start
    and end points coincide.
    """

    def compute(self, x, y):
        arc_length = geometry.arclength(x, y)
        chord = geometry.chord_length(x, y)
 
        if np.isclose(chord, 0):
            return np.inf
 
        return arc_length / chord
    

    

class MeanCurvatureTortuosity(TortuosityMeasure):
    """
    Mean absolute curvature along the curve, 
 
        curvature(s) = |kappa(s)|
        tortuosity   = mean(curvature(s))
 
    Higher values indicate a curve that bends more sharply on average.
 
    Parameters:
        smooth : float
            Smoothing parameter in [0, 1], passed to the underlying
            csaps fit. 0 = least-squares line fit, 1 = exact.
            interpolation. Default 0.99 lightly regularizes to reduce
            noise sensitivity in the 2nd derivative while staying close
            to the original curve. 
    """
 
    def __init__(self, smooth=0.99):
        self.smooth = smooth
 
    def compute(self, x, y):
        n = len(x)
        if n < 3:
            return 0.0
 
        deriv = CurveDerivatives(_ArrayCurve(x, y))
        _, kappa = deriv.curvature(method="csaps", smooth=self.smooth)
 
        return float(np.mean(np.abs(kappa)))
    



class TotalCurvatureTortuosity(TortuosityMeasure):
    """
    Total Curvature (Smedby et al., 1993): the integral of absolute curvature along 
    the curve's arc length, also known as Tau_2 by (William E. Hart et al., 1998):
    
        TC = integral of |kappa(s)| ds
 
    Parameters:
        smooth : float
            Smoothing parameter in [0, 1], passed to the underlying
            csaps fit. 0 = least-squares line fit, 1 = exact.
            interpolation. Default 0.99 lightly regularizes to reduce
            noise sensitivity in the 2nd derivative while staying close
            to the original curve. 
    """
 
    def __init__(self, smooth=0.99):
        self.smooth = smooth

 
    def compute(self, x, y):
        n = len(x)
        if n < 3:
            return 0.0

        curve_parametrization = geometry.cumulative_arclength(x, y)
        deriv = CurveDerivatives(_ArrayCurve(x, y), t=curve_parametrization)
        t, kappa = deriv.curvature(method="csaps", smooth=self.smooth)
        total_curvature = np.trapezoid(np.abs(kappa), t)
 
        return float(total_curvature)



class Tau_3_Tortuosity(TortuosityMeasure):
    """
    Total Squared Curvature (Tau_3) (William E. Hart et al., 1998):
    
        Tau_3 = TSC = integral of kappa(s)^2 ds
 
    Parameters:
        smooth : float
            Smoothing parameter in [0, 1], passed to the underlying
            csaps fit. 0 = least-squares line fit, 1 = exact.
            interpolation. Default 0.99 lightly regularizes to reduce
            noise sensitivity in the 2nd derivative while staying close
            to the original curve. 
    """
 
    def __init__(self, smooth=0.99):
        self.smooth = smooth

 
    def compute(self, x, y):
        n = len(x)
        if n < 3:
            return 0.0
 
        curve_parametrization = geometry.cumulative_arclength(x, y)
        deriv = CurveDerivatives(_ArrayCurve(x, y), t=curve_parametrization)
        t, kappa = deriv.curvature(method="csaps", smooth=self.smooth)
        Tau_3 = np.trapezoid(kappa ** 2, t)
 
        return float(Tau_3)



class Tau_5_Tortuosity(TortuosityMeasure):
    """
    Tau_5 Tortuosity (William E. Hart et al., 1998):
    
        Tau_5 = Tau_3 / arc_length
 
    Parameters:
        smooth : float
            Smoothing parameter in [0, 1], passed to the underlying
            csaps fit. 0 = least-squares line fit, 1 = exact.
            interpolation. Default 0.99 lightly regularizes to reduce
            noise sensitivity in the 2nd derivative while staying close
            to the original curve. 
    """
 
    def __init__(self, smooth=0.99):
        self.smooth = smooth

 
    def compute(self, x, y):
        n = len(x)
        if n < 3:
            return 0.0
 
        L_c = geometry.arclength(x, y)
        Tau_3 = Tau_3_Tortuosity(smooth=self.smooth).compute(x, y)
        Tau_5 = Tau_3 / L_c

        return float(Tau_5)




class AverageSquaredDerivativeCurvatureTortuosity(TortuosityMeasure):
    """
    Average Squared-Derivative-Curvature (ASDC), based on Patasius et
    al. (used in retinal vessel tortuosity literature):
 
        ASDC = (1 / L) * integral( (d(kappa)/dt)^2 ) ds


    Parameters:
        smooth : float
            Smoothing parameter in [0, 1], passed to the underlying
            csaps fit for kappa(s). 0 = least-squares line fit, 1 =
            exact interpolation. Default 0.99. Requires:
            pip install csaps.

    """

    def __init__(self, smooth=0.99):
        self.smooth = smooth


    def compute(self, x, y):
        n = len(x)
        if n < 3:
            return 0.0
 
        curve_parametrization = geometry.cumulative_arclength(x, y)
        deriv = CurveDerivatives(_ArrayCurve(x, y), t=curve_parametrization)
        t, kappa = deriv.curvature(method="csaps", smooth=self.smooth)
 
        dkappa_dt = np.gradient(kappa)
        integral = np.trapezoid(dkappa_dt**2)

        L_x = geometry.chord_length(x, y)
        if np.isclose(L_x, 0):
            return 0.0
 
        return integral / L_x




class InflectionCountTortuosity(TortuosityMeasure):
    """
    Inflection Count Metric (ICM), based on Bullitt et al. (2003):
 
        ICM = (arc_length / chord_length) * (n_inflections + 1)
 
    where n_inflections is the number of sign changes in curvature
    (i.e. the number of times the curve switches from bending one way
    to bending the other).
 
    Parameters:
        smooth : float
            Smoothing parameter in [0, 1], passed to the underlying
            csaps fit. 0 = least-squares line fit, 1 = exact
            interpolation. Default 0.99 lightly regularizes to avoid
            spurious inflections from point-to-point noise. 
        threshold : float
            only count an inflection if the "excess a curvature" of the segment     
    """
 
    def __init__(self, smooth=0.99, threshold=1e-4):
        self.smooth = smooth
        self.threshold = threshold  # minimum excess length to count an inflection
 
    def compute(self, x, y):
        n = len(x)
        if n < 3:
            return 0.0
 
        curve_parametrization = geometry.cumulative_arclength(x, y)
        deriv = CurveDerivatives(_ArrayCurve(x, y), t=curve_parametrization)
        _, kappa = deriv.curvature(method="csaps", smooth=self.smooth)
 
        signs = np.sign(kappa)
        
        i_start = 0
        n_inflections = 0        
        sign_deriv = signs[0]
        for i, sign_i in enumerate(signs[1:], start=1):
            if sign_i == 0:
                continue
            
            if sign_deriv == 0:
                sign_deriv = sign_i

            if sign_deriv != sign_i:
                sign_deriv = sign_i
                i_end = i

                if i_end - i_start < 2:
                    i_start = i
                    continue                

                DMs_i = Distance_Metric_Tortuosity().compute(x[i_start:i_end + 1], y[i_start:i_end + 1]) - 1
    
                if DMs_i > self.threshold:
                    n_inflections += 1

                i_start = i - 1
        
        DM = Distance_Metric_Tortuosity().compute(x, y)
        return DM * (n_inflections + 1)




class SOAMTortuosity(TortuosityMeasure):
    """
    Sum of Angles Metric (SOAM), based on Bullitt et al. (2003).

    Computes the discrete turning angle (in degrees) at each interior point and
    accumulates, normalized by arc length:

        SOAM = sum(|angle|) / arc_length 

    Higher values indicate more frequent/sharper directional changes
    per unit length. Sensitive to high-frequency noise, so consider
    smoothing the curve first (see smoothers.py / Curve.smooth()).
    """

    def compute(self, x, y):
        n = len(x)
        if n < 3:
            return 0.0

        dx = np.diff(x)
        dy = np.diff(y)
                
        Theta = np.degrees(np.arctan2(dy, dx))
        alpha = np.diff(Theta)

        arc_length = sum(np.sqrt(dx**2 + dy**2))
        if np.isclose(arc_length, 0):
            return 0.0

        return np.sum(np.abs(alpha)) / arc_length
    



class TortuosityDensityTortuosity(TortuosityMeasure):
    """
    Tortuosity Density (TD), based on Enrico Grisan et al. 2008
 
    It is based on partitioning each vessel in segments of constant-sign curvature
    and then combining together each evaluation of such segments and their number.
 
        TD = ((n - 1) / n) * (1 / Lc) * sum_{i=1}^{n} ((Lc_i / Lx_i) - 1)
 
    where Lc_i and Lx_i are the arc length and chord length of the
    i-th segment (n segments total, split at n-1 inflection points),
    and Lc is the arclen of the entire curve.

    Parameters:
    smooth : float
        Smoothing parameter in [0, 1], passed to the underlying
        csaps fit used to locate inflection points. 0 =
        least-squares line fit, 1 = exact interpolation. 
    """


    def __init__(self, smooth=0.99, threshold=1e-4):
        self.smooth = smooth
        self.threshold = threshold

    def compute(self, x, y):
        n = len(x)
        if n < 3:
            return 0.0
 
        curve_parametrization = geometry.cumulative_arclength(x, y)
        deriv = CurveDerivatives(_ArrayCurve(x, y), t=curve_parametrization)
        _, kappa = deriv.curvature(method="csaps", smooth=self.smooth)
 
        signs = np.sign(kappa)
        L = []

        i_start = 0
        n_ic = 0        
        sign_deriv = signs[0]
        for i, sign_i in enumerate(signs[1:], start=1):
            if sign_i == 0:
                continue
            
            if sign_deriv == 0:
                sign_deriv = sign_i

            if sign_deriv != sign_i:
                sign_deriv = sign_i
                i_end = i

                if i_end - i_start < 2:
                    i_start = i
                    continue                

                DMs_i = Distance_Metric_Tortuosity().compute(x[i_start:i_end + 1], y[i_start:i_end + 1]) - 1
    
                if DMs_i > self.threshold:
                    L.append(DMs_i)
                    n_ic += 1                    
                    i_start = i
        
        if i_start < len(x) -1:
           DMs_i = Distance_Metric_Tortuosity().compute(x[i_start:], y[i_start:]) - 1
           L.append(DMs_i)
           n_ic += 1           
        
        L_c = geometry.arclength(x, y)
        TD = ((n_ic - 1) / n_ic) * (1 / L_c) * np.sum(L)
        return TD





class SCC_Tortuosity(TortuosityMeasure):
    """
    
    """

    def scc(self, x, y):
        diff_x = np.diff(x)
        diff_y = np.diff(y)
                
        Theta = np.degrees(np.arctan2(diff_y, diff_x))
        alpha = np.diff(Theta)

        C = np.where((alpha >= 180) | (alpha <= -180))[0]
        for k in C: 
            alpha[k] = np.mod(alpha[k], np.sign(alpha[k]) * -360)
        
        alpha = alpha/180
        return alpha

    def compute(self, X, Y):
        alpha = self.scc(X, Y)
        SCC = np.sum(np.absolute(alpha))        
 
        return SCC
    


class ESCC_Tortuosity(TortuosityMeasure):
    """
    
    """

    def escc(self, X, Y):
        n = len(X)
        Xe = []
        Ye = []
    
        L_x = np.mean(geometry.segment_lengths(X, Y))
    
        vx = X[1] - X[0]
        vy = Y[1] - Y[0]
        M = np.degrees(np.arctan2(vy, vx))
    
        x_0 = X[0]
        y_0 = Y[0]
            
        x_1 = y_1 = None
        Xl = Yl = None
        Xp = Yp = None
    
        idx = 0
        while (idx >= 0) and (idx < n - 1):
            # heart shaped (box) pattern construction 
            Xe.append(x_0)
            Ye.append(y_0)

            alpha = np.radians(M)

            theta = np.linspace(alpha, alpha + np.pi, 180)
            Xp = x_0 + (L_x * (1 - ((theta - alpha) / np.pi))) * np.cos(theta)
            Yp = y_0 + (L_x * (1 - ((theta - alpha) / np.pi))) * np.sin(theta)

            theta = np.linspace(alpha, alpha - np.pi, 180)
            Xn = x_0 + (L_x * (1 + ((theta - alpha) / np.pi))) * np.cos(theta)
            Yn = y_0 + (L_x * (1 + ((theta - alpha) / np.pi))) * np.sin(theta)

            Xt = np.concatenate([Xp, Xn[:-1][::-1]])
            Yt = np.concatenate([Yp, Yn[:-1][::-1]])

            intersection = False
            n_idx = 1
            m_idx = 0

            # find intersection between the heart-shaped pattern and the original curve
            while not intersection:
                if idx < 0 or idx >= n - 1:
                    break

                p1 = (X[idx], Y[idx])
                p2 = (X[idx + 1], Y[idx + 1])
                
                xi, yi = geometry.segment_polyline_intersections(p1, p2, Xt, Yt)
 
                if len(xi) > 0:
                    if len(xi) > 1:
                        dists = np.sqrt(
                            (x_0 - np.array(xi)) ** 2 + (y_0 - np.array(yi)) ** 2
                        )
                        j = int(np.argmax(dists))
                        x_1 = xi[j]
                        y_1 = yi[j]
                    else:
                        x_1 = xi[0]
                        y_1 = yi[0]
                    intersection = True
                else:
                    if n_idx != m_idx:
                        if (idx + n_idx) < n - 1:
                            m_idx += 1
                            idx = idx + m_idx
                            n_idx += 1
                            m_idx = n_idx
                        else:
                            d1 = np.hypot(X[idx] - X[idx + 1], Y[idx] - Y[idx + 1])
                            d2 = np.hypot(Xp[0] - Xp[-1], Yp[0] - Yp[-1])
                            if d1 < d2:
                                break
                            else:
                                idx = idx - n_idx
                    else:
                        idx = idx - n_idx
                        n_idx += 1
 
            vx = x_1 - x_0
            vy = y_1 - y_0
            M = np.degrees(np.arctan2(vy, vx))
 
            idx += 1
 
            x_0 = x_1
            y_0 = y_1
 
        Xe.append(X[-1])
        Ye.append(Y[-1])

    
        return np.array(Xe), np.array(Ye)        


    def scc(self, X, Y):
        diff_x = np.diff(X)
        diff_y = np.diff(Y)

        Theta = np.degrees(np.arctan2(diff_y, diff_x))
        alpha = np.diff(Theta)

        C = np.where((alpha >= 180) | (alpha <= -180))[0]
        for k in C: 
            alpha[k] = np.mod(alpha[k], np.sign(alpha[k]) * -360)
        
        alpha = alpha/180
        return alpha



    def compute(self, X, Y):
        [Xe,Ye] = self.escc(X,Y);
        alpha = self.scc(Xe,Ye);
        ESCC = np.sum(np.absolute(alpha))        
 
        return ESCC
    
    


# ----------------------------------------------------------------------
# Registry: maps a string method name -> TortuosityMeasure class
# ----------------------------------------------------------------------

TORTUOSITY_REGISTRY = {
    "DM": Distance_Metric_Tortuosity,
    "mean_curvature": MeanCurvatureTortuosity,
    "total_curvature": TotalCurvatureTortuosity,
    "tau_3": Tau_3_Tortuosity,
    "tau_5": Tau_5_Tortuosity,
    "ASDC": AverageSquaredDerivativeCurvatureTortuosity,
    "ICM": InflectionCountTortuosity,
    "SOAM": SOAMTortuosity,    
    "TD": TortuosityDensityTortuosity,
    "SCC": SCC_Tortuosity,
    "ESCC": ESCC_Tortuosity,
}


def get_tortuosity_measure(method, **kwargs):
    """
    Factory function: builds a TortuosityMeasure instance from a
    registry key.

    Parameters:
        method : str
            One of the keys in TORTUOSITY_REGISTRY
            (e.g. 'arc_chord', 'soam', 'mean_curvature', 'icm').
        **kwargs :
            Passed to the measure's constructor.

    Returns:
        TortuosityMeasure instance
    """
    if method not in TORTUOSITY_REGISTRY:
        available = ", ".join(TORTUOSITY_REGISTRY.keys())
        raise ValueError(
            f"Unknown tortuosity method '{method}'. Available: {available}"
        )
    return TORTUOSITY_REGISTRY[method](**kwargs)
