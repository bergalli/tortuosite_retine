import numpy as np


def segment_lengths(x, y):
    """Returns the Euclidean distance between each pair of consecutive points."""
    dx = np.diff(x)
    dy = np.diff(y)
    return np.sqrt(dx**2 + dy**2)



def cumulative_arclength(x, y):
    """
    Returns the cumulative arc length at each point along the curve,
    starting at 0 for the first point.
    """
    lengths = segment_lengths(x, y)
    return np.concatenate(([0.0], np.cumsum(lengths)))



def arclength(x, y):
    """Returns the total arc length of the curve defined by x, y."""
    return cumulative_arclength(x, y)[-1]
 
 

def chord_length(x, y):
    """Returns the straight-line distance between the first and last points."""
    return np.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2)



def segment_intersection(p1, p2, p3, p4):
    """
    Computes the intersection point of segment p1-p2 with segment
    p3-p4, if one exists within both segments' bounds.
 
    Returns:
        (x, y) tuple, or None if the segments don't intersect (or are
        parallel).
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
 
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if np.isclose(denom, 0):
        return None  # parallel or degenerate
 
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / denom
 
    if 0 <= t <= 1 and 0 <= u <= 1:
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        return (ix, iy)
    return None



def segment_polyline_intersections(line_p1, line_p2, poly_x, poly_y):
    """
    Computes the intersection point of a single 2-point segment (line_p1, line_p2) 
    against a multi-point polyline (poly_x, poly_y).
 
    Returns:
        (xi, yi) : tuple of lists
            Coordinates of every intersection point found, one entry
            per intersecting polyline segment.
    """
    xi, yi = [], []
    
    paired = list(zip(poly_x, poly_y))    
    for p3, p4 in zip(paired[:-1], paired[1:]):
        pt = segment_intersection(line_p1, line_p2, p3, p4)
        if pt is not None:
            xi.append(pt[0])
            yi.append(pt[1])
    return xi, yi



def circle_line_intersection(cx, cy, r, p1, p2):
    """
    Computes the intersection points between a circle and a line segment.

    Parameters:
        cx, cy : float
            Coordinates of the circle's center.
        r : float
            Radius of the circle.
        p1, p2 : tuple (x, y)
            Start and end points of the line segment.

    Returns:
        list of tuples (x, y)
            The intersection point(s), if any:
              - []                     : no intersection
              - [(x, y)]               : tangent (one intersection point)
              - [(x1, y1), (x2, y2)]   : two intersection points
    """
    x1, y1 = p1
    x2, y2 = p2

    dx = x2 - x1
    dy = y2 - y1
    
    fx = x1 - cx
    fy = y1 - cy

    # Quadratic equation coefficients: a*t^2 + b*t + c = 0
    # where P(t) = p1 + t*(p2 - p1), t in [0, 1]
    a = dx**2 + dy**2
    b = 2 * (fx * dx + fy * dy)
    c = fx**2 + fy**2 - r**2

    # Degenerate case: p1 == p2 (zero-length segment)
    if np.isclose(a, 0):
        return []

    discriminant = b**2 - 4 * a * c

    # No real roots -> no intersection
    if discriminant < 0:
        return []

    sqrt_disc = np.sqrt(discriminant)

    t1 = (-b - sqrt_disc) / (2 * a)
    t2_param = (-b + sqrt_disc) / (2 * a)

    intersections = []

    # Keep only solutions where t is within the segment bounds [0, 1]
    for t in sorted({t1, t2_param}):
        if 0 <= t <= 1:
            ix = x1 + t * dx
            iy = y1 + t * dy
            intersections.append((ix, iy))

    return intersections


def SCC_parametrization(X, Y, n_points):
    """
    Resamples a curve into n_points points approximately equally
    spaced along arc length.

    At each step, a circle of radius `step` (= total_length / (n_points-1))
    is centered at the current resampled point and intersected with the
    original polyline to find the next resampled point — effectively
    "walking a compass" of fixed radius along the curve.

    Parameters:
        X, Y : array-like
            Coordinates of the original (densely sampled) curve.
        n_points : int
            Desired number of points in the resampled output curve.

    Returns:
        Xp, Yp : np.ndarray
            Resampled coordinates, length n_points.
        t_length : float
            Total arc length of the resampled polyline.
    """
    Xp = np.zeros(n_points)
    Yp = np.zeros(n_points)

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n = len(X)
    L_c = arclength(X, Y)

    # Target step length between resampled points
    n_points -= 1
    step = L_c / (n_points)

    cx, cy = X[0], Y[0]
    idx = 0     # index into original polyline (0-based)
    o_idx = 0   # index into output arrays (0-based)

    while o_idx < n_points - 1:
        Xp[o_idx] = cx
        Yp[o_idx] = cy

        i = idx
        while i < n - 1:
            P1 = (X[i], Y[i])
            P2 = (X[i + 1], Y[i + 1])

            intersections = circle_line_intersection(cx, cy, step, P1, P2)

            if intersections:
                if len(intersections) > 1:
                    cx, cy = intersections[1] if any(np.isclose(np.array(intersections[0]), np.array([Xp[i],Yp[i]]), rtol=1e-10)) else intersections[1]
                else:
                    cx, cy = intersections[0]
                i += 1
                break
            else:
                dist_to_next = np.sqrt((X[i + 1] - cx) ** 2 + (Y[i + 1] - cy) ** 2)
                if step > dist_to_next:
                    i += 1
                else:
                    i -= 1

        idx = i
        o_idx += 1

    Xp[o_idx] = cx
    Yp[o_idx] = cy        

    Xp[-1] = X[-1]
    Yp[-1] = Y[-1]

    t_length = arclength(Xp, Yp)

    return Xp, Yp, t_length