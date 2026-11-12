"""Image-space annotations for the saved Local-bump diagnostic geometry.

No scoring occurs here. Coordinates are inverse-scaled before drawing over the
source image; crops never rotate, stretch, or straighten an anatomical curve.
"""

from __future__ import annotations

import math

import numpy as np
from matplotlib.patches import Arc, Rectangle
from PIL import Image

INK = "#17202a"
MUTED = "#68727d"
VEIN = "#2468b4"
POSITIVE = "#b33b36"
NEGATIVE = "#215e91"


def number(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def angle_vertex_indices(diagnostics) -> np.ndarray:
    """Match local_angle_change's removal of zero-length directed segments.

    Ordinarily angle i is at smoothed vertex i+1. If smoothing produces a
    duplicate vertex, retain the scoring service's exact valid-segment pairing.
    """
    valid = np.flatnonzero(
        np.linalg.norm(np.diff(diagnostics.smoothed, axis=0), axis=1) > 1e-6
    )
    return valid[1:]


def angle_triplet(diagnostics, index: int) -> np.ndarray:
    valid = np.flatnonzero(
        np.linalg.norm(np.diff(diagnostics.smoothed, axis=0), axis=1) > 1e-6
    )
    previous, following = valid[index : index + 2]
    return diagnostics.smoothed[[previous, following, following + 1]]


def image_points(diagnostics, scale: float) -> np.ndarray:
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Facteur de normalisation invalide")
    return diagnostics.smoothed / scale


def crop_box(points, image_size, margin_fraction=0.18, min_margin=12):
    points = np.asarray(points)
    margin = max(min_margin, margin_fraction * max(np.ptp(points, axis=0)))
    low = np.maximum(np.floor(points.min(axis=0) - margin), 0)
    high = np.minimum(np.ceil(points.max(axis=0) + margin), image_size)
    return (*low.astype(int), *high.astype(int))


def backdrop(axis, source, box, fade=0.30):
    crop = source.crop(tuple(box)).convert("RGB")
    if fade:
        crop = Image.blend(crop, Image.new("RGB", crop.size, "white"), fade)
    axis.imshow(crop, extent=(box[0], box[2], box[3], box[1]))
    axis.set_xlim(box[0], box[2])
    axis.set_ylim(box[3], box[1])
    axis.set_aspect("equal")
    axis.set_axis_off()


def line(axis, points, color=VEIN, width=2, **kwargs):
    axis.plot(*np.asarray(points).T, color="white", linewidth=width + 2, **kwargs)
    axis.plot(*np.asarray(points).T, color=color, linewidth=width, **kwargs)


def context(axis, source, raw):
    box = crop_box(raw, source.size)
    backdrop(axis, source, box, fade=0)
    line(axis, raw)
    for label, p in zip(("Début", "Fin"), raw[[0, -1]]):
        axis.scatter(*p, s=30, facecolor="white", edgecolor=VEIN, zorder=5)
        axis.annotate(
            label,
            p,
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9),
            zorder=6,
        )
    j = max(1, len(raw) // 2)
    axis.annotate(
        "",
        raw[min(j + 3, len(raw) - 1)],
        raw[max(0, j - 3)],
        arrowprops=dict(arrowstyle="->", color=INK, lw=2),
        zorder=7,
    )
    return box


def retained(axis, source, diagnostics, scale, *, label_examples=False):
    points = image_points(diagnostics, scale)
    backdrop(axis, source, crop_box(points, source.size), fade=0.60)
    line(axis, points, color=INK, width=1.1)
    vertices = points[angle_vertex_indices(diagnostics)]
    values = diagnostics.filtered_angle_change
    for mask, marker, color, size in (
        (values == 0, "o", MUTED, 9),
        (values > 0, "+", POSITIVE, 45),
        (values < 0, "_", NEGATIVE, 45),
    ):
        axis.scatter(
            *vertices[mask].T,
            s=size,
            marker=marker,
            color=color,
            linewidths=1.7,
            zorder=5,
        )
    if label_examples:
        for label, index, offset in zip(
            ("a", "b"), example_indices(diagnostics), ((0.10, 0.88), (0.90, 0.12))
        ):
            if index is not None:
                axis.annotate(
                    label,
                    vertices[index],
                    xytext=offset,
                    textcoords="axes fraction",
                    fontsize=11,
                    fontweight="bold",
                    ha="center",
                    zorder=8,
                    bbox=dict(fc="white", ec=INK, boxstyle="circle,pad=.15"),
                    arrowprops=dict(arrowstyle="-", color=INK, lw=0.8),
                )
    return vertices


def transition_groups(diagnostics, group_size=6):
    """Bound label density; every transition appears in exactly one crop."""
    indices = diagnostics.sign_change_indices
    return [
        indices[i : i + group_size] for i in range(0, len(indices), group_size)
    ] or [indices]


def transitions(axis, source, diagnostics, scale, indices, first_number=1):
    points = image_points(diagnostics, scale)
    vertex_indices = angle_vertex_indices(diagnostics)
    if len(indices):
        first = max(0, vertex_indices[indices[0]] - 3)
        last = min(len(points), vertex_indices[indices[-1]] + 4)
        selected = points[first:last]
    else:
        selected = points
    box = crop_box(selected, source.size, 0.30, 18)
    backdrop(axis, source, box, fade=0.55)
    line(axis, points, width=1.5)
    positions = points[vertex_indices[indices]]
    allocated = axis.get_position(original=True)
    actual = axis.get_position()
    # Equally spaced labels in two external rails prevent text collisions.
    for k, (index, p) in enumerate(zip(indices, positions)):
        label = first_number + k
        x = (k // 2 + 0.5) / math.ceil(len(indices) / 2)
        y = 1.04 if k % 2 == 0 else -0.04
        axis.scatter(*p, s=28, facecolor="white", edgecolor=INK, zorder=5)
        current = "+" if diagnostics.filtered_angle_change[index] > 0 else "−"
        previous = "−" if current == "+" else "+"
        axis.annotate(
            f"{label}\n{previous}→{current}",
            p,
            xytext=(allocated.x0 + x * allocated.width, actual.y0 + y * actual.height),
            textcoords="figure fraction",
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=.18", fc="white", ec=INK, lw=0.8),
            arrowprops=dict(arrowstyle="-", color=INK, lw=0.7),
            annotation_clip=False,
        )
    if not len(indices):
        axis.text(
            0.5,
            0.06,
            "Aucun changement de signe : N = 0",
            transform=axis.transAxes,
            ha="center",
            fontsize=10,
            bbox=dict(fc="white", ec="none"),
        )


def example_indices(diagnostics):
    values = diagnostics.filtered_angle_change
    kept = np.flatnonzero(values)
    rejected = np.flatnonzero(values == 0)
    # Use a visible retained angle away from the endpoints where possible.
    midpoint = len(values) / 2
    select = lambda candidates: (
        int(candidates[np.argmin(abs(candidates - midpoint))])
        if len(candidates)
        else None
    )
    interior = kept[(kept >= 2) & (kept < len(values) - 2)]
    candidates = interior if len(interior) else kept
    chosen = (
        int(candidates[np.argmax(abs(values[candidates]))]) if len(candidates) else None
    )
    return chosen, select(rejected)


def preparation(axis, source, diagnostics, scale, index, settings):
    points = diagnostics.resampled / scale
    center = (
        len(points) // 2
        if index is None
        else int(angle_vertex_indices(diagnostics)[index])
    )
    lo, hi = max(0, center - 3), min(len(points), center + 4)
    box = crop_box(points[lo:hi], source.size, 0.30, 5)
    backdrop(axis, source, box, fade=0.60)
    raw = diagnostics.geometry / scale
    axis.plot(*raw.T, color=MUTED, lw=1.5, ls="--")
    line(axis, image_points(diagnostics, scale), width=1.5)
    axis.scatter(*points[lo:hi].T, s=24, facecolor="white", edgecolor=INK, zorder=5)
    if hi - lo > 1:
        pair = points[center : min(center + 2, hi)]
        if len(pair) == 2:
            axis.annotate(
                f"{number(settings.resample_step,0)} px*",
                pair.mean(axis=0),
                xytext=(28, 10),
                textcoords="offset points",
                ha="center",
                fontsize=10,
                bbox=dict(fc="white", ec="none"),
                arrowprops=dict(arrowstyle="-", lw=0.7),
            )


def angle_example(
    axis, source, diagnostics, scale, index, settings, retained_example=True
):
    if index is None:
        axis.set_axis_off()
        axis.text(
            0.5,
            0.5,
            "Aucun angle retenu" if retained_example else "Aucun angle annulé",
            ha="center",
            transform=axis.transAxes,
            fontsize=10,
        )
        return
    points = angle_triplet(diagnostics, index) / scale
    p, q, r = points
    incoming = q - p
    extension = q + incoming / np.linalg.norm(incoming) * np.linalg.norm(r - q)
    box = crop_box(np.vstack([points, extension]), source.size, 0.25, 1)
    backdrop(axis, source, box, fade=0.72)
    axis.plot(*np.vstack([q, extension]).T, "--", color=MUTED, lw=1.2)
    for a, b in ((p, q), (q, r)):
        axis.annotate(
            "",
            b,
            a,
            arrowprops=dict(arrowstyle="->", color=INK, lw=1.8, shrinkA=0, shrinkB=0),
        )
    axis.scatter(*points.T, s=22, facecolor="white", edgecolor=INK, zorder=5)
    for label, point in zip(("P₀", "P₁", "P₂"), points):
        axis.annotate(
            label,
            point,
            xytext=(-16, 0),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )
    theta = np.degrees(np.arctan2(incoming[1], incoming[0]))
    delta = float(diagnostics.angle_change[index])
    filtered = diagnostics.filtered_angle_change[index]
    angle_color = POSITIVE if filtered > 0 else NEGATIVE if filtered < 0 else MUTED
    end = theta + np.degrees(delta)
    radius = min(np.linalg.norm(incoming), np.linalg.norm(r - q)) * 0.65
    axis.add_patch(
        Arc(
            q,
            2 * radius,
            2 * radius,
            theta1=min(theta, end),
            theta2=max(theta, end),
            color=angle_color,
            lw=2,
        )
    )
    axis.annotate(
        "Δθ",
        q
        + radius
        * np.array(
            [
                np.cos(np.radians((theta + end) / 2)),
                np.sin(np.radians((theta + end) / 2)),
            ]
        ),
        xytext=(0.5, 0.95),
        textcoords="axes fraction",
        fontsize=11,
        color=angle_color,
        ha="center",
        bbox=dict(fc="white", ec="none", alpha=0.85),
        arrowprops=dict(arrowstyle="-", color=angle_color, lw=0.7),
    )
    rule = "≥" if filtered else "<"
    axis.text(
        0.5,
        -0.17,
        f"Δθ = {number(delta,6)} rad\n|Δθ| {rule} {number(settings.curvature_threshold)} → δ = {number(filtered,6)}",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=9.5,
    )


def retina_locator(axis, source, box):
    axis.imshow(source)
    axis.add_patch(
        Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            fill=False,
            ec="white",
            lw=2,
        )
    )
    axis.set_axis_off()
