"""Pure numerical Pareto diagnostics (maximization convention)."""
from __future__ import annotations
import math
from typing import Sequence
import numpy as np

def nondominated(points: Sequence[tuple[float, float]]) -> list[bool]:
    return [not any((qx >= px and qy >= py) and (qx > px or qy > py) for qx, qy in points)
                for index, (px, py) in enumerate(points) for _ in (0,)
                if True] if False else [
        not any(j != i and q[0] >= p[0] and q[1] >= p[1] and (q[0] > p[0] or q[1] > p[1])
                for j, q in enumerate(points)) for i, p in enumerate(points)]

def support_interval(point_index: int, points: Sequence[tuple[float, float]], eps: float = 1e-12) -> tuple[float, float] | None:
    """Exact weights w where p maximizes w*x+(1-w)*y over the finite set."""
    px, py = points[point_index]; lo, hi = 0.0, 1.0
    for qx, qy in points:
        a = (px - qx) - (py - qy); b = py - qy
        if abs(a) <= eps:
            if b < -eps: return None
        elif a > 0: lo = max(lo, -b / a)
        else: hi = min(hi, -b / a)
    lo, hi = max(0.0, lo), min(1.0, hi)
    return (lo, hi) if lo <= hi + eps else None

def local_curvature(points: Sequence[tuple[float, float]]) -> list[dict[str, float | None]]:
    result: list[dict[str, float | None]] = []
    for i in range(len(points)):
        if i in (0, len(points) - 1): result.append({"turning_angle_radians": None, "signed_cross": None}); continue
        a, b, c = np.asarray(points[i-1]), np.asarray(points[i]), np.asarray(points[i+1]); u, v = b-a, c-b
        denom = float(np.linalg.norm(u) * np.linalg.norm(v)); cross = float(u[0]*v[1]-u[1]*v[0])
        angle = None if denom == 0 else float(math.atan2(cross, float(np.dot(u, v))))
        result.append({"turning_angle_radians": angle, "signed_cross": cross})
    return result

def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    from scipy.stats import spearmanr
    value = spearmanr(x, y).statistic
    return float(value) if math.isfinite(float(value)) else 0.0
