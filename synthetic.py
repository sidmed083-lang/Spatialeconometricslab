"""Reproducible synthetic lattice datasets with known data-generating processes.

The simulators are intentionally independent of PySAL (pure NumPy), so that estimator
tests can compare PySAL results against parameters the data were generated from.
"""

from __future__ import annotations

from collections.abc import Sequence

import geopandas as gpd
import numpy as np
from shapely.geometry import box


def make_lattice(
    n_rows: int, n_cols: int, *, cell_size: float = 1000.0, crs: str = "EPSG:32631"
) -> gpd.GeoDataFrame:
    """Regular grid of square polygons. Row ``i`` of the frame is cell ``(r, c)`` with
    ``i = r * n_cols + c``; the default CRS is a projected one (UTM 31N, metres)."""
    if n_rows < 2 or n_cols < 2:
        raise ValueError("A lattice needs at least 2 rows and 2 columns.")
    rows, cols, polys = [], [], []
    for r in range(n_rows):
        for c in range(n_cols):
            rows.append(r)
            cols.append(c)
            polys.append(
                box(c * cell_size, r * cell_size, (c + 1) * cell_size, (r + 1) * cell_size)
            )
    return gpd.GeoDataFrame(
        {"cell_id": np.arange(len(polys)), "row": rows, "col": cols}, geometry=polys, crs=crs
    )


def lattice_weights_matrix(
    n_rows: int, n_cols: int, *, queen: bool = False, row_standardize: bool = True
) -> np.ndarray:
    """Dense contiguity matrix for a lattice built by :func:`make_lattice`."""
    n = n_rows * n_cols
    W = np.zeros((n, n))
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if queen:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for r in range(n_rows):
        for c in range(n_cols):
            for dr, dc in offsets:
                rr, cc = r + dr, c + dc
                if 0 <= rr < n_rows and 0 <= cc < n_cols:
                    W[r * n_cols + c, rr * n_cols + cc] = 1.0
    if row_standardize:
        W = W / W.sum(axis=1, keepdims=True)
    return W


def make_spatial_dataset(
    n_rows: int = 15,
    n_cols: int = 15,
    *,
    model: str = "sar",
    beta: Sequence[float] = (1.0, 2.0, -1.5),
    rho: float = 0.5,
    lam: float = 0.5,
    theta: Sequence[float] = (0.8, -0.4),
    sigma: float = 1.0,
    seed: int = 42,
    crs: str = "EPSG:32631",
) -> gpd.GeoDataFrame:
    """Simulate ``y`` on a rook-contiguity lattice with row-standardised weights ``W``.

    ``beta`` is ``(intercept, b_x1, b_x2)``. Models:

    * ``"ols"``: ``y = X b + e``
    * ``"sar"``: ``y = (I - rho W)^-1 (X b + e)``
    * ``"sem"``: ``y = X b + (I - lam W)^-1 e``
    * ``"sdm"``: ``y = (I - rho W)^-1 (X b + W Z theta + e)`` with ``Z = [x1, x2]``

    The result has columns ``y, x1, x2`` and ``true_error`` (the innovation ``e``); the
    true parameters are stored in ``gdf.attrs["true_params"]``.
    """
    if model not in {"ols", "sar", "sem", "sdm"}:
        raise ValueError(f"Unknown model '{model}'.")
    if len(beta) != 3 or len(theta) != 2:
        raise ValueError("`beta` needs 3 values (intercept, x1, x2) and `theta` needs 2.")
    for name, val in (("rho", rho), ("lam", lam)):
        if not -1.0 < val < 1.0:
            raise ValueError(f"`{name}` must lie in (-1, 1) for row-standardised weights.")

    gdf = make_lattice(n_rows, n_cols, crs=crs)
    n = len(gdf)
    W = lattice_weights_matrix(n_rows, n_cols)
    I = np.eye(n)  # noqa: E741

    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 2))
    eps = rng.normal(scale=sigma, size=n)
    X = np.column_stack([np.ones(n), Z])
    b = np.asarray(beta, dtype=float)

    if model == "ols":
        y = X @ b + eps
    elif model == "sar":
        y = np.linalg.solve(I - rho * W, X @ b + eps)
    elif model == "sem":
        y = X @ b + np.linalg.solve(I - lam * W, eps)
    else:
        y = np.linalg.solve(I - rho * W, X @ b + W @ Z @ np.asarray(theta, dtype=float) + eps)

    gdf["y"], gdf["x1"], gdf["x2"], gdf["true_error"] = y, Z[:, 0], Z[:, 1], eps
    gdf.attrs["true_params"] = {
        "model": model,
        "beta": tuple(float(v) for v in b),
        "rho": rho if model in {"sar", "sdm"} else None,
        "lambda": lam if model == "sem" else None,
        "theta": tuple(float(v) for v in theta) if model == "sdm" else None,
        "sigma": sigma,
        "seed": seed,
    }
    return gdf
