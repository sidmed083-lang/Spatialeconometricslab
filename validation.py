"""Validation of spatial datasets: CRS, geometry quality, duplicates, missing values."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
import pandas as pd

ERROR, WARNING, INFO = "error", "warning", "info"


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str
    count: int = 0
    details: dict = field(default_factory=dict)


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(i.severity == ERROR for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == WARNING for i in self.issues)

    @property
    def codes(self) -> set[str]:
        return {i.code for i in self.issues}

    def get(self, code: str) -> ValidationIssue | None:
        return next((i for i in self.issues if i.code == code), None)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [(i.severity, i.code, i.message, i.count) for i in self.issues],
            columns=["severity", "code", "message", "count"],
        )


def crs_info(gdf: gpd.GeoDataFrame) -> dict:
    """Summarise the CRS of a GeoDataFrame (all values ``None`` when no CRS is set)."""
    crs = gdf.crs
    if crs is None:
        return {
            "name": None,
            "epsg": None,
            "is_geographic": None,
            "is_projected": None,
            "units": None,
        }
    axis = crs.axis_info[0].unit_name if crs.axis_info else None
    return {
        "name": crs.name,
        "epsg": crs.to_epsg(),
        "is_geographic": bool(crs.is_geographic),
        "is_projected": bool(crs.is_projected),
        "units": axis,
    }


def dataset_metadata(gdf: gpd.GeoDataFrame) -> dict:
    """Geographic metadata for display next to the attribute table."""
    geom_col = gdf.geometry.name
    non_geom = [c for c in gdf.columns if c != geom_col]
    bounds = gdf.total_bounds
    return {
        "n_rows": int(len(gdf)),
        "n_columns": len(non_geom),
        "geometry_types": sorted(gdf.geometry.geom_type.dropna().unique().tolist()),
        "crs": crs_info(gdf),
        "bounds": [float(b) for b in bounds] if np.isfinite(bounds).all() else None,
        "numeric_columns": [c for c in non_geom if pd.api.types.is_numeric_dtype(gdf[c])],
        "dtypes": {c: str(gdf[c].dtype) for c in non_geom},
    }


def validate_geodataframe(
    gdf: gpd.GeoDataFrame,
    *,
    id_col: str | None = None,
    analysis_columns: Sequence[str] | None = None,
) -> ValidationReport:
    """Run structural and quality checks.

    Errors block analysis; warnings should be reviewed (many can be fixed in
    :mod:`spatial_econometrics.data.preprocessing`).
    """
    report = ValidationReport()
    add = report.issues.append
    if len(gdf) == 0:
        add(ValidationIssue(ERROR, "empty_dataset", "The dataset has no rows."))
        return report

    geom_col = gdf.geometry.name
    geoms = gdf.geometry

    # CRS ------------------------------------------------------------------
    if gdf.crs is None:
        add(
            ValidationIssue(
                ERROR,
                "missing_crs",
                "No CRS is defined. Assign the correct CRS before computing distances or maps.",
            )
        )
    elif gdf.crs.is_geographic:
        add(
            ValidationIssue(
                WARNING,
                "geographic_crs",
                "The CRS is geographic (degrees). Reproject to a projected CRS before "
                "building distance-based weights.",
            )
        )

    # Geometry quality -------------------------------------------------------
    missing = geoms.isna() | geoms.is_empty
    if missing.any():
        add(
            ValidationIssue(
                ERROR,
                "missing_geometry",
                f"{int(missing.sum())} observation(s) have null or empty geometry.",
                int(missing.sum()),
            )
        )
    present = ~missing

    invalid = present & ~geoms.is_valid
    if invalid.any():
        add(
            ValidationIssue(
                WARNING,
                "invalid_geometry",
                f"{int(invalid.sum())} geometry(ies) are invalid (e.g. self-intersections).",
                int(invalid.sum()),
            )
        )

    types = sorted(geoms[present].geom_type.unique().tolist())
    if len(types) > 1:
        add(
            ValidationIssue(
                WARNING,
                "mixed_geometry_types",
                f"Mixed geometry types: {', '.join(types)}.",
                len(types),
            )
        )

    if gdf.crs is not None and gdf.crs.is_geographic and present.any():
        minx, miny, maxx, maxy = geoms[present].total_bounds
        if abs(minx) > 180 or abs(maxx) > 180 or abs(miny) > 90 or abs(maxy) > 90:
            add(
                ValidationIssue(
                    ERROR,
                    "coords_out_of_range",
                    "Coordinates fall outside valid longitude/latitude ranges; the CRS is "
                    "probably mislabelled.",
                )
            )

    # Duplicates -------------------------------------------------------------
    wkb = geoms.to_wkb()
    dup_geom = wkb.duplicated(keep="first") & present
    if dup_geom.any():
        add(
            ValidationIssue(
                WARNING,
                "duplicate_geometry",
                f"{int(dup_geom.sum())} observation(s) share a geometry with an earlier one. "
                "Coincident points create zero distances and unstable neighbour sets.",
                int(dup_geom.sum()),
            )
        )

    attrs = pd.DataFrame(gdf.drop(columns=geom_col))
    attrs["__wkb__"] = wkb.to_numpy()
    try:
        n_dup_rows = int(attrs.duplicated().sum())
    except TypeError:  # unhashable cell values
        n_dup_rows = 0
    if n_dup_rows:
        add(
            ValidationIssue(
                WARNING,
                "duplicate_rows",
                f"{n_dup_rows} fully duplicated observation(s) (same attributes and geometry).",
                n_dup_rows,
            )
        )

    if id_col is not None:
        if id_col not in gdf.columns:
            raise KeyError(f"id_col '{id_col}' not found in dataset.")
        n_dup_ids = int(gdf[id_col].duplicated().sum())
        if n_dup_ids:
            add(
                ValidationIssue(
                    ERROR,
                    "duplicate_ids",
                    f"{n_dup_ids} duplicate value(s) in identifier column '{id_col}'.",
                    n_dup_ids,
                )
            )

    # Missing attribute values -------------------------------------------------
    cols = list(analysis_columns) if analysis_columns else [c for c in gdf.columns if c != geom_col]
    unknown = [c for c in cols if c not in gdf.columns]
    if unknown:
        raise KeyError(f"Columns not found in dataset: {unknown}")
    na = gdf[cols].isna().sum()
    na = na[na > 0]
    if not na.empty:
        listing = ", ".join(f"{c} ({int(n)})" for c, n in na.items())
        add(
            ValidationIssue(
                WARNING,
                "missing_values",
                f"Missing values in: {listing}.",
                int(na.sum()),
                {str(c): int(n) for c, n in na.items()},
            )
        )

    return report
