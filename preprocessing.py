"""Cleaning and transformation of spatial datasets.

Every function returns a new object and never mutates its input. Functions that
remove or alter observations also return a count so the UI can log what changed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union
from shapely.validation import make_valid


class PreprocessingError(ValueError):
    """Raised when a preprocessing step cannot be applied."""


@dataclass
class PreprocessResult:
    gdf: gpd.GeoDataFrame
    log: list[str] = field(default_factory=list)


def assign_crs(gdf: gpd.GeoDataFrame, crs: str | int) -> gpd.GeoDataFrame:
    """Declare the CRS of a dataset that has none (does not move coordinates)."""
    if gdf.crs is not None:
        raise PreprocessingError(
            f"Dataset already has a CRS ({gdf.crs.name}); use `reproject` to transform it."
        )
    return gdf.set_crs(crs)


def reproject(gdf: gpd.GeoDataFrame, crs: str | int) -> gpd.GeoDataFrame:
    """Transform coordinates to ``crs``. Requires an existing CRS."""
    if gdf.crs is None:
        raise PreprocessingError("Cannot reproject: dataset has no CRS. Use `assign_crs` first.")
    try:
        return gdf.to_crs(crs)
    except Exception as exc:  # noqa: BLE001
        raise PreprocessingError(f"Reprojection to {crs} failed: {exc}") from exc


def drop_empty_geometries(gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, int]:
    """Remove rows with null or empty geometry."""
    keep = gdf.geometry.notna() & ~gdf.geometry.is_empty
    return gdf.loc[keep].reset_index(drop=True), int((~keep).sum())


def _polygonal_part(geom):
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    if hasattr(geom, "geoms"):
        parts = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        if parts:
            return unary_union(parts)
    return geom


def fix_invalid_geometries(gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, int]:
    """Repair invalid geometries with ``shapely.make_valid``.

    For polygon inputs, non-polygonal fragments (stray lines/points) produced by the
    repair are discarded so that polygon contiguity weights remain meaningful.
    """
    out = gdf.copy()
    geom_col = out.geometry.name
    mask = out.geometry.notna() & ~out.geometry.is_empty & ~out.geometry.is_valid
    n_fixed = int(mask.sum())
    if n_fixed == 0:
        return out, 0

    def _repair(geom):
        was_polygon = geom.geom_type in ("Polygon", "MultiPolygon")
        fixed = make_valid(geom)
        return _polygonal_part(fixed) if was_polygon else fixed

    new = out.geometry.copy()
    new.loc[mask] = new.loc[mask].apply(_repair)
    out[geom_col] = new
    return out, n_fixed


def drop_duplicate_observations(gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, int]:
    """Drop fully duplicated rows (identical attributes *and* geometry), keeping the first."""
    geom_col = gdf.geometry.name
    attrs = pd.DataFrame(gdf.drop(columns=geom_col))
    attrs["__wkb__"] = gdf.geometry.to_wkb().to_numpy()
    dup = attrs.duplicated(keep="first").to_numpy()
    return gdf.loc[~dup].reset_index(drop=True), int(dup.sum())


def handle_missing(
    gdf: gpd.GeoDataFrame, columns: Sequence[str], strategy: str = "drop"
) -> tuple[gpd.GeoDataFrame, int]:
    """Handle missing values in ``columns``.

    ``strategy`` is ``"drop"`` (remove rows; the default and the safest for spatial
    weights, though dropping rows changes the neighbour structure), or ``"mean"`` /
    ``"median"`` (numeric columns only). Imputation can bias spatial statistics and
    should be reported when used. Returns the data and the number of rows removed
    (``drop``) or cells imputed.
    """
    if strategy not in {"drop", "mean", "median"}:
        raise PreprocessingError(f"Unknown missing-value strategy '{strategy}'.")
    cols = list(columns)
    unknown = [c for c in cols if c not in gdf.columns]
    if unknown:
        raise PreprocessingError(f"Columns not found: {unknown}")

    if strategy == "drop":
        keep = ~gdf[cols].isna().any(axis=1)
        return gdf.loc[keep].reset_index(drop=True), int((~keep).sum())

    non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(gdf[c])]
    if non_numeric:
        raise PreprocessingError(f"Cannot impute non-numeric columns: {non_numeric}")
    out = gdf.copy()
    n_imputed = int(out[cols].isna().sum().sum())
    for c in cols:
        fill = out[c].mean() if strategy == "mean" else out[c].median()
        out[c] = out[c].fillna(fill)
    return out, n_imputed


def join_attributes(
    gdf: gpd.GeoDataFrame,
    table: pd.DataFrame,
    *,
    left_on: str,
    right_on: str | None = None,
    how: str = "left",
) -> tuple[gpd.GeoDataFrame, dict]:
    """Attach a non-spatial table to geometries and report key mismatches.

    Keys are compared as stripped strings so ``"01"`` and ``1`` do not silently mismatch
    across file types. The report lists geometry keys with no match in the table and
    table keys with no match in the geometry (a common source of spatial mismatches).
    """
    right_on = right_on or left_on
    if left_on not in gdf.columns:
        raise PreprocessingError(f"'{left_on}' not found in spatial data.")
    if right_on not in table.columns:
        raise PreprocessingError(f"'{right_on}' not found in table.")

    left = gdf.copy()
    right = table.copy()
    left["__key__"] = left[left_on].astype(str).str.strip()
    right["__key__"] = right[right_on].astype(str).str.strip()
    if right["__key__"].duplicated().any():
        raise PreprocessingError(f"Join key '{right_on}' has duplicate values in the table.")

    report = {
        "unmatched_geometry_keys": sorted(set(left["__key__"]) - set(right["__key__"])),
        "unmatched_table_keys": sorted(set(right["__key__"]) - set(left["__key__"])),
    }
    right = right.drop(columns=[right_on]) if right_on == left_on else right
    right = right.drop(columns=[c for c in right.columns if c in left.columns and c != "__key__"])
    merged = left.merge(right, on="__key__", how=how).drop(columns="__key__")
    if not isinstance(merged, gpd.GeoDataFrame):
        merged = gpd.GeoDataFrame(merged, geometry=left.geometry.name, crs=left.crs)
    return merged.reset_index(drop=True), report


def preprocess(
    gdf: gpd.GeoDataFrame,
    *,
    target_crs: str | int | None = None,
    fix_geometries: bool = True,
    drop_empty: bool = True,
    drop_duplicates: bool = False,
    missing_columns: Sequence[str] | None = None,
    missing_strategy: str = "drop",
) -> PreprocessResult:
    """Apply the standard cleaning pipeline and log each change.

    Order: drop empty geometries, repair invalid geometries, reproject, drop
    duplicate observations, handle missing values.
    """
    log: list[str] = []
    out = gdf.copy()

    if drop_empty:
        out, n = drop_empty_geometries(out)
        if n:
            log.append(f"Dropped {n} observation(s) with null/empty geometry.")
    if fix_geometries:
        out, n = fix_invalid_geometries(out)
        if n:
            log.append(f"Repaired {n} invalid geometry(ies) with make_valid.")
    if target_crs is not None:
        out = reproject(out, target_crs)
        log.append(f"Reprojected to {target_crs}.")
    if drop_duplicates:
        out, n = drop_duplicate_observations(out)
        if n:
            log.append(f"Dropped {n} duplicate observation(s).")
    if missing_columns:
        out, n = handle_missing(out, missing_columns, missing_strategy)
        if n:
            verb = "Dropped rows" if missing_strategy == "drop" else "Imputed cells"
            log.append(f"{verb}: {n} ({missing_strategy}) for missing values.")

    return PreprocessResult(out.reset_index(drop=True), log)
