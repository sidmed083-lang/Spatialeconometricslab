"""Load spatial datasets (GeoJSON, Shapefile, GeoPackage, CSV) into GeoDataFrames."""

from __future__ import annotations

import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import BinaryIO

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS
from shapely.geometry import Point

SPATIAL_EXTENSIONS = {".geojson", ".json", ".gpkg", ".shp", ".zip"}
TABULAR_EXTENSIONS = {".csv"}
SUPPORTED_EXTENSIONS = SPATIAL_EXTENSIONS | TABULAR_EXTENSIONS

_LAT_NAMES = ("lat", "latitude", "y", "ycoord", "y_coord")
_LON_NAMES = ("lon", "lng", "long", "longitude", "x", "xcoord", "x_coord")
_WKT_NAMES = ("wkt", "geometry", "geom", "the_geom")


class DataLoadError(ValueError):
    """Raised when a dataset cannot be read or has no usable geometry."""


def detect_coordinate_columns(columns: Iterable[str]) -> dict[str, str | None]:
    """Guess latitude, longitude and WKT column names (case-insensitive).

    Returns a dict with keys ``lat``, ``lon`` and ``wkt``; values are the original
    column names or ``None``. Bare ``x``/``y`` are accepted, so callers loading
    projected coordinates must pass the matching ``csv_crs``.
    """
    lowered = {str(c).lower().strip(): c for c in columns}

    def first(names: tuple[str, ...]) -> str | None:
        for name in names:
            if name in lowered:
                return str(lowered[name])
        return None

    return {"lat": first(_LAT_NAMES), "lon": first(_LON_NAMES), "wkt": first(_WKT_NAMES)}


def load_spatial_file(
    source: str | Path | bytes | BinaryIO,
    filename: str | None = None,
    *,
    layer: str | None = None,
    lat_col: str | None = None,
    lon_col: str | None = None,
    wkt_col: str | None = None,
    csv_crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """Read a spatial dataset from a path, raw bytes or a file-like object.

    Parameters
    ----------
    source : path, bytes or file-like (e.g. a Streamlit ``UploadedFile``).
    filename : required for bytes / file-like sources (the extension selects the reader).
    layer : GeoPackage layer name (default: first layer).
    lat_col, lon_col, wkt_col : CSV geometry columns. Auto-detected when omitted.
    csv_crs : CRS of CSV coordinates (default EPSG:4326).
    """
    options = {
        "layer": layer,
        "lat_col": lat_col,
        "lon_col": lon_col,
        "wkt_col": wkt_col,
        "csv_crs": csv_crs,
    }
    if isinstance(source, str | Path):
        path = Path(source)
        if not path.exists():
            raise DataLoadError(f"File not found: {path}")
        return _load_path(path, **options)

    if filename is None:
        raise DataLoadError("`filename` is required when loading from bytes or a file object.")
    data = bytes(source) if isinstance(source, bytes | bytearray) else source.read()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / Path(filename).name
        path.write_bytes(data)
        return _load_path(path, **options)


def load_from_uploads(files: Mapping[str, bytes], **kwargs) -> gpd.GeoDataFrame:
    """Load from one or more uploaded files given as ``{filename: bytes}``.

    Multiple files are accepted so that a Shapefile's ``.shp/.shx/.dbf/.prj`` parts can be
    uploaded together. Extra keyword arguments are forwarded to :func:`load_spatial_file`.
    """
    if not files:
        raise DataLoadError("No files were provided.")
    if len(files) == 1:
        (name, data), = files.items()
        return load_spatial_file(data, filename=name, **kwargs)

    shp_names = [n for n in files if Path(n).suffix.lower() == ".shp"]
    if len(shp_names) != 1:
        raise DataLoadError(
            "Multiple files were provided; expected exactly one .shp plus its sidecar files."
        )
    with tempfile.TemporaryDirectory() as tmp:
        for name, data in files.items():
            (Path(tmp) / Path(name).name).write_bytes(data)
        return _load_path(Path(tmp) / Path(shp_names[0]).name, **_default_options(kwargs))


def list_gpkg_layers(path: str | Path) -> list[str]:
    """Return the layer names contained in a GeoPackage."""
    import pyogrio

    try:
        return [str(row[0]) for row in pyogrio.list_layers(str(path))]
    except Exception as exc:  # noqa: BLE001
        raise DataLoadError(f"Could not list layers in {path}: {exc}") from exc


def _default_options(overrides: Mapping[str, object]) -> dict[str, object]:
    options: dict[str, object] = {
        "layer": None,
        "lat_col": None,
        "lon_col": None,
        "wkt_col": None,
        "csv_crs": "EPSG:4326",
    }
    options.update(overrides)
    return options


def _load_path(path: Path, **options) -> gpd.GeoDataFrame:
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise DataLoadError(
            f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )
    if ext == ".csv":
        gdf = _load_csv(path, **{k: v for k, v in options.items() if k != "layer"})
    else:
        if ext == ".shp" and not path.with_suffix(".dbf").exists():
            raise DataLoadError(
                "Shapefile is missing its .dbf attribute file. Upload the .shp together with "
                ".shx/.dbf/.prj files, or upload a single .zip archive."
            )
        target = f"zip://{path}" if ext == ".zip" else str(path)
        try:
            layer = options.get("layer")
            gdf = gpd.read_file(target, layer=layer) if layer else gpd.read_file(target)
        except Exception as exc:  # noqa: BLE001
            raise DataLoadError(f"Could not read '{path.name}': {exc}") from exc

    if len(gdf) == 0:
        raise DataLoadError(f"'{path.name}' contains no rows.")
    if gdf.geometry.isna().all():
        raise DataLoadError(f"'{path.name}' has no usable geometries.")
    return gdf.reset_index(drop=True)


def _load_csv(
    path: Path,
    *,
    lat_col: str | None,
    lon_col: str | None,
    wkt_col: str | None,
    csv_crs: str,
) -> gpd.GeoDataFrame:
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        raise DataLoadError(f"Could not parse CSV '{path.name}': {exc}") from exc
    if df.empty:
        raise DataLoadError(f"'{path.name}' contains no rows.")

    if bool(lat_col) != bool(lon_col):
        raise DataLoadError("Provide both `lat_col` and `lon_col`, or neither.")

    detected = detect_coordinate_columns(df.columns)
    if wkt_col is None and not (lat_col and lon_col):
        if detected["wkt"] is not None:
            wkt_col = detected["wkt"]
        elif detected["lat"] is not None and detected["lon"] is not None:
            lat_col, lon_col = detected["lat"], detected["lon"]
        else:
            raise DataLoadError(
                "Could not find geometry columns in the CSV. Specify latitude/longitude "
                "columns or a WKT column."
            )

    if wkt_col is not None:
        if wkt_col not in df.columns:
            raise DataLoadError(f"WKT column '{wkt_col}' not found.")
        raw = df[wkt_col].astype(object).where(df[wkt_col].notna(), None)
        try:
            geoms = gpd.GeoSeries.from_wkt(raw)
        except Exception as exc:  # noqa: BLE001
            raise DataLoadError(f"Invalid WKT in column '{wkt_col}': {exc}") from exc
        return gpd.GeoDataFrame(df.drop(columns=[wkt_col]), geometry=geoms.values, crs=csv_crs)

    assert lat_col is not None and lon_col is not None
    for col in (lat_col, lon_col):
        if col not in df.columns:
            raise DataLoadError(f"Coordinate column '{col}' not found.")
    lat = pd.to_numeric(df[lat_col], errors="coerce").to_numpy(dtype=float)
    lon = pd.to_numeric(df[lon_col], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(lat) & np.isfinite(lon)
    if not finite.any():
        raise DataLoadError("No valid numeric coordinates found in the CSV.")
    if CRS.from_user_input(csv_crs).is_geographic:
        if (np.abs(lat[finite]) > 90).any() or (np.abs(lon[finite]) > 180).any():
            raise DataLoadError(
                "Coordinates fall outside valid longitude/latitude ranges for a geographic CRS. "
                "Check the column choice or set `csv_crs` to the correct projected CRS."
            )
    geometry = [Point(x, y) if ok else None for x, y, ok in zip(lon, lat, finite, strict=True)]
    return gpd.GeoDataFrame(df, geometry=geometry, crs=csv_crs)
