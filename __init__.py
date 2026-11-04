"""Spatial data management: loading, validation, preprocessing, synthetic data."""

from .loaders import (
    SUPPORTED_EXTENSIONS,
    DataLoadError,
    detect_coordinate_columns,
    list_gpkg_layers,
    load_from_uploads,
    load_spatial_file,
)
from .preprocessing import (
    PreprocessingError,
    PreprocessResult,
    assign_crs,
    drop_duplicate_observations,
    drop_empty_geometries,
    fix_invalid_geometries,
    handle_missing,
    join_attributes,
    preprocess,
    reproject,
)
from .synthetic import lattice_weights_matrix, make_lattice, make_spatial_dataset
from .validation import (
    ValidationIssue,
    ValidationReport,
    crs_info,
    dataset_metadata,
    validate_geodataframe,
)

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "DataLoadError",
    "PreprocessResult",
    "PreprocessingError",
    "ValidationIssue",
    "ValidationReport",
    "assign_crs",
    "crs_info",
    "dataset_metadata",
    "detect_coordinate_columns",
    "drop_duplicate_observations",
    "drop_empty_geometries",
    "fix_invalid_geometries",
    "handle_missing",
    "join_attributes",
    "lattice_weights_matrix",
    "list_gpkg_layers",
    "load_from_uploads",
    "load_spatial_file",
    "make_lattice",
    "make_spatial_dataset",
    "preprocess",
    "reproject",
    "validate_geodataframe",
]
