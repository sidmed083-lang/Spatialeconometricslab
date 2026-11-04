import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon

from spatial_econometrics.data import crs_info, dataset_metadata, validate_geodataframe


def test_clean_lattice_has_no_errors_or_warnings(lattice):
    report = validate_geodataframe(lattice, id_col="cell_id")
    assert not report.has_errors
    assert not report.has_warnings


def test_missing_crs_is_error():
    no_crs = gpd.GeoDataFrame({"a": [1]}, geometry=[Point(0, 0)])
    report = validate_geodataframe(no_crs)
    assert report.has_errors and "missing_crs" in report.codes


def test_geographic_crs_warns(points):
    report = validate_geodataframe(points)
    assert "geographic_crs" in report.codes and not report.has_errors


def test_invalid_geometry_detected():
    bowtie = Polygon([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)])
    gdf = gpd.GeoDataFrame({"a": [1]}, geometry=[bowtie], crs="EPSG:32631")
    assert validate_geodataframe(gdf).get("invalid_geometry").count == 1


def test_missing_geometry_is_error():
    gdf = gpd.GeoDataFrame({"a": [1, 2, 3]}, geometry=[Point(0, 0), None, Point()], crs="EPSG:32631")
    issue = validate_geodataframe(gdf).get("missing_geometry")
    assert issue.severity == "error" and issue.count == 2


def test_duplicates():
    gdf = gpd.GeoDataFrame(
        {"id": [1, 1, 2], "v": [5.0, 5.0, 7.0]},
        geometry=[Point(0, 0), Point(0, 0), Point(0, 0)],
        crs="EPSG:32631",
    )
    report = validate_geodataframe(gdf, id_col="id")
    assert report.get("duplicate_ids").count == 1
    assert report.get("duplicate_geometry").count == 2
    assert report.get("duplicate_rows").count == 1


def test_missing_values_scoped_to_analysis_columns(lattice):
    lattice = lattice.assign(y=1.0, x=2.0)
    lattice.loc[0, "y"] = None
    lattice.loc[1, "x"] = None
    assert validate_geodataframe(lattice, analysis_columns=["y"]).get("missing_values").count == 1
    assert validate_geodataframe(lattice).get("missing_values").count == 2


def test_coordinates_out_of_range():
    gdf = gpd.GeoDataFrame({"a": [1]}, geometry=[Point(500, 0)], crs="EPSG:4326")
    assert "coords_out_of_range" in validate_geodataframe(gdf).codes


def test_unknown_columns_raise(lattice):
    with pytest.raises(KeyError):
        validate_geodataframe(lattice, id_col="nope")
    with pytest.raises(KeyError):
        validate_geodataframe(lattice, analysis_columns=["nope"])


def test_empty_dataset():
    empty = gpd.GeoDataFrame({"a": []}, geometry=[], crs="EPSG:4326")
    assert validate_geodataframe(empty).codes == {"empty_dataset"}


def test_metadata(lattice):
    meta = dataset_metadata(lattice)
    assert meta["n_rows"] == 16 and meta["geometry_types"] == ["Polygon"]
    assert meta["crs"]["epsg"] == 32631 and meta["crs"]["is_projected"]
    assert "row" in meta["numeric_columns"]
    assert meta["bounds"] == [0.0, 0.0, 4000.0, 4000.0]


def test_crs_info_without_crs():
    gdf = gpd.GeoDataFrame({"a": [1]}, geometry=[Point(0, 0)])
    assert crs_info(gdf)["epsg"] is None
