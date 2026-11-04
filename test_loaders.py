import shutil

import pytest

from spatial_econometrics.data import (
    DataLoadError,
    detect_coordinate_columns,
    list_gpkg_layers,
    load_from_uploads,
    load_spatial_file,
)


def test_geojson_roundtrip(tmp_path, points):
    path = tmp_path / "pts.geojson"
    points.to_file(path, driver="GeoJSON")
    gdf = load_spatial_file(path)
    assert len(gdf) == 3
    assert gdf.crs.to_epsg() == 4326
    assert list(gdf["value"]) == [1.0, 2.0, 3.0]


def test_geopackage_roundtrip_preserves_crs(tmp_path, lattice):
    path = tmp_path / "grid.gpkg"
    lattice.to_file(path, driver="GPKG", layer="grid")
    gdf = load_spatial_file(path)
    assert len(gdf) == 16
    assert gdf.crs.to_epsg() == 32631
    assert list_gpkg_layers(path) == ["grid"]
    assert len(load_spatial_file(path, layer="grid")) == 16


def test_shapefile_roundtrip(tmp_path, lattice):
    path = tmp_path / "grid.shp"
    lattice.to_file(path)
    gdf = load_spatial_file(path)
    assert len(gdf) == 16
    assert gdf.crs.to_epsg() == 32631


def test_zipped_shapefile(tmp_path, lattice):
    folder = tmp_path / "shp"
    folder.mkdir()
    lattice.to_file(folder / "grid.shp")
    archive = shutil.make_archive(str(tmp_path / "zipped"), "zip", folder)
    assert len(load_spatial_file(archive)) == 16


def test_load_from_bytes_requires_filename(tmp_path, lattice):
    path = tmp_path / "grid.gpkg"
    lattice.to_file(path, driver="GPKG")
    data = path.read_bytes()
    assert len(load_spatial_file(data, filename="upload.gpkg")) == 16
    with pytest.raises(DataLoadError):
        load_spatial_file(data)


def test_load_from_uploads_multi_file_shapefile(tmp_path, lattice):
    lattice.to_file(tmp_path / "grid.shp")
    files = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert len(files) > 1
    assert len(load_from_uploads(files)) == 16


def test_shapefile_without_dbf_raises(tmp_path):
    (tmp_path / "broken.shp").write_bytes(b"not a shapefile")
    with pytest.raises(DataLoadError, match="dbf"):
        load_spatial_file(tmp_path / "broken.shp")


def test_unsupported_extension_and_missing_file(tmp_path):
    bad = tmp_path / "data.txt"
    bad.write_text("x")
    with pytest.raises(DataLoadError, match="Unsupported"):
        load_spatial_file(bad)
    with pytest.raises(DataLoadError, match="not found"):
        load_spatial_file(tmp_path / "nope.geojson")


def test_detect_coordinate_columns():
    found = detect_coordinate_columns(["ID", "Latitude", "Longitude", "WKT"])
    assert found == {"lat": "Latitude", "lon": "Longitude", "wkt": "WKT"}
    assert detect_coordinate_columns(["a", "b"]) == {"lat": None, "lon": None, "wkt": None}


def test_csv_lat_lon_autodetect(tmp_path):
    path = tmp_path / "pts.csv"
    path.write_text("id,lat,lon,v\n1,10.0,20.0,5\n2,11.0,21.0,6\n")
    gdf = load_spatial_file(path)
    assert gdf.crs.to_epsg() == 4326
    assert (gdf.geometry.x.tolist(), gdf.geometry.y.tolist()) == ([20.0, 21.0], [10.0, 11.0])


def test_csv_missing_coordinates_become_null_geometry(tmp_path):
    path = tmp_path / "pts.csv"
    path.write_text("id,lat,lon\n1,10.0,20.0\n2,,21.0\n")
    gdf = load_spatial_file(path)
    assert gdf.geometry.isna().tolist() == [False, True]


def test_csv_wkt(tmp_path):
    path = tmp_path / "wkt.csv"
    path.write_text('id,wkt\n1,"POINT (1 2)"\n2,"POINT (3 4)"\n')
    gdf = load_spatial_file(path, csv_crs="EPSG:32631")
    assert gdf.crs.to_epsg() == 32631
    assert gdf.geometry.x.tolist() == [1.0, 3.0]


def test_csv_errors(tmp_path):
    no_geo = tmp_path / "a.csv"
    no_geo.write_text("a,b\n1,2\n")
    with pytest.raises(DataLoadError, match="geometry columns"):
        load_spatial_file(no_geo)

    out_of_range = tmp_path / "b.csv"
    out_of_range.write_text("lat,lon\n95.0,10.0\n")
    with pytest.raises(DataLoadError, match="outside valid"):
        load_spatial_file(out_of_range)

    with pytest.raises(DataLoadError, match="both"):
        load_spatial_file(no_geo, lat_col="a")
