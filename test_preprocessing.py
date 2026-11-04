import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon

from spatial_econometrics.data import (
    PreprocessingError,
    assign_crs,
    drop_duplicate_observations,
    drop_empty_geometries,
    fix_invalid_geometries,
    handle_missing,
    join_attributes,
    preprocess,
    reproject,
)

BOWTIE = Polygon([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)])


def test_reproject_and_assign_crs(lattice):
    assert reproject(lattice, "EPSG:4326").crs.to_epsg() == 4326
    no_crs = gpd.GeoDataFrame({"a": [1]}, geometry=[Point(0, 0)])
    with pytest.raises(PreprocessingError):
        reproject(no_crs, "EPSG:4326")
    assert assign_crs(no_crs, "EPSG:4326").crs.to_epsg() == 4326
    with pytest.raises(PreprocessingError):
        assign_crs(lattice, "EPSG:4326")


def test_drop_empty_geometries():
    gdf = gpd.GeoDataFrame({"a": [1, 2, 3]}, geometry=[Point(0, 0), None, Point()])
    out, n = drop_empty_geometries(gdf)
    assert n == 2 and len(out) == 1


def test_fix_invalid_geometries_does_not_mutate_input():
    gdf = gpd.GeoDataFrame({"a": [1]}, geometry=[BOWTIE], crs="EPSG:32631")
    out, n = fix_invalid_geometries(gdf)
    assert n == 1 and out.geometry.is_valid.all()
    assert not gdf.geometry.is_valid.any()
    assert out.geometry.iloc[0].geom_type in {"Polygon", "MultiPolygon"}


def test_fix_invalid_noop_when_all_valid(lattice):
    out, n = fix_invalid_geometries(lattice)
    assert n == 0 and len(out) == len(lattice)


def test_drop_duplicate_observations():
    gdf = gpd.GeoDataFrame(
        {"v": [1.0, 1.0, 2.0]}, geometry=[Point(0, 0), Point(0, 0), Point(0, 0)]
    )
    out, n = drop_duplicate_observations(gdf)
    assert n == 1 and out["v"].tolist() == [1.0, 2.0]


def test_handle_missing_strategies(lattice):
    df = lattice.assign(y=range(16)).astype({"y": float})
    df.loc[[0, 1], "y"] = None

    dropped, n = handle_missing(df, ["y"], "drop")
    assert n == 2 and len(dropped) == 14

    mean, n = handle_missing(df, ["y"], "mean")
    assert n == 2 and mean["y"].isna().sum() == 0
    assert mean.loc[0, "y"] == pytest.approx(df["y"].mean())

    median, _ = handle_missing(df, ["y"], "median")
    assert median.loc[1, "y"] == pytest.approx(df["y"].median())

    with pytest.raises(PreprocessingError):
        handle_missing(df, ["y"], "magic")
    with pytest.raises(PreprocessingError):
        handle_missing(df.assign(s="a"), ["s"], "mean")
    with pytest.raises(PreprocessingError):
        handle_missing(df, ["nope"], "drop")


def test_join_attributes_reports_mismatches(lattice):
    table = pd.DataFrame({"code": ["0", "1", "2", "99"], "gdp": [10.0, 20.0, 30.0, 40.0]})
    out, report = join_attributes(lattice, table, left_on="cell_id", right_on="code")
    assert isinstance(out, gpd.GeoDataFrame) and len(out) == 16
    assert out["gdp"].notna().sum() == 3
    assert report["unmatched_table_keys"] == ["99"]
    assert len(report["unmatched_geometry_keys"]) == 13
    with pytest.raises(PreprocessingError):
        join_attributes(lattice, pd.DataFrame({"cell_id": [0, 0], "a": [1, 2]}), left_on="cell_id")


def test_preprocess_pipeline_logs_each_step():
    gdf = gpd.GeoDataFrame(
        {"y": [1.0, None, 3.0, 4.0, 4.0]},
        geometry=[BOWTIE, Point(0, 0), None, Point(1, 1), Point(1, 1)],
        crs="EPSG:32631",
    )
    result = preprocess(
        gdf,
        target_crs="EPSG:4326",
        drop_duplicates=True,
        missing_columns=["y"],
    )
    assert len(result.gdf) == 2
    assert result.gdf.crs.to_epsg() == 4326
    assert result.gdf.geometry.is_valid.all()
    assert len(result.log) == 5
