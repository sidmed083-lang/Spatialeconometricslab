import geopandas as gpd
import pytest
from shapely.geometry import Point

from spatial_econometrics.data import make_lattice


@pytest.fixture
def lattice():
    """4x4 projected lattice (UTM 31N)."""
    return make_lattice(4, 4)


@pytest.fixture
def points():
    """Three geographic points with a numeric attribute."""
    return gpd.GeoDataFrame(
        {"id": [1, 2, 3], "value": [1.0, 2.0, 3.0]},
        geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
        crs="EPSG:4326",
    )
