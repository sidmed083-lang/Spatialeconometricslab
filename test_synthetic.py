import numpy as np
import pytest

from spatial_econometrics.data import lattice_weights_matrix, make_lattice, make_spatial_dataset


def test_make_lattice_shape_and_order():
    gdf = make_lattice(3, 4, cell_size=10.0)
    assert len(gdf) == 12 and gdf.crs.to_epsg() == 32631
    assert gdf.loc[5, "row"] == 1 and gdf.loc[5, "col"] == 1
    assert gdf.geometry.iloc[5].bounds == (10.0, 10.0, 20.0, 20.0)
    with pytest.raises(ValueError):
        make_lattice(1, 5)


def test_rook_and_queen_weights():
    rook = lattice_weights_matrix(3, 3, row_standardize=False)
    assert rook.sum(axis=1).tolist() == [2, 3, 2, 3, 4, 3, 2, 3, 2]
    assert np.array_equal(rook, rook.T)
    queen = lattice_weights_matrix(3, 3, queen=True, row_standardize=False)
    assert queen[4].sum() == 8 and queen[0].sum() == 3
    std = lattice_weights_matrix(3, 3)
    assert np.allclose(std.sum(axis=1), 1.0)


def test_reproducible_and_seed_sensitive():
    a, b = make_spatial_dataset(6, 6, seed=1), make_spatial_dataset(6, 6, seed=1)
    c = make_spatial_dataset(6, 6, seed=2)
    assert np.array_equal(a["y"], b["y"])
    assert not np.array_equal(a["y"], c["y"])


@pytest.mark.parametrize("model", ["ols", "sar", "sem", "sdm"])
def test_dgp_reconstructs_innovations(model):
    n_rows = n_cols = 8
    gdf = make_spatial_dataset(n_rows, n_cols, model=model, rho=0.6, lam=0.4)
    p = gdf.attrs["true_params"]
    W = lattice_weights_matrix(n_rows, n_cols)
    y, eps = gdf["y"].to_numpy(), gdf["true_error"].to_numpy()
    Z = gdf[["x1", "x2"]].to_numpy()
    Xb = np.column_stack([np.ones(len(gdf)), Z]) @ np.array(p["beta"])
    if model == "ols":
        recovered = y - Xb
    elif model == "sar":
        recovered = y - p["rho"] * W @ y - Xb
    elif model == "sem":
        recovered = (y - Xb) - p["lambda"] * W @ (y - Xb)
    else:
        recovered = y - p["rho"] * W @ y - Xb - W @ Z @ np.array(p["theta"])
    assert np.allclose(recovered, eps)


def test_parameter_validation():
    with pytest.raises(ValueError):
        make_spatial_dataset(model="bogus")
    with pytest.raises(ValueError):
        make_spatial_dataset(rho=1.2)
    with pytest.raises(ValueError):
        make_spatial_dataset(beta=(1.0, 2.0))
