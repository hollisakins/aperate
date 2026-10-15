"""Regression tests for postprocess.apply_psf_corrections (review finding F17).

Inverse filters (PSF broader than the target) are scaled to the target resolution per source at
photometry time, so postprocess must apply the *target* filter's aperture-to-total grid to them,
not their own. Applying their own grid corrected them twice (F770W fluxes +15% in cosmos_v1.3).
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.table import Table

from aperate.commands.postprocess import apply_psf_corrections


def _write_grid(path: Path, factor: float):
    """A flat aperture-to-total grid over (a, q) with one constant correction factor."""
    a = np.repeat(np.linspace(1, 20, 10), 5)
    q = np.tile(np.linspace(0.2, 1.0, 5), 10)
    np.savetxt(path, np.column_stack([a, q, np.full_like(a, factor)]))


@pytest.fixture
def grids(tmp_path):
    _write_grid(tmp_path / 'f444w_psf_corr_grid.txt', 1.10)   # target
    _write_grid(tmp_path / 'f770w_psf_corr_grid.txt', 1.50)   # inverse filter's own grid
    _write_grid(tmp_path / 'f277w_psf_corr_grid.txt', 1.30)   # homogenized filter's own grid (unused)
    return tmp_path


def _catalog(n=5):
    cat = Table()
    cat['kron2_a'] = np.linspace(2, 10, n)
    cat['kron2_b'] = 0.8 * cat['kron2_a']
    for f in ['f444w', 'f277w', 'f770w']:
        cat[f'f_auto_{f}'] = np.ones(n)
        cat[f'e_auto_{f}'] = np.full(n, 0.1)
        cat[f'f_aper_hom_{f}'] = np.ones((n, 3))
        cat[f'e_aper_hom_{f}'] = np.full((n, 3), 0.1)
    cat['aper_corr'] = np.ones((n, 3))
    return cat


def test_inverse_filter_uses_target_grid(grids):
    psfs_config = SimpleNamespace(target_filter='f444w', inverse_filters=['f770w'], master_psf=True)
    cat = apply_psf_corrections(_catalog(), grids, ['T1'], ['f444w', 'f277w', 'f770w'], psfs_config, [0.3, 0.5, 1.0])
    for f in ['f444w', 'f277w', 'f770w']:
        assert np.allclose(cat[f'f_auto_{f}'], 1.10), f
        assert np.allclose(cat[f'e_auto_{f}'], 0.11), f
        assert np.allclose(cat[f'f_aper_hom_{f}'], 1.10), f
        assert np.allclose(cat[f'e_aper_hom_{f}'], 0.11), f


def test_per_tile_grids_use_target(tmp_path):
    _write_grid(tmp_path / 'f444w_T1_psf_corr_grid.txt', 1.20)
    _write_grid(tmp_path / 'f770w_T1_psf_corr_grid.txt', 2.00)
    psfs_config = SimpleNamespace(target_filter='f444w', inverse_filters=['f770w'], master_psf=False)
    cat = apply_psf_corrections(_catalog(), tmp_path, ['T1'], ['f770w'], psfs_config, [0.3, 0.5, 1.0])
    assert np.allclose(cat['f_auto_f770w'], 1.20)
