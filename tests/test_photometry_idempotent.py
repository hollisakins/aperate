"""Partial reruns must not apply kron_corr / aper_corr twice (review finding F03)."""
import numpy as np
from astropy.io import fits
from astropy.table import Table

from aperate.commands.photometry import (
    _apply_multiplicative_correction,
    _mark_raw,
    load_tile_catalog,
    save_tile_catalog,
)


def test_rerun_of_subset_leaves_other_filters_unchanged():
    n = 4
    raw_a, raw_b = np.arange(1, n + 1, dtype=float), np.arange(10, 10 + n, dtype=float)
    corr1 = np.array([1.5, 1.2, 1.0, 2.0])
    cat = Table({'f_auto_a': raw_a.copy(), 'e_auto_a': 0.1 * raw_a, 'f_auto_b': raw_b.copy(), 'e_auto_b': 0.1 * raw_b})

    # first run: both filters measured, correction applied once
    _apply_multiplicative_correction(cat, ('f_auto_', 'e_auto_'), corr1, None, 'KRONCORR')
    cat['kron_corr'] = corr1
    assert cat.meta['KRONCORR'] == ['a', 'b']
    assert np.allclose(cat['f_auto_b'], raw_b * corr1)

    # second run: only filter a re-measured (raw again), same correction recomputed
    cat['f_auto_a'] = raw_a.copy(); cat['e_auto_a'] = 0.1 * raw_a
    _mark_raw(cat, 'KRONCORR', 'a')
    assert cat.meta['KRONCORR'] == ['b']
    old = np.asarray(cat['kron_corr'])
    _apply_multiplicative_correction(cat, ('f_auto_', 'e_auto_'), corr1, old, 'KRONCORR')
    assert np.allclose(cat['f_auto_a'], raw_a * corr1)
    assert np.allclose(cat['f_auto_b'], raw_b * corr1)          # not corr1**2
    assert np.allclose(cat['e_auto_b'], 0.1 * raw_b * corr1)

    # third run: correction values changed (e.g. new kron params); untouched filter follows
    corr2 = corr1 * 1.1
    _apply_multiplicative_correction(cat, ('f_auto_', 'e_auto_'), corr2, corr1, 'KRONCORR')
    assert np.allclose(cat['f_auto_b'], raw_b * corr2)


def test_two_dimensional_columns_and_nan_corrections():
    n, nd = 3, 2
    raw = np.ones((n, nd))
    corr1 = np.array([[2.0, np.nan], [1.5, 1.5], [1.0, 3.0]])
    cat = Table({'f_aper_hom_x': raw.copy(), 'e_aper_hom_x': raw.copy()})
    _apply_multiplicative_correction(cat, ('f_aper_hom_', 'e_aper_hom_'), corr1, None, 'APERCORR')
    assert np.isnan(cat['f_aper_hom_x'][0, 1])
    # re-applying the same correction is a no-op (NaN stays NaN, nothing is squared)
    _apply_multiplicative_correction(cat, ('f_aper_hom_', 'e_aper_hom_'), corr1, corr1, 'APERCORR')
    assert np.allclose(cat['f_aper_hom_x'][1], 1.5) and cat['f_aper_hom_x'][2, 1] == 3.0
    assert np.isnan(cat['f_aper_hom_x'][0, 1])


def test_correction_state_round_trips_through_fits(tmp_path):
    (tmp_path / 'catalogs').mkdir()
    path = tmp_path / 'catalogs' / 'catalog_proj_T1.fits'
    cat = Table({'x': np.arange(3.0), 'y': np.arange(3.0), 'f_auto_f444w': np.ones(3)})
    primary = fits.PrimaryHDU(); primary.header['WINDOWED'] = False
    fits.HDUList([primary, fits.BinTableHDU(cat, name='SOURCES')]).writeto(path)

    loaded, windowed = load_tile_catalog(tmp_path, 'proj', 'T1')
    assert loaded.meta['KRONCORR'] == [] and loaded.meta['APERCORR'] == []
    loaded.meta['KRONCORR'] = [f'f{i:03d}w' for i in range(40)]      # long list -> CONTINUE cards
    loaded.meta['APERCORR'] = ['f444w']
    save_tile_catalog(tmp_path, 'proj', 'T1', loaded)
    again, _ = load_tile_catalog(tmp_path, 'proj', 'T1')
    assert again.meta['KRONCORR'] == loaded.meta['KRONCORR']
    assert again.meta['APERCORR'] == ['f444w']
