"""Running a postprocess step twice on the same merged catalog must not apply it twice (F14)."""
from types import SimpleNamespace

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from aperate.commands.postprocess import apply_galactic_extinction_corr, apply_psf_corrections

from test_psf_corrections import _catalog, _write_grid


def test_psf_corrections_applied_once(tmp_path):
    _write_grid(tmp_path / 'f444w_psf_corr_grid.txt', 1.10)
    psfs_config = SimpleNamespace(target_filter='f444w', inverse_filters=[], master_psf=True)
    cat = _catalog()
    for _ in range(2):
        cat = apply_psf_corrections(cat, tmp_path, ['T1'], ['f444w', 'f277w'], psfs_config, [0.3, 0.5, 1.0])
    assert np.allclose(cat['f_auto_f444w'], 1.10) and np.allclose(cat['f_auto_f277w'], 1.10)
    assert cat.meta['PSFCORR'] == 'f277w,f444w'
    # a filter not yet corrected is still processed
    cat = apply_psf_corrections(cat, tmp_path, ['T1'], ['f770w'], psfs_config, [0.3, 0.5, 1.0])
    assert np.allclose(cat['f_auto_f770w'], 1.10) and cat.meta['PSFCORR'] == 'f277w,f444w,f770w'


def test_state_survives_fits_round_trip(tmp_path):
    cat = Table({'f_auto_f444w': np.ones(3)})
    cat.meta['PSFCORR'] = ','.join(f'f{i:03d}w' for i in range(40))
    path = tmp_path / 'merged.fits'
    cat.write(path)
    again = Table.read(path)
    assert again.meta['PSFCORR'] == cat.meta['PSFCORR']


def test_extinction_applied_once(tmp_path):
    # constant E(B-V) = 0.1 dust map with a TAN WCS around the sources
    w = WCS(naxis=2)
    w.wcs.crpix = [5, 5]; w.wcs.cdelt = [-0.1, 0.1]; w.wcs.crval = [150.0, 2.0]; w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    hdu = fits.PrimaryHDU(np.full((10, 10), 0.1), header=w.to_header())
    dust = tmp_path / 'dust.fits'; hdu.writeto(dust)
    n = 3
    cat = Table({'ra': np.full(n, 150.0), 'dec': np.full(n, 2.0), 'f_auto_f444w': np.ones(n), 'e_auto_f444w': np.full(n, 0.1)})
    for _ in range(2):
        cat = apply_galactic_extinction_corr(cat, dust, {'f444w': 44000.0}, ['f444w'])
    expected = 10 ** (0.4 * cat['A_f444w'])
    assert np.allclose(cat['f_auto_f444w'], expected) and np.allclose(cat['e_auto_f444w'], 0.1 * expected)
    assert cat.meta['EXTCORR'] == 'f444w'
