"""The 'exp' image extension: a multi-plane exposure map (EXPTIME, NEXP) whose readers must
select the exposure-count plane by name; the first image HDU is the exposure time."""
import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from aperate.commands import photometry as ph
from aperate.commands.postprocess import _sample_tile_task, load_map
from aperate.config.images import EXP_NEXP_HDU, ImageFiles


N = 600


@pytest.fixture(scope='module')
def exp_file(tmp_path_factory):
    d = tmp_path_factory.mktemp('exp')
    nexp = np.full((N, N), 3, np.int16)
    nexp[:, N // 2:] = 6
    nexp[:40, :40] = 0                       # no coverage
    exptime = (100.0 * nexp).astype(np.float32)
    hdus = fits.HDUList([fits.PrimaryHDU(),
                         fits.CompImageHDU(exptime, name='EXPTIME', compression_type='GZIP_2'),
                         fits.CompImageHDU(nexp, name='NEXP', compression_type='GZIP_2')])
    for h in hdus:
        h.header['EXPSRC'] = 'context'
        h.header['NINPUTS'] = 60
    hdus['EXPTIME'].header['BUNIT'] = 's'
    p = d / 'tile_exp.fits'
    hdus.writeto(p)
    return p, nexp, exptime


def test_load_map_selects_the_nexp_plane(exp_file):
    p, nexp, exptime = exp_file
    assert np.array_equal(load_map(p, EXP_NEXP_HDU), nexp)
    assert np.allclose(load_map(p), exptime, rtol=0.01)   # first image HDU: the exposure time
    with pytest.raises(ValueError, match="no HDU named 'BOGUS'"):
        load_map(p, 'BOGUS')


def test_image_files_know_the_exp_extension(exp_file):
    p = exp_file[0]
    files = ImageFiles('f277w', 'T1', {'sci': '/x/sci.fits', 'exp': str(p)})
    assert files.has_extension('exp') and files.get_exp_path() == p
    assert ImageFiles('f277w', 'T1', {'sci': '/x/sci.fits'}).get_exp_path() is None
    assert not hasattr(files, 'get_nexp_path')


def test_photometry_nexp_column_is_the_local_mean_of_the_nexp_plane(exp_file):
    p = exp_file[0]
    cat = Table({'x': [100.0, 500.0, 10.0], 'y': [100.0, 500.0, 10.0]})
    ctx = ph.PhotometryContext(catalog=cat, segmap=None, tile='T1', images_config=None, photometry_config=None,
                               psfs_config=None, main_config=None, windowed=False, overwrite=False)
    files = ImageFiles('f277w', 'T1', {'sci': '/x/sci.fits', 'exp': str(p)})
    ph.compute_weights(ctx, 'f277w', files)
    col = ctx.catalog['nexp_f277w']
    assert col.dtype == np.float32
    assert np.allclose(col[:2], [3.0, 6.0])              # NEXP, not EXPTIME (300, 600)
    assert col[2] == 0.0                                  # uncovered corner
    assert f'{p}#{EXP_NEXP_HDU}' in ctx._image_cache


def test_random_apertures_take_the_exposure_count_from_the_exp_file(exp_file, tmp_path):
    p, nexp, _ = exp_file
    rng = np.random.default_rng(3)
    sci = rng.normal(0, 1, (N, N)).astype(np.float32)
    sci[nexp == 0] = np.nan
    wht = np.where(nexp > 0, 1.0, 0.0).astype(np.float32)
    fits.writeto(tmp_path / 'sci.fits', sci)
    fits.writeto(tmp_path / 'wht.fits', wht)
    out = _sample_tile_task((str(tmp_path / 'sci.fits'), None, str(tmp_path / 'wht.fits'), str(p), None,
                             'wht', 'wht', False, [0.3, 0.6], 0.03, 3000, 100, 0, 1, None, 0))
    levels = np.exp(np.concatenate(out['logp']['nexp']))
    assert len(levels) > 100
    assert levels.min() >= 1.0 and levels.max() <= 6.0 + 1e-3        # counts, not seconds
    assert (np.abs(levels - 3) < 0.01).any() and (np.abs(levels - 6) < 0.01).any()
