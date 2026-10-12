"""Tests for the aperture-correction gating (review finding F01) and the final NaN sweep."""
import numpy as np
from astropy.table import Table

from aperate.commands.photometry import _finalize_aper_corr
from aperate.commands.postprocess import set_missing_to_nan


def test_finalize_gates_bounds_and_flags():
    ac = np.array([[2.0, -1.5, 30.0, 0.5, np.nan],
                   [1.2, 1.3, 1.4, 1.5, 1.6]])
    snr = np.array([[10.0, 1.0, 10.0, 10.0, 10.0],
                    [10.0, 2.0, np.nan, 10.0, 10.0]])
    out, flag = _finalize_aper_corr(ac, snr, bounds=[1.0, 5.0], min_snr=3.0)
    # row 0: kept; low S/N -> NaN; above bounds (flagged, kept); below bounds (flagged, kept); undefined -> NaN
    assert out[0, 0] == 2.0 and flag[0, 0] == 0
    assert np.isnan(out[0, 1]) and flag[0, 1] == 1
    assert out[0, 2] == 30.0 and flag[0, 2] == 2
    assert out[0, 3] == 0.5 and flag[0, 3] == 2
    assert np.isnan(out[0, 4]) and flag[0, 4] == 1
    # row 1: low S/N and undefined S/N both block the correction
    assert np.isnan(out[1, 1]) and flag[1, 1] == 1
    assert np.isnan(out[1, 2]) and flag[1, 2] == 1
    assert out[1, 0] == 1.2 and flag[1, 0] == 0
    # nothing negative survives
    assert not np.any(out[np.isfinite(out)] < 0)


def test_set_missing_to_nan_catches_nonpositive_calibrated_errors():
    cat = Table()
    cat['f_aper_hom_x'] = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    cat['e_aper_hom_x_uncal'] = np.full((3, 2), 0.1)
    cat['e_aper_hom_x'] = np.array([[0.1, -0.2], [0.3, 0.4], [np.nan, 0.6]])
    cat = set_missing_to_nan(cat, ['x'])
    assert np.isnan(cat['f_aper_hom_x'][0, 1]) and np.isnan(cat['e_aper_hom_x'][0, 1])
    assert np.isnan(cat['f_aper_hom_x'][2, 0])
    assert cat['f_aper_hom_x'][1, 0] == 3.0 and cat['e_aper_hom_x'][1, 1] == 0.4
