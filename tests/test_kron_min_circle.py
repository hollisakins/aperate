"""The minimum-circle rule of the Kron (AUTO) aperture (review finding F20).

An object whose scaled Kron ellipse has a geometric-mean radius kron_params[0]*kronrad*sqrt(a*b)
below kron_params[1] pixels must be measured in a circle of radius kron_params[1]; one above must
keep its ellipse.  The original code applied the kron_params[0]*kronrad factor a second time in
the test, so the circle was used far too rarely.
"""
from types import SimpleNamespace

import numpy as np
from astropy.table import Table

from aperate.commands.photometry import _compute_auto_photometry


def _context(a, b, kronrad):
    n = len(a)
    cat = Table()
    cat['id'] = np.arange(1, n + 1, dtype=np.int32)
    cat['a'] = np.asarray(a, float)
    cat['b'] = np.asarray(b, float)
    cat['theta'] = np.zeros(n)
    cat['kronrad'] = np.asarray(kronrad, float)
    return SimpleNamespace(catalog=cat, segmap=np.zeros((200, 200), dtype=np.int32))


def test_minimum_circle_uses_scaled_axes_once():
    kron_params = [1.1, 1.6]
    # object 0: 1.1*1.4*1.0 = 1.54 px  -> below the minimum: circle of 1.6 px
    # object 1: 1.1*1.5*1.0 = 1.65 px  -> above the minimum: keep the ellipse
    # object 2: 1.1*0.5*3.0 = 1.65 px  -> above the minimum, but the old (double-scaled) test
    #           gave 0.55*1.65 = 0.91 < 1.6 and wrongly used the circle
    a = [1.0, 1.0, 3.0]
    b = [1.0, 1.0, 3.0]
    kronrad = [1.4, 1.5, 0.5]
    ctx = _context(a, b, kronrad)
    sci = np.ones((200, 200), dtype=np.float32)
    mask = np.zeros_like(sci, dtype=bool)
    x = np.array([50.0, 100.0, 150.0])
    y = np.array([50.0, 100.0, 150.0])

    flux, fluxerr, flag, a_out, b_out = _compute_auto_photometry(sci, None, mask, x, y, ctx, kron_params)

    assert a_out[0] == 1.6 and b_out[0] == 1.6
    assert np.isclose(a_out[1], 1.65) and np.isclose(b_out[1], 1.65)
    assert np.isclose(a_out[2], 1.65) and np.isclose(b_out[2], 1.65)
    # on an image of ones the flux is the (pixelised) aperture area: the circle object collects
    # less than the two ellipse objects, which are identical apertures
    assert np.allclose(flux, np.pi * np.asarray(a_out) * np.asarray(b_out), rtol=0.06)
    assert flux[0] < flux[1] and np.isclose(flux[1], flux[2])


def test_minimum_circle_boundary():
    kron_params = [2.5, 3.5]
    # geometric-mean radii 3.49 (circle) and 3.51 (ellipse) px
    ctx = _context([1.0, 1.0], [1.0, 1.0], [3.49 / 2.5, 3.51 / 2.5])
    sci = np.ones((200, 200), dtype=np.float32)
    mask = np.zeros_like(sci, dtype=bool)
    _, _, _, a_out, b_out = _compute_auto_photometry(sci, None, mask, np.array([60.0, 140.0]), np.array([60.0, 140.0]), ctx, kron_params)
    assert a_out[0] == 3.5 and b_out[0] == 3.5
    assert np.isclose(a_out[1], 3.51) and np.isclose(b_out[1], 3.51)
