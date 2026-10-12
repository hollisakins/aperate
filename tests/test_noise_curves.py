"""Stratified noise curves (aperate.core.noise_curves) on synthetic images.

Two regions with the same pixel noise but different correlated-noise fractions: a single curve
cannot be calibrated in both; two strata of a depth proxy that separates the regions can.
"""
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from aperate.core import noise_curves as nc


def make_image(rng, n=1200):
    white = rng.normal(0, 1, (n, n)).astype(np.float32)
    # left half: white noise; right half: correlated noise with the same pixel rms
    corr = gaussian_filter(rng.normal(0, 1, (n, n)), 1.5).astype(np.float32)
    corr *= 1.0 / corr.std()
    sci = white.copy()
    sci[:, n // 2:] = corr[:, n // 2:]
    scale = np.ones((n, n), np.float32)
    proxy = np.ones((n, n), np.float32)
    proxy[:, n // 2:] = 10.0         # "deeper" (larger weight) region holds the correlated noise
    return sci, scale, proxy


@pytest.fixture(scope='module')
def samples():
    rng = np.random.default_rng(3)
    sci, scale, proxy = make_image(rng)
    valid = np.ones(sci.shape, bool)
    radii = [1.0, 2.0, 4.0, 8.0]
    out = nc.sample_image(sci, scale, {'wht': proxy}, valid, radii, [8000] * 4, None, 1, rng)
    sqrtN = np.sqrt(np.pi) * np.array(radii)
    s = {i: {'f': out['f'][i], 'logp': out['logp']['wht'][i]} for i in range(4)}
    _, rms1 = 0.0, float(np.std(out['pixels']))
    return s, sqrtN, rms1


def test_single_curve_is_miscalibrated_but_strata_are_not(samples):
    s, sqrtN, rms1 = samples
    one = nc.fit_model(s, sqrtN, rms1, k=1, holdout_fraction=0.3, width_method='biweight')
    two = nc.fit_model(s, sqrtN, rms1, k=2, holdout_fraction=0.3, width_method='biweight')
    # at least the two regions (apertures straddling the boundary may form a small middle stratum)
    assert len(two['strata']['log_centres']) >= 2
    # the correlated region needs a steeper curve
    c0, c1 = two['strata']['curves'][0], two['strata']['curves'][-1]
    assert nc.evaluate_curve(c1, sqrtN[-1]) > 1.5 * nc.evaluate_curve(c0, sqrtN[-1])
    # held-out calibration in the two regions at the largest aperture: within 10% with strata
    for w in (two['validation']['by_stratum'][0], two['validation']['by_stratum'][-1]):
        assert abs(w[-1] - 1) < 0.10, w
    # the single curve is off by more than 25% in at least one region at the largest aperture
    f, logp = s[3]['f'], s[3]['logp']
    pred1 = rms1 * nc.evaluate_curve(one, np.full(len(f), sqrtN[-1]))
    widths = [np.std(f[logp < 1] / pred1[logp < 1]), np.std(f[logp > 1] / pred1[logp > 1])]
    assert max(abs(np.array(widths) - 1)) > 0.25


def test_predict_interpolates_between_strata():
    sqrtN = np.array([2.0, 4.0, 8.0, 16.0])
    white = nc.fit_curve(sqrtN, sqrtN)                 # sigma_N/sigma_1 = sqrt N
    corr = nc.fit_curve(sqrtN, 2.0 * sqrtN)            # twice as large
    model = dict(white, strata={'log_centres': [0.0, 1.0], 'curves': [white, corr], 'edges': [0.5]})
    p = nc.predict(model, np.full(3, 8.0), np.array([0.0, 0.5, 1.0]))
    assert p[0] < p[1] < p[2]
    assert np.isclose(p[1], 0.5 * (p[0] + p[2]))
    # outside the range: clamped to the outer strata
    p_out = nc.predict(model, np.full(2, 8.0), np.array([-5.0, 6.0]))
    assert np.isclose(p_out[0], p[0]) and np.isclose(p_out[1], p[2])
    # no proxy: the pooled curve
    assert np.isclose(nc.predict(model, 8.0, None)[0], nc.evaluate_curve(model, 8.0))


def test_strata_edges_merge_sparse_bins():
    rng = np.random.default_rng(0)
    logp = np.concatenate([rng.normal(0, 0.1, 5000), rng.normal(3, 0.1, 100)])
    edges = nc.strata_edges(logp, 4, min_count=300)
    counts = np.bincount(np.digitize(logp, edges), minlength=len(edges) + 1)
    assert counts.min() >= 300
    assert nc.strata_edges(logp, 1).size == 0


def test_weight_map_check_detects_a_bad_map():
    rng = np.random.default_rng(5)
    n = 600
    emp = np.ones((n, n), np.float32); emp[:, n // 2:] = 2.0      # right half twice as noisy
    good = 1 / emp ** 2                                           # a correct weight map
    bad = rng.uniform(0.5, 2, (n, n)).astype(np.float32)          # unrelated to the noise
    valid = np.ones((n, n), bool)
    assert nc.weight_map_check(good, emp, valid, rng, n=20000)['ok']
    assert not nc.weight_map_check(bad, emp, valid, rng, n=20000)['ok']
