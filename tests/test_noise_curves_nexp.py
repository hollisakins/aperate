"""Two-dimensional strata (exposure count x depth per exposure) in aperate.core.noise_curves.

A synthetic field with four quadrants: NEXP in {1, 4} and weight per exposure in {1, 4}. The
correlated-noise fraction depends on both (few exposures and deep per-exposure weight give the
most correlated noise). Two quadrants share the total weight 4 (one exposure of weight 4, four of
weight 1) with very different noise, so strata of the total weight alone cannot be calibrated
in every quadrant, while groups of NEXP with strata of weight/NEXP can.
"""
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from aperate.core import noise_curves as nc


def _noise(rng, shape, corr_fraction):
    """Unit-variance noise, a mix of white and smoothed (correlated) noise."""
    white = rng.normal(0, 1, shape)
    corr = gaussian_filter(rng.normal(0, 1, shape), 1.5)
    corr /= corr.std()
    out = np.sqrt(1 - corr_fraction) * white + np.sqrt(corr_fraction) * corr
    return (out / out.std()).astype(np.float32)


def make_field(rng, n=1000):
    h = n // 2
    sci = np.empty((n, n), np.float32)
    nexp = np.empty((n, n), np.float32)
    wht = np.empty((n, n), np.float32)
    # (rows, cols): top-left NEXP 1 wpe 1 (mild), top-right NEXP 1 wpe 4 (strongly correlated),
    # bottom-left NEXP 4 wpe 1 (white; same total weight as top-right), bottom-right NEXP 4 wpe 4 (moderate)
    spec = {(0, 0): (1, 1.0, 0.4), (0, 1): (1, 4.0, 0.9), (1, 0): (4, 1.0, 0.0), (1, 1): (4, 4.0, 0.6)}
    for (i, j), (ne, wpe, cf) in spec.items():
        sl = (slice(i * h, (i + 1) * h), slice(j * h, (j + 1) * h))
        sci[sl] = _noise(rng, (h, h), cf)
        nexp[sl] = ne
        wht[sl] = wpe * ne
    return sci, nexp, wht


@pytest.fixture(scope='module')
def field():
    rng = np.random.default_rng(11)
    sci, nexp, wht = make_field(rng)
    valid = np.ones(sci.shape, bool)
    radii = [1.0, 2.0, 4.0, 8.0]
    out = nc.sample_image(sci, np.ones_like(sci), {'wht': wht, 'nexp': nexp}, valid, radii, [12000] * 4, None, 1, rng)
    sqrtN = np.sqrt(np.pi) * np.array(radii)
    s = {i: {'f': out['f'][i], 'logp': out['logp']['wht'][i], 'lognexp': out['logp']['nexp'][i]} for i in range(4)}
    return s, sqrtN, float(np.std(out['pixels']))


def test_sampling_balances_the_cells(field):
    s, _, _ = field
    # every (NEXP level, weight) quadrant gets a fair share of apertures at every size
    for i in range(4):
        level = np.rint(np.exp(s[i]['lognexp'])); wpe = np.exp(s[i]['logp']) / level
        for ne in (1, 4):
            for w in (1.0, 4.0):
                frac = np.mean((level == ne) & (np.abs(wpe - w) < 0.5))
                assert 0.15 < frac < 0.35, (i, ne, w, frac)


def test_two_dimensional_strata_calibrate_every_quadrant(field):
    s, sqrtN, rms1 = field
    two_d = nc.fit_model(s, sqrtN, rms1, k=4, holdout_fraction=0.3, width_method='biweight', k_groups=2, k_per_group=2)
    assert len(two_d['groups']['log_centres']) == 2
    assert all(len(m['log_centres']) >= 2 for m in two_d['groups']['members'])
    # held out, per cell, largest aperture: within 10%
    for g, member in enumerate(two_d['groups']['members']):
        for c in range(len(member['log_centres'])):
            w = two_d['validation']['by_cell'][g][c][-1]
            assert w is not None and abs(w - 1) < 0.10, (g, c, w)
    # the 1-D strata of the total weight mix the top-right (NEXP 1, wht 4) and bottom-left (NEXP 4,
    # wht 4) quadrants: at least one of them is off by more than 15% with them
    f, logp, lognexp = s[3]['f'], s[3]['logp'], s[3]['lognexp']
    pred_1d = rms1 * nc.predict(two_d, np.full(len(f), sqrtN[-1]), logp)
    pred_2d = rms1 * nc.predict(two_d, np.full(len(f), sqrtN[-1]), logp, lognexp)
    level = np.rint(np.exp(lognexp)); wpe = np.exp(logp) / level
    worst_1d, worst_2d = 0.0, 0.0
    for ne in (1, 4):
        for w in (1.0, 4.0):
            sel = (level == ne) & (np.abs(wpe - w) < 0.5)
            worst_1d = max(worst_1d, abs(np.std(f[sel] / pred_1d[sel]) - 1))
            worst_2d = max(worst_2d, abs(np.std(f[sel] / pred_2d[sel]) - 1))
    assert worst_1d > 0.15
    assert worst_2d < 0.10


def test_predict_is_continuous_across_groups_and_falls_back():
    sqrtN = np.array([2.0, 4.0, 8.0, 16.0])
    base = nc.fit_curve(sqrtN, sqrtN)
    tilt = lambda a: {'base': base, 'tilt': [a, 0.0]}
    member = lambda a: {'log_centres': [0.0], 'curves': [tilt(a)], 'edges': []}
    model = dict(base, strata={'log_centres': [0.0, 1.0], 'curves': [tilt(0.0), tilt(0.5)], 'edges': [0.5]},
                 groups={'log_centres': [0.0, np.log(4.0)], 'members': [member(np.log(2.0)), member(np.log(1.0))], 'edges': [np.log(2.0)]})
    sq = np.full(3, 8.0)
    # between the group centres the prediction is the linear blend of the two groups
    p = nc.predict(model, sq, np.zeros(3), np.array([0.0, 0.5 * np.log(4.0), np.log(4.0)]))
    assert np.isclose(p[0], 16.0) and np.isclose(p[2], 8.0) and np.isclose(p[1], 12.0)
    # beyond the outer groups: clamped
    p_out = nc.predict(model, sq[:2], np.zeros(2), np.array([-3.0, 5.0]))
    assert np.isclose(p_out[0], 16.0) and np.isclose(p_out[1], 8.0)
    # no NEXP (or a non-finite one): the 1-D strata; no proxy at all: the pooled curve
    p_1d = nc.predict(model, sq, np.array([0.0, 0.5, 1.0]))
    assert np.allclose(nc.predict(model, sq, np.array([0.0, 0.5, 1.0]), np.array([np.nan] * 3)), p_1d)
    assert np.isclose(p_1d[1], 0.5 * (p_1d[0] + p_1d[2]))
    assert np.isclose(nc.predict(model, 8.0)[0], 8.0)


def test_group_edges_merge_small_groups_and_respect_k():
    rng = np.random.default_rng(2)
    nexp = np.concatenate([np.full(5000, 1.0), np.full(5000, 2.0), np.full(100, 3.0), np.full(5000, 7.0), np.full(5000, 12.0)])
    nexp += rng.normal(0, 0.05, len(nexp))     # fractional near edges
    lognexp = np.log(nexp)
    edges = nc.group_edges(lognexp, 5, min_count=300)
    counts = np.bincount(np.digitize(lognexp, edges), minlength=len(edges) + 1)
    assert counts.min() >= 300 and len(edges) + 1 == 4       # the 100-member level was merged
    assert len(nc.group_edges(lognexp, 2, min_count=300)) == 1
    assert nc.group_edges(np.zeros(10000), 5).size == 0        # one level: no groups
    assert nc.group_edges(lognexp, 1).size == 0


def test_fit_model_without_nexp_has_no_groups(field):
    s, sqrtN, rms1 = field
    one_d = nc.fit_model({i: {'f': s[i]['f'], 'logp': s[i]['logp']} for i in s}, sqrtN, rms1, k=4,
                         holdout_fraction=0.3, width_method='biweight', k_groups=5)
    assert one_d['groups']['log_centres'] == [] and one_d['validation']['by_group'] == []
    assert len(one_d['strata']['log_centres']) >= 2


def test_strata_edges_handle_a_discrete_proxy():
    rng = np.random.default_rng(4)
    # three exposure times, unequal shares, slightly blurred at the boundaries
    logp = np.concatenate([np.zeros(6000), np.full(3000, 1.0), np.full(3000, 2.0)]) + rng.normal(0, 1e-3, 12000)
    edges = nc.strata_edges(logp, 4, min_count=300)
    counts = np.bincount(np.digitize(logp, edges), minlength=len(edges) + 1)
    assert len(edges) == 2 and counts.min() >= 300
    assert np.all((edges > 0.1) & (edges < 1.9))


def test_validation_and_production_fits_share_the_structure(field):
    s, sqrtN, rms1 = field
    kw = dict(k=4, width_method='biweight', k_groups=2, k_per_group=2)
    with_holdout = nc.fit_model(s, sqrtN, rms1, holdout_fraction=0.3, **kw)
    production = nc.fit_model(s, sqrtN, rms1, holdout_fraction=0.0, **kw)
    assert with_holdout['strata']['edges'] == production['strata']['edges']
    assert with_holdout['groups']['edges'] == production['groups']['edges']
    assert [m['edges'] for m in with_holdout['groups']['members']] == [m['edges'] for m in production['groups']['members']]
    assert production['validation']['n_test'] == [0] * len(sqrtN)


def test_a_level_whose_apertures_thin_out_with_diameter_keeps_its_group(field):
    s, sqrtN, rms1 = field
    rng = np.random.default_rng(5)
    # single-exposure strips: plenty of small apertures, few large ones (5% at the two largest sizes)
    thinned = {}
    for i in s:
        level = np.rint(np.exp(s[i]['lognexp']))
        keep = (level != 1) | (rng.random(len(level)) < (0.05 if i >= 2 else 1.0))
        thinned[i] = {k: v[keep] for k, v in s[i].items()}
    assert (np.rint(np.exp(thinned[3]['lognexp'])) == 1).sum() < 500
    model = nc.fit_model(thinned, sqrtN, rms1, k=4, holdout_fraction=0.2, width_method='biweight',
                         k_groups=2, k_per_group=2, min_count_cell=500)
    assert len(model['groups']['log_centres']) == 2
    assert np.isclose(np.exp(model['groups']['log_centres'][0]), 1.0, atol=0.1)
    # the single-exposure group is still calibrated where it has held-out apertures
    for c in range(len(model['groups']['members'][0]['log_centres'])):
        w = model['validation']['by_cell'][0][c][0]
        assert w is not None and abs(w - 1) < 0.10, (c, w)
