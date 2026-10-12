"""Stratified empirical noise curves from random apertures.

The error of an aperture sum is modelled as

    sigma(N, x) = sigma1 * c_s(x)(sqrt N) * S(x)

where S(x) is a per-pixel *scale* map (the rms map, 1/sqrt(wht), or an empirical noise map
measured from the source-masked image), sqrt N is the square root of the number of pixels
summed, and c_s is an empirical curve (robust width of random-aperture sums divided by S) fitted
separately in K *strata* of a depth proxy p(x) (the weight map, used only to order pixels by
depth, or the empirical noise map). Between strata the curves are interpolated linearly in
log p. With K = 1 this is exactly the single-curve ("legacy") model; with K ~ 8 plus tail bins
it captures the change of the correlated-noise fraction with depth that a single curve cannot
(review finding F06; see noise_model_design.md).

This module holds the pure numerical parts (sampling one tile, fitting, predicting, validating)
so that they can be unit-tested; the orchestration over tiles and filters and the file formats
live in aperate.commands.postprocess.
"""

from typing import Dict, List, Optional

import numpy as np
import sep
from astropy.io import fits
from astropy.stats import biweight_scale
from scipy.interpolate import BSpline, make_splrep
from scipy.optimize import curve_fit

TAIL_PERCENTILES = (96.0, 98.0, 99.0, 99.5)


# ---------------------------------------------------------------------------------------------
# curves in sqrt(N)

def fit_curve(sqrtN: np.ndarray, ratio: np.ndarray, ratio_err: Optional[np.ndarray] = None) -> dict:
    """Fit the aperture curve sigma_N/sigma_1 versus sqrt(N).

    The curve is a cubic polynomial in log-log space through (1, 1) (white noise is the line
    ratio = sqrt N; correlated noise bends it up smoothly), fitted with inverse-variance weights
    after one pass of outlier rejection, and continued beyond the fitted range with the local
    power-law slope. Four parameters are enough for the measured curves and, unlike a smoothing
    spline, do not oscillate when the points are noisy (few apertures per stratum). The legacy
    'spline' and 'plaw' entries are also stored so that older readers of the coefficient files
    keep working.
    """
    sqrtN = np.asarray(sqrtN, float); ratio = np.asarray(ratio, float)
    ok = np.isfinite(ratio) & (ratio > 0) & (sqrtN > 0)
    sqrtN, ratio = sqrtN[ok], ratio[ok]
    if ratio_err is None:
        ratio_err = 0.02 * ratio
    else:
        ratio_err = np.asarray(ratio_err, float)[ok]
        ratio_err = np.where(np.isfinite(ratio_err) & (ratio_err > 0), ratio_err, 0.02 * ratio)

    def plaw(x, alpha, beta):
        return alpha * np.power(x, beta)
    popt, _ = curve_fit(plaw, sqrtN, ratio, p0=[1, 1], maxfev=int(1e5))

    # log-log cubic through (1, 1): log r = a1 L + a2 L^2 + a3 L^3,  L = log sqrtN
    L = np.log(sqrtN); y = np.log(ratio); w = ratio / ratio_err          # d(log r) = dr / r
    keep = np.ones(len(L), bool)
    coef = np.array([1.0, 0.0, 0.0])
    for _ in range(2):
        A = np.vstack([L[keep], L[keep] ** 2, L[keep] ** 3]).T * w[keep, None]
        coef = np.linalg.lstsq(A, y[keep] * w[keep], rcond=None)[0]
        resid = y - (coef[0] * L + coef[1] * L ** 2 + coef[2] * L ** 3)
        keep = np.abs(resid) < np.maximum(3 / w, 0.25)                   # 3 sigma or 25%
        if keep.sum() < 4:
            keep = np.ones(len(L), bool); break
    Lmax = float(np.max(L))
    slope_max = float(coef[0] + 2 * coef[1] * Lmax + 3 * coef[2] * Lmax ** 2)

    x_all = np.append([0, 1], sqrtN); y_all = np.append([0, 1], ratio)
    w_all = np.append([500, 100], 1 / ratio_err)
    bspline = make_splrep(x=x_all, y=y_all, w=w_all, k=3, s=len(x_all))
    return {
        'loglog': {'coef': [float(c) for c in coef], 'Lmax': Lmax, 'slope_max': slope_max},
        'spline': {'c': [float(v) for v in bspline.c], 't': [float(v) for v in bspline.t], 'k': int(bspline.k)},
        'plaw': {'alpha': float(popt[0]), 'beta': float(popt[1])},
        'sqrtN_max': float(np.max(sqrtN)),
    }


def evaluate_curve(curve: dict, sqrtN) -> np.ndarray:
    """sigma_N/sigma_1 at sqrtN for one fitted curve."""
    sqrtN = np.asarray(sqrtN, float)
    if 'tilt' in curve:
        # stratum curve = base (pooled) curve x exp(a + b log sqrtN): see fit_model
        a, b = curve['tilt']
        with np.errstate(divide='ignore'):
            L = np.log(np.maximum(sqrtN, 1e-9))
        return evaluate_curve(curve['base'], sqrtN) * np.exp(a + b * L)
    if 'loglog' in curve:
        c = curve['loglog']['coef']; Lmax = curve['loglog']['Lmax']
        with np.errstate(divide='ignore'):
            L = np.log(np.maximum(sqrtN, 1e-9))
        inside = c[0] * L + c[1] * L ** 2 + c[2] * L ** 3
        at_max = c[0] * Lmax + c[1] * Lmax ** 2 + c[2] * Lmax ** 3
        outside = at_max + curve['loglog']['slope_max'] * (L - Lmax)
        return np.where(L <= Lmax, np.exp(inside), np.exp(outside))
    # coefficient files written by the single-curve code: spline inside, power law outside
    bspline = BSpline(c=curve['spline']['c'], t=curve['spline']['t'], k=curve['spline']['k'])
    plaw = curve['plaw']['alpha'] * np.power(sqrtN, curve['plaw']['beta'])
    return np.where(sqrtN < 0.85 * curve['sqrtN_max'], bspline(sqrtN), plaw)


def fit_tilt(base: dict, sqrtN: np.ndarray, ratio: np.ndarray, ratio_err: Optional[np.ndarray] = None) -> dict:
    """A stratum's curve as the base (pooled) curve times exp(a + b log sqrtN).

    The difference between depth strata is a slowly varying factor (e.g. 1.15-1.25 in the deep
    tail of B1), so two parameters per stratum are enough and are far less noisy than an
    independent curve when a stratum holds only ~1000 apertures. Weighted least squares with
    one pass of outlier rejection; the tilt is pinned to zero where the stratum has no data.
    """
    sqrtN = np.asarray(sqrtN, float); ratio = np.asarray(ratio, float)
    base_vals = evaluate_curve(base, sqrtN)
    ok = np.isfinite(ratio) & (ratio > 0) & (base_vals > 0) & (sqrtN > 0)
    if ratio_err is None:
        ratio_err = 0.03 * ratio
    ratio_err = np.asarray(ratio_err, float)
    L = np.log(sqrtN[ok]); y = np.log(ratio[ok] / base_vals[ok])
    w = np.where(np.isfinite(ratio_err[ok]) & (ratio_err[ok] > 0), ratio[ok] / ratio_err[ok], 1.0 / 0.03)
    if ok.sum() < 3:
        return {'base': {k: base[k] for k in ('loglog', 'spline', 'plaw', 'sqrtN_max') if k in base}, 'tilt': [0.0, 0.0]}
    keep = np.ones(len(L), bool)
    coef = np.array([0.0, 0.0])
    for _ in range(2):
        A = np.vstack([np.ones(keep.sum()), L[keep]]).T * w[keep, None]
        coef = np.linalg.lstsq(A, y[keep] * w[keep], rcond=None)[0]
        resid = y - (coef[0] + coef[1] * L)
        keep = np.abs(resid) < np.maximum(3 / w, 0.15)
        if keep.sum() < 3:
            keep = np.ones(len(L), bool); break
    return {'base': {k: base[k] for k in ('loglog', 'spline', 'plaw', 'sqrtN_max') if k in base},
            'tilt': [float(coef[0]), float(coef[1])]}


# ---------------------------------------------------------------------------------------------
# strata

def strata_edges(logp: np.ndarray, k: int, min_count: int = 300, tail_percentiles=TAIL_PERCENTILES) -> np.ndarray:
    """Bin edges in log(proxy): k quantile bins plus tail bins at the given upper percentiles.
    Edges that would leave a bin with fewer than min_count members are removed."""
    logp = np.asarray(logp, float)
    logp = logp[np.isfinite(logp)]
    if k <= 1 or len(logp) < 2 * min_count:
        return np.array([])
    edges = list(np.percentile(logp, np.linspace(0, 100, k + 1)[1:-1]))
    if tail_percentiles:
        edges += [e for e in np.percentile(logp, tail_percentiles) if e > edges[-1]]
    # distinct edges strictly inside the range (an edge at the maximum would make an empty bin)
    edges = sorted({float(e) for e in edges if logp.min() < e < logp.max()})
    while edges:
        counts = np.bincount(np.digitize(logp, edges), minlength=len(edges) + 1)
        if counts.min() >= min_count:
            break
        i = int(np.argmin(counts))
        edges.pop(min(i, len(edges) - 1) if i > 0 else 0)
    return np.array(edges)


def robust_width(x: np.ndarray, method: str = 'negative_tail') -> float:
    """Width of a sample of aperture sums: Gaussian fit to the negative tail (robust against
    undetected sources; the legacy estimator) or the biweight scale."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return np.nan
    if method == 'biweight':
        return float(biweight_scale(x))
    return negative_tail_width(x)


def negative_tail_width(x: np.ndarray, maxiters: int = 4) -> float:
    """Gaussian width from the negative side of the distribution (undetected sources only add a
    positive tail). A Gaussian is fitted to the histogram below median + 1 sigma, with a bin
    count adapted to the sample size; falls back to the scaled MAD of the lower half."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    med = np.median(x); std = 1.4826 * np.median(np.abs(x - med))
    if not np.isfinite(std) or std <= 0:
        return float(np.std(x)) if len(x) else np.nan

    def gauss(t, A, mu, sigma):
        return A * np.exp(-0.5 * (t - mu) ** 2 / sigma ** 2)
    nbins = int(np.clip(len(x) // 40, 15, 100))
    for _ in range(maxiters):
        lo, hi = med - 4 * std, med + 1.0 * std
        counts, edges = np.histogram(x[(x > lo) & (x < hi)], bins=nbins, range=(lo, hi))
        centres = 0.5 * (edges[1:] + edges[:-1])
        if counts.max() <= 0:
            break
        try:
            popt, _ = curve_fit(gauss, centres, counts / counts.max(), p0=[1, med, std], maxfev=2000)
            new_med, new_std = float(popt[1]), float(abs(popt[2]))
        except Exception:
            break
        if not np.isfinite(new_std) or new_std <= 0 or new_std > 3 * std:
            break
        converged = abs(new_std - std) < 0.01 * std
        med, std = new_med, new_std
        if converged:
            break
    return float(std)


# ---------------------------------------------------------------------------------------------
# model

def fit_model(samples: Dict[int, dict], sqrtN: np.ndarray, rms1: float, k: int,
              holdout_fraction: float = 0.2, min_count: int = 200, width_method: str = 'negative_tail',
              rng: Optional[np.random.Generator] = None) -> dict:
    """Fit the stratified curves.

    samples[i] = {'f': normalised aperture sums (sum / S) for diameter index i, 'logp': log proxy at
    the aperture}. rms1 is the single-pixel width of (image / S) (= 1 if S is the true pixel rms).
    Returns the model dictionary (JSON/pickle friendly); the held-out apertures (holdout_fraction
    of each sample) give the validation table: robust width of f / prediction per stratum and
    diameter, 1 for a calibrated model.
    """
    rng = rng or np.random.default_rng(0)
    sqrtN = np.asarray(sqrtN, float)
    train, test = {}, {}
    for i, s in samples.items():
        f = np.asarray(s['f'], float); logp = np.asarray(s['logp'], float)
        ok = np.isfinite(f) & np.isfinite(logp)
        f, logp = f[ok], logp[ok]
        is_test = rng.random(len(f)) < holdout_fraction
        train[i] = (f[~is_test], logp[~is_test]); test[i] = (f[is_test], logp[is_test])

    # pooled (single-curve) fit: the legacy model, always stored
    pooled = np.array([robust_width(train[i][0], width_method) / rms1 for i in range(len(sqrtN))])
    pooled_err = np.array([pooled[i] / np.sqrt(2 * max(len(train[i][0]), 1)) for i in range(len(sqrtN))])
    model = {'sqrtN': sqrtN, 'rmsN/rms1': pooled, 'rmsN_err/rms1': pooled_err, 'k': int(k)}
    model.update(fit_curve(sqrtN, pooled, pooled_err))

    # strata on the diameter with the fewest training apertures (so every stratum is populated at
    # every diameter)
    i_min = min(train, key=lambda i: len(train[i][0]))
    edges = strata_edges(train[i_min][1], k, min_count=min_count)
    strata = {'edges': edges.tolist(), 'log_centres': [], 'curves': [], 'rmsN/rms1': [], 'n': []}
    n_strata = len(edges) + 1
    if n_strata > 1:
        all_logp = np.concatenate([train[i][1] for i in train])
        all_bin = np.digitize(all_logp, edges)
        for s in range(n_strata):
            strata['log_centres'].append(float(np.median(all_logp[all_bin == s])))
            widths, errs, ns = [], [], []
            for i in range(len(sqrtN)):
                f, logp = train[i]
                sel = np.digitize(logp, edges) == s
                w = robust_width(f[sel], width_method) / rms1
                widths.append(w); errs.append(w / np.sqrt(2 * max(sel.sum(), 1))); ns.append(int(sel.sum()))
            strata['rmsN/rms1'].append(widths); strata['n'].append(ns)
            strata['curves'].append(fit_tilt(model, sqrtN, np.array(widths), np.array(errs)))
    model['strata'] = strata

    # validation on the held-out apertures
    val = {'overall': [], 'by_stratum': [[None] * len(sqrtN) for _ in range(n_strata)],
           'overall_single_curve': [], 'n_test': []}
    for i in range(len(sqrtN)):
        f, logp = test[i]
        if len(f) < 50:
            val['overall'].append(None); val['overall_single_curve'].append(None); val['n_test'].append(int(len(f)))
            continue
        pred = rms1 * predict(model, np.full(len(f), sqrtN[i]), logp)
        pred1 = rms1 * evaluate_curve(model, np.full(len(f), sqrtN[i]))
        # same width estimator as the fit, so that undetected sources bias both the same way
        val['overall'].append(robust_width(f / pred, width_method))
        val['overall_single_curve'].append(robust_width(f / pred1, width_method))
        val['n_test'].append(int(len(f)))
        if n_strata > 1:
            b = np.digitize(logp, edges)
            for s in range(n_strata):
                sel = b == s
                if sel.sum() >= 50:
                    val['by_stratum'][s][i] = robust_width(f[sel] / pred[sel], width_method)
    model['validation'] = val
    return model


def predict(model: dict, sqrtN, logp=None) -> np.ndarray:
    """sigma_N/sigma_1 for apertures of sqrt(N) = sqrtN at depth proxy log p (arrays of the same
    length, or scalar logp). Without strata, or without logp, the pooled curve is used."""
    sqrtN = np.atleast_1d(np.asarray(sqrtN, float))
    strata = model.get('strata') or {}
    centres = np.asarray(strata.get('log_centres', []), float)
    if logp is None or len(centres) < 2:
        return evaluate_curve(model, sqrtN)
    logp = np.broadcast_to(np.asarray(logp, float), sqrtN.shape)
    values = np.stack([evaluate_curve(c, sqrtN) for c in strata['curves']])     # (S, n)
    pos = np.interp(logp, centres, np.arange(len(centres)))                   # fractional stratum index
    pos = np.where(np.isfinite(pos), pos, 0.0)
    lo = np.clip(np.floor(pos).astype(int), 0, len(centres) - 1)
    hi = np.clip(lo + 1, 0, len(centres) - 1)
    w = pos - lo
    idx = np.arange(len(sqrtN))
    return (1 - w) * values[lo, idx] + w * values[hi, idx]


# ---------------------------------------------------------------------------------------------
# sampling one image

def empirical_noise_map(image: np.ndarray, bad: np.ndarray, box: int = 256) -> np.ndarray:
    """Robust pixel-noise map of the source-masked image (sep.Background in box x box cells)."""
    data = np.ascontiguousarray(np.where(bad, 0, image).astype(np.float32))
    bkg = sep.Background(data, mask=np.ascontiguousarray(bad), bw=box, bh=box, fw=3, fh=3)
    return np.ascontiguousarray(bkg.rms().astype(np.float32))


def local_mean(image: np.ndarray, x, y, r: float) -> np.ndarray:
    """Mean of `image` inside circles of radius r (pixels); invalid pixels must already be 0."""
    total, _, _ = sep.sum_circle(image, x, y, r, subpix=0)
    npix = np.pi * r * r
    return total / npix


def sample_image(sci: np.ndarray, scale: np.ndarray, proxies: Dict[str, np.ndarray], valid: np.ndarray,
                 radii_pix: List[float], n_apertures: List[int], source_mask: Optional[np.ndarray] = None,
                 mask_dilate: int = 3, rng: Optional[np.random.Generator] = None,
                 n_pixel_sample: int = 50000, balance_bins: int = 12) -> dict:
    """Random-aperture sums on one image.

    sci, scale and the proxy maps: native-order float32 arrays of the same shape; valid: bool
    array of usable pixels; source_mask: bool array of pixels to avoid (segmentation map > 0) or
    None. Returns {'f': [sums / scale per radius], 'logp': {proxy name: [log proxy per radius]},
    'pixels': sample of sci/scale on valid pixels for the single-pixel width, 'x', 'y'}.

    balance_bins > 0: the apertures are drawn so that `balance_bins` equal-width bins of the log
    of the first proxy map get (as far as the image allows) the same number of apertures, i.e.
    rare depths (the deep tail, where the correlated noise differs most) are oversampled instead
    of being represented by a few per cent of area-weighted apertures.
    """
    rng = rng or np.random.default_rng()
    ny, nx = sci.shape
    with np.errstate(divide='ignore', invalid='ignore'):
        nsci = np.where(valid, sci / scale, 0).astype(np.float32)
    nsci = np.ascontiguousarray(nsci)
    bad = ~valid if source_mask is None else (~valid | source_mask)
    badf = np.ascontiguousarray(bad.astype(np.float32))
    proxies0 = {name: np.ascontiguousarray(np.where(valid, p, 0).astype(np.float32)) for name, p in proxies.items()}
    n_valid = int(valid.sum())
    frac_ok = max(n_valid / valid.size, 0.01)

    # pixel sample for the single-pixel width
    iy = rng.integers(0, ny, 4 * n_pixel_sample); ix = rng.integers(0, nx, 4 * n_pixel_sample)
    ok = valid[iy, ix]
    pixels = nsci[iy[ok], ix[ok]][:n_pixel_sample]

    out = {'f': [], 'logp': {name: [] for name in proxies0}, 'x': [], 'y': [], 'pixels': pixels, 'n_valid': n_valid}
    first_proxy = next(iter(proxies0)) if proxies0 else None
    for r, nap in zip(radii_pix, n_apertures):
        xs, ys = [], []
        want = nap * (4 if balance_bins and first_proxy else 1)    # candidates for the depth balancing
        n_draw = int(want / frac_ok) + 1000
        for _ in range(30):
            xd = rng.uniform(r + mask_dilate + 1, nx - r - mask_dilate - 2, n_draw)
            yd = rng.uniform(r + mask_dilate + 1, ny - r - mask_dilate - 2, n_draw)
            # keep apertures whose dilated footprint contains no invalid/source pixel
            nbad, _, _ = sep.sum_circle(badf, xd, yd, r + mask_dilate, subpix=0)
            keep = nbad < 0.5
            xs.append(xd[keep]); ys.append(yd[keep])
            if sum(len(a) for a in xs) >= want:
                break
        x = np.concatenate(xs)[:want]; y = np.concatenate(ys)[:want]
        if balance_bins and first_proxy and len(x) > nap:
            p = local_mean(proxies0[first_proxy], x, y, max(r, 5.0))
            with np.errstate(divide='ignore', invalid='ignore'):
                lp = np.where(p > 0, np.log(p), np.nan)
            lp = np.where(np.isfinite(lp), lp, np.nanmedian(lp))
            lo, hi = np.percentile(lp, [0.5, 99.5])
            b = np.clip(((lp - lo) / max(hi - lo, 1e-9) * balance_bins).astype(int), 0, balance_bins - 1)
            target = int(np.ceil(nap / balance_bins))
            chosen = []
            for k in range(balance_bins):
                members = np.flatnonzero(b == k)
                chosen.append(rng.choice(members, min(len(members), target), replace=False))
            chosen = np.concatenate(chosen)
            if len(chosen) < nap:        # fill up from the remaining candidates
                rest = np.setdiff1d(np.arange(len(x)), chosen)
                chosen = np.concatenate([chosen, rng.choice(rest, min(len(rest), nap - len(chosen)), replace=False)])
            x, y = x[chosen[:nap]], y[chosen[:nap]]
        f, _, _ = sep.sum_circle(nsci, x, y, r, subpix=5)
        out['f'].append(f); out['x'].append(x); out['y'].append(y)
        for name, p0 in proxies0.items():
            p = local_mean(p0, x, y, max(r, 5.0))
            with np.errstate(divide='ignore', invalid='ignore'):
                out['logp'][name].append(np.where(p > 0, np.log(p), np.nan))
    return out


def weight_map_check(wht: np.ndarray, emp: np.ndarray, valid: np.ndarray, rng=None, n: int = 20000,
                     radius: float = 128.0) -> dict:
    """Does the weight map track the noise? On random valid positions the empirical noise
    log(emp) is regressed on log(1/sqrt(wht)), both averaged over circles of `radius` pixels
    (the resolution of the empirical map, so that the two are compared at the same scale): a
    map that tracks the noise gives a slope near 1 (a noise floor lowers it), an uninformative
    map a slope near 0. Also reports the Spearman correlation and the robust scatter (dex) of
    the log ratio. 'ok' = slope within [0.5, 1.5]."""
    from scipy.stats import spearmanr
    rng = rng or np.random.default_rng(1)
    ny, nx = wht.shape
    m = int(min(radius, nx // 4, ny // 4))
    iy = rng.integers(m, ny - m, n); ix = rng.integers(m, nx - m, n)
    ok = valid[iy, ix] & (wht[iy, ix] > 0) & (emp[iy, ix] > 0)
    iy, ix = iy[ok], ix[ok]
    w = np.ascontiguousarray(np.where(valid & (wht > 0), wht, 0).astype(np.float32))
    e = np.ascontiguousarray(np.where(valid, emp, 0).astype(np.float32))
    wm = local_mean(w, ix.astype(float), iy.astype(float), float(m))
    em = local_mean(e, ix.astype(float), iy.astype(float), float(m))
    good = (wm > 0) & (em > 0)
    lw = -0.5 * np.log10(wm[good]); le = np.log10(em[good])
    if good.sum() < 100:
        return {'n': int(good.sum()), 'spearman': np.nan, 'scatter_dex': np.nan, 'ok': False}
    rho = float(spearmanr(lw, le).correlation)
    scatter = float(biweight_scale(le - lw))
    # slope by least squares with one round of 3-sigma clipping
    keep = np.ones(len(lw), bool)
    slope = 0.0
    for _ in range(2):
        A = np.vstack([lw[keep], np.ones(keep.sum())]).T
        slope, icpt = np.linalg.lstsq(A, le[keep], rcond=None)[0]
        resid = le - (slope * lw + icpt)
        keep = np.abs(resid) < 3 * max(biweight_scale(resid), 1e-6)
    return {'n': int(good.sum()), 'spearman': rho, 'scatter_dex': scatter, 'slope': float(slope),
            'ok': bool(0.5 < slope < 1.5)}
