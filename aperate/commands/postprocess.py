"""Postprocessing: merge tiles, random aperture calibration, and PSF corrections."""

import click
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import glob, warnings, tqdm

import numpy as np
from astropy.table import Table, vstack
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord
from scipy.interpolate import griddata, BSpline, make_splrep

from ..core.catalog import find_catalog_in_directory, AperateCatalog
from ..core.logging import get_logger
from ..config.parser import load_config
from ..config.images import load_images_config
from ..config.psfs import load_psfs_config
from ..config.schema import PostprocessConfig, AperateConfig
from ..utils.helpers import get_unit_conversion

from .detect import fit_pixel_dist

import json
import pickle
import sep
from scipy.optimize import curve_fit
from photutils.aperture import EllipticalAperture, CircularAperture, aperture_photometry
from astropy.wcs import WCS, FITSFixedWarning
warnings.simplefilter('ignore', category=FITSFixedWarning)

def distance_to_nearest_edge(x, y, image_width, image_height):
    """
    Compute the distance to the nearest edge for a point (x, y) in an image.

    Parameters:
    x (float): X-coordinate of the point.
    y (float): Y-coordinate of the point.
    image_width (int): Width of the image.
    image_height (int): Height of the image.

    Returns:
    float: Distance to the nearest edge.
    """
    left_edge_dist = x
    right_edge_dist = image_width - x
    top_edge_dist = y
    bottom_edge_dist = image_height - y

    if np.ndim(x)==1:
        return np.min([left_edge_dist, right_edge_dist, top_edge_dist, bottom_edge_dist], axis=0)
    else:
        return min(left_edge_dist, right_edge_dist, top_edge_dist, bottom_edge_dist)


def merge_tile_catalogs(
    project_dir: Path,
    project_name: str,
    tiles: List[str],
    matching_radius: float,
    edge_mask: int,
    output_path: Path,
    overwrite: bool
) -> Table:
    """
    Merge individual tile catalogs into a master catalog.
    
    Args:
        project_dir: Project directory
        project_name: Project name
        tiles: List of tile names to merge
        matching_radius: Radius for matching overlapping sources (arcsec)
        edge_mask: Edge mask size to exclude sources near tile boundaries (pixels)
        output_path: Path for output master catalog
        
    Returns:
        Merged catalog table
        
    Implementation notes for user:
    - Load individual tile catalogs from: project_dir / "catalogs" / f"catalog_{project_name}_{tile}.fits"
    - For each source, check if it's within edge_mask pixels of tile boundary
    - Match sources between overlapping tiles using RA/Dec and matching_radius
    - For matched sources, combine measurements (weighted average, error propagation)
    - Handle duplicate source removal
    - Add master catalog metadata (tile provenance, merge statistics)
    - Write final catalog to output_path
    """
    
    logger = get_logger()
    logger.info(f"Merging {len(tiles)} tile catalogs into master catalog")
    images_config = load_images_config(project_dir)
    tiles = sorted(tiles)
    if output_path is None:
        output_path = project_dir / "catalogs" / f"catalog_{project_name}_merged.fits"

    if output_path.exists() and not overwrite:
        logger.info(f"Output catalog {output_path} already exists, skipping merge...")
        return Table.read(output_path)

    if len(tiles) == 0:
        logger.error('No catalogs to merge!')
        return None
        
    if len(tiles) == 1:
        tile = tiles[0]
        logger.info(f'Only one catalog found (tile {tile}), skipping merge...')
        catalog_path = project_dir / "catalogs" / f"catalog_{project_name}_{tile}.fits"
        return Table.read(catalog_path)
    
    # Start with first catalog
    tile0 = tiles[0]
    catalog0_path = project_dir / "catalogs" / f"catalog_{project_name}_{tile0}.fits"
    catalog0 = Table.read(catalog0_path)
    catalog0['tile'] = tile0
    catalog0['id_tile'] = catalog0['id']

    # Remove objects too close to tile edges
    detection_image_path = images_config.get_tile_detection_image_path(tile0)
    with fits.open(detection_image_path) as detec_hdul:
        height0, width0 = np.shape(detec_hdul[1].data)

    catalog0_distance_to_edge = distance_to_nearest_edge(catalog0['x'], catalog0['y'], width0, height0)
    catalog0 = catalog0[catalog0_distance_to_edge > edge_mask]
    
    logger.info(f'{tile0}: {len(catalog0)} objects')
    coords0 = SkyCoord(catalog0['ra'], catalog0['dec'], unit='deg')

    # Merge with remaining catalogs
    for tile1 in tiles[1:]:
        catalog1_path = project_dir / "catalogs" / f"catalog_{project_name}_{tile1}.fits"
        catalog1 = Table.read(catalog1_path)
        catalog1['tile'] = tile1
        catalog1['id_tile'] = catalog1['id']

        # Remove objects too close to tile edges
        detection_image_path = images_config.get_tile_detection_image_path(tile1)
        with fits.open(detection_image_path) as detec_hdul:
            height1, width1 = np.shape(detec_hdul[1].data)

        catalog1_distance_to_edge = distance_to_nearest_edge(catalog1['x'], catalog1['y'], width1, height1)
        catalog1 = catalog1[catalog1_distance_to_edge > edge_mask]
        
        logger.info(f'{tile1}: {len(catalog1)} objects')
        coords1 = SkyCoord(catalog1['ra'], catalog1['dec'], unit='deg')

        # Perform source matching
        idx, d2d, d3d = coords0.match_to_catalog_sky(coords1)
        match = d2d < matching_radius*u.arcsec
        
        catalog0_unique = catalog0[~match]
        catalog0_matched = catalog0[match]
        catalog1_matched = catalog1[idx[match]]
        unique = np.ones(len(catalog1), dtype=bool)
        unique[idx[match]] = False
        catalog1_unique = catalog1[unique]

        # Handle matched sources - keep the one farther from tile edge
        if len(catalog0_matched) > 0:
            catalog0_distance_to_edge = distance_to_nearest_edge(catalog0_matched['x'], catalog0_matched['y'], width0, height0)
            catalog1_distance_to_edge = distance_to_nearest_edge(catalog1_matched['x'], catalog1_matched['y'], width1, height1)
            which = np.argmax([catalog0_distance_to_edge, catalog1_distance_to_edge], axis=0)
            
            # Combine all sources
            catalog_merged = vstack([catalog0_unique, catalog0_matched[which==0], catalog1_matched[which==1], catalog1_unique], join_type='outer')
        else:
            # No matches, just combine unique sources
            catalog_merged = vstack([catalog0_unique, catalog1_unique], join_type='outer')

        # Update for next iteration
        tile0 = tile1
        catalog0 = catalog_merged
        coords0 = SkyCoord(catalog0['ra'], catalog0['dec'], unit='deg')
        height0, width0 = height1, width1

    logger.info(f'Final merged catalog: {len(catalog_merged)} objects')
    
    # Reassign IDs
    catalog_merged['id'] = np.arange(len(catalog_merged)).astype(int)
    
    catalog_merged.write(output_path, overwrite=True)
    
    return catalog_merged


def _random_aperture_products(output_dir: Path, filt: str, homogenized: bool):
    suffix = '_psfhom' if homogenized else ''
    base = output_dir / f'random_apertures_{filt}{suffix}'
    return (base.with_name(base.name + '_coeffs.pickle'), base.with_name(base.name + '_validation.json'),
            base.with_name(base.name + '_empirical.npz'), base.with_name(base.name + '_coeffs.png'))


def _resolve_noise_maps(image_files, depth_proxy: str, scale_map: str):
    """Which maps provide the scale S(x) and the depth proxy p(x) for this filter's images."""
    has = {e: bool(image_files.has_extension(e)) for e in ('rms', 'wht')}
    scale = scale_map
    if scale == 'auto':
        scale = 'rms' if has['rms'] else ('wht' if has['wht'] else 'empirical')
    elif scale in has and not has[scale]:
        raise ValueError(f"random_apertures.scale_map = '{scale}' but the images have no '{scale}' extension")
    proxy = depth_proxy
    if proxy == 'auto':
        proxy = 'wht' if has['wht'] else ('rms' if has['rms'] else 'empirical')
    elif proxy in has and not has[proxy]:
        raise ValueError(f"random_apertures.depth_proxy = '{proxy}' but the images have no '{proxy}' extension")
    return scale, proxy


def _sample_tile_task(args):
    """Worker: random apertures on one tile for one filter (native or PSF-homogenized image).

    Returns the normalised sums per radius, the log depth proxies (one array per available proxy
    map) and a pixel sample; see aperate.core.noise_curves.sample_image.
    """
    (sci_path, rms_path, wht_path, segmap_path, scale_kind, proxy_kind, check_wht, diameters, pixel_scale,
     density, min_density, mask_dilate, seed, source_positions, balance_bins) = args
    from ..core import noise_curves as nc

    def load(path):
        with fits.open(path) as hdul:
            return np.ascontiguousarray(hdul[0].data.astype(np.float32))

    sci = load(sci_path)
    valid = np.isfinite(sci)
    maps = {}
    if rms_path is not None and (scale_kind == 'rms' or proxy_kind == 'rms'):
        maps['rms'] = load(rms_path)
        valid &= np.isfinite(maps['rms']) & (maps['rms'] > 0)
    if wht_path is not None and (scale_kind == 'wht' or proxy_kind == 'wht' or check_wht):
        maps['wht'] = load(wht_path)
        valid &= np.isfinite(maps['wht']) & (maps['wht'] > 0)
    source_mask = None
    if segmap_path is not None:
        source_mask = np.asarray(fits.getdata(segmap_path)) > 0
    emp = None
    if scale_kind == 'empirical' or proxy_kind == 'empirical' or (check_wht and 'wht' in maps):
        bad = ~valid if source_mask is None else (~valid | source_mask)
        emp = nc.empirical_noise_map(sci, bad)
        valid &= np.isfinite(emp) & (emp > 0)
        maps['empirical'] = emp
    if scale_kind == 'wht':
        with np.errstate(divide='ignore', invalid='ignore'):
            scale = np.where(valid, 1.0 / np.sqrt(maps['wht']), 0).astype(np.float32)
    else:
        scale = maps[scale_kind]
    proxies = {k: maps[k] for k in ('wht', 'rms', 'empirical') if k in maps}

    n_valid = int(valid.sum())
    area = n_valid * pixel_scale ** 2 / 3600       # arcmin^2
    # the same number of apertures at every diameter (sep makes large apertures cheap); the
    # strata need a few hundred apertures each at the largest diameter too
    nap = max(int(density * area), int(min_density * area), 0)
    radii_pix = [d / 2 / pixel_scale for d in diameters]
    n_apertures = [nap] * len(diameters)

    rng = np.random.default_rng(seed)
    # the proxy to balance on is the first map in the dict: order it first
    proxies = {k: proxies[k] for k in ([proxy_kind] if proxy_kind in proxies else []) + [k for k in proxies if k != proxy_kind]}
    out = nc.sample_image(sci, scale, proxies, valid, radii_pix, n_apertures, source_mask, mask_dilate, rng,
                          balance_bins=balance_bins)
    out['area_arcmin2'] = area
    out['wht_check'] = (nc.weight_map_check(maps['wht'], emp, valid, rng)
                        if (check_wht and 'wht' in maps and emp is not None) else None)
    if source_positions is not None and emp is not None:
        from .photometry import _local_median
        ids, x, y = source_positions
        out['empirical_at_sources'] = (np.asarray(ids), _local_median(emp, x, y))
    return out


def measure_random_apertures(
    images_config,
    psfs_config,
    tiles: List[str],
    filters: List[str],
    min_radius: float,
    max_radius: float,
    num_radii: int,
    num_apertures_per_sq_arcmin: int,
    output_dir: Path,
    overwrite: bool = False,
    min_num_apertures_per_sq_arcmin: int = 30,
    depth_strata: int = 8,
    depth_proxy: str = 'auto',
    scale_map: str = 'auto',
    mask_sources: bool = True,
    mask_dilate: int = 3,
    holdout_fraction: float = 0.2,
    check_weight_map: bool = True,
    parallel: int = 1,
    project_dir: Optional[Path] = None,
    project_name: Optional[str] = None,
) -> None:
    """
    Measure random apertures and fit the stratified noise curves for every filter.

    For each filter (and for its PSF-homogenized image when one is used for photometry) the
    apertures of all tiles are pooled and `aperate.core.noise_curves.fit_model` fits one curve
    sigma_N/sigma_1 versus sqrt(N) per depth stratum (depth_strata quantile bins of the depth
    proxy plus tail bins; depth_strata = 1 is the single-curve model). Apertures avoid the
    segmentation map (mask_sources) and invalid pixels. 20% of the apertures (holdout_fraction)
    are kept back to report the calibration of the fitted model per stratum and aperture size
    (random_apertures_<filter>[_psfhom]_validation.json and the check plot).

    Maps: scale_map = rms map | 1/sqrt(wht) | empirical noise map (sep.Background of the masked
    image); depth_proxy = wht | rms | empirical; 'auto' takes the first available. When
    check_weight_map is set and the weight map is the proxy, it is compared with the empirical
    noise map on the fly and replaced by it if it does not order pixels by depth.

    Output: random_apertures_<filter>[_psfhom]_coeffs.pickle (model dict, readable by
    _get_random_aperture_curve), the validation JSON, a PNG check plot, and, when the empirical
    map is used, random_apertures_<filter>[_psfhom]_empirical.npz with the map value at every
    catalogued source (tile, id, value) for calibrate_flux_uncertainties.
    """
    logger = get_logger()
    from ..core import noise_curves as nc
    logger.info(f"Measuring random apertures for {len(tiles)} tiles x {len(filters)} filters")
    logger.info(f"Aperture range: {min_radius:.3f} - {max_radius:.3f} arcsec ({num_radii} steps)")
    logger.info(f"Density: {num_apertures_per_sq_arcmin} apertures per sq arcmin; depth strata: {depth_strata}; "
                f"proxy: {depth_proxy}; scale: {scale_map}; parallel: {parallel}")

    index = 2
    aperture_diameters = np.power(np.linspace(np.power(min_radius*2, 1/index), np.power(max_radius*2, 1/index), num_radii), index)  # arcsec

    for filt in filters:
        if psfs_config is None or filt == psfs_config.target_filter or filt in psfs_config.inverse_filters:
            runs = [False]
        else:
            runs = [False, True]

        for homogenized in runs:
            pickle_file, validation_file, empirical_file, plot_file = _random_aperture_products(output_dir, filt, homogenized)
            imgtype = 'psf-homogenized' if homogenized else 'native resolution'
            if pickle_file.exists() and not overwrite:
                logger.info(f'Skipping random aperture measurement for {filt} ({imgtype}): {pickle_file.name} exists')
                if not plot_file.exists():
                    _plot_random_apertures(pickle.load(open(pickle_file, 'rb')), filt, imgtype, plot_file)
                continue

            logger.info(f'Measuring random aperture scaling for {filt} ({imgtype})')
            filter_tiles = images_config.get_tiles_for_filter(filt)
            tasks, kinds, conversions, pixel_scales = [], set(), [], []
            for n, tile in enumerate(filter_tiles):
                image_files = images_config.get_image_files(filt, tile)
                sci_path = image_files.get_homogenized_path(psfs_config.target_filter) if homogenized else image_files.get_science_path()
                scale_kind, proxy_kind = _resolve_noise_maps(image_files, depth_proxy, scale_map)
                kinds.add((scale_kind, proxy_kind))
                header = fits.getheader(sci_path)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', FITSFixedWarning)
                    pixel_scale = WCS(header).proj_plane_pixel_scales()[0].to('arcsec').value
                pixel_scales.append(pixel_scale)
                conversions.append(get_unit_conversion(header, 'nJy'))
                segmap_path = images_config.get_tile_segmap_path(tile) if mask_sources else None
                if mask_sources and (segmap_path is None or not Path(segmap_path).exists()):
                    logger.warning(f'{filt} {tile}: no segmentation map found, random apertures will not avoid sources')
                    segmap_path = None
                positions = None
                need_emp = scale_kind == 'empirical' or proxy_kind == 'empirical' or (check_weight_map and proxy_kind == 'wht')
                if need_emp and project_dir is not None:
                    from .photometry import load_tile_catalog, get_source_positions
                    cat, windowed = load_tile_catalog(project_dir, project_name, tile)
                    if cat is not None:
                        x, y = get_source_positions(cat, windowed)
                        positions = (np.asarray(cat['id']), np.asarray(x, float), np.asarray(y, float))
                tasks.append((str(sci_path),
                              str(image_files.get_rms_path()) if image_files.has_extension('rms') else None,
                              str(image_files.get_weight_path()) if image_files.has_extension('wht') else None,
                              str(segmap_path) if segmap_path is not None else None,
                              scale_kind, proxy_kind, check_weight_map and proxy_kind == 'wht',
                              aperture_diameters.tolist(), pixel_scale,
                              num_apertures_per_sq_arcmin, min_num_apertures_per_sq_arcmin, mask_dilate,
                              1000 * n + 7, positions, 12 if depth_strata > 1 else 0))
            if len(kinds) != 1:
                raise ValueError(f'{filt}: the tiles do not have the same map extensions: {kinds}')
            scale_kind, proxy_kind = kinds.pop()

            if parallel > 1 and len(tasks) > 1:
                import multiprocessing as mp
                with mp.get_context('fork').Pool(min(parallel, len(tasks))) as pool:
                    results = pool.map(_sample_tile_task, tasks)
            else:
                results = []
                for tile, task in zip(filter_tiles, tqdm.tqdm(tasks)):
                    results.append(_sample_tile_task(task))
            results = [(tile, r) for tile, r in zip(filter_tiles, results) if r['n_valid'] > 0]
            if not results:
                raise ValueError(f'{filt}: no valid pixels in any tile for random apertures')
            for tile, r in results:
                logger.debug(f'{filt} {tile}: {r["area_arcmin2"]:.2f} arcmin^2, {len(r["f"][0])} apertures at the smallest radius')

            # single-pixel width of image/scale (= 1 for a perfect scale map)
            pixels = np.concatenate([r['pixels'] for _, r in results])
            _, rms1 = fit_pixel_dist(pixels, sigma_upper=1.0, maxiters=5)

            # weight-map check: fall back to the empirical proxy if the weight map does not order depth
            checks = [r['wht_check'] for _, r in results if r.get('wht_check')]
            wht_check = None
            proxy_used = proxy_kind
            if checks:
                wht_check = {'spearman': float(np.median([c['spearman'] for c in checks])),
                             'slope': float(np.median([c['slope'] for c in checks])),
                             'scatter_dex': float(np.median([c['scatter_dex'] for c in checks])),
                             'n_tiles_ok': int(sum(c['ok'] for c in checks)), 'n_tiles': len(checks)}
                logger.info(f'{filt}: weight map vs empirical noise: slope {wht_check["slope"]:.2f}, Spearman {wht_check["spearman"]:.2f}, '
                            f'scatter {wht_check["scatter_dex"]:.3f} dex ({wht_check["n_tiles_ok"]}/{wht_check["n_tiles"]} tiles pass)')
                if wht_check['n_tiles_ok'] < 0.5 * wht_check['n_tiles'] and all('empirical' in r['logp'] for _, r in results):
                    logger.warning(f'{filt}: the weight map does not track the empirical noise; using the empirical '
                                   f'noise map as the depth proxy instead')
                    proxy_used = 'empirical'

            samples = {i: {'f': np.concatenate([r['f'][i] for _, r in results]),
                           'logp': np.concatenate([r['logp'][proxy_used][i] for _, r in results])}
                       for i in range(num_radii)}
            pixel_scale = float(np.median(pixel_scales))
            sqrtN = np.sqrt(np.pi) * aperture_diameters / 2 / pixel_scale
            logger.info(f'{filt}: fitting {depth_strata} depth strata on {sum(len(s["f"]) for s in samples.values())} apertures')
            # validation fit (holdout_fraction of the apertures kept back), then the production fit on
            # every aperture; the catalog uses the latter, the validation table comes from the former
            validation = None
            if holdout_fraction > 0:
                validation = nc.fit_model(samples, sqrtN, rms1, depth_strata, holdout_fraction=holdout_fraction)['validation']
            model = nc.fit_model(samples, sqrtN, rms1, depth_strata, holdout_fraction=0.0)
            if validation is not None:
                model['validation'] = validation
            model.update({
                'sigma1': float(rms1 * np.median(conversions)),
                'rms1': float(rms1),
                'pixel_scale': pixel_scale,
                'proxy': proxy_used,
                'scale': scale_kind,
                'tiles': [tile for tile, _ in results],
                'n_apertures': [int(len(s['f'])) for s in samples.values()],
                'wht_check': wht_check,
                'mask_sources': mask_sources,
            })
            with open(pickle_file, 'wb') as f:
                pickle.dump(model, f)
            with open(validation_file, 'w') as f:
                json.dump({'filter': filt, 'image': imgtype, 'proxy': proxy_used, 'scale': scale_kind,
                           'depth_strata_requested': depth_strata, 'strata_edges_logp': model['strata']['edges'],
                           'strata_log_centres': model['strata']['log_centres'], 'strata_n': model['strata']['n'],
                           'diameters_arcsec': aperture_diameters.tolist(), 'sqrtN': sqrtN.tolist(),
                           'rms1': float(rms1), 'wht_check': wht_check, 'validation': model['validation']}, f, indent=1)
            if proxy_used == 'empirical' or scale_kind == 'empirical':
                parts = [(tile, r['empirical_at_sources']) for tile, r in results if 'empirical_at_sources' in r]
                if parts:
                    np.savez_compressed(empirical_file,
                                        tile=np.concatenate([np.full(len(ids), tile) for tile, (ids, _) in parts]),
                                        id=np.concatenate([ids for _, (ids, _) in parts]),
                                        value=np.concatenate([v for _, (_, v) in parts]))
            v = model['validation']
            summary = ', '.join(f'{d:.2f}"={w:.2f}' for d, w in zip(aperture_diameters[::max(1, num_radii // 6)], v['overall'][::max(1, num_radii // 6)]) if w is not None)
            logger.info(f'{filt} ({imgtype}): {len(model["strata"]["log_centres"]) or 1} strata; held-out width measured/predicted: {summary}')
            logger.info(f'Saving results to {pickle_file}')
            _plot_random_apertures(model, filt, imgtype, plot_file)


def _plot_random_apertures(model: dict, filt: str, imgtype: str, plot_file: Path) -> None:
    """Check plot: the fitted curves per stratum and the held-out calibration."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from ..core import noise_curves as nc
    sqrtN = np.asarray(model['sqrtN']); pixel_scale = model['pixel_scale']
    strata = model.get('strata') or {}
    n_strata = len(strata.get('log_centres', []))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    ax = axes[0]
    x = np.linspace(0, 1.1 * np.max(sqrtN), 400)
    ax.errorbar(sqrtN, model['rmsN/rms1'], yerr=model['rmsN_err/rms1'], fmt='s', mfc='none', mec='k', ms=4, ecolor='k', elinewidth=0.6, label='all apertures')
    ax.plot(x, nc.evaluate_curve(model, x), color='k', lw=1)
    if n_strata > 1:
        colors = plt.cm.viridis(np.linspace(0, 1, n_strata))
        for s in range(n_strata):
            ax.plot(x, nc.evaluate_curve(strata['curves'][s], x), color=colors[s], lw=0.8,
                    label=f'stratum {s} (log p={strata["log_centres"][s]:.2f})' if n_strata <= 6 or s in (0, n_strata - 1) else None)
    ax.plot(x, x, color='0.6', ls=':', lw=0.8, label='white noise')
    ax.set_xlabel(r'$\sqrt{N_{\rm pix}}$' + f' ({int(pixel_scale*1000)} mas)'); ax.set_ylabel(r'$\sigma_N/\sigma_1$')
    ax.set_xlim(0, 1.1 * np.max(sqrtN)); ax.legend(fontsize=6, frameon=False, title=f'{filt} ({imgtype})')
    ax = axes[1]
    v = model['validation']
    diam = 2 * sqrtN / np.sqrt(np.pi) * pixel_scale
    ax.plot(diam, [np.nan if w is None else w for w in v['overall_single_curve']], 'k--', lw=0.8, label='single curve')
    ax.plot(diam, [np.nan if w is None else w for w in v['overall']], 'k-', lw=1.2, label='stratified (all)')
    if n_strata > 1:
        for s in range(n_strata):
            ax.plot(diam, [np.nan if w is None else w for w in v['by_stratum'][s]], color=colors[s], lw=0.7)
    ax.axhline(1, color='0.5', lw=0.6); ax.set_ylim(0.6, 1.5)
    ax.set_xlabel('aperture diameter [arcsec]'); ax.set_ylabel('held-out: measured / predicted')
    ax.legend(fontsize=6, frameon=False)
    plt.savefig(plot_file, dpi=120)
    plt.close(fig)


def _get_random_aperture_curve(
    sqrtN: np.ndarray,
    coeff_file: Path,
    log_proxy: Optional[np.ndarray] = None,
) -> np.ndarray:
    """sigma_N in nJy per unit of the scale map, for apertures of sqrt(N) pixels at depth proxy
    log_proxy (None: the single pooled curve)."""
    from ..core import noise_curves as nc
    with open(coeff_file, 'rb') as f:
        model = pickle.load(f)
    return model['sigma1'] * nc.predict(model, sqrtN, log_proxy)


def _object_noise_inputs(catalog: Table, filt: str, model: dict, empirical_file: Path):
    """Per-object scale (in image units) and log depth proxy for the model's map choices.
    Returns (scale, log_proxy); log_proxy is None when the catalog has no column for it."""
    logger = get_logger()
    n = len(catalog)
    emp = None

    def empirical_values():
        nonlocal emp
        if emp is None:
            if not empirical_file.exists():
                raise ValueError(f'{filt}: the noise model uses the empirical noise map but {empirical_file.name} is missing')
            z = np.load(empirical_file, allow_pickle=False)
            lookup = {(str(t), int(i)): v for t, i, v in zip(z['tile'], z['id'], z['value'])}
            tiles = np.asarray(catalog['tile']).astype(str) if 'tile' in catalog.colnames else np.full(n, str(z['tile'][0]))
            ids = np.asarray(catalog['id_tile']) if 'id_tile' in catalog.colnames else np.asarray(catalog['id'])
            emp = np.array([lookup.get((t, int(i)), np.nan) for t, i in zip(tiles, ids)], float)
        return emp

    def column(kind):
        if kind == 'rms' and f'rms_{filt}' in catalog.colnames:
            return np.asarray(catalog[f'rms_{filt}'], float)
        if kind == 'wht' and f'wht_{filt}' in catalog.colnames:
            return np.asarray(catalog[f'wht_{filt}'], float)
        if kind == 'empirical':
            return empirical_values()
        return None

    scale_kind, proxy_kind = model.get('scale', 'rms'), model.get('proxy')
    scale = column(scale_kind)
    if scale is None:
        raise ValueError(f"{filt}: catalog has no column for the noise scale '{scale_kind}'")
    if scale_kind == 'wht':
        with np.errstate(divide='ignore', invalid='ignore'):
            scale = 1.0 / np.sqrt(np.maximum(scale, 1e-30))
    log_proxy = None
    if proxy_kind is not None and model.get('strata', {}).get('log_centres'):
        p = column(proxy_kind)
        if p is None:
            logger.warning(f"{filt}: catalog has no column for the depth proxy '{proxy_kind}'; using the single pooled curve")
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                log_proxy = np.where(p > 0, np.log(p), np.nan)
    return scale, log_proxy


def calibrate_flux_uncertainties(
    catalog: Table,
    random_aperture_dir: Path,
    filters: List[str],
    psfs_config,
    aperture_diameters,
    flux_unit,
) -> Table:
    """
    Replace the measured flux errors by the random-aperture model: for every object,
    sigma = sigma1 * c(sqrt N, depth proxy) * scale(x), where N is the number of pixels actually
    summed (npix_* columns written by photometry; the Kron ellipse area / aperture area if they
    are missing) and the multiplicative corrections applied to the fluxes (kron_corr,
    inverse-filter factors, aper_corr) are applied to the errors as well. The measured errors
    are kept as e_*_uncal.
    """
    logger = get_logger()
    from ..core import noise_curves as nc
    logger.info("Calibrating uncertainties using random aperture measurements")
    filters = _filters_to_apply(catalog, 'uncertainty_calibration', filters)
    models_used = set()

    def model_for(filt, homogenized):
        use_native = psfs_config is None or filt == psfs_config.target_filter or filt in psfs_config.inverse_filters
        pickle_file, _, empirical_file, _ = _random_aperture_products(random_aperture_dir, filt, homogenized and not use_native)
        with open(pickle_file, 'rb') as f:
            model = pickle.load(f)
        if 'strata' not in model:     # coefficients written by the single-curve code
            model['strata'] = {}; model.setdefault('scale', 'rms'); model.setdefault('proxy', None)
        return model, empirical_file

    def to_unit(values):
        return (values * u.nJy).to(flux_unit).value

    for filt in filters:
        logger.info(f'Applying random aperture calibration for {filt}')

        # auto (kron) photometry: PSF-homogenized image unless target/inverse filter
        if f'f_auto_{filt}' in catalog.columns:
            model, empirical_file = model_for(filt, True)
            scale, log_proxy = _object_noise_inputs(catalog, filt, model, empirical_file)
            if f'npix_auto_{filt}' in catalog.columns:
                sqrtN = np.sqrt(np.maximum(np.asarray(catalog[f'npix_auto_{filt}'], float), 1.0))
            else:
                sqrtN = np.sqrt(np.asarray(catalog['kron1_area'], float))
            rmsN = to_unit(model['sigma1'] * nc.predict(model, sqrtN, log_proxy) * scale)
            if f'e_auto_{filt}_uncal' not in catalog.columns:
                catalog.rename_column(f'e_auto_{filt}', f'e_auto_{filt}_uncal')
            catalog[f'e_auto_{filt}'] = np.where(np.isfinite(catalog[f'e_auto_{filt}_uncal']), rmsN, np.nan)
            if psfs_config is not None and filt in psfs_config.inverse_filters and f'psf_corr_auto_{filt}' in catalog.columns:
                catalog[f'e_auto_{filt}'] *= catalog[f'psf_corr_auto_{filt}']
            if 'kron_corr' in catalog.columns:
                catalog[f'e_auto_{filt}'] *= catalog['kron_corr']
            models_used.add((model.get('scale'), model.get('proxy'), len(model['strata'].get('log_centres', [])) or 1))

        for phot_type, homogenized in (('aper_nat', False), ('aper_hom', True)):
            if f'f_{phot_type}_{filt}' not in catalog.columns:
                continue
            model, empirical_file = model_for(filt, homogenized)
            scale, log_proxy = _object_noise_inputs(catalog, filt, model, empirical_file)
            pixel_scale = model['pixel_scale']
            n_diam = len(aperture_diameters)
            if f'npix_{phot_type}_{filt}' in catalog.columns:
                npix = np.maximum(np.asarray(catalog[f'npix_{phot_type}_{filt}'], float), 1.0)
            else:
                npix = np.broadcast_to(np.pi * (np.asarray(aperture_diameters) / pixel_scale / 2) ** 2, (len(catalog), n_diam))
            rmsN = np.empty((len(catalog), n_diam))
            for j in range(n_diam):
                rmsN[:, j] = model['sigma1'] * nc.predict(model, np.sqrt(npix[:, j]), log_proxy) * scale
            rmsN = to_unit(rmsN)
            if f'e_{phot_type}_{filt}_uncal' not in catalog.columns:
                catalog.rename_column(f'e_{phot_type}_{filt}', f'e_{phot_type}_{filt}_uncal')
            catalog[f'e_{phot_type}_{filt}'] = np.where(np.isfinite(catalog[f'e_{phot_type}_{filt}_uncal']), rmsN, np.nan)
            if phot_type == 'aper_hom':
                if psfs_config is not None and filt in psfs_config.inverse_filters and f'psf_corr_aper_{filt}' in catalog.columns:
                    catalog[f'e_aper_hom_{filt}'] *= catalog[f'psf_corr_aper_{filt}']
                if 'aper_corr' in catalog.columns:
                    catalog[f'e_aper_hom_{filt}'] *= catalog['aper_corr']

    if models_used:
        catalog.meta['NOISEMOD'] = '; '.join(f'scale={s} proxy={p} strata={k}' for s, p, k in sorted(models_used, key=str))
    _record_applied(catalog, STEP_STATE_KEYS['uncertainty_calibration'], filters)
    return catalog


def compute_psf_correction_grids(
    images_config,
    psfs_config,
    filters: List[str],
    output_dir: Path,
    overwrite: bool = False
) -> None:
    """
    Compute PSF correction grids for total flux estimation.
    
    Args:
        images_config: ImagesConfig object
        psfs_config: PSFsConfig object  
        filters: List of filters to process
        output_dir: Directory for output correction grids
        overwrite: Whether to overwrite existing files
        
    """
    logger = get_logger()
    logger.info(f"Computing PSF correction grids")

    for filt in filters:
        logger.info(f'Computing PSF correction for {filt}')

        tiles = images_config.get_tiles_for_filter(filt)
        if not tiles:
            logger.error(f"No tiles found for filter {filt}")
            continue
    
        if psfs_config and psfs_config.master_psf:
            
            # Master PSF mode - compute one grid per filter
            output_file = output_dir / f'{filt}_psf_corr_grid.txt'
            if output_file.exists() and not overwrite:
                logger.info(f'Skipping PSF correction for {filt} - file exists')
                continue

            # Get PSF from first available tile
            psf_path = images_config.get_psf_path(filt, tiles[0])
            _compute_psf_corr_grid(psf_path, output_file)

        else:
            # Individual PSF mode - compute one grid per filter/tile
            for tile in tiles:
                output_file = output_dir / f'{filt}_{tile}_psf_corr_grid.txt'
                if output_file.exists() and not overwrite:
                    logger.info(f'Skipping PSF correction for {filt}-{tile} - file exists')
                    continue

                psf_path = images_config.get_psf_path(filt, tile)
                _compute_psf_corr_grid(psf_path, output_file)

    
def _compute_psf_corr_grid(psf_path, output_file):
    a = np.linspace(0.05, 2.0, 50)
    q = np.linspace(0.05, 1, 50)
    
    with fits.open(psf_path) as hdul:
        psf = hdul[0].data
        pixel_scale = WCS(hdul[0].header).proj_plane_pixel_scales()[0].to('arcsec').value
    
    a /= pixel_scale

    psf_corrs = np.zeros((len(a),len(q)))
    for i, ai in enumerate(a):
        for j,qj in enumerate(q):
            bi = qj*ai
            ap = EllipticalAperture((np.shape(psf)[0]/2, np.shape(psf)[1]/2), a=ai, b=bi, theta=0)
            tab = aperture_photometry(psf, ap)
            psf_corrs[i,j] = np.sum(psf)/float(tab['aperture_sum'][0])

    psf_corrs = psf_corrs.T
    a, q = np.meshgrid(a, q)
    
    np.savetxt(
        output_file, 
        np.array([a.flatten()*pixel_scale, q.flatten(), psf_corrs.flatten()]).T
    )

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    fig = plt.figure(figsize=(4.2,3.5), constrained_layout=True)
    gs = mpl.gridspec.GridSpec(nrows=1,ncols=2,width_ratios=[1,0.06], figure=fig)
    ax = fig.add_subplot(gs[0])
    im = ax.scatter(a.flatten()*pixel_scale, q.flatten(), c=psf_corrs.flatten(), vmax=3.5, marker='s', s=10)
    ax.set_xlabel('semi-major axis [arcsec]')
    ax.set_ylabel('axis ratio $q$')
    ax = fig.add_subplot(gs[1])
    fig.colorbar(im, cax=ax, label='PSF correction')
    plt.savefig(output_file.with_suffix('.pdf'))
    plt.close()

    return output_file


# Header keywords of the merged catalog recording which filters each postprocess step has
# already been applied to (comma-separated lists). Each step is applied once per merged
# catalog; `postprocess --overwrite` rebuilds the merge from the tile catalogs and so resets
# them. (Review finding F14: a second `postprocess` run re-applied the PSF grid, x1.126.)
STEP_STATE_KEYS = {'psf_corrections': 'PSFCORR', 'uncertainty_calibration': 'ERRCAL',
                   'extinction_correction': 'EXTCORR'}


def _applied_filters(catalog: Table, key: str) -> set:
    return {f for f in str(catalog.meta.get(key, '')).split(',') if f}


def _record_applied(catalog: Table, key: str, filters) -> None:
    catalog.meta[key] = ','.join(sorted(_applied_filters(catalog, key) | set(filters)))


def _filters_to_apply(catalog: Table, step: str, filters: List[str]) -> List[str]:
    """Drop the filters a step has already been applied to in this merged catalog."""
    logger = get_logger()
    done = _applied_filters(catalog, STEP_STATE_KEYS[step])
    skipped = [f for f in filters if f in done]
    if skipped:
        logger.warning(f"{step}: already applied to {len(skipped)} filter(s) of this merged catalog "
                       f"({', '.join(skipped[:6])}{', ...' if len(skipped) > 6 else ''}); skipping them. "
                       f"Run `postprocess --overwrite` to rebuild the merged catalog from the tile "
                       f"catalogs and apply every step once.")
    return [f for f in filters if f not in done]


def apply_psf_corrections(
    catalog: Table,
    psf_correction_dir: Path,
    tiles: List[str],
    filters: List[str],
    psfs_config,
    aperture_diameters: List[float]
) -> Table:
    """
    Apply PSF corrections to catalog fluxes.
    
    Args:
        catalog: Master catalog table
        psf_correction_dir: Directory containing PSF correction grids
        tiles: List of tiles
        filters: List of filters
        
    Returns:
        Catalog with PSF-corrected fluxes
    """
    logger = get_logger()

    # Every filter is corrected with the TARGET filter's aperture-to-total grid. Homogenized
    # filters are measured on images convolved to the target PSF; inverse filters (broader PSF
    # than the target) are measured on their native image and then scaled at photometry time by
    # the per-source factor flux(target, native) / flux(target, convolved to this filter), which
    # already brings them to the target resolution. Using the inverse filter's own grid here
    # applied that correction a second time (F17 in the 2026-10 review: F770W fluxes +15%).
    target = psfs_config.target_filter
    filters = _filters_to_apply(catalog, 'psf_corrections', filters)
    for filt in filters:
        logger.info(filt)
        if psfs_config.master_psf:
            logger.debug(f'{filt} - master PSF ({target} grid)')
            psf_corr_file = psf_correction_dir / f'{target}_psf_corr_grid.txt'
            catalog = _apply_psf_corr(catalog, filt, 'all', psf_corr_file, aperture_diameters)
        else:
            logger.debug(f'{filt} - per-tile PSF ({target} grids)')
            for tile in tiles:
                logger.debug(f'{filt} - {tile}')
                psf_corr_file = psf_correction_dir / f'{target}_{tile}_psf_corr_grid.txt'
                catalog = _apply_psf_corr(catalog, filt, tile, psf_corr_file, aperture_diameters)

    _record_applied(catalog, STEP_STATE_KEYS['psf_corrections'], filters)
    return catalog


def _apply_psf_corr(catalog, filt, tile, psf_corr_file, aperture_diameters):
    a, q, c = np.loadtxt(psf_corr_file).T

    if f'f_auto_{filt}' in catalog.columns and 'kron2_a' in catalog.columns and 'kron2_b' in catalog.columns:
        ai = catalog['kron2_a']
        bi = catalog['kron2_b']
        qi = bi/ai
        kron_psf_corr = griddata(np.array([a,q]).T, c, (ai,qi), fill_value=1.0)

        catalog[f'f_auto_{filt}'] *= kron_psf_corr
        if f'e_auto_{filt}' in catalog.columns:
            catalog[f'e_auto_{filt}'] *= kron_psf_corr

    if f'f_aper_hom_{filt}' in catalog.columns:
        if 'aper_corr' in catalog.columns:
            ai = catalog['kron2_a']
            bi = catalog['kron2_b']
            qi = bi/ai
            aper_psf_corr = griddata(np.array([a,q]).T, c, (ai,qi), fill_value=1.0)
            aper_psf_corr = aper_psf_corr[:,np.newaxis]
        else:
            aperture_diameters = np.array(aperture_diameters)
            aper_psf_corr = griddata(np.array([a,q]).T, c, (aperture_diameters/2, [1]*len(aperture_diameters)), fill_value=1.0)
            
        catalog[f'f_aper_hom_{filt}'] *= aper_psf_corr
        if f'e_aper_hom_{filt}' in catalog.columns:
            catalog[f'e_aper_hom_{filt}'] *= aper_psf_corr


    return catalog


def set_missing_to_nan(catalog, filters):

    for filt in filters:

        for phot_type in ['auto','aper_nat','aper_hom']:
            if f'f_{phot_type}_{filt}' in catalog.columns:
                f = catalog[f'f_{phot_type}_{filt}']
                e_uncal = catalog[f'e_{phot_type}_{filt}_uncal']
                cond = ~np.isfinite(f) | (f == 0) | ~np.isfinite(e_uncal) | (e_uncal == 0)
                # A calibrated error that is not finite or not positive marks an entry whose
                # correction factors were undefined or negative (F01): no usable measurement.
                e = catalog[f'e_{phot_type}_{filt}']
                with np.errstate(invalid='ignore'):
                    cond |= ~np.isfinite(e) | (e <= 0)
                catalog[f'f_{phot_type}_{filt}'][cond] = np.nan
                catalog[f'e_{phot_type}_{filt}'][cond] = np.nan
                catalog[f'e_{phot_type}_{filt}_uncal'][cond] = np.nan
    
    return catalog


def _cardelli(wavs):
    """ Calculate the ratio A(lambda)/A(V) for the Cardelli et al.
    (1989) extinction curve. """

    A_lambda = np.zeros_like(wavs)

    inv_mic = 1./(wavs*10.**-4.)

    mask1 = (inv_mic < 1.1)
    mask2 = (inv_mic >= 1.1) & (inv_mic < 3.3)
    mask3 = (inv_mic >= 3.3) & (inv_mic < 5.9)
    mask4 = (inv_mic >= 5.9) & (inv_mic < 8.)
    mask5 = (inv_mic >= 8.)

    A_lambda[mask1] = (0.574*inv_mic[mask1]**1.61
                        + (-0.527*inv_mic[mask1]**1.61)/3.1)

    y = inv_mic[mask2] - 1.82

    A_lambda[mask2] = (1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3
                        + 0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6
                        + 0.32999*y**7 + (1.41388*y + 2.28305*y**2
                                            + 1.07233*y**3 - 5.38434*y**4
                                            - 0.62251*y**5 + 5.30260*y**6
                                            - 2.09002*y**7)/3.1)

    A_lambda[mask3] = ((1.752 - 0.316*inv_mic[mask3]
                        - (0.104)/((inv_mic[mask3]-4.67)**2 + 0.341))
                        + (-3.09 + 1.825*inv_mic[mask3]
                            + 1.206/((inv_mic[mask3]-4.62)**2 + 0.263))/3.1)

    A_lambda[mask4] = ((1.752 - 0.316*inv_mic[mask4]
                        - (0.104)/((inv_mic[mask4]-4.67)**2 + 0.341)
                        - 0.04473*(inv_mic[mask4] - 5.9)**2
                        - 0.009779*(inv_mic[mask4]-5.9)**3)
                        + (-3.09 + 1.825*inv_mic[mask4]
                            + 1.206/((inv_mic[mask4]-4.62)**2 + 0.263)
                            + 0.2130*(inv_mic[mask4]-5.9)**2
                            + 0.1207*(inv_mic[mask4]-5.9)**3)/3.1)

    A_lambda[mask5] = ((-1.073
                        - 0.628*(inv_mic[mask5] - 8.)
                        + 0.137*(inv_mic[mask5] - 8.)**2
                        - 0.070*(inv_mic[mask5] - 8.)**3)
                        + (13.670
                            + 4.257*(inv_mic[mask5] - 8.)
                            - 0.420*(inv_mic[mask5] - 8.)**2
                            + 0.374*(inv_mic[mask5] - 8.)**3)/3.1)

    return A_lambda

def apply_galactic_extinction_corr(
    catalog: Table,
    reddening_map_file: Path,
    pivot_wavelengths: Dict[str, float],
    filters: List[str],
    max_correction_factor: float = 10.0
) -> Table:
    """
    Apply galactic extinction correction to catalog fluxes.
    
    Args:
        catalog: Master catalog table
        reddening_map_file: Path to dust map FITS file
        pivot_wavelengths: Dictionary mapping filter names to pivot wavelengths in Angstroms
        filters: List of filters to process
        max_correction_factor: Maximum allowed correction factor as safety limit
        
    Returns:
        Updated catalog with extinction-corrected fluxes
    """
    logger = get_logger()
    logger.info(f"Applying galactic extinction correction using {reddening_map_file.name}")
    filters = _filters_to_apply(catalog, 'extinction_correction', filters)

    # Load dust map
    with fits.open(reddening_map_file) as hdul:
        ebv_map = hdul[0].data
        wcs = WCS(hdul[0].header)
        logger.debug(f"    Dust map shape: {ebv_map.shape}")
    
    # Convert RA/Dec to pixel coordinates
    from astropy.coordinates import SkyCoord
    coords = SkyCoord(catalog['ra'], catalog['dec'], unit='deg')
    x_pix, y_pix = wcs.world_to_pixel(coords)
    
    # Interpolate E(B-V) at source positions
    from scipy.interpolate import RegularGridInterpolator
    y_grid, x_grid = np.arange(ebv_map.shape[0]), np.arange(ebv_map.shape[1])
    interp = RegularGridInterpolator((y_grid, x_grid), ebv_map, 
                                    bounds_error=False, fill_value=0.0)
    
    # Get E(B-V) values at source positions
    ebv_sources = interp((y_pix, x_pix))
    ebv_sources *= 0.86  # SFD1998 to SF2011 scaling
    Av_sources = ebv_sources * 3.1
    
    # Store Av in catalog
    catalog['Av'] = Av_sources
    logger.debug(f"    Computed Av for {len(catalog)} sources")
    
    # Apply correction for each filter
    n_corrected = 0
    for filt in filters:
        if filt not in pivot_wavelengths:
            logger.warning(f"    No pivot wavelength for filter {filt}, skipping extinction correction")
            continue
            
        # Calculate A_lambda for this filter
        A_lambda_over_Av = _cardelli(np.array([pivot_wavelengths[filt]]))
        A_lambda = Av_sources * A_lambda_over_Av[0]
        
        # Calculate correction factor
        corr_factor = 10**(0.4 * A_lambda)
        
        # Apply safety limit
        corr_factor = np.clip(corr_factor, 1.0/max_correction_factor, max_correction_factor)
        n_clipped = np.sum((corr_factor == 1.0/max_correction_factor) | (corr_factor == max_correction_factor))
        
        # Apply to all flux columns for this filter
        filter_cols_corrected = 0
        for col in catalog.colnames:
            if col.startswith(f'f_') and col.endswith(f'_{filt}'):
                # Handle aperture photometry columns (2D arrays) vs other columns (1D arrays)
                if catalog[col].ndim == 2:
                    # Aperture photometry: reshape correction factor to broadcast (sources, 1)
                    catalog[col] *= corr_factor.reshape(-1, 1)
                else:
                    # Other photometry: direct multiplication
                    catalog[col] *= corr_factor
                filter_cols_corrected += 1
                n_corrected += 1
            elif col.startswith(f'e_') and col.endswith(f'_{filt}'):
                # Handle aperture photometry error columns (2D arrays) vs other columns (1D arrays)
                if catalog[col].ndim == 2:
                    # Aperture photometry: reshape correction factor to broadcast (sources, 1)
                    catalog[col] *= corr_factor.reshape(-1, 1)
                else:
                    # Other photometry: direct multiplication
                    catalog[col] *= corr_factor
                filter_cols_corrected += 1
        
        # Store A_lambda for reference
        catalog[f'A_{filt}'] = A_lambda
        
        logger.info(f"    {filt}: median A_lambda = {np.median(A_lambda):.3f} mag, corrected {filter_cols_corrected} columns")
        if n_clipped > 0:
            logger.warning(f"    {filt}: {n_clipped}/{len(catalog)} sources had correction factors clipped to safety limits")
    
    logger.info(f"Extinction correction applied to {n_corrected} columns")
    logger.info(f"  Median A_V = {np.median(Av_sources):.3f} mag")
    logger.info(f"  A_V range: [{np.min(Av_sources):.3f}, {np.max(Av_sources):.3f}] mag")

    _record_applied(catalog, STEP_STATE_KEYS['extinction_correction'], filters)
    return catalog


@click.command("postprocess")
@click.option(
    "--project-dir", 
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path.cwd(),
    help="Project directory (default: current directory)"
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite existing postprocessing outputs"
)
@click.option(
    "--merge-only",
    is_flag=True,
    help="Only run tile merging step"
)
@click.option(
    "--random-apertures-only",
    is_flag=True,
    help="Only run random apertures step"
)
@click.option(
    "--psf-corrections-only",
    is_flag=True,
    help="Only run PSF corrections step"
)
@click.option(
    "--extinction-correction-only",
    is_flag=True,
    help="Only run extinction correction step"
)
@click.option(
    "--skip-merge",
    is_flag=True,
    help="Skip tile merging"
)
@click.option(
    "--skip-random-apertures",
    is_flag=True,
    help="Skip random aperture calibration"
)
@click.option(
    "--skip-psf-corrections",
    is_flag=True,
    help="Skip PSF corrections"
)
@click.option(
    "--skip-extinction-correction",
    is_flag=True,
    help="Skip extinction correction"
)
@click.option(
    "--filters",
    type=str,
    default='',
    help="Run only for a subset of filters (comma-separated)"
)
@click.option(
    "--parallel",
    type=int,
    default=0,
    help="Tiles sampled in parallel for the random apertures (default: config value)"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable debug output"
)
def postprocess_cmd(
    project_dir,
    overwrite,
    merge_only,
    random_apertures_only,
    psf_corrections_only,
    extinction_correction_only,
    skip_merge,
    skip_random_apertures,
    skip_psf_corrections,
    skip_extinction_correction,
    filters,
    parallel,
    verbose
):
    """
    Postprocessing: merge tiles, calibrate uncertainties, apply PSF corrections, extinction correction.
    
    This command performs the final steps of the catalog generation pipeline:
    
    1. MERGE TILES: Combine individual tile catalogs into master catalog
    2. RANDOM APERTURES: Measure background for uncertainty calibration  
    3. PSF CORRECTIONS: Apply total flux corrections using PSF models
    4. EXTINCTION CORRECTION: Apply galactic extinction correction using dust maps
    
    The steps run in sequence and can be controlled with --skip-* or --*-only options.
    """
    logger = get_logger()

    # Set debug level if verbose flag is used
    if verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    
    # Validate step selection options
    only_flags = [merge_only, random_apertures_only, psf_corrections_only, extinction_correction_only]
    if sum(only_flags) > 1:
        logger.error("Cannot specify multiple --*-only options")
        raise click.ClickException("Invalid option combination")
    
    # Determine which steps to run
    run_merge = not skip_merge and (not any(only_flags) or merge_only)
    run_random_apertures = not skip_random_apertures and (not any(only_flags) or random_apertures_only)
    run_psf_corrections = not skip_psf_corrections and (not any(only_flags) or psf_corrections_only)
    run_extinction_correction = not skip_extinction_correction and (not any(only_flags) or extinction_correction_only)
    
    if not any([run_merge, run_random_apertures, run_psf_corrections, run_extinction_correction]):
        logger.error("No postprocessing steps selected")
        raise click.ClickException("Nothing to do")
    
    logger.info(f"[bold green]Starting postprocessing in {project_dir}[/bold green]")
    logger.info(f"Steps: merge={run_merge}, random_apertures={run_random_apertures}, psf_corrections={run_psf_corrections}, extinction_correction={run_extinction_correction}")
    
    # Load configuration
    config_path = project_dir / "config.toml"
    if not config_path.exists():
        logger.error(f"No config.toml found in {project_dir}")
        raise click.ClickException("Config not found")
    
    config = load_config(config_path)
    postprocess_config = config.postprocess
    
    # Parse filter override
    if filters:
        filters_to_process = [f.strip() for f in filters.split(',') if f.strip() in config.filters]
    else:
        filters_to_process = config.filters
    
    if not filters_to_process:
        logger.error("[bold red]Error:[/bold red] No filters available for postprocessing")
        raise click.ClickException("No filters to process")
    
    # Load images config
    images_config = load_images_config(project_dir)
    
    # Load PSF config (needed for PSF corrections)
    psfs_config = load_psfs_config(project_dir)
    if psfs_config is None:
        logger.warning("No psfs.toml found - PSF corrections may not work properly")
    
    # Create output directories
    output_dirs = {
        'random_apertures': project_dir / "random_apertures",
        'psf_corrections': project_dir / "psf_corrections"
    }
    
    for dir_path in output_dirs.values():
        dir_path.mkdir(exist_ok=True)
    
    # Determine tiles and filters to process
    available_tiles = []
    for filter_name in filters_to_process:
        available_tiles.extend(images_config.get_tiles_for_filter(filter_name))
    available_tiles = list(set(available_tiles))  # Remove duplicates
    
    logger.info(f"Processing {len(available_tiles)} tiles: {available_tiles}")
    logger.info(f"Processing {len(filters_to_process)} filters: {filters_to_process}")
    
    # Step 1: Merge tile catalogs
    master_catalog_path = project_dir / f"catalog_{config.name}_merged.fits"
    master_catalog = None
    
    if run_merge:
        if postprocess_config.merge_tiles.run:
            logger.info("[bold blue]Step 1: Merging tile catalogs[/bold blue]")
            
            # Check if tile catalogs exist
            catalogs_dir = project_dir / "catalogs"
            if not catalogs_dir.exists():
                logger.error(f"No catalogs directory found: {catalogs_dir}")
                raise click.ClickException("Tile catalogs not found")
            
            existing_tiles = []
            for tile in available_tiles:
                tile_catalog_path = catalogs_dir / f"catalog_{config.name}_{tile}.fits"
                if tile_catalog_path.exists():
                    existing_tiles.append(tile)
                else:
                    logger.warning(f"Tile catalog not found: {tile_catalog_path}")
            
            if not existing_tiles:
                logger.error("No tile catalogs found for merging")
                raise click.ClickException("No catalogs to merge")
            
            logger.info(f"Found {len(existing_tiles)} tile catalogs to merge")
            
            try:
                master_catalog = merge_tile_catalogs(
                    project_dir=project_dir,
                    project_name=config.name,
                    tiles=existing_tiles,
                    matching_radius=postprocess_config.merge_tiles.matching_radius,
                    edge_mask=postprocess_config.merge_tiles.edge_mask,
                    output_path=master_catalog_path,
                    overwrite=overwrite,
                )
                logger.info(f"Merged catalog saved: {master_catalog_path}")
            except NotImplementedError:
                logger.error("Tile merging not yet implemented")
                logger.info("User needs to implement merge_tile_catalogs function")
                return
            except Exception as e:
                logger.error(f"Tile merging failed: {e}")
                raise
                raise click.ClickException("Merge step failed")
        else:
            logger.info("Tile merging disabled in config")
    
    # Step 2: Random aperture measurements
    if run_random_apertures:
        if postprocess_config.random_apertures.run:
            logger.info("[bold blue]Step 2: Measuring random apertures[/bold blue]")
            
            try:
                measure_random_apertures(
                    images_config=images_config,
                    psfs_config=psfs_config,
                    tiles=available_tiles,
                    filters=filters_to_process,
                    min_radius=postprocess_config.random_apertures.min_radius,
                    max_radius=postprocess_config.random_apertures.max_radius,
                    num_radii=postprocess_config.random_apertures.num_radii,
                    num_apertures_per_sq_arcmin=postprocess_config.random_apertures.num_apertures_per_sq_arcmin,
                    output_dir=output_dirs['random_apertures'],
                    overwrite=overwrite,
                    depth_strata=postprocess_config.random_apertures.depth_strata,
                    depth_proxy=postprocess_config.random_apertures.depth_proxy,
                    scale_map=postprocess_config.random_apertures.scale_map,
                    mask_sources=postprocess_config.random_apertures.mask_sources,
                    mask_dilate=postprocess_config.random_apertures.mask_dilate,
                    holdout_fraction=postprocess_config.random_apertures.holdout_fraction,
                    check_weight_map=postprocess_config.random_apertures.check_weight_map,
                    parallel=parallel or postprocess_config.random_apertures.parallel,
                    project_dir=project_dir,
                    project_name=config.name,
                )
                logger.info("Random aperture measurements completed")
                
                # Apply uncertainty calibration if we have a master catalog
                if master_catalog is None and master_catalog_path.exists():
                    # Load existing master catalog
                    master_catalog = Table.read(master_catalog_path)
                
                if master_catalog is not None:
                    logger.info("Applying uncertainty calibration")
                    master_catalog = calibrate_flux_uncertainties(
                        catalog=master_catalog,
                        random_aperture_dir=output_dirs['random_apertures'],
                        filters=filters_to_process,
                        psfs_config=psfs_config,
                        aperture_diameters=config.photometry.aperture.diameters,
                        flux_unit=config.flux_unit
                    )
                    # Save updated catalog
                    master_catalog = set_missing_to_nan(master_catalog, filters_to_process)
                    master_catalog.write(master_catalog_path, overwrite=True)
                    logger.info("Uncertainty calibration applied")
                
            except NotImplementedError:
                logger.error("Random aperture measurement not yet implemented")
                logger.info("User needs to implement measure_random_apertures function")
                return
            except Exception as e:
                logger.error(f"Random aperture step failed: {e}")
                raise e
                # raise click.ClickException("Random apertures step failed")
        else:
            logger.info("Random aperture calibration disabled in config")
    
    # Step 3: PSF corrections
    if run_psf_corrections:
        if postprocess_config.psf_corrections.run:
            logger.info("[bold blue]Step 3: Computing and applying PSF corrections[/bold blue]")
            
            if psfs_config is None:
                logger.warning("No PSF configuration found - PSF corrections may not work")
            
            try:

                psf_corr_filters = [psfs_config.target_filter] + psfs_config.inverse_filters

                # Compute PSF correction grids
                compute_psf_correction_grids(
                    images_config=images_config,
                    psfs_config=psfs_config,
                    filters=psf_corr_filters,
                    output_dir=output_dirs['psf_corrections'],
                    overwrite=overwrite
                )
                
                # Apply PSF corrections if we have a master catalog
                if master_catalog is None and master_catalog_path.exists():
                    # Load existing master catalog
                    master_catalog = Table.read(master_catalog_path)
                
                if master_catalog is not None:
                    logger.info("Applying PSF corrections to catalog")
                    master_catalog = apply_psf_corrections(
                        catalog=master_catalog,
                        psf_correction_dir=output_dirs['psf_corrections'],
                        tiles=available_tiles,
                        filters=filters_to_process,
                        psfs_config=psfs_config,
                        aperture_diameters=config.photometry.aperture.diameters
                    )
                    # Save final catalog
                    master_catalog.write(master_catalog_path, overwrite=True)
                    logger.info("PSF corrections applied")
                else:
                    logger.warning("No master catalog found - PSF corrections not applied")
                
                logger.info("PSF correction step completed")
                
            except NotImplementedError:
                logger.error("PSF corrections not yet implemented") 
                logger.error("User needs to implement compute_psf_correction_grids and apply_psf_corrections functions")
                return
            except Exception as e:
                logger.error(f"PSF corrections step failed: {e}")
                raise click.ClickException("PSF corrections step failed")
        else:
            logger.info("PSF corrections disabled in config")
    
    # Step 4: Galactic extinction correction
    if run_extinction_correction:
        if postprocess_config.extinction_correction.run:
            logger.info("[bold blue]Step 4: Applying galactic extinction correction[/bold blue]")
            
            if not postprocess_config.extinction_correction.dust_map:
                logger.error("Dust map file not specified in config")
                raise click.ClickException("Extinction correction step failed")
            
            if not postprocess_config.extinction_correction.pivot_wavelengths:
                logger.error("No pivot wavelengths specified in config")
                raise click.ClickException("Extinction correction step failed")
            
            # Load or verify master catalog exists
            if master_catalog is None and master_catalog_path.exists():
                master_catalog = Table.read(master_catalog_path)
            
            if master_catalog is not None:
                dust_map_path = project_dir / postprocess_config.extinction_correction.dust_map
                
                if not dust_map_path.exists():
                    logger.error(f"Dust map not found: {dust_map_path}")
                    raise click.ClickException("Extinction correction step failed")
                else:
                    try:
                        master_catalog = apply_galactic_extinction_corr(
                            catalog=master_catalog,
                            reddening_map_file=dust_map_path,
                            pivot_wavelengths=postprocess_config.extinction_correction.pivot_wavelengths,
                            filters=filters_to_process,
                            max_correction_factor=postprocess_config.extinction_correction.max_correction_factor
                        )
                        
                        # Save updated catalog
                        master_catalog.write(master_catalog_path, overwrite=True)
                        logger.info("Extinction correction applied successfully")
                        
                    except Exception as e:
                        logger.error(f"Extinction correction failed: {e}")
                        raise click.ClickException("Extinction correction step failed")
            else:
                logger.warning("No master catalog found - extinction correction not applied")
        else:
            logger.info("Extinction correction disabled in config")
    

    # Final summary
    logger.info("[bold green]Postprocessing completed successfully![/bold green]")
    if master_catalog is not None:
        logger.info(f"Final master catalog: {master_catalog_path}")
        logger.info(f"Total sources: {len(master_catalog)}")
    
    logger.info("Output directories:")
    logger.info(f"  Random apertures: {output_dirs['random_apertures']}")
    logger.info(f"  PSF corrections: {output_dirs['psf_corrections']}")