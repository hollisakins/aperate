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

import pickle
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
) -> None:
    """
    Measure random aperture fluxes for uncertainty calibration.
    
    This function measures random apertures on both native-resolution and PSF-homogenized
    images to calibrate flux uncertainties separately for each image type.
    
    Args:
        images_config: ImagesConfig object
        psfs_config: PSFsConfig object (or None if no PSF homogenization)
        tiles: List of tiles to process
        filters: List of filters to process
        min_radius: Minimum aperture radius (arcsec)
        max_radius: Maximum aperture radius (arcsec)
        num_radii: Number of aperture sizes to sample
        num_apertures_per_sq_arcmin: Number of apertures per sq arcmin
        output_dir: Directory for output files
        overwrite: Whether to overwrite existing files
        
    Implementation notes for user:
    - Create logarithmically spaced aperture radii from min_radius to max_radius
    - For each tile/filter combination, measure random apertures on:
      1. Native-resolution images (always)
      2. PSF-homogenized images (if psfs_config is not None and filter has homogenized images)
    - For PSF-homogenized measurements:
      - Target filter and inverse filters use native images
      - Other filters use homogenized images (if available)
    - Output files should be named:
      - random_apertures_{tile}_{filter}_native.fits
      - random_apertures_{tile}_{filter}_homogenized.fits
    - Include metadata about which image type was used for each measurement
    """
    logger = get_logger()
    logger.info(f"Measuring random apertures for {len(tiles)} tiles x {len(filters)} filters")
    logger.info(f"Aperture range: {min_radius:.3f} - {max_radius:.3f} arcsec ({num_radii} steps)")
    logger.info(f"Density: {num_apertures_per_sq_arcmin} apertures per sq arcmin")

    index = 2
    aperture_diameters = np.power(np.linspace(np.power(min_radius*2, 1/index), np.power(max_radius*2, 1/index), num_radii), index) # arcsec

    for filt in filters:

        if psfs_config is None or filt == psfs_config.target_filter or filt in psfs_config.inverse_filters:
            runs = [False]
        else:
            runs = [False, True]

        for homogenized in runs:

            if homogenized:
                output_file = output_dir / f'random_apertures_{filt}_psfhom_coeffs.pickle'
            else:
                output_file = output_dir / f'random_apertures_{filt}_coeffs.pickle'

            imgtype = 'psf-homogenized' if homogenized else 'native resolution'
            
            if not output_file.exists() or overwrite:
                logger.info(f'Measuring random aperture scaling for {filt} ({imgtype})')


                tiles = images_config.get_tiles_for_filter(filt)
            
                # First, fit the pixel distribution of the NSCI image to get the baseline single-pixel RMS 
                logger.info('Fitting pixel distribution')
                nrandom_pixels_per_tile = 50000

                fluxes = np.zeros(nrandom_pixels_per_tile*len(tiles))
                i = 0
                for tile in tqdm.tqdm(tiles):

                    image_files = images_config.get_image_files(filt, tile)

                    if homogenized: 
                        sci_path = image_files.get_homogenized_path(psfs_config.target_filter)
                        with fits.open(sci_path) as sci_hdul:
                            sci = sci_hdul[0].data
                    else:
                        sci_path = image_files.get_science_path()
                        with fits.open(sci_path) as sci_hdul:
                            sci = sci_hdul[0].data

                    if image_files.has_extension('rms'):
                        rms_path = image_files.get_rms_path()
                        logger.debug(f"Using rms image -> {rms_path.name}")
                        with fits.open(rms_path) as hdul:
                            rms = hdul[0].data
                    else: 
                        wht_path = image_files.get_weight_path()
                        logger.debug(f"No rms image found, using wht image -> {wht_path.name}")
                        with fits.open(wht_path) as hdul:
                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore', category=RuntimeWarning)
                                rms = 1/np.sqrt(hdul[0].data)
                    
                    sci = sci.astype(sci.dtype.newbyteorder('='))
                    rms = rms.astype(rms.dtype.newbyteorder('='))

                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', category=RuntimeWarning)
                        nsci = sci/rms
                    
                    nsci[(rms <= 0)|~np.isfinite(rms)] = np.nan
                    nsci = nsci[np.isfinite(nsci)]

                    # Select a subset of pixels to do the fitting — no need to use all of them (its slow)
                    nsci = np.random.choice(nsci, size=nrandom_pixels_per_tile)
                    fluxes[i:i+nrandom_pixels_per_tile] = nsci
                    i += nrandom_pixels_per_tile

                _, rms1 = fit_pixel_dist(fluxes, sigma_upper=1.0, maxiters=5) 
                rms1_random = 1


                logger.info('Getting random aperture fluxes')
                fluxes = {i:[] for i in range(len(aperture_diameters))}
                fluxes_random = {i:[] for i in range(len(aperture_diameters))}

                for tile in tqdm.tqdm(tiles):

                    image_files = images_config.get_image_files(filt, tile)

                    if homogenized: 
                        sci_path = image_files.get_homogenized_path(psfs_config.target_filter)
                        with fits.open(sci_path) as sci_hdul:
                            sci = sci_hdul[0].data
                            header = sci_hdul[0].header
                    else:
                        sci_path = image_files.get_science_path()
                        with fits.open(sci_path) as sci_hdul:
                            sci = sci_hdul[0].data
                            header = sci_hdul[0].header

                    if image_files.has_extension('rms'):
                        rms_path = image_files.get_rms_path()
                        logger.debug(f"Using rms image -> {rms_path.name}")
                        with fits.open(rms_path) as hdul:
                            rms = hdul[0].data
                    else: 
                        wht_path = image_files.get_weight_path()
                        logger.debug(f"No rms image found, using wht image -> {wht_path.name}")
                        with fits.open(wht_path) as hdul:
                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore', category=RuntimeWarning)
                                rms = 1/np.sqrt(hdul[0].data)
                    
                    sci = sci.astype(sci.dtype.newbyteorder('='))
                    rms = rms.astype(rms.dtype.newbyteorder('='))

                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', category=RuntimeWarning)
                        nsci = sci/rms

                    nsci[(rms <= 0)|~np.isfinite(rms)] = np.nan

                    nsci_random = np.random.normal(loc=0, scale=rms1_random, size=nsci.shape)

                    pixel_scale = WCS(header).proj_plane_pixel_scales()[0].to('arcsec').value

                    n_valid_pixels = np.sum(np.isfinite(nsci))
                    area = n_valid_pixels * pixel_scale**2 / 3600

                    x, y = np.arange(np.shape(nsci)[1]), np.arange(np.shape(nsci)[0])
                    x, y = np.meshgrid(x, y)
                    x, y = x.flatten(), y.flatten()
                    x = x[np.isfinite(nsci.flatten())]
                    y = y[np.isfinite(nsci.flatten())]

                    for i in range(num_radii):
                        diameter = aperture_diameters[i]
                        Nap = int(num_apertures_per_sq_arcmin * area)
                        if diameter > 1.0: 
                            Nap = int(Nap * 0.25/(diameter/2)**2)
                        if Nap/area < min_num_apertures_per_sq_arcmin:
                            Nap = int(min_num_apertures_per_sq_arcmin * area)
                            
                        idx = np.random.randint(low=0,high=len(x),size=Nap)
                        xi, yi = x[idx].astype(float), y[idx].astype(float)
                        xi += np.random.normal(loc=0, scale=1, size=Nap)
                        yi += np.random.normal(loc=0, scale=1, size=Nap)

                        apertures = CircularAperture(np.array([xi,yi]).T, r=diameter/2/pixel_scale)

                        tbl = aperture_photometry(nsci, apertures, method='subpixel', subpixels=5)
                        fluxes[i].extend(list(tbl['aperture_sum']))
                        
                        tbl = aperture_photometry(nsci_random, apertures, method='subpixel', subpixels=5)
                        fluxes_random[i].extend(list(tbl['aperture_sum']))
                
                logger.info('Fitting distribution')
                rmsN = np.zeros(num_radii)
                rmsN_err = np.zeros(num_radii)
                
                rmsN_random = np.zeros(num_radii)
                rmsN_err_random = np.zeros(num_radii)

                for i in tqdm.tqdm(range(num_radii)):
                    f = np.array(fluxes[i])
                    _, std, _, std_err = fit_pixel_dist(f, sigma_upper=1.0, maxiters=3, return_err=True)
                    rmsN[i] = std
                    rmsN_err[i] = std_err
                    
                    rmsN_random[i] = np.nanstd(np.array(fluxes_random[i]))
                    rmsN_err_random[i] = 0

                logger.info('Fitting curve')        
                N = np.pi*(aperture_diameters/2/pixel_scale)**2
                sqrtN = np.sqrt(N)
                conversion = get_unit_conversion(header, 'nJy')

                # First compute the simple power law method
                def func(sqrtN, alpha, beta):
                    return alpha*np.power(sqrtN, beta)
                popt, pcov = curve_fit(func, sqrtN, rmsN/rms1, p0=[1,1], maxfev=int(1e5))
                plaw = lambda x: func(x, *popt)
                
                # Then do the spline method
                x_all = np.append([0,1], sqrtN)
                y_all = np.append([0,1], rmsN/rms1)
                w_all = np.append([500,100], rms1/rmsN_err)

                bspline = make_splrep(
                    x = x_all, 
                    y = y_all, 
                    w = w_all, 
                    k = 3, 
                    s = len(x_all)
                )

                results = {
                    'sigma1': float(rms1 * conversion),
                    'sqrtN': sqrtN,
                    'rmsN/rms1': rmsN/rms1,
                    'rmsN_err/rms1': rmsN_err/rms1,
                    'rmsN_random/rms1': rmsN_random/rms1_random,
                    'rmsN_err_random/rms1': rmsN_err_random/rms1_random,
                    'pixel_scale': pixel_scale,
                    'spline': {
                        'c': list(bspline.c.astype(float)),
                        't': list(bspline.t.astype(float)),
                        'k': int(bspline.k),
                    },
                    'plaw': {
                        'alpha': float(popt[0]),
                        'beta': float(popt[1]),
                    },
                    'sqrtN_max': float(np.max(sqrtN)),
                }

                with open(output_file, 'wb') as pickle_file:
                    pickle.dump(results, pickle_file)
                logger.info(f'Saving results to {output_file}')

            else:
                logger.info(f'Skipping random aperture measurement for {filt} ({imgtype})')

            plot_file = output_file.with_suffix('.pdf')
            if not plot_file.exists() or overwrite: 
                with open(output_file, 'rb') as pickle_file:
                    results = pickle.load(pickle_file)

                sqrtN = results['sqrtN']
                N = sqrtN**2
                pixel_scale = results['pixel_scale']

                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(3.5,3), constrained_layout=True)
                
                ax.errorbar(sqrtN, results['rmsN/rms1'], yerr=results['rmsN_err/rms1'],
                    linewidth=0, marker='s', mfc='none', mec='k', 
                    mew=0.8, ms=5, elinewidth=0.6, ecolor='k', 
                    capsize=1.5, capthick=0.6, zorder=1000)
                x = np.linspace(0, 1.1*np.max(sqrtN), 1000)
                

                bspline = BSpline(c=results['spline']['c'], t=results['spline']['t'], k=results['spline']['k'])
                y = bspline(x)
                ax.plot(x, y, color='b', label='Spline')

                y = results['plaw']['alpha'] * np.power(x, results['plaw']['beta'])
                ax.plot(x, y, color='m', label='Power law')
                
                ax.errorbar(sqrtN, results['rmsN_random/rms1'], yerr=results['rmsN_err_random/rms1'], 
                    linewidth=0, marker='s', mfc='none', mec='0.7', mew=0.8, ms=5, 
                    elinewidth=0.6, ecolor='0.7', capsize=1.5, capthick=0.6, zorder=1000)
                x = np.linspace(0, 1.1*np.max(sqrtN), 1000)

                ax.set_xlim(0, np.max(np.sqrt(N))*1.1)
                ax.set_ylim(-0.03*np.max(results['rmsN/rms1']), 1.1*np.max(results['rmsN/rms1']))
                ax.set_xlabel(r'$\sqrt{N_{\rm pix}}$' + f' ({int(pixel_scale*1000)} mas)')
                ax.set_ylabel(r'$\sigma_N/\sigma_1$')

                ax.legend(loc='upper left', title=filt.upper(), frameon=False)

                ax.annotate('Gaussian random field', (0.95,0.05), 
                    xycoords='axes fraction', ha='right', va='bottom', color='0.7')

                logger.info(f"Saving plot to {plot_file}")
                plt.savefig(plot_file)
                plt.close()


def _get_random_aperture_curve(
    sqrtN: np.ndarray,
    coeff_file: Path
) -> np.ndarray:
    
    with open(coeff_file, 'rb') as f:
        coeffs = pickle.load(f)

    sigma1 = coeffs['sigma1']
    alpha = coeffs['plaw']['alpha']
    beta = coeffs['plaw']['beta']
    
    c = coeffs['spline']['c']
    t = coeffs['spline']['t']
    k = coeffs['spline']['k']
    sqrtN_max = coeffs['sqrtN_max']

    plaw_result = sigma1 * alpha * np.power(sqrtN, beta)

    from scipy.interpolate import BSpline
    bspline = BSpline(c=coeffs['spline']['c'], t=coeffs['spline']['t'], k=coeffs['spline']['k'])
    spline_result = sigma1 * bspline(sqrtN)

    result = np.where(sqrtN<0.85*sqrtN_max, spline_result, plaw_result)
    return result

def calibrate_flux_uncertainties(
    catalog: Table,
    random_aperture_dir: Path,
    filters: List[str],
    psfs_config,
    aperture_diameters, 
    flux_unit,
) -> Table:
    """
    Apply uncertainty calibration using random aperture measurements.
    
    Args:
        catalog: Master catalog table
        random_aperture_dir: Directory containing random aperture files
        tiles: List of tiles
        filters: List of filters
        
    Returns:
        Catalog with calibrated uncertainties
        
    Implementation notes for user:
    - Load random aperture measurements for all tiles/filters
    - For each aperture diameter in catalog:
      - Find corresponding random aperture measurements
      - Calculate empirical scatter vs theoretical errors
      - Fit scaling relations (e.g., scatter = a * theoretical_error + b)
      - Apply corrections to catalog flux errors
    - Update error columns with calibrated values
    - Add metadata about calibration (scaling factors, RMS, etc.)
    """
    logger = get_logger()
    logger.info("Calibrating uncertainties using random aperture measurements")
    
    for filt in filters:
        logger.info(f'Applying random aperture calibration for {filt}')

        # pixel_scale = self.images.get(filter=filt)[0].pixel_scale
        # err = self.objects[f'err_{filt}']

        if f'rms_{filt}' in catalog.columns:
            rms = catalog[f'rms_{filt}']
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                rms = 1/np.sqrt(np.maximum(catalog[f'wht_{filt}'], 1e-10))

        # auto (kron) photometry
        if f'f_auto_{filt}' in catalog.columns:
            sqrtN_kron = np.sqrt(catalog['kron1_area'])
            
            coeff_file = random_aperture_dir / f'random_apertures_{filt}_psfhom_coeffs.pickle'
            if psfs_config is not None: 
                if filt == psfs_config.target_filter or filt in psfs_config.inverse_filters:
                    coeff_file = random_aperture_dir / f'random_apertures_{filt}_coeffs.pickle'

            result = _get_random_aperture_curve(sqrtN_kron, coeff_file)
            rmsN = result * rms
            rmsN = (rmsN*u.nJy).to(flux_unit).value

            catalog.rename_column(f'e_auto_{filt}', f'e_auto_{filt}_uncal')
            catalog[f'e_auto_{filt}'] = np.where(np.isfinite(catalog[f'e_auto_{filt}_uncal']), rmsN, np.nan)
            
            if psfs_config is not None:
                if filt in psfs_config.inverse_filters:
                    catalog[f'e_auto_{filt}'] *= catalog[f'psf_corr_auto_{filt}']

            catalog[f'e_auto_{filt}'] *= catalog[f'kron_corr']

        if f'f_aper_nat_{filt}' in catalog.columns:
            # native resolution aperture photometry
            coeff_file = random_aperture_dir / f'random_apertures_{filt}_coeffs.pickle'
            with open(coeff_file, 'rb') as f:
                coeffs = pickle.load(f)
            pixel_scale = coeffs['pixel_scale']
            
            sqrtN_aper = np.sqrt(np.pi)*np.array(aperture_diameters)/pixel_scale/2

            result = _get_random_aperture_curve(sqrtN_aper, coeff_file)
            rmsN = np.outer(rms, result)
            rmsN = (rmsN*u.nJy).to(flux_unit).value

            catalog.rename_column(f'e_aper_nat_{filt}', f'e_aper_nat_{filt}_uncal')
            catalog[f'e_aper_nat_{filt}'] = np.where(np.isfinite(catalog[f'e_aper_nat_{filt}_uncal']), rmsN, np.nan)

        
        if f'f_aper_hom_{filt}' in catalog.columns:
            # psf-homogenized aperture photometry
            coeff_file = random_aperture_dir / f'random_apertures_{filt}_psfhom_coeffs.pickle'
            if psfs_config is not None: 
                if filt == psfs_config.target_filter or filt in psfs_config.inverse_filters:
                    coeff_file = random_aperture_dir / f'random_apertures_{filt}_coeffs.pickle'

            with open(coeff_file, 'rb') as f:
                coeffs = pickle.load(f)
            pixel_scale = coeffs['pixel_scale']
            
            sqrtN_aper = np.sqrt(np.pi)*np.array(aperture_diameters)/pixel_scale/2

            result = _get_random_aperture_curve(sqrtN_aper, coeff_file)
            rmsN = np.outer(rms, result)
            rmsN = (rmsN*u.nJy).to(flux_unit).value

            catalog.rename_column(f'e_aper_hom_{filt}', f'e_aper_hom_{filt}_uncal')
            catalog[f'e_aper_hom_{filt}'] = np.where(np.isfinite(catalog[f'e_aper_hom_{filt}_uncal']), rmsN, np.nan)

            if psfs_config is not None:
                if filt in psfs_config.inverse_filters:
                    catalog[f'e_aper_hom_{filt}'] *= catalog[f'psf_corr_aper_{filt}']

            catalog[f'e_aper_hom_{filt}'] *= catalog[f'aper_corr']

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
            psf_corrs[i,j] = np.sum(psf)/float(tab['aperture_sum'])

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

    for filt in filters:
        logger.info(filt)
        if filt in psfs_config.inverse_filters:
            if psfs_config.master_psf:
                logger.debug(f'{filt} - master PSF')
                psf_corr_file = psf_correction_dir / f'{filt}_psf_corr_grid.txt'
                catalog = _apply_psf_corr(catalog, filt, 'all', psf_corr_file, aperture_diameters)
            else:
                logger.debug(f'{filt} - per-tile PSF')
                for tile in tiles:
                    logger.debug(f'{filt} - {tile}')
                    psf_corr_file = psf_correction_dir / f'{filt}_{tile}_psf_corr_grid.txt'
                    catalog = _apply_psf_corr(catalog, filt, tile, psf_corr_file, aperture_diameters)
        else:
            if psfs_config.master_psf:
                logger.debug(f'{filt} - master PSF')
                psf_corr_file = psf_correction_dir / f'{psfs_config.target_filter}_psf_corr_grid.txt'
                catalog = _apply_psf_corr(catalog, filt, 'all', psf_corr_file, aperture_diameters)
            else:
                logger.debug(f'{filt} - per-tile PSF')
                for tile in tiles:
                    logger.debug(f'{filt} - {tile}')
                    psf_corr_file = psf_correction_dir / f'{psfs_config.target_filter}_{tile}_psf_corr_grid.txt'
                    catalog = _apply_psf_corr(catalog, filt, tile, psf_corr_file, aperture_diameters)

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
                cond = ~np.isfinite(catalog[f'f_{phot_type}_{filt}']) | (catalog[f'f_{phot_type}_{filt}'] == 0) | ~np.isfinite(catalog[f'e_{phot_type}_{filt}_uncal']) | (catalog[f'e_{phot_type}_{filt}_uncal'] == 0)
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
                    overwrite=overwrite
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