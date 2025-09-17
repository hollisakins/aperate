"""PSF generation by stacking stars."""

import gc
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import click
import numpy as np
import sep
sep.set_extract_pixstack(int(1e7))
sep.set_sub_object_limit(4096)

import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.nddata import Cutout2D, NoOverlapError, PartialOverlapError
from astropy.stats import sigma_clipped_stats
from scipy.stats import binned_statistic_2d
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from photutils.segmentation import make_2dgaussian_kernel

from ..core.catalog import find_catalog_in_directory, AperateCatalog
from ..core.logging import get_logger
from ..config.parser import load_config
from ..config.images import load_images_config, update_psf_paths_in_images_toml
from ..utils.helpers import Gaussian

from astropy.wcs import FITSFixedWarning
warnings.simplefilter('ignore', category=FITSFixedWarning)

# Nominal PSF FWHMs (in arcsec) for different filters
# These are used for aperture radius calculations
NOMINAL_PSF_FWHMS = {
    'vis': 0.140,
    'f435w': 0.045,
    'f606w': 0.075,
    'f814w': 0.100,
    'f098m': 0.210,
    'f070w': 0.023,
    'f090w': 0.030,
    'f115w': 0.037,
    'f140m': 0.046,
    'f150w': 0.049,
    'f162m': 0.053,
    'f164n': 0.054,
    'f150w2': 0.045,
    'f182m': 0.060,
    'f187n': 0.061,
    'f200w': 0.064,
    'f210m': 0.068,
    'f212n': 0.069,
    'f250m': 0.082,
    'f277w': 0.088,
    'f300m': 0.097,
    'f322w2': 0.096,
    'f323n': 0.106,
    'f335m': 0.109,
    'f356w': 0.114,
    'f360m': 0.118,
    'f405n': 0.132,
    'f410m': 0.133,
    'f430m': 0.139,
    'f444w': 0.140,
    'f460m': 0.151,
    'f466n': 0.152,
    'f470n': 0.154,
    'f480m': 0.157,
}

def process_tile_psf(project_dir: Path, filter_name: str, tile: str, 
                     fwhm_min: float, fwhm_max: float, max_ellip: float, 
                     min_snr: float, max_snr: float, psf_size: int, 
                     checkplots: bool, az_average: bool) -> Optional[Dict]:
    """
    Process a single tile to extract PSF star data.
    
    Returns a dictionary containing PSF star positions and metadata
    for later combination into a master PSF.
    """
    logger = get_logger()
    images_config = load_images_config(project_dir)
    image_files = images_config.get_image_files(filter_name, tile)
    
    if not image_files or not image_files.has_extension('sci'):
        logger.warning(f"    No science image found for {filter_name} {tile}")
        return None
    
    sci_path = image_files.get_science_path()
    if not sci_path.exists():
        logger.warning(f"    Science image does not exist: {sci_path}")
        return None
    
    # Read science data with context manager
    with fits.open(sci_path) as hdul:
        sci_data = hdul[0].data.astype(np.float32, order='C')  # C-order for SEP
        header = hdul[0].header
        wcs = WCS(header)
        pixel_scale = np.abs(np.round(wcs.proj_plane_pixel_scales()[0].to('arcsec').value,3)) # arcsec/pixel
    
    # Read error data if available, or compute from weight for HST
    err_data = None
    if image_files.has_extension('err'):
        # JWST case: direct error array
        err_path = image_files.get_error_path()
        if err_path.exists():
            with fits.open(err_path) as hdul:
                err_data = hdul[0].data.astype(np.float32, order='C')
                
    elif image_files.has_extension('wht'):
        # HST case: compute error from weight array
        wht_path = image_files.get_weight_path()
        if wht_path.exists():
            with fits.open(wht_path) as hdul:
                wht_data = hdul[0].data.astype(np.float32, order='C')
                # Compute error as 1/sqrt(weight) where weight > 0
                err_data = np.full_like(wht_data, np.inf)
                valid_mask = wht_data > 0
                err_data[valid_mask] = 1.0 / np.sqrt(wht_data[valid_mask])
    
    # Set up sep options (PRESERVE EXACT ORIGINAL VALUES)
    sep.set_extract_pixstack(int(1e7))
    sep.set_sub_object_limit(2048)
    
    # Create detection kernel (PRESERVE EXACT ORIGINAL VALUES)
    kernel = make_2dgaussian_kernel(fwhm=1.5, size=5).array
    
    # Detect sources in the image to identify stars
    logger.info(f'    [bold cyan]{filter_name}-{tile}[/bold cyan]: Detecting sources')
    mask = np.isnan(sci_data)

    objs, segmap = sep.extract(
        sci_data, 
        err=err_data,
        thresh=2.5,
        minarea=8, 
        deblend_nthresh=32, 
        deblend_cont=0.005,
        mask=mask,
        filter_type='matched',
        filter_kernel=kernel,
        clean=True,
        clean_param=1.0,
        segmentation_map=True)
    
    seg_id = np.arange(1, len(objs)+1, dtype=np.int32)
    objs['theta'][objs['theta']>np.pi/2] -= np.pi
    nanparams = np.isnan(objs['a']) | np.isnan(objs['b'])
    objs['a'][nanparams] = 5
    objs['b'][nanparams] = 5
    
    # Compute fluxes (iteration 1) - PRESERVE EXACT CALCULATIONS
    logger.info(f'    [bold cyan]{filter_name}-{tile}[/bold cyan]: Computing kron radius (iteration 1)')
    kronrad, krflag = sep.kron_radius(
        sci_data, objs['x'], objs['y'], 
        objs['a'], objs['b'], objs['theta'], 
        6.0, mask=mask, seg_id=seg_id, segmap=segmap,
    )

    logger.info(f'    [bold cyan]{filter_name}-{tile}[/bold cyan]: Computing fluxes (iteration 1)')
    flux, fluxerr, flag = sep.sum_ellipse(
        sci_data, objs['x'], objs['y'], 
        objs['a'], objs['b'], objs['theta'], 
        2.5*kronrad, subpix=1, mask=mask,
        seg_id=seg_id, segmap=segmap,
    )
    flag |= krflag
    
    logger.info(f'    [bold cyan]{filter_name}-{tile}[/bold cyan]: Computing FWHM (iteration 1)')
    rhalf, rflag = sep.flux_radius(
        sci_data, objs['x'], objs['y'], 
        6.*objs['a'], 0.5, 
        seg_id=seg_id, segmap=segmap,
        mask=mask, normflux=flux, subpix=5
    )

    logger.info(f'    [bold cyan]{filter_name}-{tile}[/bold cyan]: Computing windowed positions')
    sig = 2. / 2.35 * rhalf  
    xwin, ywin, flag = sep.winpos(
        sci_data, objs['x'], objs['y'], sig, 
        mask=mask)

    # Compute fluxes (iteration 2) - PRESERVE EXACT CALCULATIONS
    logger.info(f'    [bold cyan]{filter_name}-{tile}[/bold cyan]: Computing kron radius (iteration 2)')
    kronrad, krflag = sep.kron_radius(
        sci_data, xwin, ywin,
        objs['a'], objs['b'], objs['theta'], 
        6.0, mask=mask, seg_id=seg_id, segmap=segmap,
    )
    
    logger.info(f'    [bold cyan]{filter_name}-{tile}[/bold cyan]: Computing fluxes (iteration 2)')
    flux, fluxerr, flag = sep.sum_ellipse(
        sci_data, xwin, ywin,
        objs['a'], objs['b'], objs['theta'], 
        2.5*kronrad, subpix=1, mask=mask,
        seg_id=seg_id, segmap=segmap,
    )
    flag |= krflag

    logger.info(f'    [bold cyan]{filter_name}-{tile}[/bold cyan]: Computing FWHM (iteration 2)')
    rhalf, rflag = sep.flux_radius(
        sci_data, xwin, ywin, 
        6.*objs['a'], 0.5, 
        seg_id=seg_id, segmap=segmap,
        mask=mask, normflux=flux, subpix=5
    )
    flag |= rflag
    fwhm = 2 * rhalf

    # Filter for PSF stars - PRESERVE EXACT CALCULATIONS
    aperture_radius = 3*NOMINAL_PSF_FWHMS[filter_name]/pixel_scale
    flux, fluxerr, cflag = sep.sum_circle(
        sci_data, xwin, ywin, 
        aperture_radius, err=err_data
    )
    flag |= cflag
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        snr = flux/fluxerr
    ellip = (objs['a']-objs['b'])/(objs['a']+objs['b'])
    cond = np.logical_and.reduce((
        fwhm > fwhm_min, 
        fwhm < fwhm_max,
        snr > min_snr, 
        snr < max_snr, 
        ellip < max_ellip,
    ))
    
    if len(cond[cond]) == 0:
        logger.warning(f'    [bold cyan]{filter_name}-{tile}[/bold cyan]: Could not identify any PSF stars')
        return None
    else:
        logger.info(f'    [bold cyan]{filter_name}-{tile}[/bold cyan]: Identified {len(cond[cond])} PSF stars')

    # Generate checkplot if requested
    if checkplots:
        generate_psf_checkplot(project_dir, filter_name, tile, fwhm, snr, cond, 
                             fwhm_min, fwhm_max, min_snr, max_snr, pixel_scale)

    # Extract PSF star positions
    x = xwin[cond]
    y = ywin[cond]
    
    # Return PSF data for this tile (no image data to save memory)
    return {
        'filter': filter_name,
        'tile': tile,
        'x': x,
        'y': y,
        'psf_size': psf_size,
        'az_average': az_average,
        'sci_path': str(sci_path),
        'wcs': wcs,
        'pixel_scale': pixel_scale
    }


def generate_psf_checkplot(project_dir: Path, filter_name: str, tile: str, 
                          fwhm, snr, cond, fwhm_min: float, fwhm_max: float, 
                          min_snr: float, max_snr: float, pixel_scale: float):
    """Generate diagnostic checkplot for PSF star selection"""
    psf_dir = project_dir / "psfs"
    psf_dir.mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(constrained_layout=True)
    ax.scatter(fwhm, snr, linewidths=0, s=1, color='k', marker='o')
    ax.scatter(fwhm[cond], snr[cond], linewidths=0, s=2, color='tab:red', marker='o')
    ax.set_xlabel('FWHM [pix]')
    ax.set_xlim(0.7, 100)
    ax.set_ylabel('SNR')
    ax.set_ylim(3, 1e5)
    ax.loglog()
    ax.plot([fwhm_min, fwhm_min], [min_snr, max_snr], linewidth=1, color='r', linestyle='--')
    ax.plot([fwhm_max, fwhm_max], [min_snr, max_snr], linewidth=1, color='r', linestyle='--')
    ax.plot([fwhm_min, fwhm_max], [min_snr, min_snr], linewidth=1, color='r', linestyle='--')
    ax.plot([fwhm_min, fwhm_max], [max_snr, max_snr], linewidth=1, color='r', linestyle='--')
    ax.axvline(NOMINAL_PSF_FWHMS[filter_name]/pixel_scale, linewidth=1, color='b', linestyle=':')
    
    checkplot_path = psf_dir / f"psf_{filter_name}_{tile}_fwhm.pdf"
    plt.savefig(checkplot_path)

    logger = get_logger()
    logger.info(f"    [bold cyan]{filter_name}-{tile}[/bold cyan]: Saved FWHM checkplot to {checkplot_path}")

    plt.close()

def plot_psf(psf_data: np.ndarray, plot_path: Path):
    """
    Plot the PSF.
    """
    cmap = plt.colormaps['inferno']
    cmap.set_bad('k')
    norm = mpl.colors.LogNorm(vmin=1e-5, vmax=1)

    fig, ax = plt.subplots(figsize=(6,5),constrained_layout=True)
    im = ax.imshow(psf_data/np.max(psf_data), cmap=cmap, norm=norm)
    cbar = fig.colorbar(mappable=im)
    ax.set_title(plot_path.stem, fontsize=8)

    ax.axis('off')
    plt.savefig(plot_path, dpi=300)
    logger = get_logger()
    logger.info(f"Saved PSF plot to {plot_path}")
    plt.close()
        


def create_individual_psf_from_tile(project_dir: Path, filter_name: str, tile: str,
                                   psf_data: Dict) -> np.ndarray:
    """Create PSF from individual tile data"""
    logger = get_logger()
    
    x = psf_data['x']
    y = psf_data['y']
    psf_size = psf_data['psf_size']
    az_average = psf_data['az_average']
    sci_path = Path(psf_data['sci_path'])
    
    logger.info(f'Computing PSF model for {filter_name} {tile}')
    
    # Re-read the image data for PSF creation
    with fits.open(sci_path) as hdul:
        sci_data = hdul[0].data.astype(np.float32)
    
    # Create PSF grid - PRESERVE EXACT ORIGINAL CALCULATIONS
    x_grid, y_grid = np.zeros(psf_size**2 * len(x)), np.zeros(psf_size**2 * len(y))
    z_grid = np.zeros(psf_size**2 * len(y))
    
    for i in range(len(x)):
        xi, yi = np.arange(psf_size)+0.5, np.arange(psf_size)+0.5
        xi, yi = np.meshgrid(xi, yi)
        xi, yi = xi.flatten(), yi.flatten()
        try:
            cutout = Cutout2D(sci_data, position=(x[i],y[i]), size=psf_size, mode='strict')
            # Correct for sub-pixel offset in the windowed position 
            dx = cutout.input_position_cutout[0] - psf_size//2
            dy = cutout.input_position_cutout[1] - psf_size//2
            xi += dx
            yi += dy
            zi = cutout.data.flatten()
        except (NoOverlapError, PartialOverlapError):
            zi = np.full(psf_size**2, np.nan)

        if az_average:
            theta = np.random.uniform(0, np.pi)
            xp = (xi-psf_size/2) * np.cos(theta) - (yi-psf_size/2) * np.sin(theta) + psf_size/2
            yp = (xi-psf_size/2) * np.sin(theta) + (yi-psf_size/2) * np.cos(theta) + psf_size/2
            xi = xp
            yi = yp

        x_grid[i*(psf_size**2):(i+1)*(psf_size)**2] = xi
        y_grid[i*(psf_size**2):(i+1)*(psf_size)**2] = yi
        z_grid[i*(psf_size**2):(i+1)*(psf_size)**2] = zi

    x_bins = np.arange(psf_size)
    y_bins = np.arange(psf_size) 
    x_bins = np.append(x_bins, x_bins[-1]+1)
    y_bins = np.append(y_bins, y_bins[-1]+1)

    psf, _, _, _ = binned_statistic_2d(
        x_grid, y_grid, z_grid, 
        bins=(x_bins, y_bins), 
        statistic=np.nanmedian
    )
    psf = psf.T

    # Subtract the background from the PSF - PRESERVE EXACT CALCULATIONS
    logger.info('Background-subtracting PSF')
    mean, median, std = sigma_clipped_stats(psf)
    bins = np.linspace(-3*std+median, 5*std+median, 80)
    bc = 0.5*(bins[1:]+bins[:-1])
    h, _ = np.histogram(psf.flatten(), bins)
    h = h / np.max(h)
    p0 = [1, bc[np.argmax(h)], std]
    
    try:
        popt, pcov = curve_fit(Gaussian, bc[bc<2*std+median], h[bc<2*std+median], p0=p0)
        popt, pcov = curve_fit(Gaussian, bc[bc<1*std+median], h[bc<1*std+median], p0=popt)
        bkg = popt[1]
    except:
        bkg = median
    
    psf = psf - bkg
    psf[psf < 0] = 0
    xg, yg = np.arange(psf_size), np.arange(psf_size)
    xg, yg = np.meshgrid(xg, yg)
    dist = np.sqrt((xg-psf_size/2)**2 + (yg-psf_size/2)**2)
    psf[dist > psf_size/2] = 0
    
    logger.info('PSF model computed successfully')
    return psf


def create_master_psf(project_dir: Path, filter_name: str, 
                     psf_data_list: List[Dict]) -> np.ndarray:
    """Combine PSF data from multiple tiles into a master PSF"""
    logger = get_logger()
    
    # Collect all star positions
    x_all = np.array([])
    y_all = np.array([])
    which_tile = np.array([])
    
    for i, psf_data in enumerate(psf_data_list):
        x_all = np.append(x_all, psf_data['x'])
        y_all = np.append(y_all, psf_data['y'])
        which_tile = np.append(which_tile, [i]*len(psf_data['x']))
    
    which_tile = which_tile.astype(int)
    logger.info(f'Combining {len(x_all)} stars from {len(psf_data_list)} tiles')
    
    # Use psf_size from first tile (should be consistent)
    psf_size = psf_data_list[0]['psf_size']
    az_average = psf_data_list[0]['az_average']
    
    logger.info('Computing master PSF model')
    
    # Create PSF grid - PRESERVE EXACT ORIGINAL CALCULATIONS
    x_grid, y_grid = np.zeros(psf_size**2 * len(x_all)), np.zeros(psf_size**2 * len(y_all))
    z_grid = np.zeros(psf_size**2 * len(y_all))
    
    # Load tile images as needed (memory efficient)
    tile_data_cache = {}
    
    for i in range(len(x_all)):
        xi, yi = np.arange(psf_size)+0.5, np.arange(psf_size)+0.5
        xi, yi = np.meshgrid(xi, yi)
        xi, yi = xi.flatten(), yi.flatten()
        
        # Get the tile for this star
        tile_idx = which_tile[i]
        tile_info = psf_data_list[tile_idx]
        sci_path = Path(tile_info['sci_path'])
        
        # Load tile data if not cached
        if tile_idx not in tile_data_cache:
            with fits.open(sci_path) as hdul:
                tile_data_cache[tile_idx] = hdul[0].data.astype(np.float32)
        
        sci_data = tile_data_cache[tile_idx]
        
        try:
            cutout = Cutout2D(sci_data, position=(x_all[i],y_all[i]), size=psf_size, mode='strict')
            dx = cutout.input_position_cutout[0] - psf_size//2
            dy = cutout.input_position_cutout[1] - psf_size//2
            xi += dx
            yi += dy
            zi = cutout.data.flatten()
        except (NoOverlapError, PartialOverlapError):
            zi = np.full(psf_size**2, np.nan)

        if az_average:
            theta = np.random.uniform(0, np.pi)
            xp = (xi-psf_size/2) * np.cos(theta) - (yi-psf_size/2) * np.sin(theta) + psf_size/2
            yp = (xi-psf_size/2) * np.sin(theta) + (yi-psf_size/2) * np.cos(theta) + psf_size/2
            xi = xp
            yi = yp

        x_grid[i*(psf_size**2):(i+1)*(psf_size)**2] = xi
        y_grid[i*(psf_size**2):(i+1)*(psf_size)**2] = yi
        z_grid[i*(psf_size**2):(i+1)*(psf_size)**2] = zi
        
        # Clear cache if this is the last star from this tile
        remaining_tiles = [which_tile[j] for j in range(i+1, len(x_all))]
        if tile_idx not in remaining_tiles:
            del tile_data_cache[tile_idx]

    x_bins = np.arange(psf_size)
    y_bins = np.arange(psf_size) 
    x_bins = np.append(x_bins, x_bins[-1]+1)
    y_bins = np.append(y_bins, y_bins[-1]+1)

    psf, _, _, _ = binned_statistic_2d(
        x_grid, y_grid, z_grid, 
        bins=(x_bins, y_bins), 
        statistic=np.nanmedian
    )
    psf = psf.T

    # Background subtract - PRESERVE EXACT CALCULATIONS
    logger.info('Background-subtracting master PSF')
    mean, median, std = sigma_clipped_stats(psf)
    bins = np.linspace(-3*std+median, 5*std+median, 80)
    bc = 0.5*(bins[1:]+bins[:-1])
    h, _ = np.histogram(psf.flatten(), bins)
    h = h / np.max(h)
    p0 = [1, bc[np.argmax(h)], std]
    
    try:
        popt, pcov = curve_fit(Gaussian, bc[bc<2*std+median], h[bc<2*std+median], p0=p0)
        popt, pcov = curve_fit(Gaussian, bc[bc<1*std+median], h[bc<1*std+median], p0=popt)
        bkg = popt[1]
    except:
        bkg = median
    
    psf = psf - bkg
    psf[psf < 0] = 0
    xg, yg = np.arange(psf_size), np.arange(psf_size)
    xg, yg = np.meshgrid(xg, yg)
    dist = np.sqrt((xg-psf_size/2)**2 + (yg-psf_size/2)**2)
    psf[dist > psf_size/2] = 0
    
    logger.info('Master PSF created successfully')
    return psf


def save_psf(psf_data: np.ndarray, psf_file: Path, wcs: WCS, pixel_scale: float):
    """Save PSF to FITS file with proper WCS"""
    logger = get_logger()
    
    logger.info(f'Writing to {psf_file}')
    
    psf_size = psf_data.shape[0]
    new_wcs = WCS(naxis=2)
    new_wcs.wcs.crpix = [psf_size/2, psf_size/2]
    ps = pixel_scale / 3600  # Convert arcsec to degrees
    new_wcs.wcs.cdelt = [-ps, ps]
    new_wcs.wcs.crval = wcs.wcs.crval
    new_wcs.wcs.ctype = wcs.wcs.ctype

    # Ensure directory exists
    psf_file.parent.mkdir(parents=True, exist_ok=True)
    
    fits.writeto(psf_file, data=psf_data, header=new_wcs.to_header(), overwrite=True)
    logger.info(f'PSF saved successfully')


def generate_psfs_for_filter(project_dir: Path, filter_name: str, 
                            config, overwrite: bool, checkplots: bool):
    """Generate PSFs for a single filter, processing one tile at a time."""
    logger = get_logger()
    images_config = load_images_config(project_dir)
    psf_config = config.psf_generation

    # Get tiles for this filter
    tiles = images_config.get_tiles_for_filter(filter_name)
    if not tiles:
        logger.warning(f"No tiles found for filter {filter_name}")
        return
    
    # Check if PSF files already exist
    if psf_config.master_psf:
        master_psf_path = project_dir / "psfs" / f"psf_{filter_name}_master.fits"
        tile_psf_paths = [project_dir / "psfs" / f"psf_{filter_name}_{tile}.fits" for tile in tiles]

        if master_psf_path.exists() and not overwrite:
            logger.info(f"    [bold]{filter_name}[/bold]: Master PSF already exists, skipping")
            return
    else:
        tile_psf_paths = [project_dir / "psfs" / f"psf_{filter_name}_{tile}.fits" for tile in tiles]
        if all([path.exists() for path in tile_psf_paths]) and not overwrite:
            logger.info(f"    [bold]{filter_name}[/bold]: All tile PSFs already exist, skipping")
            return

    logger.info(f"Processing {len(tiles)} tiles for filter [bold cyan]{filter_name}[/bold cyan]")
        
    # Check if filter has specific parameters
    if filter_name in psf_config.filter_params:
        filter_params = psf_config.filter_params[filter_name]
        fwhm_min = filter_params.fwhm_min
        fwhm_max = filter_params.fwhm_max
    else:
        logger.error(f"No fwhm_min,fwhm_max parameters found for {filter_name}")
        raise ValueError(f"No fwhm_min,fwhm_max parameters found for {filter_name}")

    psf_size = psf_config.psf_size
    master_psf = psf_config.master_psf
    
    # Process each tile individually
    psf_data_list = []
    
    for tile in tiles:
        logger.info(f"Working on [bold cyan]{filter_name}[/bold cyan], tile [bold cyan]{tile}[/bold cyan]")
      
        # Process this tile
        psf_data = process_tile_psf(
            project_dir=project_dir,
            filter_name=filter_name, 
            tile=tile,
            fwhm_min=fwhm_min,
            fwhm_max=fwhm_max,
            max_ellip=psf_config.max_ellip,
            min_snr=psf_config.min_snr,
            max_snr=psf_config.max_snr,
            psf_size=psf_size,
            checkplots=checkplots,
            az_average=psf_config.az_average
        )
        
        if psf_data:
            psf_data_list.append(psf_data)

        # Force garbage collection after each tile
        gc.collect()

    for tile, psf_data in zip(tiles, psf_data_list):

        logger.info(f"Creating individual PSF for {filter_name}-{tile}")
        psf = create_individual_psf_from_tile(
            project_dir, 
            filter_name, 
            tile, 
            psf_data)

        psf_path = project_dir / "psfs" / f"psf_{filter_name}_{tile}.fits"
        save_psf(psf, psf_path, psf_data['wcs'], psf_data['pixel_scale'])

        if checkplots:
            psf_plot_path = project_dir / "psfs" / f"psf_{filter_name}_{tile}.pdf"
            plot_psf(psf, psf_plot_path)
        
        logger.info(f"[bold green]Success:[/bold green] PSF created for {filter_name}-{tile}")

    if master_psf:
        logger.info(f"Creating master PSF for {filter_name}")

        master_psf_data = create_master_psf(project_dir, filter_name, psf_data_list)
        
        # Use WCS from first tile
        master_path = project_dir / "psfs" / f"psf_{filter_name}_master.fits"
        save_psf(master_psf_data, master_path, psf_data_list[0]['wcs'], 
                psf_data_list[0]['pixel_scale'])
        
        if checkplots: 
            master_plot_path = project_dir / "psfs" / f"psf_{filter_name}_master.pdf"
            plot_psf(master_psf_data, master_plot_path)
        
        logger.info(f"[bold green]Success:[/bold green] Master PSF created for {filter_name}")
    
    # Final cleanup
    gc.collect()


@click.command("psf")
@click.option(
    "--project-dir", 
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path.cwd(),
    help="Project directory (default: current directory)"
)
@click.option(
    "--filters", 
    type=str, 
    default='', 
    help="Run only for a subset of filters (comma-separated)"
)
@click.option(
    "--overwrite", 
    is_flag=True,
    help="Overwrite existing PSF files"
)
@click.option(
    "--checkplots", 
    is_flag=True,
    help="Generate diagnostic checkplots"
)
def psf_cmd(project_dir, filters, overwrite, checkplots):
    """Generate PSF models by stacking stars."""
    logger = get_logger()
    
    # Load config
    config_path = project_dir / "config.toml"
    if not config_path.exists():
        logger.error(f"[bold red]Error:[/bold red] No config.toml found in {project_dir}")
        raise click.ClickException("Config not found")
    
    config = load_config(config_path)
    logger.info(f"Loaded config for project: [bold cyan]{config.name}[/bold cyan]")
    
    # Validate PSF size
    if not config.psf_generation.psf_size % 2:
        logger.error('PSF size must be odd.')
        raise ValueError('PSF size must be odd.')

    # Check images.toml exists
    images_config = load_images_config(project_dir)
    available_filters = images_config.get_filters()
    override_filters = filters.split(',') if filters != '' else available_filters
    
    # Process filters that are both in config and available in images
    filters_to_process = [f for f in config.filters if f in available_filters and f in override_filters]
    
    if not filters_to_process:
        logger.error(f"[bold red]Error:[/bold red] No filters available for PSF generation")
        raise click.ClickException("No filters available")
    
    logger.info(f"Processing PSFs for filters: {filters_to_process}")
    
    # Implement CLI override logic for checkplots
    # CLI flag takes precedence over config file setting
    use_checkplots = checkplots or config.psf_generation.checkplots
    
    # Process each filter
    for filter_name in filters_to_process:
        try:
            generate_psfs_for_filter(
                project_dir=project_dir,
                filter_name=filter_name,
                config=config,
                overwrite=overwrite,
                checkplots=use_checkplots,
            )
        except Exception as e:
            logger.error(f"Error generating PSF for {filter_name}: {str(e)}")
            raise
            if len(filters_to_process) == 1:
                raise
            else:
                logger.warning(f"Continuing with remaining filters...")
                continue
    
    # ALWAYS update images.toml with PSF paths for all processed filters
    # This runs even if PSF generation was skipped because files already exist
    logger.info("Updating images.toml with PSF file paths...")
    for filter_name in filters_to_process:
        try:
            # Get tiles for this filter
            images_config = load_images_config(project_dir)
            tiles = images_config.get_tiles_for_filter(filter_name)
            
            if tiles:
                update_psf_paths_in_images_toml(
                    project_dir=project_dir,
                    filter_name=filter_name,
                    tiles=tiles,
                    master_psf=config.psf_generation.master_psf
                )
        except Exception as e:
            logger.warning(f"Failed to update images.toml for {filter_name}: {e}")
    
    # TODO: Update state tracking when implemented
    # This would track PSF generation status, timestamps, etc.
    
    logger.info(f"[bold green]Success:[/bold green] PSF generation completed")
  