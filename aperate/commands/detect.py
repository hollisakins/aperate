"""Source detection and catalog creation."""

import gc
import logging
import multiprocessing as mp
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
import sep
sep.set_extract_pixstack(int(1e7))
sep.set_sub_object_limit(4096)

from ..core.catalog import find_catalog_in_directory
from ..core.logging import get_logger
from ..config.parser import load_config
from ..config.images import load_images_config, update_detection_paths_in_images_toml, check_detection_paths_completeness
from ..config.schema import BuildDetectionImageConfig, SourceDetectionConfig, DetectionParams

from astropy.wcs import FITSFixedWarning
warnings.simplefilter('ignore', category=FITSFixedWarning)

from scipy.optimize import curve_fit

def fit_pixel_dist(nsci, sigma_upper=1.0, maxiters=5, return_err=False):
    """
    Robust pixel distribution fitting using iterative Gaussian fitting to negative tail.
    
    Args:
        nsci: Input image data (will be flattened)
        sigma_upper: Upper sigma threshold for negative tail fitting
        maxiters: Maximum number of iterations
        return_err: Return uncertainties on the median/sigma (default: False)
        
    Returns:
        Tuple of (median, sigma) from robust Gaussian fit
    """
    logger = get_logger()

    if np.any(np.isnan(nsci)):
        nsci = nsci[np.isfinite(nsci)]

    def gauss(x, A, mu, sigma):
        return A * np.exp(-0.5 * (x - mu)**2 / sigma**2)

    # Initial guess using astropy.stats.sigma_clipped_stats
    _, median, std = sigma_clipped_stats(nsci, sigma_upper=sigma_upper, maxiters=maxiters) 

    # Iterative refinement by fitting Gaussian to negative tail
    for i in range(maxiters):
        logger.debug(f"    Fitting pixel distribution, iteration {i+1}/{maxiters}")
        # Center bins around current median estimate  
        if i == 0:
            bins = np.linspace(median - 5*std, median + 5*std, 100)
        else:
            bins = np.linspace(median - 3*std, median + 3*std, 100)
            
        y, bins = np.histogram(nsci, bins=bins)
        y = y / np.max(y)
        bc = 0.5 * (bins[1:] + bins[:-1])
        
        # Fit only negative tail (below median + sigma_upper*std)
        bins_to_fit = bc < median + sigma_upper * std
        
        try:
            popt, pcov = curve_fit(gauss, bc[bins_to_fit], y[bins_to_fit], 
                                 p0=[1, median, std])
            median, std = popt[1], abs(popt[2])  # Ensure positive std
        except Exception as e:
            logger.warning(f"    Gaussian fit failed on iteration {i+1}/{maxiters}: {str(e)}. Using current estimates.")
            break
    
    if return_err:
        perr = np.sqrt(np.diag(pcov))
        median_err, std_err = perr[1], perr[2]
        return median, std, median_err, std_err
    else:
        return median, std


def _validate_detection_paths(
    project_dir: Path,
    tile: str,
    detection_config,
    target_filter: Optional[str],
    inverse_filters: List[str]
) -> bool:
    """
    Validate file paths for detection image building, warn about missing data.
    
    Args:
        project_dir: Project directory
        tile: Tile name
        detection_config: Detection configuration
        target_filter: Target filter for homogenization
        inverse_filters: List of filters that used inverse homogenization
        
    Returns:
        True if at least one filter has valid data, False if none do
    """
    from ..config.images import load_images_config
    
    logger = get_logger()
    images_config = load_images_config(project_dir)
    
    valid_filters = 0
    
    for filter_name in detection_config.filters:
        image_files = images_config.get_image_files(filter_name, tile)
        if not image_files:
            logger.warning(f"    Images not found for filter {filter_name}, tile {tile}. Will skip this filter in detection.")
            continue
        
        # Determine which science path we'll need
        if detection_config.homogenized and filter_name not in inverse_filters and filter_name != target_filter:
            # Forward homogenization: filter homogenized to target
            sci_path = image_files.get_homogenized_path(target_filter)
            if not sci_path or not sci_path.exists():
                logger.warning(f"    Homogenized image missing for {filter_name} -> {target_filter}. Will skip this filter in detection.")
                continue
        else:
            # Target filter or inverse filter - use original science image
            sci_path = image_files.get_science_path()
            if not sci_path or not sci_path.exists():
                logger.warning(f"    Science image missing for {filter_name}, tile {tile}. Will skip this filter in detection.")
                continue
        
        # Validate weight/RMS path
        if image_files.has_extension('rms'):
            rms_path = image_files.get_rms_path()
            if not rms_path or not rms_path.exists():
                logger.warning(f"    RMS image missing for {filter_name}, tile {tile}. Will skip this filter in detection.")
                continue
        else:
            wht_path = image_files.get_weight_path()
            if not wht_path or not wht_path.exists():
                logger.warning(f"    Weight image missing for {filter_name}, tile {tile}. Will skip this filter in detection.")
                continue
        
        # If we get here, this filter has valid data
        valid_filters += 1
    
    if valid_filters == 0:
        logger.error(f"    No valid filters found for tile {tile}. Cannot create detection image.")
        return False
    
    if valid_filters < len(detection_config.filters):
        logger.warning(f"    Only {valid_filters}/{len(detection_config.filters)} filters have valid data for tile {tile}.")
    
    return True


def build_detection_image_for_tile(
    project_dir: Path,
    tile: str,
    detection_config: BuildDetectionImageConfig,
    overwrite: bool
) -> Optional[Path]:
    """
    Build detection image for a single tile.
    
    Args:
        project_dir: Project directory
        tile: Tile name
        detection_config: Detection image configuration
        overwrite: Whether to overwrite existing files
        
    Returns:
        Path to detection image, or None if failed
    """
    logger = get_logger()
    
    # Create detection_images directory
    detection_dir = project_dir / "detection_images"
    detection_dir.mkdir(exist_ok=True)
    
    # Construct output path
    output_path = detection_dir / f"detection_image_{detection_config.type}_{tile}.fits"
    
    # Check if already exists
    if output_path.exists() and not overwrite:
        logger.info(f"    Using existing detection image at {output_path}")
        return output_path
    
    logger.info(f"    Building {detection_config.type.upper()} detection image for tile {tile}")
    logger.debug(f"    Filters: {detection_config.filters}")
    logger.debug(f"    Homogenized: {detection_config.homogenized}")
    
    try:
        # Get PSF configuration for inverse filter info
        from ..config.psfs import load_psfs_config
        psfs_config = load_psfs_config(project_dir)
        
        if not psfs_config and detection_config.homogenized:
            raise ValueError("psfs.toml not found - run homogenization first if using homogenized images")
        
        target_filter = psfs_config.target_filter if psfs_config else None
        inverse_filters = psfs_config.inverse_filters if psfs_config else []
        
        # Validate paths and check if we have enough data to proceed
        if not _validate_detection_paths(project_dir, tile, detection_config, target_filter, inverse_filters):
            raise ValueError(f"No valid filters found for tile {tile}")
        
        # Route to appropriate builder
        if detection_config.type == 'ivw':
            detection_image_sci, detection_image_err, header = build_ivw_detection_image(
                project_dir, tile, detection_config.filters, detection_config.homogenized,
                target_filter, inverse_filters, detection_config.sigma_upper, detection_config.maxiters
            )
        elif detection_config.type == 'chisq':
            detection_image_sci, detection_image_err, header = build_chisq_detection_image(
                project_dir, tile, detection_config.filters, detection_config.homogenized,
                target_filter, inverse_filters
            )
        else:
            raise ValueError(f"Unknown detection image type: {detection_config.type}")
        
        # Add configuration to header
        header['DETTYPE'] = detection_config.type
        header['DETHOM'] = detection_config.homogenized
        header['DETFILT'] = ','.join(detection_config.filters)
        header['HISTORY'] = f'Detection image created with type={detection_config.type}'
        
        # Save detection image
        primary_hdu = fits.PrimaryHDU(header=fits.Header({'EXTEND':True}))
        sci_hdu = fits.ImageHDU(data=detection_image_sci, header=header)
        err_hdu = fits.ImageHDU(data=detection_image_err, header=header)
        hdul = fits.HDUList([primary_hdu, sci_hdu, err_hdu])
        hdul.writeto(output_path, overwrite=True)
        logger.info(f"    [bold green]Success:[/bold green] Saved detection image to {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"    [bold red]Failed:[/bold red] {str(e)}")
        return None


def build_ivw_detection_image(
    project_dir: Path,
    tile: str,
    filters: List[str],
    homogenized: bool,
    target_filter: str,
    inverse_filters: List[str],
    sigma_upper: float = 1.0,
    maxiters: int = 5
) -> Tuple[np.ndarray, np.ndarray, fits.Header]:
    """
    Build inverse-variance weighted detection image.
    
    Args:
        project_dir: Project directory
        tile: Tile name
        filters: List of filters to combine
        homogenized: Whether to use homogenized images
        target_filter: Target filter for homogenization
        inverse_filters: List of filters that used inverse homogenization
        sigma_upper: Upper sigma threshold for robust pixel distribution fitting
        maxiters: Maximum iterations for robust pixel distribution fitting
        
    Returns:
        Tuple of (detection_image_sci, detection_image_err, header)
    """
    logger = get_logger()
    inverse_filters = [i for i in inverse_filters if i in filters]
    
    logger.info(f"    [bold cyan]{tile}[/bold cyan]: Building IVW detection image")
    logger.info(f"    [bold cyan]{tile}[/bold cyan]: Detection bands: {', '.join(filters)}")
    if homogenized: 
        logger.info(f"    [bold cyan]{tile}[/bold cyan]: PSF-homogenized to target filter: {target_filter} (inverse filters {', '.join(inverse_filters)})")
    
    # Load images config for file access
    images_config = load_images_config(project_dir)
    
    shape = None
    # Initialize accumulators for incremental processing
    num = None  # numerator: sum(sci * wht) 
    den = None  # denominator: sum(wht)
    
    for filter_name in filters:
        # Get ImageFiles object for this filter/tile
        image_files = images_config.get_image_files(filter_name, tile)
        
        if not image_files:
            logger.debug(f"    [bold cyan]{tile}[/bold cyan]: Images not found for filter {filter_name}, tile {tile}. Skipping (already validated).")
            continue
        
        # Determine which science image to use
        if homogenized and filter_name not in inverse_filters and filter_name != target_filter:
            # Use homogenized image (forward homogenization)
            sci_path = image_files.get_homogenized_path(target_filter)
            logger.debug(f"    [bold cyan]{filter_name}-{tile}[/bold cyan]: using homogenized image -> {sci_path.name if sci_path else 'NOT FOUND'}")
        else:
            # Use original science image (target filter, inverse filter, or homogenized=False)
            sci_path = image_files.get_science_path()  
            if filter_name == target_filter:
                reason = "target filter"
            elif filter_name in inverse_filters:
                reason = "inverse filter"
            else:
                reason = "homogenized=False"
            logger.debug(f"    [bold cyan]{filter_name}-{tile}[/bold cyan]: using original science image ({reason}) -> {sci_path.name if sci_path else 'NOT FOUND'}")
        
        logger.info(f"    [bold cyan]{filter_name}-{tile}[/bold cyan]: Loading {sci_path}")
        with fits.open(sci_path) as hdul:
            sci_data = hdul[0].data.astype(np.float32)
            header = hdul[0].header
         
        if shape is None:
            shape = sci_data.shape
            # Initialize accumulators
            num = np.zeros(shape, dtype=np.float32)
            den = np.zeros(shape, dtype=np.float32)
        elif sci_data.shape != shape:
            logger.error(f"    [bold cyan]{filter_name}-{tile}[/bold cyan]: Image shape mismatch: {filter_name} has {sci_data.shape}, expected {shape}. All detection filters must have identical dimensions.")
            raise ValueError(f"Image shape mismatch: {filter_name} has {sci_data.shape}, expected {shape}. All detection filters must have identical dimensions.")

        if image_files.has_extension('rms'):
            rms_path = image_files.get_rms_path()
            logger.debug(f"    [bold cyan]{filter_name}-{tile}[/bold cyan]: rms image -> {rms_path.name if rms_path else 'NOT FOUND'}")
            with fits.open(rms_path) as hdul:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=RuntimeWarning)
                    wht_data = 1/(hdul[0].data**2)
        else: 
            logger.warning(f"    [bold cyan]{filter_name}-{tile}[/bold cyan]: no rms image found, falling back to wht")
            wht_path = image_files.get_weight_path()
            logger.debug(f"    [bold cyan]{filter_name}-{tile}[/bold cyan]: wht image -> {wht_path.name if wht_path else 'NOT FOUND'}")
            with fits.open(wht_path) as hdul:
                wht_data = hdul[0].data
        
        # Incremental accumulation - same logic as before but per-filter
        valid_mask = np.isfinite(sci_data) & np.isfinite(wht_data) & (wht_data > 0)
        
        # Accumulate: num += sci * wht, den += wht (only for valid pixels)
        num += np.where(valid_mask, sci_data * wht_data, 0)
        den += np.where(valid_mask, wht_data, 0)
        
        # Explicit cleanup after each filter to reduce memory usage
        del sci_data, wht_data, valid_mask
        gc.collect()

    logger.info(f"    [bold cyan]{tile}[/bold cyan]: Stacking images")
    # Set pixels with no valid data to NaN
    no_data_mask = den == 0
    num[no_data_mask] = np.nan
    den[no_data_mask] = np.nan
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        
        ivw = num/den
        nivw = ivw * np.sqrt(den)

    # Pre-process data once to eliminate redundancy
    data = nivw[np.isfinite(nivw)]
    
    # Robust pixel distribution fitting using iterative Gaussian fitting to negative tail
    logger.info(f"    [bold cyan]{tile}[/bold cyan]: Computing robust background statistics")
    median, sigma = fit_pixel_dist(data, sigma_upper=sigma_upper, maxiters=maxiters)

    with warnings.catch_warnings(): 
        warnings.simplefilter('ignore')    
        detec_sci = (nivw - median) / np.sqrt(den)
        detec_err = sigma / np.sqrt(den)

    return detec_sci, detec_err, header


def build_chisq_detection_image(
    project_dir: Path,
    tile: str,
    filters: List[str],
    homogenized: bool,
    target_filter: str,
    inverse_filters: List[str]
) -> Tuple[np.ndarray, fits.Header]:
    """
    Build chi-squared detection image.
    
    Args:
        project_dir: Project directory
        tile: Tile name
        filters: List of filters to combine
        homogenized: Whether to use homogenized images
        target_filter: Target filter for homogenization
        inverse_filters: List of filters that used inverse homogenization
        
    Returns:
        Tuple of (detection_image, header)
    """
    raise NotImplementedError


def _detect_sources_single_stage(
    detec,
    mask,
    kernel_type,
    kernel_params,
    thresh, 
    minarea, 
    deblend_nthresh, 
    deblend_cont, 
    filter_type='matched', 
    clean=True,
    clean_param=1.0,
):

    if kernel_type == 'tophat':
        from astropy.convolution import Tophat2DKernel
        if 'radius' not in kernel_params:
            raise ValueError('radius must be specified for Tophat2DKernel')
        if 'mode' not in kernel_params:
            kernel_params['mode'] = 'oversample'
        kernel = Tophat2DKernel(**kernel_params)
        kernel = kernel.array
        kernel = kernel/np.max(kernel)
    
    elif kernel_type == 'gaussian':
        if 'fwhm' not in kernel_params:
            raise ValueError('`fwhm` must be specified for Gaussian2DKernel')
        if 'size' not in kernel_params:
            kernel_params['size'] = kernel_params['fwhm'] * 3
        from photutils.segmentation import make_2dgaussian_kernel
        kernel = make_2dgaussian_kernel(**kernel_params).array

    else:
        raise ValueError('kernel_type must be one of `gaussian` or `tophat`')

    objs, segmap = sep.extract(
        detec, 
        thresh=thresh, minarea=minarea,
        deblend_nthresh=deblend_nthresh, deblend_cont=deblend_cont, mask=mask,
        filter_type=filter_type, filter_kernel=kernel, clean=clean, clean_param=clean_param,
        segmentation_map=True)

    ids = np.arange(len(objs))+1
    ids = ids.astype(int)
    objs = Table(objs)
    objs['id'] = ids

    # Fix angles
    objs['theta'][objs['theta']>np.pi/2] -= np.pi

    return objs, segmap    

def _merge_detections(
    objs1, segmap1,
    objs2, segmap2,
    dilate_kernel_size=None,
):
    '''
    Merges two lists of detections, objs1 and objs2, where all detections 
    in objs1 are included and detections in objs2 are included only if 
    they do not fall in a mask created as the union of segmap1 and an elliptical 
    mask derived from mask_scale_factor and mask_min_radius. 
    '''

    mask = segmap1 > 0

    if dilate_kernel_size is not None:
        from astropy.convolution import Tophat2DKernel
        from scipy.ndimage import binary_dilation
        kernel = Tophat2DKernel(dilate_kernel_size)
        mask = binary_dilation(mask.astype(int), kernel.array).astype(bool)
    
    # Mask objects in objs2 with segmap pixels that overlap the mask
    ids_to_mask = np.unique(segmap2 * mask)[1:] 
    segmap2[np.isin(segmap2,ids_to_mask)] = 0 # set segmap to 0
    objs2 = objs2[~np.isin(objs2['id'], ids_to_mask)]

    segmap2[segmap2>0] += np.max(segmap1)
    segmap = segmap1 + segmap2

    objs2['id'] += np.max(objs1['id'])
    objs = vstack((objs1,objs2))

    return objs, segmap

def load_existing_catalog(
    project_dir: Path, 
    project_name: str, 
    tile: str,
    detection_config: SourceDetectionConfig
) -> Optional[Tuple[Table, Optional[np.ndarray]]]:
    """
    Load existing catalog and segmentation map if they exist.
    
    Args:
        project_dir: Project directory
        project_name: Project name
        tile: Tile name  
        detection_config: Detection configuration for segmap path construction
        
    Returns:
        Tuple of (objs, segmentation_map) or None if not found
    """
    logger = get_logger()
    
    # Check for catalog
    catalog_path = project_dir / "catalogs" / f"catalog_{project_name}_{tile}.fits"
    if not catalog_path.exists():
        return None
    
    try:
        # Load catalog
        with fits.open(catalog_path) as hdul:
            objs = Table(hdul['SOURCES'].data)
        
        segmap = None
        if detection_config.save_segmap:
            # Try to load segmentation map
            detection_dir = project_dir / "detection_images"
            
            # Get detection type from catalog header 
            with fits.open(catalog_path) as hdul:
                detection_type = hdul[0].header.get('DETTYPE')
                if detection_type is None:
                    raise ValueError(f"Cannot determine detection type from catalog {catalog_path}. Catalog may be from older version - recreate with --overwrite.")
            
            segmap_path = detection_dir / f"segmap_{detection_type}_{detection_config.scheme}_{tile}.fits"
            if segmap_path.exists():
                with fits.open(segmap_path) as hdul:
                    segmap = hdul[0].data
                logger.debug(f"    Loaded existing segmentation map: {segmap_path}")
            else:
                logger.warning(f"    Catalog exists but segmentation map not found: {segmap_path}")
        
        logger.info(f"    Using existing catalog with {len(objs)} sources for {tile}")
        return objs, segmap
        
    except Exception as e:
        logger.warning(f"    Failed to load existing catalog {catalog_path}: {str(e)}")
        return None


def plot_detections(detec, segmap, objs, plot_path, windowed=False):
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from matplotlib.patches import Ellipse
    from photutils.utils.colormaps import make_random_cmap

    cmap1 = plt.colormaps['Greys_r']
    cmap1.set_bad('k')
    background_color='#000000ff'

    cmap2 = make_random_cmap(len(objs)+1)
    cmap2.colors[0] = colors.to_rgba(background_color)
    
    shape = np.shape(detec)
    ny, nx = shape

    # Determine aspect ratio and orientation
    if nx >= ny:
        # Landscape: panels top-and-bottom
        orientation = 'vertical'
        fig_height = 12
        fig_width = 12 * nx / ny
        fig, ax = plt.subplots(2, 1, figsize=(fig_width, fig_height), sharex=True, sharey=True, constrained_layout=True)
    else:
        # Portrait: panels side-by-side
        orientation = 'horizontal'
        fig_width = 12
        fig_height = 12 * ny / (2 * nx)
        fig, ax = plt.subplots(1, 2, figsize=(fig_width, fig_height), sharex=True, sharey=True, constrained_layout=True)

    ax[0].imshow(detec, norm=colors.LogNorm(vmin=1, vmax=300), cmap=cmap1, origin='lower')
    ax[1].imshow(segmap, cmap=cmap2, origin='lower')

    kronrad = objs['kronrad']
    if windowed:
        x, y = objs['xwin'], objs['ywin']
    else:
        x, y = objs['x'], objs['y']
    for i in range(len(objs)):
        e = Ellipse(xy=(x[i], y[i]), 
                    width=2.5*kronrad[i]*objs['a'][i], height=2.5*kronrad[i]*objs['b'][i], angle=np.degrees(objs['theta'][i]))
        e.set_facecolor('none')
        e.set_linewidth(0.15)
        e.set_edgecolor('lime')

        ax[0].add_artist(e)

    ax[0].axis('off')
    ax[1].axis('off')

    plt.savefig(plot_path, dpi=1000)
    plt.close()



def detect_sources_in_tile(
    detection_image_path: Path,
    detection_config: SourceDetectionConfig,
    project_dir: Path,
    tile: str,
    project_name: str,
    overwrite: bool
) -> Tuple[Optional[Table], Optional[np.ndarray]]:
    """
    Run source detection on detection image.
    
    Args:
        detection_image_path: Path to detection image
        detection_config: Source detection configuration
        project_dir: Project directory  
        tile: Tile name
        project_name: Project name for catalog paths
        overwrite: Whether to overwrite existing files
        
    Returns:
        Tuple of (objs, segmentation_map)
    """
    logger = get_logger()
    
    # Check for existing catalog if not overwriting
    if not overwrite:
        existing_result = load_existing_catalog(project_dir, project_name, tile, detection_config)
        if existing_result is not None:
            return existing_result
    
    logger.info(f"    Running {detection_config.scheme} source detection on {tile}")
    
    # Load detection image
    with fits.open(detection_image_path) as hdul:
        detection_image_sci = hdul[1].data
        detection_image_err = hdul[2].data
        detection_header = hdul[1].header

    detec = detection_image_sci / detection_image_err
    
    try:
        # Route to appropriate detection scheme
        if detection_config.scheme == 'single':
            objs, segmap = _detect_sources_single_stage(
                detec,
                mask=~np.isfinite(detec),
                # TODO update config handling to parse the following params from config steps.detection.source_detection directly 
                kernel_type=detection_config.kernel_type, 
                kernel_params=detection_config.kernel_params,
                thresh=detection_config.thresh, 
                minarea=detection_config.minarea, 
                deblend_nthresh=detection_config.deblend_nthresh, 
                deblend_cont=detection_config.deblend_cont, 
                filter_type=detection_config.filter_type, 
                clean=detection_config.clean,
                clean_param=detection_config.clean_param,
            )

        elif detection_config.scheme == 'hot+cold':
            logger.info('    Running cold-mode detection...')
            # First perform cold-mode detection
            objs_cold, segmap_cold = _detect_sources_single_stage(
                detec,
                mask=~np.isfinite(detec),
                kernel_type=detection_config.cold.kernel_type, 
                kernel_params=detection_config.cold.kernel_params,
                thresh=detection_config.cold.thresh, 
                minarea=detection_config.cold.minarea, 
                deblend_nthresh=detection_config.cold.deblend_nthresh, 
                deblend_cont=detection_config.cold.deblend_cont, 
                filter_type=detection_config.cold.filter_type, 
                clean=detection_config.cold.clean,
                clean_param=detection_config.cold.clean_param,
            )
            objs_cold['mode'] = 'cold'
            logger.info(f'    Detected {len(objs_cold)} objects from cold-mode detection')

            logger.info('    Running hot-mode detection...')
            objs_hot, segmap_hot = _detect_sources_single_stage(
                detec,
                mask=~np.isfinite(detec),
                kernel_type=detection_config.hot.kernel_type, 
                kernel_params=detection_config.hot.kernel_params,
                thresh=detection_config.hot.thresh, 
                minarea=detection_config.hot.minarea, 
                deblend_nthresh=detection_config.hot.deblend_nthresh, 
                deblend_cont=detection_config.hot.deblend_cont, 
                filter_type=detection_config.hot.filter_type, 
                clean=detection_config.hot.clean,
                clean_param=detection_config.hot.clean_param,
            )
            objs_hot['mode'] = 'hot'
            logger.info(f'    Detected {len(objs_hot)} objects from hot-mode detection')

            logger.info('    Merging hot+cold detections')
            objs, segmap = _merge_detections(
                objs_cold, segmap_cold,
                objs_hot, segmap_hot, 
                dilate_kernel_size = detection_config.cold_mask_dilate_size,
            )

        else:
            raise ValueError(f"Unknown detection scheme: {detection_config.scheme}")
        
        # Save segmentation map if requested
        if detection_config.save_segmap and segmap is not None:
            save_segmentation_map(
                project_dir, tile, segmap, detection_header, 
                detection_config, overwrite
            )
        
        logger.info(f"    [bold green]Success:[/bold green] Detected {len(objs)} sources in {tile}")
        return objs, segmap
        
    except Exception as e:
        logger.error(f"    [bold red]Failed:[/bold red] Source detection failed: {str(e)}")
        return None, None



def save_segmentation_map(
    project_dir: Path,
    tile: str,
    segmap: np.ndarray,
    detection_header: fits.Header,
    detection_config: SourceDetectionConfig,
    overwrite: bool
) -> None:
    """Save segmentation map to FITS file."""
    logger = get_logger()
    
    detection_dir = project_dir / "detection_images"
    
    # Get detection image type from header
    detection_type = detection_header.get('DETTYPE', 'unknown')
    
    segmap_path = detection_dir / f"segmap_{detection_type}_{detection_config.scheme}_{tile}.fits"
    
    if segmap_path.exists() and not overwrite:
        logger.debug(f"    Segmentation map exists for {tile}, skipping")
        return
    
    # Create header for segmap
    header = detection_header.copy()
    header['HISTORY'] = f'Segmentation map from {detection_config.scheme} detection'
    header['SCHEME'] = detection_config.scheme
    
    fits.writeto(segmap_path, segmap, header, overwrite=True)
    logger.debug(f"    Saved segmentation map to {segmap_path}")


def check_catalog_metadata_completeness(
    catalog_path: Path,
    source_config: SourceDetectionConfig,
    build_config: BuildDetectionImageConfig
) -> bool:
    """
    Check if catalog metadata is complete and up-to-date.
    
    Args:
        catalog_path: Path to catalog file
        source_config: Source detection configuration
        build_config: Detection image build configuration
        
    Returns:
        True if metadata needs updating, False if complete
    """
    logger = get_logger()
    
    try:
        with fits.open(catalog_path) as hdul:
            header = hdul[0].header
            
            # Check for required metadata fields
            required_fields = {
                'DETTYPE': build_config.type,
                'DETSCHEM': source_config.scheme,
                'DETFILT': ','.join(build_config.filters),
                'DETHOM': build_config.homogenized,
                'WINDOWED': source_config.windowed_positions
            }
            
            for field, expected_value in required_fields.items():
                if field not in header:
                    logger.debug(f"    Missing metadata field: {field}")
                    return True
                if header[field] != expected_value:
                    logger.debug(f"    Outdated metadata field: {field} = {header[field]} (expected {expected_value})")
                    return True
            
            return False  # All metadata is complete and up-to-date
            
    except Exception as e:
        logger.warning(f"    Could not check catalog metadata: {e}")
        return True  # Assume needs update if we can't read it


def update_catalog_metadata(
    catalog_path: Path, 
    source_config: SourceDetectionConfig,
    build_config: BuildDetectionImageConfig,
    project_name: str,
    tile: str
) -> None:
    """
    Update catalog metadata without regenerating the catalog.
    Used to add missing metadata to existing catalogs.
    
    Args:
        catalog_path: Path to catalog file
        source_config: Source detection configuration
        build_config: Detection image build configuration
        project_name: Project name
        tile: Tile name
    """
    logger = get_logger()
    
    try:
        # Load existing catalog
        with fits.open(catalog_path) as hdul:
            primary_hdu = hdul[0].copy()
            table_hdu = hdul[1].copy()  # Preserve existing source data
        
        # Update primary header with missing/outdated metadata
        header = primary_hdu.header
        header['PROJECT'] = project_name
        header['TILE'] = tile
        header['DETTYPE'] = build_config.type
        header['DETSCHEM'] = source_config.scheme
        header['DETFILT'] = ','.join(build_config.filters)
        header['DETHOM'] = build_config.homogenized
        header['WINDOWED'] = source_config.windowed_positions
        
        # Add update history
        header['HISTORY'] = f'Metadata updated by aperate detect for tile {tile}'
        
        # Save updated catalog
        hdu_list = fits.HDUList([primary_hdu, table_hdu])
        hdu_list.writeto(catalog_path, overwrite=True)
        
        logger.debug(f"    Updated catalog metadata: {catalog_path}")
        
    except Exception as e:
        logger.error(f"    Failed to update catalog metadata {catalog_path}: {str(e)}")
        raise


def create_tile_catalog(
    project_dir: Path,
    project_name: str,
    tile: str,
    objs: Table,
    metadata: Dict[str, any]
) -> Path:
    """
    Create FITS catalog for a single tile.
    
    Args:
        project_dir: Project directory
        project_name: Project name from config
        tile: Tile name
        objs: Detected sources table
        metadata: Additional metadata to store
        
    Returns:
        Path to created catalog
    """
    logger = get_logger()
    
    # Create catalogs directory
    catalogs_dir = project_dir / "catalogs"
    catalogs_dir.mkdir(exist_ok=True)
    
    catalog_path = catalogs_dir / f"catalog_{project_name}_{tile}.fits"
    
    # Create primary HDU with metadata
    primary_hdu = fits.PrimaryHDU()
    header = primary_hdu.header
    
    # Add metadata to header
    header['PROJECT'] = project_name
    header['TILE'] = tile
    header['NSOURCES'] = len(objs)
    
    for key, value in metadata.items():
        header[key.upper()] = value
    
    header['HISTORY'] = f'Catalog created by aperate detect for tile {tile}'
    
    # Create table HDU for sources
    table_hdu = fits.BinTableHDU(objs, name='SOURCES')
    
    # Create HDU list and save
    hdu_list = fits.HDUList([primary_hdu, table_hdu])
    hdu_list.writeto(catalog_path, overwrite=True)
    
    logger.info(f"    [bold green]Success:[/bold green] Created catalog {catalog_path}")
    return catalog_path

def compute_kron_radius(detec, segmap, objs, windowed=False):
    mask = np.isnan(detec)

    if windowed:
        x, y = objs['xwin'], objs['ywin']
    else:
        x, y = objs['x'], objs['y']

    a, b, theta = objs['a'], objs['b'], objs['theta']
    kronrad, krflag = sep.kron_radius(detec, x, y, a, b, theta, r=6.0, mask=mask, seg_id=objs['id'], segmap=segmap)
    kronrad[kronrad < 1.] = 1.
    objs['kronrad'] = kronrad
    objs['flag'] |= krflag
    return objs

def compute_windowed_positions(detec, segmap, objs):

    if 'kronrad' not in objs.columns:
        raise Exception("Compute kron radius before computing windowed positions!")

    mask = np.isnan(detec)

    flux, fluxerr, flag = sep.sum_ellipse(
        detec, objs['x'], objs['y'], 
        objs['a'], objs['b'], objs['theta'], 
        2.5*objs['kronrad'], subpix=1, mask=mask,
        seg_id=objs['id'], segmap=segmap,
    )

    rhalf, rflag = sep.flux_radius(
        detec, objs['x'], objs['y'], 
        6.*objs['a'], 0.5, 
        seg_id=objs['id'], segmap=segmap,
        mask=mask, normflux=flux, subpix=5
    )

    sig = 2. / 2.35 * rhalf  
    xwin, ywin, flag = sep.winpos(
        detec, objs['x'], objs['y'], sig, 
        mask=mask)

    objs['xwin'] = xwin
    objs['ywin'] = ywin
    return objs


def process_detection_tile(project_dir, tile, build_config, source_config, config_name, overwrite):
    """Process a single tile for detection."""
    logger = get_logger()
    
    logger.info(f"Processing tile [bold cyan]{tile}[/bold cyan]")
    
    try:
        # Step 1: Build detection image
        detection_image_path = build_detection_image_for_tile(
            project_dir, tile, build_config, overwrite
        )
        
        if detection_image_path is None:
            logger.warning(f"Skipping tile {tile} due to detection image failure")
            return False
        
        # Step 2: Run source detection
        objs, segmap = detect_sources_in_tile(
            detection_image_path, source_config, project_dir, tile, config_name, overwrite
        )
        
        if objs is None:
            logger.warning(f"Skipping tile {tile} due to source detection failure")
            return False
        
        logger.info(f"    Processing {len(objs)} sources for {tile}")

        # If we don't have kronrad (i.e., we created a fresh catalog instead of loading an existing one)
        if 'kronrad' not in objs.columns:
            with fits.open(detection_image_path) as hdul:
                detec = hdul[1].data / hdul[2].data
                header = hdul[1].header
                wcs = WCS(header)

            # Compute the kron radius
            logger.info(f"    Computing kron radius")
            compute_kron_radius(detec, segmap, objs, windowed=False)

            # Compute windowed positions if requested
            if source_config.windowed_positions:
                logger.info(f"    Computing windowed positions")
                compute_windowed_positions(detec, segmap, objs)
                # Recompute kron radius using new windowed positions
                logger.info(f"    Re-computing kron radius")
                compute_kron_radius(detec, segmap, objs, windowed=True)
            
            # Add RA/Dec coordinates
            if source_config.windowed_positions:
                coords = wcs.pixel_to_world(objs['xwin'], objs['ywin'])
            else:
                coords = wcs.pixel_to_world(objs['x'], objs['y'])
            ra = coords.ra.value
            dec = coords.dec.value
            objs['ra'] = ra
            objs['dec'] = dec

        if source_config.plot:
            plot_path = detection_image_path.with_suffix('.pdf')
            if not plot_path.exists() or overwrite:
                logger.info(f"    Plotting detections")
                
                with fits.open(detection_image_path) as hdul:
                    detec = hdul[1].data / hdul[2].data
                    header = hdul[1].header
                    wcs = WCS(header)

                plot_detections(detec, segmap, objs, plot_path, windowed=source_config.windowed_positions)

        # Step 3: Create catalog
        catalog_path = project_dir / "catalogs" / f"catalog_{config_name}_{tile}.fits"
        
        if not catalog_path.exists() or overwrite:
            metadata = {
                'dettype': build_config.type,
                'detschem': source_config.scheme,
                'windowed': source_config.windowed_positions
            }
            if source_config.scheme == "master":
                metadata.update({
                    'detfilte': build_config.filters[0] if build_config.filters else 'unknown',
                })
            
            # Create catalog with proper extension naming
            catalog_path = create_tile_catalog(
                project_dir, config_name, tile, objs, metadata
            )
        
        # Force garbage collection after each tile
        gc.collect()
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing tile {tile}: {str(e)}")
        return False


def process_detection_tile_wrapper(args):
    """Wrapper function for multiprocessing."""
    project_dir, tile, build_config, source_config, config_name, overwrite = args
    return process_detection_tile(project_dir, tile, build_config, source_config, config_name, overwrite)


@click.command("detect")
@click.option(
    "--project-dir", 
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path.cwd(),
    help="Project directory (default: current directory)"
)
@click.option(
    "--tiles",
    type=str,
    default='',
    help="Run only for a subset of tiles (comma-separated)"
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite existing detection images and catalogs"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable debug output"
)
@click.option(
    "--parallel",
    type=int,
    default=1,
    help="Number of parallel processes (default: 1, no parallelization)"
)
def detect_cmd(project_dir, tiles, overwrite, verbose, parallel):
    """Build detection images and detect sources."""
    logger = get_logger()
    
    # Set debug level if verbose flag is used
    if verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Load config
    config_path = project_dir / "config.toml"
    if not config_path.exists():
        logger.error(f"[bold red]Error:[/bold red] No config.toml found in {project_dir}")
        raise click.ClickException("Config not found")
    
    config = load_config(config_path)
    logger.info(f"Loaded config for project: [bold cyan]{config.name}[/bold cyan]")
    
    # Load images config
    images_config = load_images_config(project_dir)
    available_tiles = []
    for filter_name in config.filters:
        available_tiles.extend(images_config.get_tiles_for_filter(filter_name))
    available_tiles = list(set(available_tiles))  # Remove duplicates
    
    # Parse tile override
    if tiles:
        tiles_to_process = [t.strip() for t in tiles.split(',') if t.strip() in available_tiles]
    else:
        tiles_to_process = available_tiles
    
    if not tiles_to_process:
        logger.error("[bold red]Error:[/bold red] No tiles available for detection")
        raise click.ClickException("No tiles to process")
    
    logger.info(f"Processing detection for {len(tiles_to_process)} tiles: {tiles_to_process}")
    
    # Get detection configuration
    detection_config = config.detection
    build_config = detection_config.build_detection_image
    source_config = detection_config.source_detection
    
    logger.info(f"Detection image type: [bold cyan]{build_config.type}[/bold cyan]")
    logger.info(f"Detection filters: {build_config.filters}")
    logger.info(f"Homogenized images: {build_config.homogenized}")
    logger.info(f"Detection scheme: [bold cyan]{source_config.scheme}[/bold cyan]")
    
    # Load and display PSF configuration info if using homogenized images
    if build_config.homogenized:
        from ..config.psfs import load_psfs_config
        psfs_config = load_psfs_config(project_dir)
        
        if psfs_config:
            logger.info(f"Target filter: [bold cyan]{psfs_config.target_filter}[/bold cyan]")
            if psfs_config.inverse_filters:
                logger.info(f"Inverse filters: [bold yellow]{[i for i in psfs_config.inverse_filters if i in build_config.filters]}[/bold yellow]")
                logger.info("Note: Inverse filters will use original science images")
            else:
                logger.info("No inverse filters detected")
        else:
            logger.warning("psfs.toml not found - homogenized images may not be available")
    
    # Process tiles
    if parallel > 1 and len(tiles_to_process) > 1:
        logger.info(f"Processing {len(tiles_to_process)} tiles using {parallel} processes")
        # Prepare tasks for multiprocessing
        tasks = [(project_dir, tile, build_config, source_config, config.name, overwrite) 
                 for tile in tiles_to_process]
        
        with mp.Pool(processes=parallel) as pool:
            results = pool.map(process_detection_tile_wrapper, tasks)
        
        successful_tiles = [tile for tile, success in zip(tiles_to_process, results) if success]
    else:
        # Sequential processing
        successful_tiles = []
        for tile in tiles_to_process:
            success = process_detection_tile(project_dir, tile, build_config, source_config, config.name, overwrite)
            if success:
                successful_tiles.append(tile)
    
    # Update images.toml with detection image and segmap paths
    if successful_tiles:
        # Check if detection paths need updating
        needs_detection_path_update = check_detection_paths_completeness(
            project_dir=project_dir,
            tiles=successful_tiles,
            detection_type=build_config.type,
            detection_scheme=source_config.scheme
        )
        
        if needs_detection_path_update:
            logger.info("Updating images.toml with detection file paths...")
            try:
                update_detection_paths_in_images_toml(
                    project_dir=project_dir,
                    tiles=successful_tiles,
                    detection_type=build_config.type,
                    detection_scheme=source_config.scheme
                )
            except Exception as e:
                logger.warning(f"Failed to update images.toml with detection paths: {e}")
        else:
            logger.debug("Detection paths in images.toml are up-to-date")
    
    # Summary
    if successful_tiles:
        logger.info(f"[bold green]Success:[/bold green] Detection completed for {len(successful_tiles)} tiles")
        logger.info(f"Processed tiles: {successful_tiles}")
    else:
        logger.error("[bold red]Error:[/bold red] No tiles were processed successfully")
        raise click.ClickException("Detection failed for all tiles")