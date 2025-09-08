"""Core PSF homogenization functionality."""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.modeling import models, fitting
import sep

from ..core.logging import get_logger


# def measure_psf_fwhm_old(psf_data: np.ndarray, pixel_scale: float) -> Tuple[float, float]:
#     """
#     Measure PSF FWHM using flux radius method.
    
#     Args:
#         psf_data: 2D PSF array
#         pixel_scale: Pixel scale in arcsec/pixel
        
#     Returns:
#         Tuple of (fwhm_pixels, fwhm_arcsec)
#     """
#     logger = get_logger()
    
#     # Debug info
#     logger.debug(f"PSF shape: {psf_data.shape}")
#     logger.debug(f"PSF min/max: {np.min(psf_data):.6f} / {np.max(psf_data):.6f}")
#     logger.debug(f"PSF sum: {np.sum(psf_data):.6f}")
#     logger.debug(f"Pixel scale: {pixel_scale:.4f} arcsec/pixel")
    
#     # Ensure PSF is normalized and positive
#     psf_data = psf_data.copy()
#     psf_data[psf_data < 0] = 0
#     psf_norm = psf_data / np.max(psf_data)
    
#     # Find center of PSF (center of mass)
#     y, x = np.indices(psf_data.shape)
#     xcen = np.sum(x * psf_norm)
#     ycen = np.sum(y * psf_norm)
    
#     logger.debug(f"PSF center: ({xcen:.2f}, {ycen:.2f})")
#     logger.debug(f"Expected center: ({psf_data.shape[1]/2:.2f}, {psf_data.shape[0]/2:.2f})")
    
#     # Method 1: Try SEP flux radius
#     try:
#         # Convert to C-contiguous array for SEP
#         psf_c = np.ascontiguousarray(psf_data, dtype=np.float32)
        
#         # Create a simple background (zero)
#         bkg = np.zeros_like(psf_c)
        
#         # Extract the PSF as a single source
#         objects = sep.extract(psf_c - bkg, thresh=1e-6, minarea=5)
        
#         logger.debug(f"SEP found {len(objects)} objects")
        
#         if len(objects) > 0:
#             # Take the brightest (should be our PSF)
#             idx = np.argmax(objects['flux'])
#             obj = objects[idx]
            
#             logger.debug(f"Brightest object at ({obj['x']:.2f}, {obj['y']:.2f})")
#             logger.debug(f"Object a={obj['a']:.2f}, b={obj['b']:.2f}, flux={obj['flux']:.6f}")
            
#             # Measure flux radius at 0.5 to get half-light radius
#             # SEP expects arrays, so wrap scalars
#             rhalf, flag = sep.flux_radius(
#                 psf_c, 
#                 [obj['x']], 
#                 [obj['y']], 
#                 [6.0 * obj['a']],  # max radius
#                 [0.5],  # fraction of light
#                 normflux=[1.],
#                 subpix=5
#             )
            
#             # FWHM = 2 * half-light radius
#             fwhm_pixels = 2.0 * rhalf[0]
#             fwhm_arcsec = fwhm_pixels * pixel_scale
            
#             logger.debug(f"SEP measured PSF FWHM: {fwhm_pixels:.2f} pixels ({fwhm_arcsec:.3f} arcsec)")
            
#             return fwhm_pixels, fwhm_arcsec
#     except Exception as e:
#         logger.debug(f"SEP method failed: {e}")
    
#     # Method 2: Direct calculation using cumulative flux
#     # This is more robust for PSF models
#     try:
#         # Calculate radial distances from center
#         r = np.sqrt((x - xcen)**2 + (y - ycen)**2)
        
#         # Sort pixels by radius
#         r_flat = r.flatten()
#         psf_flat = psf_norm.flatten()
#         sort_idx = np.argsort(r_flat)
#         r_sorted = r_flat[sort_idx]
#         psf_sorted = psf_flat[sort_idx]
        
#         # Calculate cumulative flux
#         cumflux = np.cumsum(psf_sorted)
        
#         # Find radius containing 50% of flux
#         idx_half = np.searchsorted(cumflux, 0.5)
#         rhalf = r_sorted[idx_half]
        
#         logger.debug(f"Direct method: total flux = {cumflux[-1]:.6f}")
#         logger.debug(f"Direct method: half-light radius = {rhalf:.2f} pixels")
#         logger.debug(f"Direct method: flux at rhalf = {cumflux[idx_half]:.6f}")
        
#         # FWHM = 2 * half-light radius
#         fwhm_pixels = 2.0 * rhalf
#         fwhm_arcsec = fwhm_pixels * pixel_scale
        
#         logger.debug(f"Direct method PSF FWHM: {fwhm_pixels:.2f} pixels ({fwhm_arcsec:.3f} arcsec)")
        
#         return fwhm_pixels, fwhm_arcsec
        
#     except Exception as e:
#         logger.warning(f"Direct method also failed: {e}")
        
#         # Final fallback: Gaussian second moment
#         r2 = (x - xcen)**2 + (y - ycen)**2
#         sigma2 = np.sum(r2 * psf_norm)
#         sigma = np.sqrt(sigma2)
#         fwhm_pixels = 2.355 * sigma
#         fwhm_arcsec = fwhm_pixels * pixel_scale
        
#         logger.debug(f"Fallback method: sigma = {sigma:.2f} pixels")
#         logger.debug(f"Fallback method PSF FWHM: {fwhm_pixels:.2f} pixels ({fwhm_arcsec:.3f} arcsec)")
        
#         return fwhm_pixels, fwhm_arcsec


def measure_psf_fwhm(psf_data: np.ndarray, pixel_scale: float) -> Tuple[float, float]:
    """
    Measure PSF FWHM using 2D Gaussian fitting.
    
    Uses astropy.modeling to fit a 2D Gaussian to the PSF and extract FWHM.
    FWHM = 2.355 * sigma for Gaussian profiles.
    
    Args:
        psf_data: 2D PSF array
        pixel_scale: Pixel scale in arcsec/pixel
        
    Returns:
        Tuple of (fwhm_pixels, fwhm_arcsec)
        
    Raises:
        Exception if Gaussian fitting fails (no fallbacks)
    """
    logger = get_logger()
    
    # Debug info
    logger.debug(f"PSF shape: {psf_data.shape}")
    logger.debug(f"PSF min/max: {np.min(psf_data):.6f} / {np.max(psf_data):.6f}")
    logger.debug(f"PSF sum: {np.sum(psf_data):.6f}")
    logger.debug(f"Pixel scale: {pixel_scale:.4f} arcsec/pixel")
    
    # Ensure PSF is positive and normalized
    psf_clean = psf_data.copy()
    psf_clean[psf_clean < 0] = 0
    
    if np.sum(psf_clean) == 0:
        raise ValueError("PSF contains no positive flux")
    
    # Find peak location for initial center guess
    peak_idx = np.unravel_index(np.argmax(psf_clean), psf_clean.shape)
    y_peak, x_peak = peak_idx[0], peak_idx[1]
    
    # Use fixed initial sigma estimate (reasonable for space telescope PSFs)
    sigma_est = 3.0  # pixels
    
    logger.debug(f"Peak location: ({x_peak}, {y_peak})")
    logger.debug(f"Initial sigma estimate: {sigma_est:.2f} pixels (fixed)")
    
    # Set up 2D Gaussian model with initial parameters
    gaussian_init = models.Gaussian2D(
        amplitude=np.max(psf_clean),
        x_mean=x_peak,
        y_mean=y_peak,
        x_stddev=sigma_est,
        y_stddev=sigma_est
    )
    
    logger.debug(f"Initial model parameters:")
    logger.debug(f"  amplitude: {gaussian_init.amplitude.value:.6f}")
    logger.debug(f"  x_mean: {gaussian_init.x_mean.value:.2f}")
    logger.debug(f"  y_mean: {gaussian_init.y_mean.value:.2f}")
    logger.debug(f"  x_stddev: {gaussian_init.x_stddev.value:.2f}")
    logger.debug(f"  y_stddev: {gaussian_init.y_stddev.value:.2f}")
    
    # Create coordinate grids
    y_grid, x_grid = np.mgrid[0:psf_clean.shape[0], 0:psf_clean.shape[1]]
    
    # Fit the model
    fitter = fitting.LevMarLSQFitter()
    gaussian_fit = fitter(gaussian_init, x_grid, y_grid, psf_clean)
    
    # Check if fitting was successful
    if fitter.fit_info['fvec'] is None:
        raise RuntimeError("Gaussian fitting failed to converge")
    
    logger.debug(f"Fitted model parameters:")
    logger.debug(f"  amplitude: {gaussian_fit.amplitude.value:.6f}")
    logger.debug(f"  x_mean: {gaussian_fit.x_mean.value:.2f}")
    logger.debug(f"  y_mean: {gaussian_fit.y_mean.value:.2f}")
    logger.debug(f"  x_stddev: {gaussian_fit.x_stddev.value:.2f}")
    logger.debug(f"  y_stddev: {gaussian_fit.y_stddev.value:.2f}")
    
    # Extract sigma values
    sigma_x = gaussian_fit.x_stddev.value
    sigma_y = gaussian_fit.y_stddev.value
    
    # Handle elliptical PSFs by averaging sigma values
    sigma_avg = (sigma_x + sigma_y) / 2
    
    # Convert to FWHM
    fwhm_pixels = 2.355 * sigma_avg
    fwhm_arcsec = fwhm_pixels * pixel_scale
    
    logger.debug(f"Fitted sigma_x: {sigma_x:.2f}, sigma_y: {sigma_y:.2f}")
    logger.debug(f"Average sigma: {sigma_avg:.2f} pixels")
    logger.debug(f"Gaussian fit FWHM: {fwhm_pixels:.2f} pixels ({fwhm_arcsec:.3f} arcsec)")
    
    return fwhm_pixels, fwhm_arcsec


def determine_inverse_filters(
    filter_fwhms: Dict[str, float], 
    target_filter: str
) -> List[str]:
    """
    Determine which filters need inverse homogenization.
    
    Args:
        filter_fwhms: Dictionary of filter -> FWHM (pixels)
        target_filter: Target filter name
        
    Returns:
        List of filter names that need inverse homogenization
    """
    
    if target_filter not in filter_fwhms:
        raise ValueError(f"Target filter {target_filter} not found in FWHM measurements")
    
    target_fwhm = filter_fwhms[target_filter]
    inverse_filters = []
    
    for filt, fwhm in filter_fwhms.items():
        if filt == target_filter:
            continue
            
        if fwhm > target_fwhm:
            inverse_filters.append(filt)
    
    return inverse_filters


def create_matching_kernel(
    source_psf_path: Path,
    target_psf_path: Path,
    reg_fact: float = 1e-4,
    angle: float = 0.0
) -> np.ndarray:
    """
    Create convolution kernel to match source PSF to target PSF using pypher.
    
    Args:
        source_psf_path: Path to source PSF FITS file
        target_psf_path: Path to target PSF FITS file  
        reg_fact: Regularization factor for pypher
        angle: Rotation angle for kernel
        
    Returns:
        Convolution kernel array
    """
    try:
        from pypher.pypher import homogenization_kernel
    except ImportError:
        raise ImportError("pypher package required for PSF homogenization")
    
    logger = get_logger()
    
    # Load PSFs
    with fits.open(source_psf_path) as hdul:
        source_psf = hdul[0].data.astype(np.float64)
    
    with fits.open(target_psf_path) as hdul:
        target_psf = hdul[0].data.astype(np.float64)
    
    # Ensure PSFs are same size
    if source_psf.shape != target_psf.shape:
        raise ValueError(f"PSF size mismatch: {source_psf.shape} vs {target_psf.shape}")
    
    # Normalize PSFs
    source_psf = source_psf / np.sum(source_psf)
    target_psf = target_psf / np.sum(target_psf)
        
    # Use pypher to create kernel
    # Note: pypher expects (target, source) order
    kernel, kernel_fft = homogenization_kernel(
        target_psf, 
        source_psf,
        reg_fact=reg_fact,
        clip=True
    )
    
    return kernel


def homogenize_image(
    image_path: Path,
    kernel: np.ndarray,
    output_path: Path,
) -> None:
    """
    Convolve image with kernel to homogenize PSF.
    
    Args:
        image_path: Path to input science image
        kernel: Convolution kernel
        output_path: Path for output homogenized image
    """
    logger = get_logger()
    
    # Load science image
    with fits.open(image_path) as hdul:
        sci_data = hdul[0].data.astype(np.float32)
        header = hdul[0].header.copy()
    
    # Apply convolution using FFT
    from astropy.convolution import convolve_fft

    # Convolve science image
    homogenized = convolve_fft(sci_data, kernel, normalize_kernel=False, allow_huge=True)
    
    # Ensure that pixels that were originally nan stay nan
    homogenized[~np.isfinite(sci_data)] = np.nan

    # Ensure that the dtype remains float32
    homogenized = homogenized.astype(sci_data.dtype)
    
    # Update header
    header['HISTORY'] = f'PSF-homogenized with pypher'
    header['HOMPSF'] = True
    header['HOMKERN'] = f'Kernel sum: {np.sum(kernel):.6f}'
    
    # Save homogenized image
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(output_path, homogenized, header, overwrite=True)


def get_psf_paths_for_filter(
    project_dir: Path,
    filter_name: str,
    tile: str,
    master_psf: bool
) -> Path:
    """
    Get PSF file path for a given filter/tile combination.
    
    Args:
        project_dir: Project directory
        filter_name: Filter name
        tile: Tile name
        master_psf: Whether using master PSF
        
    Returns:
        Path to PSF file
    """
    psf_dir = project_dir / "psfs"
    
    if master_psf:
        return psf_dir / f"psf_{filter_name}_master.fits"
    else:
        return psf_dir / f"psf_{filter_name}_{tile}.fits"