"""Aperture photometry measurements."""

import gc
import logging
import multiprocessing as mp
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import click
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from ..core.catalog import find_catalog_in_directory
from ..core.logging import get_logger
from ..config.parser import load_config
from ..config.images import load_images_config, load_segmentation_map, ImagesConfig, ImageFiles
from ..config.psfs import load_psfs_config, PSFsConfig
from ..config.schema import PhotometryConfig, AperateConfig
from ..utils.helpers import get_unit_conversion

from astropy.wcs import FITSFixedWarning
warnings.simplefilter('ignore', category=FITSFixedWarning)

import sep

@dataclass
class PhotometryContext:
    """Encapsulates all resources needed for photometry operations.
    
    This context object provides a clean interface to access all necessary
    data and configuration for photometry measurements, avoiding the need
    to pass many parameters between functions.
    """
    catalog: Table
    segmap: np.ndarray
    tile: str
    images_config: ImagesConfig
    photometry_config: PhotometryConfig
    psfs_config: Optional[PSFsConfig]
    main_config: AperateConfig
    windowed: bool
    overwrite: bool
    _image_cache: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    _flux_cache: Dict[str, object] = field(default_factory=dict, repr=False)
    _persistent: set = field(default_factory=set, repr=False)

    def load_image(self, path, hdu_index=0, persistent=False) -> np.ndarray:
        """Load a FITS image with byteorder conversion, caching by path and HDU.

        persistent=True keeps the array across clear_image_cache() calls; used for images that
        every filter needs again (the target filter's native image for the inverse-filter
        corrections), so a tile reads them once instead of once per filter (F10).
        """
        key = f'{path}#{hdu_index}'
        if key not in self._image_cache:
            with fits.open(path) as hdul:
                data = hdul[hdu_index].data
                data = data.astype(data.dtype.newbyteorder('='))
            self._image_cache[key] = data
        if persistent:
            self._persistent.add(key)
        return self._image_cache[key]

    def load_error(self, image_files) -> np.ndarray:
        """Load error map (from ERR or WHT extension), with caching."""
        if image_files.has_extension('err'):
            return self.load_image(image_files.get_error_path())
        elif image_files.has_extension('wht'):
            wht_path = image_files.get_weight_path()
            key = f"err_from_{wht_path}"
            if key not in self._image_cache:
                wht = self.load_image(wht_path)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    self._image_cache[key] = 1.0 / np.sqrt(wht)
            return self._image_cache[key]
        else:
            return None

    def ones_image(self, shape) -> np.ndarray:
        """A persistent float32 image of ones (for counting the pixels sep sums)."""
        key = f'ones#{shape[0]}x{shape[1]}'
        if key not in self._image_cache:
            self._image_cache[key] = np.ones(shape, dtype=np.float32)
            self._persistent.add(key)
        return self._image_cache[key]

    def clear_image_cache(self, all=False):
        """Free cached image arrays (except the persistent ones unless all=True)."""
        for key in list(self._image_cache):
            if all or key not in self._persistent:
                del self._image_cache[key]
        if all:
            self._persistent.clear()

    def get_image_files(self, filter_name: str) -> Optional[ImageFiles]:
        """Get image files for any filter in the current tile.
        
        Args:
            filter_name: Filter name to get images for
            
        Returns:
            ImageFiles object or None if not found
        """
        return self.images_config.get_image_files(filter_name, self.tile)
    
    def is_target_filter(self, filter_name: str) -> bool:
        """Check if filter is the PSF homogenization target filter."""
        if not self.psfs_config:
            return False
        return filter_name == self.psfs_config.target_filter
    
    def is_inverse_filter(self, filter_name: str) -> bool:
        """Check if filter requires inverse PSF homogenization."""
        if not self.psfs_config:
            return False
        return filter_name in self.psfs_config.inverse_filters
    
    def get_logger(self) -> logging.Logger:
        """Get the logger instance."""
        return get_logger()


# Primary-header keywords recording which filters' columns carry each multiplicative
# correction (comma-separated filter lists). They make partial reruns safe: see
# _apply_multiplicative_correction (review finding F03).
CORRECTION_STATE_KEYS = {'KRONCORR': 'kron_corr', 'APERCORR': 'aper_corr'}


def _mark_raw(catalog: Table, meta_key: str, filter_name: str) -> None:
    """Record that filter_name's columns have just been (re)measured and are uncorrected."""
    applied = [f for f in catalog.meta.get(meta_key, []) if f != filter_name]
    catalog.meta[meta_key] = applied


def _apply_multiplicative_correction(catalog: Table, column_prefixes, new_corr, old_corr, meta_key: str):
    """Multiply the per-filter columns by new_corr so that the result does not depend on how
    many times, or for which subset of filters, photometry has been run.

    column_prefixes: e.g. ('f_auto_', 'e_auto_'); every column starting with the first prefix
        defines a filter (the part after the prefix).
    new_corr: correction to apply (shape broadcastable to the columns).
    old_corr: the correction column stored by the previous run (None if there was none).
    meta_key: catalog.meta entry listing the filters whose columns were corrected by the
        previous run. Those columns are divided by old_corr first (entries where old_corr is
        not finite or zero become NaN), so a rerun of `photometry --filters <subset>` never
        multiplies an untouched filter twice (review finding F03: F277W x1.64 after an
        F444W-only rerun in cosmos_v1.3). Filters re-measured in this run were removed from
        the list by _mark_raw. Returns the list of filters now corrected.
    """
    applied = set(catalog.meta.get(meta_key, []))
    first = column_prefixes[0]
    filters = [c[len(first):] for c in catalog.colnames if c.startswith(first)]
    for filt in filters:
        for prefix in column_prefixes:
            col = f'{prefix}{filt}'
            if col not in catalog.colnames:
                continue
            values = np.asarray(catalog[col], dtype=float)
            if filt in applied:
                if old_corr is None:
                    raise ValueError(f'{col} is recorded as corrected but the previous {meta_key} column is missing')
                with np.errstate(divide='ignore', invalid='ignore'):
                    ok = np.isfinite(old_corr) & (old_corr != 0)
                    values = np.where(ok, values / np.where(ok, old_corr, 1.0), np.nan)
            catalog[col] = values * new_corr
    catalog.meta[meta_key] = filters
    return filters


def load_tile_catalog(project_dir: Path, project_name: str, tile: str) -> Tuple[Optional[Table], bool]:
    """
    Load detection catalog for a specific tile.
    
    Args:
        project_dir: Project directory
        project_name: Project name
        tile: Tile name
        
    Returns:
        Tuple of (catalog_table, windowed_positions_flag)
    """
    logger = get_logger()
    catalog_path = project_dir / "catalogs" / f"catalog_{project_name}_{tile}.fits"
    
    if not catalog_path.exists():
        return None, False
    
    try:
        with fits.open(catalog_path) as hdul:
            catalog = Table(hdul['SOURCES'].data)
            # Get windowed flag from metadata, default to False for backward compatibility
            windowed = bool(hdul[0].header.get('WINDOWED', False))
            # Which filters already carry the multiplicative corrections (see
            # _apply_multiplicative_correction); comma-separated lists in the primary header.
            for key, corr_col in CORRECTION_STATE_KEYS.items():
                if key in hdul[0].header:
                    catalog.meta[key] = [f for f in str(hdul[0].header[key]).split(',') if f]
                elif corr_col in catalog.colnames:
                    # Catalog written before the state keywords existed: a stored correction
                    # column means a full run applied it to every filter present.
                    prefix = 'f_auto_' if key == 'KRONCORR' else 'f_aper_hom_'
                    catalog.meta[key] = [c[len(prefix):] for c in catalog.colnames if c.startswith(prefix)]
                else:
                    catalog.meta[key] = []
        logger.debug(f"    Loaded catalog from {catalog_path} (windowed={windowed})")
        return catalog, windowed
    except Exception as e:
        logger.warning(f"    Failed to load catalog {catalog_path}: {str(e)}")
        raise
        return None, False


def get_source_positions(catalog: Table, windowed: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get source positions based on windowed flag.
    
    Args:
        catalog: Source catalog table
        windowed: Whether to use windowed positions
        
    Returns:
        Tuple of (x_positions, y_positions)
        
    Raises:
        ValueError: If windowed positions requested but columns missing
    """
    if windowed:
        if 'xwin' not in catalog.columns or 'ywin' not in catalog.columns:
            raise ValueError(
                "Windowed positions requested but xwin/ywin columns not found in catalog. "
                "This catalog may be from an older version. Options:\n"
                "1. Re-run detection with windowed_positions=true\n" 
                "2. Use windowed=False in photometry config\n"
                "3. Use --overwrite to regenerate catalog"
            )
        return catalog['xwin'].data, catalog['ywin'].data
    else:
        if 'x' not in catalog.columns or 'y' not in catalog.columns:
            raise ValueError("Basic position columns x/y not found in catalog")
        return catalog['x'].data, catalog['y'].data


def validate_catalog_positions(catalog: Table, windowed: bool) -> None:
    """
    Validate that catalog has required position columns.
    
    Args:
        catalog: Source catalog table
        windowed: Whether windowed positions are expected
        
    Raises:
        ValueError: If required position columns are missing
    """
    try:
        get_source_positions(catalog, windowed)
    except ValueError as e:
        raise ValueError(f"Catalog position validation failed: {e}")


def save_tile_catalog(project_dir: Path, project_name: str, tile: str, catalog: Table) -> None:
    """
    Save updated catalog for a specific tile.
    
    Args:
        project_dir: Project directory
        project_name: Project name
        tile: Tile name
        catalog: Updated catalog table
    """
    logger = get_logger()
    catalog_path = project_dir / "catalogs" / f"catalog_{project_name}_{tile}.fits"
    
    try:
        # Load existing FITS file to preserve header information
        with fits.open(catalog_path) as hdul:
            primary_hdu = hdul[0].copy()  # Preserve primary header
        for key in CORRECTION_STATE_KEYS:
            applied = catalog.meta.get(key, [])
            primary_hdu.header[key] = (','.join(applied), 'filters with this correction applied')

        # Create new table HDU with updated catalog
        table_hdu = fits.BinTableHDU(catalog, name='SOURCES')
        
        # Create new HDU list and save
        hdu_list = fits.HDUList([primary_hdu, table_hdu])
        hdu_list.writeto(catalog_path, overwrite=True)
        
        logger.debug(f"    Saved updated catalog to {catalog_path}")
    except Exception as e:
        logger.error(f"    Failed to save catalog {catalog_path}: {str(e)}")
        raise


def _local_median(image: np.ndarray, x_pos, y_pos, r: float = 5.0, chunk: int = 200000) -> np.ndarray:
    """Median of the finite pixels whose centres lie within r pixels of each position.

    Vectorised gather over the (2r+1)^2 offsets, in chunks of sources; equivalent to the
    per-source cutout median of photutils.ApertureStats but ~100x faster (F10).
    """
    ri = int(np.ceil(r))
    dy, dx = np.mgrid[-ri:ri + 1, -ri:ri + 1]
    inside = (dx ** 2 + dy ** 2) <= r ** 2
    dx, dy = dx[inside], dy[inside]
    ny, nx = image.shape
    xi = np.round(np.asarray(x_pos, dtype=float)).astype(int)
    yi = np.round(np.asarray(y_pos, dtype=float)).astype(int)
    out = np.full(len(xi), np.nan)
    for s in range(0, len(xi), chunk):
        xx = xi[s:s + chunk, None] + dx[None, :]
        yy = yi[s:s + chunk, None] + dy[None, :]
        ok = (xx >= 0) & (xx < nx) & (yy >= 0) & (yy < ny)
        vals = np.full(xx.shape, np.nan)
        vals[ok] = image[yy[ok], xx[ok]]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            out[s:s + chunk] = np.nanmedian(vals, axis=1)
    return out


def compute_weights(ctx: PhotometryContext, filter_name: str, image_files) -> Table:
    """
    Compute the local weight and rms values at source positions.

    Stores the median of the weight map (and of the rms map, when present) within r = 5 px of
    each source in 'wht_{filter_name}' / 'rms_{filter_name}'; these are the per-object scale of
    the error model applied in postprocess. The median is computed with a vectorised gather
    instead of photutils.ApertureStats (20-30 s per filter and tile before, F10), and the maps
    go through the context cache so that the error map derived from the weight map does not
    read the file a second time.
    """
    logger = ctx.get_logger()
    logger.debug(f"        Computing weights for {filter_name} (windowed={ctx.windowed})")
    x_pos, y_pos = get_source_positions(ctx.catalog, ctx.windowed)

    if image_files.has_extension('wht'):
        wht = ctx.load_image(image_files.get_weight_path())
        ctx.catalog[f'wht_{filter_name}'] = _local_median(wht, x_pos, y_pos)
    else:
        logger.warning(f"        Could not find weight map for {filter_name}")

    if image_files.has_extension('rms'):
        rms = ctx.load_image(image_files.get_rms_path())
        ctx.catalog[f'rms_{filter_name}'] = _local_median(rms, x_pos, y_pos)
    else:
        logger.warning(f"        Could not find rms map for {filter_name}")

    return ctx.catalog

def compute_rhalf(ctx: PhotometryContext, filter_name: str, image_files) -> Table:
    """
    Measure half-light radii for each source.

    Computes the radius containing half of the total flux for each source
    and stores it in a new column named 'rh_{filter_name}'.

    Args:
        ctx: PhotometryContext containing all resources
        filter_name: Filter name for column naming
        image_files: ImageFiles object containing file paths

    Returns:
        Updated catalog with half-light radius column
    """
    logger = ctx.get_logger()
    logger.debug(f"        Computing half-light radii for {filter_name} (windowed={ctx.windowed})")

    # Get source positions using helper function
    x_pos, y_pos = get_source_positions(ctx.catalog, ctx.windowed)

    sci = ctx.load_image(image_files.get_science_path())
    mask = np.isnan(sci)

    flux, fluxerr, flag = sep.sum_ellipse(
        sci, x_pos, y_pos, ctx.catalog['a'], ctx.catalog['b'], ctx.catalog['theta'],
        2.5*ctx.catalog['kronrad'], subpix=1, mask=mask,
        seg_id=ctx.catalog['id'], segmap=ctx.segmap,
    )

    rhalf, rflag = sep.flux_radius(
        sci, x_pos, y_pos, 6.*ctx.catalog['a'], 0.5,
        seg_id=ctx.catalog['id'], segmap=ctx.segmap,
        mask=mask, normflux=flux, subpix=5
    )

    ctx.catalog[f'rh_{filter_name}'] = rhalf

    return ctx.catalog


def compute_aper_photometry(
    ctx: PhotometryContext,
    filter_name: str, 
    diameters: List[float], 
    homogenized: bool = False
) -> Table:
    """
    Perform circular aperture photometry.
    
    Measures flux within circular apertures of specified diameters for each source.
    Creates columns with appropriate suffixes based on homogenization status.
    Uses windowed positions (xwin, ywin) if ctx.windowed=True, otherwise uses (x, y).
    
    Args:
        ctx: PhotometryContext containing all resources
        filter_name: Filter name for column naming
        diameters: List of aperture diameters in arcsec
        homogenized: If True, use PSF-homogenized images; if False, use native images
        
    Returns:
        Updated catalog with aperture photometry columns
    """
    logger = ctx.get_logger()
    image_type = "PSF-homogenized" if homogenized else "native"
    logger.info(f"        Computing {image_type} aperture photometry for {filter_name}")
    logger.debug(f"        Aperture diameters: {diameters}")
    
    # Get source positions using helper function
    x_pos, y_pos = get_source_positions(ctx.catalog, ctx.windowed)
    
    # Get image files for this filter
    image_files = ctx.get_image_files(filter_name)
    if not image_files:
        logger.warning(f"No image files found for {filter_name}")
        return ctx.catalog
    
    # Determine which image to use based on homogenization status
    if homogenized and ctx.psfs_config:
        # For PSF-homogenized photometry
        if ctx.is_target_filter(filter_name) or ctx.is_inverse_filter(filter_name):
            # Target and inverse filters use native images
            sci_path = image_files.get_science_path()
        else:
            # Other filters use homogenized images
            sci_path = image_files.get_homogenized_path(ctx.psfs_config.target_filter)
    else:
        # For native photometry, always use native images
        sci_path = image_files.get_science_path()
    
    
    header = fits.getheader(sci_path)
    pixel_scale = WCS(header).proj_plane_pixel_scales()[0].to('arcsec').value
    
    # Convert aperture diameters from arcsec to pixels
    aperture_diameters_pixels = np.array(diameters) / pixel_scale
    
    # Load science and error images via cache
    sci = ctx.load_image(sci_path)
    err = ctx.load_error(image_files)
    if err is None:
        logger.error('No ERR or WHT data found')
        return ctx.catalog
    mask = ~np.isfinite(sci)

    flux_list = []
    fluxerr_list = []

    for diam_pix in aperture_diameters_pixels:
        flux_i, fluxerr_i, flag = sep.sum_circle(
            sci, x_pos, y_pos,
            diam_pix/2,  # sep expects radius, not diameter
            err=err,
            mask=mask,
            segmap=ctx.segmap,
            seg_id=ctx.catalog['id'],
        )
        flux_list.append(flux_i)
        fluxerr_list.append(fluxerr_i)

    # Stack results
    flux = np.column_stack(flux_list)
    fluxerr = np.column_stack(fluxerr_list)

    # Number of pixels actually summed (invalid pixels and other segments excluded): the
    # aperture size the error model uses (postprocess)
    ones = ctx.ones_image(sci.shape)
    npix = np.column_stack([
        sep.sum_circle(ones, x_pos, y_pos, diam_pix / 2, mask=mask, segmap=ctx.segmap, seg_id=ctx.catalog['id'])[0]
        for diam_pix in aperture_diameters_pixels])
    ctx.catalog[f"npix_aper_{'hom' if homogenized else 'nat'}_{filter_name}"] = np.round(npix).astype(np.int32)

    # Cache raw (pre-unit-conversion) aperture fluxes for aper_corr reuse
    if homogenized and ctx.photometry_config.aperture.aper_corr:
        aper_corr_filter = ctx.photometry_config.aperture.aper_corr_filter
        if aper_corr_filter != 'detection' and filter_name == aper_corr_filter:
            ctx._flux_cache[f'aper_raw_{filter_name}'] = flux.copy()

    # Apply PSF corrections for inverse filters if doing PSF-homogenized photometry
    if homogenized and ctx.psfs_config and ctx.is_inverse_filter(filter_name):
        logger.info(f'        Correcting {filter_name} based on {ctx.psfs_config.target_filter} homogenization')

        # Get target filter image files
        target_image_files = ctx.get_image_files(ctx.psfs_config.target_filter)
        if not target_image_files:
            logger.warning(f"Cannot perform PSF correction: target filter {ctx.psfs_config.target_filter} not found")
        else:
            # Compute aperture flux in target filter's PSF-homogenized image (homogenized to current filter)
            target_hom_path = image_files.get_homogenized_path(ctx.psfs_config.target_filter)
            target_sci_path = target_image_files.get_science_path()

            if target_hom_path and target_hom_path.exists():
                hom_data = ctx.load_image(target_hom_path)
                sci_data = ctx.load_image(target_sci_path, persistent=True)
                corr_mask = ~np.isfinite(hom_data)

                flux1_list = []
                flux2_list = []

                for diam_pix in aperture_diameters_pixels:
                    # Flux in target filter homogenized to current filter
                    flux1, _, _ = sep.sum_circle(
                        hom_data, x_pos, y_pos, diam_pix/2,
                        mask=corr_mask, segmap=ctx.segmap, seg_id=ctx.catalog['id']
                    )
                    # Flux in target filter native resolution
                    flux2, _, _ = sep.sum_circle(
                        sci_data, x_pos, y_pos, diam_pix/2,
                        mask=corr_mask, segmap=ctx.segmap, seg_id=ctx.catalog['id']
                    )
                    flux1_list.append(flux1)
                    flux2_list.append(flux2)

                flux1 = np.column_stack(flux1_list)
                flux2 = np.column_stack(flux2_list)

                # Correction factor for flux lost due to larger PSF. As in the AUTO path: the
                # ratio is positive by definition, so take abs() (a negative value, from a
                # noise-dominated flux1 or flux2, would flip the sign of flux and error) and
                # leave the flux unchanged where the ratio is undefined. (F01; the factor
                # itself remains a ratio of noisy fluxes for faint sources -- review F09.)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    corr_fact = np.abs(flux2 / flux1)
                corr_fact = np.where(np.isfinite(corr_fact), corr_fact, 1.0)
                flux *= corr_fact
                fluxerr *= corr_fact

                # Store correction factors
                ctx.catalog[f'psf_corr_aper_{filter_name}'] = corr_fact
                logger.debug(f'        Correction factors: {np.median(corr_fact, axis=1)}')
            else:
                logger.warning(f"PSF correction skipped: homogenized target image not found")
    
    # Apply unit conversion
    try:
        conversion = get_unit_conversion(header, ctx.main_config.flux_unit)
        flux *= conversion
        fluxerr *= conversion
        logger.debug(f"        Applied unit conversion: {header.get('BUNIT', 'unknown')} -> {ctx.main_config.flux_unit} (factor: {conversion:.4e})")
    except Exception as e:
        logger.error(f"        Unit conversion failed for {filter_name}: {e}. Using native image units.")
        raise
    
    # Add results to catalog (store aperture fluxes as 2D array with columns = aperture sizes)
    col_suffix = 'hom' if homogenized else 'nat'
    ctx.catalog[f'f_aper_{col_suffix}_{filter_name}'] = flux
    ctx.catalog[f'e_aper_{col_suffix}_{filter_name}'] = fluxerr
    if homogenized:
        _mark_raw(ctx.catalog, 'APERCORR', filter_name)
    
    logger.debug(f"        Added columns {f'f_aper_{col_suffix}_{filter_name}'}, {f'e_aper_{col_suffix}_{filter_name}'}")

    return ctx.catalog


def _compute_auto_photometry(sci, err, mask, x_pos, y_pos, ctx, kron_params):
    a, b = ctx.catalog['a'], ctx.catalog['b'] 
    theta = ctx.catalog['theta']
    kronrad = ctx.catalog['kronrad']
    ids = ctx.catalog['id']

    flux, fluxerr, flag = sep.sum_ellipse(
        sci, x_pos, y_pos, a, b, theta, 
        err=err,
        mask=mask, 
        r=kron_params[0]*kronrad, 
        segmap=ctx.segmap, 
        seg_id=ids
    )

    # semi-axes of the Kron ellipse actually summed above, in pixels
    a = kron_params[0]*kronrad*a
    b = kron_params[0]*kronrad*b

    # Minimum-circle rule: ellipses whose geometric-mean radius falls below the minimum radius
    # kron_params[1] (pixels) are replaced by a circle of that radius.  a and b are already
    # scaled by kron_params[0]*kronrad, so the factor must not be applied again (F20).
    use_circle = np.sqrt(a * b) < kron_params[1]

    cflux, cfluxerr, cflag = sep.sum_circle(
        sci, 
        x_pos[use_circle], y_pos[use_circle],
        kron_params[1], 
        err=err, 
        mask=mask, 
        segmap=ctx.segmap, 
        seg_id=ids[use_circle]
    )

    flux[use_circle] = cflux
    fluxerr[use_circle] = cfluxerr
    flag[use_circle] |= cflag
    a[use_circle] = kron_params[1]
    b[use_circle] = kron_params[1]

    return flux, fluxerr, flag, a, b

def compute_auto_photometry(
    ctx: PhotometryContext,
    filter_name: str, 
    kron_params: List[float]
) -> Table:
    """
    Perform elliptical Kron aperture photometry.
    
    Measures flux within elliptical Kron apertures for each source using
    PSF-homogenized images. Creates columns for AUTO flux measurements.
    Uses windowed positions (xwin, ywin) if ctx.windowed=True, otherwise uses (x, y).
    
    Args:
        ctx: PhotometryContext containing all resources
        filter_name: Filter name for column naming
        kron_params: Kron radius parameters [min_radius, scale_factor]
        
    Returns:
        Updated catalog with AUTO photometry columns
    """
    logger = ctx.get_logger()
    logger.info(f"        Computing AUTO photometry for {filter_name}")
    logger.debug(f"        Kron parameters: {kron_params}")
    
    # Get source positions using helper function
    x_pos, y_pos = get_source_positions(ctx.catalog, ctx.windowed)
    
    # Get image files for this filter
    image_files = ctx.get_image_files(filter_name)
    if not image_files:
        logger.warning(f"No image files found for {filter_name}")
        return ctx.catalog


    if ctx.is_target_filter(filter_name) or ctx.is_inverse_filter(filter_name):
        # Target and inverse filters use native images
        sci_path = image_files.get_science_path()
    else:
        # Other filters use homogenized images
        sci_path = image_files.get_homogenized_path(ctx.psfs_config.target_filter)
    
    
    header = fits.getheader(sci_path)

    # Load science and error images via cache
    sci = ctx.load_image(sci_path)
    err = ctx.load_error(image_files)
    if err is None:
        logger.error('No ERR or WHT data found')
        return ctx.catalog
    mask = ~np.isfinite(sci)

    # Run the auto photometry calculation
    flux, fluxerr, flag, a, b = _compute_auto_photometry(sci, err, mask, x_pos, y_pos, ctx, kron_params)
    # Number of pixels actually summed in the Kron aperture (same mask and segment exclusion):
    # the aperture size the error model uses (postprocess)
    npix, _, _, _, _ = _compute_auto_photometry(ctx.ones_image(sci.shape), None, mask, x_pos, y_pos, ctx, kron_params)
    ctx.catalog[f'npix_auto_{filter_name}'] = np.round(npix).astype(np.int32)

    # Cache raw (pre-unit-conversion) auto flux for kron_corr and aper_corr reuse
    # Use filter-specific keys since kron_corr_filter and aper_corr_filter may differ
    cache_filters = set()
    if ctx.photometry_config.auto.kron_corr:
        kf = ctx.photometry_config.auto.kron_corr_filter
        if kf != 'detection':
            cache_filters.add(kf)
    if ctx.photometry_config.aperture.aper_corr:
        af = ctx.photometry_config.aperture.aper_corr_filter
        if af != 'detection':
            cache_filters.add(af)
    if filter_name in cache_filters:
        ctx._flux_cache[f'auto_raw_{filter_name}'] = (flux.copy(), fluxerr.copy(), flag.copy(), a.copy(), b.copy())

    # Corrections
    if ctx.is_inverse_filter(filter_name):
        logger.info(f'        Correcting {filter_name} based on {ctx.psfs_config.target_filter} homogenization')

        # Get target filter image files
        target_image_files = ctx.get_image_files(ctx.psfs_config.target_filter)

        if not target_image_files:
            logger.warning(f"Cannot perform PSF correction: target filter {ctx.psfs_config.target_filter} not found")
        else:
            # Compute auto flux in target filter's PSF-homogenized image (homogenized to current filter)
            target_hom_path = image_files.get_homogenized_path(ctx.psfs_config.target_filter)
            target_sci_path = target_image_files.get_science_path()

            if target_hom_path and target_hom_path.exists():
                hom_data = ctx.load_image(target_hom_path)
                sci_data = ctx.load_image(target_sci_path, persistent=True)
                corr_mask = ~np.isfinite(hom_data)

                flux1, _, _, _, _ = _compute_auto_photometry(
                    hom_data, None, corr_mask, x_pos, y_pos, ctx, kron_params,
                ) # flux1 = auto flux in <target_filter>, psf-homogenized to <filter>

                flux2, _, _, _, _ = _compute_auto_photometry(
                    sci_data, None, corr_mask, x_pos, y_pos, ctx, kron_params,
                ) # flux2 = auto flux in <target_filter>, native-resolution

                # correction factor for the flux lost due to the larger PSF
                # Use abs() because this is a positive aperture ratio by definition;
                # negative values arise when flux1 or flux2 is noise-dominated and
                # would incorrectly flip the sign of flux and error.
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    corr_fact = np.abs(flux2/flux1)
                corr_fact = np.where(np.isfinite(corr_fact), corr_fact, 1.0)

                flux *= corr_fact
                fluxerr *= corr_fact
                ctx.catalog[f'psf_corr_auto_{filter_name}'] = corr_fact
            else:
                logger.warning(f"PSF correction skipped: homogenized target image not found")
    
    # Apply unit conversion
    try:
        conversion = get_unit_conversion(header, ctx.main_config.flux_unit)
        flux *= conversion
        fluxerr *= conversion
        logger.debug(f"        Applied unit conversion: {header.get('BUNIT', 'unknown')} -> {ctx.main_config.flux_unit} (factor: {conversion:.4e})")
    except Exception as e:
        logger.error(f"        Unit conversion failed for {filter_name}: {e}. Using native image units.")
        raise
    
    # Add results to catalog (store aperture fluxes as 2D array with columns = aperture sizes)
    ctx.catalog[f'f_auto_{filter_name}'] = flux
    ctx.catalog[f'e_auto_{filter_name}'] = fluxerr
    _mark_raw(ctx.catalog, 'KRONCORR', filter_name)

    logger.debug(f"        Added columns {f'f_auto_{filter_name}'}, {f'e_auto_{filter_name}'}")
    return ctx.catalog



def apply_kron_corr(
    ctx: PhotometryContext,
    kron_corr_filter: str,
    kron_params1: List[float],
    kron_params2: List[float],
    kron_corr_bounds: List[float]
) -> None:
    """
    Apply Kron aperture correction to AUTO photometry measurements.
    
    Computes AUTO photometry with two different Kron parameters and uses the ratio
    to correct for flux missed in crowded fields or for blended sources.
    
    Args:
        ctx: PhotometryContext containing all resources
        kron_corr_filter: Filter to use for correction ('detection' or filter name)
        kron_params1: Original Kron parameters [min_radius, scale_factor]
        kron_params2: Larger Kron parameters for correction
        kron_corr_bounds: [min, max] bounds for correction factor
    """
    logger = ctx.get_logger()
    logger.info(f"Applying Kron correction using {kron_corr_filter}")
    
    # Check prerequisites
    if 'kronrad' not in ctx.catalog.columns:
        raise ValueError('Kron radius must be computed before applying Kron correction')
    
    # Get source positions
    x_pos, y_pos = get_source_positions(ctx.catalog, ctx.windowed)
    
    # Determine which image to use for correction
    if kron_corr_filter == 'detection':
        # Use detection image
        det_path = ctx.images_config.get_tile_detection_image_path(ctx.tile)
        if not det_path or not det_path.exists():
            logger.error("Detection image not found for Kron correction")
            return

        logger.debug(f"Using detection image: {det_path}")
        sci = ctx.load_image(det_path, hdu_index=1)
        mask = ~np.isfinite(sci)
        err = None

        # Compute fluxes with both Kron parameters
        flux1, _, _, a1, b1 = _compute_auto_photometry(sci, err, mask, x_pos, y_pos, ctx, kron_params1)
        flux2, _, _, a2, b2 = _compute_auto_photometry(sci, err, mask, x_pos, y_pos, ctx, kron_params2)

        # Get pixel scale from header
        with fits.open(det_path) as det_hdul:
            header = det_hdul[1].header
            pixel_scale = WCS(header).proj_plane_pixel_scales()[0].to('arcsec').value

    else:
        # Use specific filter image
        image_files = ctx.get_image_files(kron_corr_filter)
        if not image_files:
            logger.error(f"Filter {kron_corr_filter} not found for Kron correction")
            return

        # Determine which image to use (prefer homogenized if appropriate)
        if ctx.psfs_config and not ctx.is_target_filter(kron_corr_filter) and not ctx.is_inverse_filter(kron_corr_filter):
            hom_path = image_files.get_homogenized_path(ctx.psfs_config.target_filter)
            if hom_path and hom_path.exists():
                sci_path = hom_path
                logger.debug(f"Using homogenized image for {kron_corr_filter}")
            else:
                sci_path = image_files.get_science_path()
                logger.debug(f"Using native image for {kron_corr_filter} (homogenized not found)")
        else:
            sci_path = image_files.get_science_path()
            logger.debug(f"Using native image for {kron_corr_filter}")

        header = fits.getheader(sci_path)
        pixel_scale = WCS(header).proj_plane_pixel_scales()[0].to('arcsec').value

        # Reuse cached flux1 from compute_auto_photometry if available
        cache_key = f'auto_raw_{kron_corr_filter}'
        if cache_key in ctx._flux_cache:
            logger.debug(f"Using cached auto flux for kron_params1")
            flux1, _, _, a1, b1 = ctx._flux_cache[cache_key]
        else:
            sci = ctx.load_image(sci_path)
            mask = ~np.isfinite(sci)
            flux1, _, _, a1, b1 = _compute_auto_photometry(sci, None, mask, x_pos, y_pos, ctx, kron_params1)

        # flux2 always needs computing (different kron_params)
        sci = ctx.load_image(sci_path)
        mask = ~np.isfinite(sci)
        err = None
        flux2, _, _, a2, b2 = _compute_auto_photometry(sci, err, mask, x_pos, y_pos, ctx, kron_params2)
    
    # Calculate correction factor
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        kron_corr = flux2 / flux1
    
    # Apply bounds to correction factor
    kron_corr = np.clip(kron_corr, kron_corr_bounds[0], kron_corr_bounds[1])
    
    # Flag sources that hit the upper bound (likely blended)
    flag_blend = (kron_corr == kron_corr_bounds[1])
    
    # Apply correction to all AUTO photometry columns (idempotently: columns corrected by a
    # previous run are first divided by that run's kron_corr, F03)
    old_corr = np.asarray(ctx.catalog['kron_corr'], dtype=float) if 'kron_corr' in ctx.catalog.colnames else None
    corrected = _apply_multiplicative_correction(ctx.catalog, ('f_auto_', 'e_auto_'), kron_corr, old_corr, 'KRONCORR')
    n_corrected = 2 * len(corrected)

    # Store correction information in catalog
    ctx.catalog['kron_corr'] = kron_corr
    ctx.catalog['flag_blend'] = flag_blend
    ctx.catalog['kron1_a'] = a1 * pixel_scale  # arcsec
    ctx.catalog['kron1_b'] = b1 * pixel_scale  # arcsec
    ctx.catalog['kron1_area'] = np.pi * a1 * b1  # pixels^2
    ctx.catalog['kron2_a'] = a2 * pixel_scale  # arcsec
    ctx.catalog['kron2_b'] = b2 * pixel_scale  # arcsec
    ctx.catalog['kron2_area'] = np.pi * a2 * b2  # pixels^2
    
    # Log summary statistics
    median_corr = np.median(kron_corr)
    n_blend = np.sum(flag_blend)
    logger.info(f"Kron correction applied to {n_corrected} columns")
    logger.info(f"  Median correction: {median_corr:.3f}")
    logger.info(f"  Sources at upper bound: {n_blend}/{len(kron_corr)} ({100*n_blend/len(kron_corr):.1f}%)")
    logger.info(f"  Correction range: [{np.min(kron_corr):.3f}, {np.max(kron_corr):.3f}]")


def _finalize_aper_corr(aper_corr, snr, bounds, min_snr):
    """Gate and bound the per-source aperture corrections (review finding F01).

    aper_corr: (n_src, n_diam) ratio AUTO / fixed-aperture flux on the correction image
    snr:       (n_src, n_diam) S/N of the fixed-aperture flux that forms the denominator
    Returns (aper_corr, flag) where flag is an int16 array with bit 1 set where the ratio was
    not applied (denominator below min_snr, or undefined -> aper_corr NaN) and bit 2 where it
    lies outside bounds (flagged, not clipped: the plausible range depends strongly on the
    aperture diameter -- on the review cutout the 5-95% range is 15-190 at 0.05" and 0.8-1.8
    at 1.0" -- so one pair of bounds cannot be a clip for all diameters). The S/N gate is what
    removes the noise-dominated ratios: in cosmos_v1.3 the ungated ratio was negative for
    0.7-2.8% of entries, sign-flipping flux and error.
    """
    aper_corr = np.array(aper_corr, dtype=float)
    snr = np.asarray(snr, dtype=float)
    flag = np.zeros(aper_corr.shape, dtype=np.int16)
    with np.errstate(invalid='ignore'):
        low = ~np.isfinite(aper_corr) | ~np.isfinite(snr) | (snr < min_snr)
        out = np.isfinite(aper_corr) & ((aper_corr < bounds[0]) | (aper_corr > bounds[1]))
    flag[low] |= 1
    flag[out & ~low] |= 2
    aper_corr[low] = np.nan
    return aper_corr, flag


def apply_aper_corr(
    ctx: PhotometryContext,
    aper_corr_filter: str,
    kron_params: List[float],
    aper_corr_bounds: List[float],
    aperture_diameters: List[float],
    aper_corr_min_snr: float = 3.0,
) -> None:
    """
    Apply aperture correction to PSF-homogenized aperture photometry measurements.

    Computes aperture photometry and Kron photometry on the same image and uses the flux ratio
    to correct for flux missed in fixed apertures. The correction is applied only where the
    fixed-aperture flux on the correction image has S/N >= aper_corr_min_snr (otherwise the
    corrected flux and error are NaN); values outside aper_corr_bounds are flagged but kept.
    Both cases are recorded in the 'flag_aper_corr' column (bit 1: not applied, bit 2: outside
    bounds).
    """
    logger = ctx.get_logger()
    logger.info(f"Applying aperture correction using {aper_corr_filter}")
    
    # Check prerequisites
    if 'kronrad' not in ctx.catalog.columns:
        raise ValueError('Kron radius must be computed before applying aperture correction')
    
    # Get source positions
    x_pos, y_pos = get_source_positions(ctx.catalog, ctx.windowed)
    
    # Determine which image to use for correction
    if aper_corr_filter == 'detection':
        # Use detection image
        det_path = ctx.images_config.get_tile_detection_image_path(ctx.tile)
        if not det_path or not det_path.exists():
            logger.error("Detection image not found for aperture correction")
            return

        logger.debug(f"Using detection image: {det_path}")
        sci = ctx.load_image(det_path, hdu_index=1)
        # The detection image carries its error map in HDU 2 (see detect.py); it gives the
        # S/N of the fixed-aperture flux that the correction divides by.
        with fits.open(det_path) as det_hdul:
            header = det_hdul[1].header
            pixel_scale = WCS(header).proj_plane_pixel_scales()[0].to('arcsec').value
            has_err = len(det_hdul) > 2 and det_hdul[2].data is not None
        err = ctx.load_image(det_path, hdu_index=2) if has_err else None
        if err is None:
            logger.warning("Detection image has no error HDU: aperture-correction S/N gate disabled")
        mask = ~np.isfinite(sci)
        if err is not None:
            mask |= ~np.isfinite(err)

        # Convert aperture diameters from arcsec to pixels
        aperture_diameters_pixels = np.array(aperture_diameters) / pixel_scale

        flux_list, fluxerr_list = [], []
        for diam_pix in aperture_diameters_pixels:
            flux_i, fluxerr_i, flag = sep.sum_circle(
                sci, x_pos, y_pos,
                diam_pix/2,
                err=err,
                mask=mask,
                segmap=ctx.segmap,
                seg_id=ctx.catalog['id'],
            )
            flux_list.append(flux_i)
            fluxerr_list.append(fluxerr_i)

        flux1 = np.column_stack(flux_list)
        fluxerr1 = np.column_stack(fluxerr_list) if err is not None else None
        flux2, _, _, a2, b2 = _compute_auto_photometry(sci, None, mask, x_pos, y_pos, ctx, kron_params)

    else:
        # Reuse fluxes cached by the main photometry pass where they are valid:
        #  - aper_raw_<filter>: fixed-aperture fluxes on the same (homogenized) image -> always reusable
        #  - auto_raw_<filter>: AUTO flux measured with auto.kron_params, which is only the
        #    numerator we need if aper_corr_params happen to equal auto.kron_params.
        aper_key = f'aper_raw_{aper_corr_filter}'
        auto_key = f'auto_raw_{aper_corr_filter}'
        have_aper_cache = aper_key in ctx._flux_cache
        have_auto_cache = (
            auto_key in ctx._flux_cache
            and list(ctx.photometry_config.auto.kron_params) == list(kron_params)
        )

        if have_aper_cache and have_auto_cache:
            logger.debug("Using cached aperture and auto fluxes for correction")
            flux1 = ctx._flux_cache[aper_key]
            flux2, _, _, _, _ = ctx._flux_cache[auto_key]
        else:
            # Load the image and compute whatever is not cached
            image_files = ctx.get_image_files(aper_corr_filter)
            if not image_files:
                logger.error(f"Filter {aper_corr_filter} not found for aperture correction")
                return

            if ctx.psfs_config and not ctx.is_target_filter(aper_corr_filter) and not ctx.is_inverse_filter(aper_corr_filter):
                hom_path = image_files.get_homogenized_path(ctx.psfs_config.target_filter)
                if hom_path and hom_path.exists():
                    sci_path = hom_path
                    logger.debug(f"Using homogenized image for {aper_corr_filter}")
                else:
                    sci_path = image_files.get_science_path()
                    logger.debug(f"Using native image for {aper_corr_filter} (homogenized not found)")
            else:
                sci_path = image_files.get_science_path()
                logger.debug(f"Using native image for {aper_corr_filter}")

            header = fits.getheader(sci_path)
            pixel_scale = WCS(header).proj_plane_pixel_scales()[0].to('arcsec').value

            sci = ctx.load_image(sci_path)
            mask = ~np.isfinite(sci)
            err = None

            if have_aper_cache:
                logger.debug("Using cached aperture fluxes for correction")
                flux1 = ctx._flux_cache[aper_key]
            else:
                aperture_diameters_pixels = np.array(aperture_diameters) / pixel_scale
                flux_list = []
                for diam_pix in aperture_diameters_pixels:
                    flux_i, fluxerr_i, flag = sep.sum_circle(
                        sci, x_pos, y_pos,
                        diam_pix/2,
                        err=err,
                        mask=mask,
                        segmap=ctx.segmap,
                        seg_id=ctx.catalog['id'],
                    )
                    flux_list.append(flux_i)
                flux1 = np.column_stack(flux_list)

            # AUTO flux with the aperture-correction Kron parameters (not auto.kron_params)
            flux2, _, _, a2, b2 = _compute_auto_photometry(sci, err, mask, x_pos, y_pos, ctx, kron_params)

        fluxerr1 = None

    # S/N of the denominator (fixed-aperture flux on the correction image)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        if fluxerr1 is not None:
            snr1 = flux1 / fluxerr1
        elif (f'f_aper_hom_{aper_corr_filter}' in ctx.catalog.colnames
              and f'e_aper_hom_{aper_corr_filter}' in ctx.catalog.colnames):
            # same apertures on the same image, already in catalog units
            snr1 = np.asarray(ctx.catalog[f'f_aper_hom_{aper_corr_filter}']) / np.asarray(ctx.catalog[f'e_aper_hom_{aper_corr_filter}'])
        else:
            logger.warning("No error available for the aperture-correction denominator: S/N gate disabled")
            snr1 = np.full(flux1.shape, np.inf)

        # Calculate correction factor
        aper_corr = flux2[:, np.newaxis] / flux1

    # Gate on S/N, apply bounds, flag (F01)
    aper_corr, flag_aper_corr = _finalize_aper_corr(aper_corr, snr1, aper_corr_bounds, aper_corr_min_snr)

    # Apply correction to all APER_HOM photometry columns (idempotently, see F03 above)
    old_corr = np.asarray(ctx.catalog['aper_corr'], dtype=float) if 'aper_corr' in ctx.catalog.colnames else None
    corrected = _apply_multiplicative_correction(ctx.catalog, ('f_aper_hom_', 'e_aper_hom_'), aper_corr, old_corr, 'APERCORR')
    n_corrected = 2 * len(corrected)

    # Store correction information in catalog
    ctx.catalog['aper_corr'] = aper_corr
    ctx.catalog['flag_aper_corr'] = flag_aper_corr

    # Log summary statistics
    median_corr = np.nanmedian(aper_corr)
    isnan = np.sum(np.isnan(aper_corr))
    frac_low = np.mean(flag_aper_corr & 1, axis=0)
    frac_out = np.mean((flag_aper_corr & 2) > 0, axis=0)
    logger.info(f"Aperture correction applied to {n_corrected} columns")
    logger.info("  Not applied (S/N < %.1f) per diameter: %s", aper_corr_min_snr,
                ', '.join(f'{d}"={100*f:.1f}%' for d, f in zip(aperture_diameters, frac_low)))
    logger.info("  Flagged outside bounds %s per diameter: %s", list(aper_corr_bounds),
                ', '.join(f'{d}"={100*f:.1f}%' for d, f in zip(aperture_diameters, frac_out)))
    logger.info(f"  Median correction: {median_corr:.3f}")
    logger.info(f"  Percent nan: {isnan/aper_corr.size*100:.1f}%")
    # logger.info(f"  Sources at upper bound: {n_at_bound}/{len(aper_corr)} ({100*n_at_bound/len(aper_corr):.1f}%)")
    # logger.info(f"  Correction range: [{np.min(aper_corr):.3f}, {np.max(aper_corr):.3f}]")



def process_photometry_for_filter(ctx: PhotometryContext, filter_name: str) -> Table:
    """
    Process all photometry measurements for a single filter.
    
    Args:
        ctx: PhotometryContext containing all resources
        filter_name: Filter name to process
        
    Returns:
        Updated catalog with photometry measurements
    """
    logger = ctx.get_logger()
    
    # Get image files for this filter
    image_files = ctx.get_image_files(filter_name)
    if not image_files:
        logger.warning(f"No image files found for {filter_name}/{ctx.tile}")
        return ctx.catalog
    
    # Log PSF homogenization status for this filter
    if ctx.psfs_config:
        logger.debug(f"        PSF homogenization: target={ctx.psfs_config.target_filter}, inverse={ctx.psfs_config.inverse_filters}")
        logger.debug(f"        Filter {filter_name}: is_target={ctx.is_target_filter(filter_name)}, is_inverse={ctx.is_inverse_filter(filter_name)}")
    
    # Always compute weights and RMS (basic measurements)
    ctx.catalog = compute_weights(ctx, filter_name, image_files)
    # note: removed compute_rms as its included within compute_weights now
    
    # Compute half-light radii if requested
    if ctx.photometry_config.compute_rhalf:
        ctx.catalog = compute_rhalf(ctx, filter_name, image_files)
    
    # Compute aperture photometry on native images if requested
    if ctx.photometry_config.aperture.run_native and ctx.photometry_config.aperture.diameters:
        ctx.catalog = compute_aper_photometry(
            ctx, filter_name, 
            ctx.photometry_config.aperture.diameters, 
            homogenized=False
        )
    # Compute aperture photometry on PSF-homogenized images if requested
    if ctx.photometry_config.aperture.run_psf_homogenized and ctx.photometry_config.aperture.diameters:
        ctx.catalog = compute_aper_photometry(
            ctx, filter_name,
            ctx.photometry_config.aperture.diameters, 
            homogenized=True
        )
    
    # Compute AUTO photometry if requested
    if ctx.photometry_config.auto.run:
        ctx.catalog = compute_auto_photometry(
            ctx, filter_name,
            ctx.photometry_config.auto.kron_params
        )
    
    return ctx.catalog


def process_photometry_tile(project_dir, tile, filters_to_process, config, images_config, photometry_config, psfs_config, overwrite):
    """Process photometry for a single tile."""
    logger = get_logger()
    
    logger.info(f"Processing tile [bold cyan]{tile}[/bold cyan]")
    
    try:
        # Load detection catalog for this tile
        catalog, windowed = load_tile_catalog(project_dir, config.name, tile)
        if catalog is None:
            logger.warning(f"No detection catalog found for tile {tile}, skipping")
            return False
        
        logger.info(f"    Loaded catalog with {len(catalog)} sources")
        
        # Validate catalog has required position columns
        validate_catalog_positions(catalog, windowed)

        # Sort catalog by y (slow/row axis) for better cache locality in sep calls
        pos_col = 'ywin' if windowed else 'y'
        sort_idx = np.argsort(catalog[pos_col])
        catalog = catalog[sort_idx]

        # Load segmentation map (per-tile, shared across filters)
        segmap = load_segmentation_map(images_config, tile)
        segmap = segmap.astype(segmap.dtype.newbyteorder('='))
        if segmap is None:
            logger.warning(f"No segmentation map found for tile {tile}, skipping")
            return False
        
        # Create photometry context for this tile
        ctx = PhotometryContext(
            catalog=catalog,
            segmap=segmap,
            tile=tile,
            images_config=images_config,
            photometry_config=photometry_config,
            psfs_config=psfs_config,
            main_config=config,
            windowed=windowed,
            overwrite=overwrite
        )

        # Process each filter
        for filter_name in filters_to_process:
            logger.info(f"    [bold cyan]{tile}[/bold cyan]: Processing filter [bold cyan]{filter_name}[/bold cyan]")
            
            # Check if this filter/tile combination exists
            if not ctx.images_config.filter_tile_exists(filter_name, tile):
                logger.warning(f"    Filter {filter_name} not available for tile {tile}, skipping")
                continue
            
            # Run photometry functions for this filter
            ctx.catalog = process_photometry_for_filter(ctx, filter_name)
            ctx.clear_image_cache()  # Free image arrays between filters

        # Apply Kron correction after all filters processed
        if ctx.photometry_config.auto.run and ctx.photometry_config.auto.kron_corr:
            logger.info("  Applying Kron correction to AUTO photometry")
            apply_kron_corr(
                ctx,
                ctx.photometry_config.auto.kron_corr_filter,
                ctx.photometry_config.auto.kron_params,
                ctx.photometry_config.auto.kron_corr_params,
                ctx.photometry_config.auto.kron_corr_bounds
            )
        
        # Apply aperture correction after all filters processed
        if ctx.photometry_config.aperture.run_psf_homogenized and ctx.photometry_config.aperture.aper_corr:
            logger.info("  Applying aperture correction to PSF-homogenized aperture photometry")
            apply_aper_corr(
                ctx,
                ctx.photometry_config.aperture.aper_corr_filter,
                ctx.photometry_config.aperture.aper_corr_params,
                ctx.photometry_config.aperture.aper_corr_bounds,
                ctx.photometry_config.aperture.diameters,
                ctx.photometry_config.aperture.aper_corr_min_snr,
            )
        
        # Save updated catalog
        save_tile_catalog(project_dir, config.name, tile, ctx.catalog)
        
        # Force garbage collection after tile processing
        gc.collect()
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing tile {tile}: {str(e)}")
        raise
        return False


def process_photometry_tile_wrapper(args):
    """Wrapper function for multiprocessing."""
    project_dir, tile, filters_to_process, config, images_config, photometry_config, psfs_config, overwrite = args
    return process_photometry_tile(project_dir, tile, filters_to_process, config, images_config, photometry_config, psfs_config, overwrite)


@click.command("photometry")
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
    "--filters",
    type=str,
    default='',
    help="Run only for a subset of filters (comma-separated)"
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite existing photometry measurements"
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
def photometry_cmd(project_dir, tiles, filters, overwrite, verbose, parallel):
    """Perform aperture photometry measurements."""
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
    
    # Parse filter override
    if filters:
        filters_to_process = [f.strip() for f in filters.split(',') if f.strip() in config.filters]
    else:
        filters_to_process = config.filters
    
    if not tiles_to_process:
        logger.error("[bold red]Error:[/bold red] No tiles available for photometry")
        raise click.ClickException("No tiles to process")
        
    if not filters_to_process:
        logger.error("[bold red]Error:[/bold red] No filters available for photometry")
        raise click.ClickException("No filters to process")
    
    logger.info(f"Processing photometry for {len(tiles_to_process)} tiles: {tiles_to_process}")
    logger.info(f"Processing {len(filters_to_process)} filters: {filters_to_process}")
    
    # Get photometry configuration
    photometry_config = config.photometry
    
    # Load PSF homogenization configuration
    psfs_config = load_psfs_config(project_dir)
    if psfs_config:
        logger.info(f"PSF homogenization: target filter = {psfs_config.target_filter}")
        if psfs_config.inverse_filters:
            logger.info(f"PSF homogenization: inverse filters = {psfs_config.inverse_filters}")
    else:
        logger.info("No psfs.toml found - PSF homogenization info not available")
    
    # Log photometry configuration
    logger.info("Photometry configuration:")
    logger.info(f"  Compute half-light radii: {photometry_config.compute_rhalf}")
    logger.info(f"  Aperture photometry (native): {photometry_config.aperture.run_native}")
    logger.info(f"  Aperture photometry (PSF-homogenized): {photometry_config.aperture.run_psf_homogenized}")
    if photometry_config.aperture.diameters:
        logger.info(f"  Aperture diameters: {photometry_config.aperture.diameters}")
    logger.info(f"  Auto photometry: {photometry_config.auto.run}")
    if photometry_config.auto.run:
        logger.info(f"  Kron parameters: {photometry_config.auto.kron_params}")
    
    # Process tiles
    if parallel > 1 and len(tiles_to_process) > 1:
        logger.info(f"Processing {len(tiles_to_process)} tiles using {parallel} processes")
        # Prepare tasks for multiprocessing
        tasks = [(project_dir, tile, filters_to_process, config, images_config, photometry_config, psfs_config, overwrite) 
                 for tile in tiles_to_process]
        
        with mp.Pool(processes=parallel) as pool:
            results = pool.map(process_photometry_tile_wrapper, tasks)
        
        successful_tiles = [tile for tile, success in zip(tiles_to_process, results) if success]
    else:
        # Sequential processing
        successful_tiles = []
        for tile in tiles_to_process:
            success = process_photometry_tile(project_dir, tile, filters_to_process, config, images_config, photometry_config, psfs_config, overwrite)
            if success:
                successful_tiles.append(tile)
    
    # Summary
    if successful_tiles:
        logger.info(f"[bold green]Success:[/bold green] Photometry completed for {len(successful_tiles)} tiles")
        logger.info(f"Processed tiles: {successful_tiles}")
    else:
        logger.error("[bold red]Error:[/bold red] No tiles were processed successfully")
        raise click.ClickException("Photometry failed for all tiles")