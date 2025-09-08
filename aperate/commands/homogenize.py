"""PSF homogenization to a target filter."""

import gc
import logging
import warnings
import time 
from pathlib import Path
from typing import Dict, List, Optional

import click
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from ..core.catalog import find_catalog_in_directory
from ..core.logging import get_logger
from ..core.homogenize import (
    measure_psf_fwhm, 
    determine_inverse_filters,
    create_matching_kernel,
    homogenize_image,
    get_psf_paths_for_filter
)
from ..config.parser import load_config
from ..config.images import load_images_config, update_homogenized_paths_in_images_toml
from ..config.psfs import load_psfs_config, create_psfs_toml, update_psfs_toml_with_missing_fwhms

from astropy.wcs import FITSFixedWarning
warnings.simplefilter('ignore', category=FITSFixedWarning)


def measure_all_psf_fwhms(
    project_dir: Path,
    filters: List[str],
    master_psf: bool
) -> Dict[str, float]:
    """
    Measure FWHM for all PSF files.
    
    Returns dict of filter -> FWHM in pixels
    """
    logger = get_logger()
    logger.info("Measuring PSF FWHMs...")
    
    fwhm_measurements = {}
    
    for filter_name in filters:
        # For master PSF, just check one file
        if master_psf:
            psf_path = project_dir / "psfs" / f"psf_{filter_name}_master.fits"
            if psf_path.exists():
                with fits.open(psf_path) as hdul:
                    psf_data = hdul[0].data
                    header = hdul[0].header
                    wcs = WCS(header)
                    pixel_scale = np.abs(wcs.proj_plane_pixel_scales()[0].to('arcsec').value)
                
                fwhm_pix, fwhm_arcsec = measure_psf_fwhm(psf_data, pixel_scale)
                fwhm_measurements[filter_name] = round(float(fwhm_arcsec), 5)  # FWHM in arcsec, 5 decimals
                logger.info(f"    [bold cyan]{filter_name}[/bold cyan]: FWHM = {fwhm_pix:.2f} pix ({fwhm_arcsec:.3f} arcsec)")
            else:
                logger.warning(f"    [bold cyan]{filter_name}[/bold cyan]: PSF file not found")
        else:
            # For per-tile PSFs, measure first available tile
            images_config = load_images_config(project_dir)
            tiles = images_config.get_tiles_for_filter(filter_name)
            
            for tile in tiles:
                psf_path = project_dir / "psfs" / f"psf_{filter_name}_{tile}.fits"
                if psf_path.exists():
                    with fits.open(psf_path) as hdul:
                        psf_data = hdul[0].data
                        header = hdul[0].header
                        wcs = WCS(header)
                        pixel_scale = np.abs(wcs.proj_plane_pixel_scales()[0].to('arcsec').value)
                    
                    fwhm_pix, fwhm_arcsec = measure_psf_fwhm(psf_data, pixel_scale)
                    fwhm_measurements[filter_name] = round(float(fwhm_arcsec), 5)  # FWHM in arcsec, 5 decimals
                    logger.info(f"    [bold cyan]{filter_name}[/bold cyan]: FWHM = {fwhm_pix:.2f} pix ({fwhm_arcsec:.3f} arcsec)")
                    break
            else:
                logger.warning(f"    [bold cyan]{filter_name}[/bold cyan]: No PSF files found")
    
    return fwhm_measurements


def homogenize_filter_tile(
    project_dir: Path,
    filter_name: str,
    tile: str,
    target_filter: str,
    is_inverse: bool,
    master_psf: bool,
    reg_fact: float,
    overwrite: bool,
    output_prefix: str = ''
) -> bool:
    """
    Homogenize a single filter/tile combination.
    
    Returns True if successful.
    """
    logger = get_logger()
    images_config = load_images_config(project_dir)
    
    # Construct paths - filename should reflect the destination filter
    # Add prefix if provided (e.g., 'mock' for simulations)
    if output_prefix:
        prefix = f"{output_prefix}-"
    else:
        prefix = ""
    
    if is_inverse:
        hom_ext = f"{prefix}hom-{filter_name}"  # Inverse: target → filter, so name reflects filter
    else:
        hom_ext = f"{prefix}hom-{target_filter}"  # Forward: filter → target, so name reflects target
    
    if is_inverse:
        # For inverse filters, we homogenize the target image
        source_filter = target_filter
        dest_filter = filter_name
        log_direction = f"{target_filter} → {filter_name}"
    else:
        # Normal: homogenize this filter to target
        source_filter = filter_name
        dest_filter = target_filter
        log_direction = f"{filter_name} → {target_filter}"
    
    # Get image files
    image_files = images_config.get_image_files(source_filter, tile)
    if not image_files:
        logger.warning(f"No image files found for {source_filter} {tile}")
        return False
    
    # Get science and error paths
    sci_path = image_files.get_science_path()
    err_path = image_files.get_error_path() or image_files.get_weight_path()
    
    if not sci_path or not sci_path.exists():
        logger.warning(f"Science image not found: {sci_path}")
        return False
    
    # Construct output paths
    base_name = sci_path.stem
    output_dir = sci_path.parent
    
    # Replace extension in filename
    if "_sci" in base_name:
        hom_name = base_name.replace("_sci", f"_{hom_ext}")
    elif "_drz" in base_name:
        hom_name = base_name.replace("_drz", f"_{hom_ext}")
    elif "_mock" in base_name:
        # Handle mock files - try to replace the exact prefix first
        if output_prefix and f"_{output_prefix}" in base_name:
            # Replace exact prefix match (e.g., _mock1)
            hom_name = base_name.replace(f"_{output_prefix}", f"_{hom_ext}")
        else:
            # Fallback: replace generic _mock
            hom_name = base_name.replace("_mock", f"_{hom_ext}")
    else:
        hom_name = f"{base_name}_{hom_ext}"
    
    output_path = output_dir / f"{hom_name}.fits"
    
    # Check if already exists
    if output_path.exists() and not overwrite:
        logger.info(f"    [bold cyan]{filter_name}-{tile}[/bold cyan]: Homogenized file exists, skipping")
        return True
    
    # Get PSF paths
    source_psf_path = get_psf_paths_for_filter(project_dir, source_filter, tile, master_psf)
    target_psf_path = get_psf_paths_for_filter(project_dir, dest_filter, tile, master_psf)
    
    if not source_psf_path.exists() or not target_psf_path.exists():
        logger.error(f"    [bold cyan]{filter_name}-{tile}[/bold cyan]: Missing PSF files")
        return False
    
    logger.info(f"    [bold cyan]{filter_name}-{tile}[/bold cyan]: Homogenizing {log_direction}")
    
    try:
        # Create kernel
        logger.debug(f"    [bold cyan]{filter_name}-{tile}[/bold cyan]: Creating kernel {source_psf_path.name} → {target_psf_path.name}")
        logger.debug(f"    [bold cyan]{filter_name}-{tile}[/bold cyan]: Regularization factor: {reg_fact}")

        kernel = create_matching_kernel(
            source_psf_path,
            target_psf_path,
            reg_fact=reg_fact
        )

        logger.debug(f"    [bold cyan]{filter_name}-{tile}[/bold cyan]: Kernel shape: {kernel.shape}, sum: {np.sum(kernel):.6f}")
        
        
        logger.info(f"    [bold cyan]{filter_name}-{tile}[/bold cyan]: Convolving {sci_path.name} → {output_path.name}")

        start = time.time()
        homogenize_image(
            sci_path,
            kernel,
            output_path,
        )
        end = time.time()
        t = end-start
        logger.info(f"    [bold cyan]{filter_name}-{tile}[/bold cyan]: Done in {int(np.floor(t/60))}m{int(t-60*int(np.floor(t/60)))}s")
        logger.info(f"    [bold cyan]{filter_name}-{tile}[/bold cyan]: [bold green]Success![/bold green] Saved homogenized image to {output_path}")

        return True
        
    except Exception as e:
        logger.error(f"    [bold cyan]{filter_name}-{tile}[/bold cyan]: [bold red]Failed[/bold red] - {str(e)}")
        return False


@click.command("homogenize")
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
    "--tiles",
    type=str,
    default='',
    help="Run only for a subset of tiles (comma-separated)"
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite existing homogenized files"
)
@click.option(
    "--target-filter",
    type=str,
    help="Target filter for PSF homogenization (overrides config)"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable debug output"
)
def homogenize_cmd(project_dir, filters, tiles, overwrite, target_filter, verbose):
    """Homogenize PSF models to a target filter."""
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
    
    # Determine target filter
    if not target_filter:
        target_filter = config.psf_homogenization.target_filter
    
    if not target_filter:
        logger.error("[bold red]Error:[/bold red] No target filter specified")
        raise click.ClickException("Target filter must be specified in config or CLI")
    
    logger.info(f"Target filter: [bold cyan]{target_filter}[/bold cyan]")
    logger.info(f"Regularization factor: {config.psf_homogenization.reg_fact}")
    
    # Load images config
    images_config = load_images_config(project_dir)
    available_filters = images_config.get_filters()
    
    # Parse filter/tile overrides
    if filters:
        filters_to_process = [f.strip() for f in filters.split(',') if f.strip() in available_filters]
    else:
        filters_to_process = [f for f in config.filters if f in available_filters]
    
    # Remove target filter from processing list
    if target_filter in filters_to_process:
        filters_to_process.remove(target_filter)
    
    if not filters_to_process:
        logger.error("[bold red]Error:[/bold red] No filters available for homogenization")
        raise click.ClickException("No filters to process")
    
    # Parse tile override
    if tiles:
        tiles_to_process = [t.strip() for t in tiles.split(',')]
    else:
        tiles_to_process = None  # Process all tiles
    
    # Check or create psfs.toml
    psfs_config = load_psfs_config(project_dir)

    # Decide whether we need to measure PSF FWHMs
    measure = False
    if psfs_config:
        # Check that all filters to process are included in psfs.toml
        all_filters = filters_to_process + [target_filter]
        filters_to_measure = [f for f in all_filters if f not in psfs_config.fwhm]
        if len(filters_to_measure) >= 1:
            logger.info(f"No PSF information found for {', '.join(filters_to_measure)}. Measuring PSF FWHMs... ")
            measure = True
    else:
        logger.info("No psfs.toml found, measuring PSF FWHMs...")
        filters_to_measure = filters_to_process + [target_filter]
        measure = True
    
    if measure:
        # Only measure the missing filters (not all filters)
        fwhm_measurements = measure_all_psf_fwhms(
            project_dir, 
            filters_to_measure,
            config.psf_generation.master_psf
        )
        
        if target_filter not in fwhm_measurements and target_filter in filters_to_measure:
            logger.error(f"[bold red]Error:[/bold red] Could not measure FWHM for target filter {target_filter}")
            raise click.ClickException("Target PSF not found")
        
        if psfs_config:
            # If psfs.toml already exists, update it (preserves existing FWHM data)
            update_psfs_toml_with_missing_fwhms(
                project_dir,
                missing_filters=filters_to_measure,
                new_fwhm_measurements=fwhm_measurements,
                target_filter=target_filter,
                master_psf=config.psf_generation.master_psf
            )
        else:
            inverse_filters = determine_inverse_filters(fwhm_measurements, target_filter)
            create_psfs_toml(
                project_dir,
                inverse_filters=inverse_filters,
                fwhm_measurements=fwhm_measurements,
                target_filter=target_filter,
                master_psf=config.psf_generation.master_psf,
            )
        
        # Reload the updated config to get the complete dataset
        psfs_config = load_psfs_config(project_dir)
    else:
        # Use existing measurements
        logger.info("Loading PSF information from psfs.toml")

    inverse_filters = psfs_config.inverse_filters
    target_fwhm = psfs_config.fwhm[target_filter]
    
    # Log filter types
    logger.info("Filter homogenization directions:")
    for filt in filters_to_process:
        fwhm = psfs_config.fwhm[filt]
        if filt in inverse_filters:
            logger.info(f'    [bold cyan]{filt}[/bold cyan]: [bold yellow]inverse[/bold yellow] (filter FWHM {fwhm}" > target FWHM {target_fwhm}")')
        else:
            logger.info(f'    [bold cyan]{filt}[/bold cyan]: [bold green]forward[/bold green] (filter FWHM {fwhm}" < target FWHM {target_fwhm}")')
    
    # Process each filter
    if len(filters_to_process)>1:
        logger.info(f"Processing {len(filters_to_process)} filters...")
    
    for filter_name in filters_to_process:
        # Get tiles for this filter
        filter_tiles = images_config.get_tiles_for_filter(filter_name)
        
        if tiles_to_process:
            # Filter to requested tiles
            filter_tiles = [t for t in filter_tiles if t in tiles_to_process]
        
        if not filter_tiles:
            logger.warning(f"No tiles found for filter {filter_name}")
            continue
        
        logger.info(f"Processing filter [bold cyan]{filter_name}[/bold cyan] ({len(filter_tiles)} tiles)")
        
        is_inverse = filter_name in inverse_filters
        
        # Process each tile
        for tile in filter_tiles:
            try:
                success = homogenize_filter_tile(
                    project_dir=project_dir,
                    filter_name=filter_name,
                    tile=tile,
                    target_filter=target_filter,
                    is_inverse=is_inverse,
                    master_psf=config.psf_generation.master_psf,
                    reg_fact=config.psf_homogenization.reg_fact,
                    overwrite=overwrite,
                    output_prefix=config.psf_homogenization.output_prefix
                )
                
                # Force garbage collection after each tile
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error processing {filter_name}-{tile}: {str(e)}")
                if len(filter_tiles) == 1:
                    raise
                else:
                    logger.warning("Continuing with remaining tiles...")
                    continue
    
    # ALWAYS update images.toml with homogenized paths
    logger.info("Updating images.toml with homogenized file paths...")
    
    for filter_name in filters_to_process:
        try:
            # Get tiles for this filter
            filter_tiles = images_config.get_tiles_for_filter(filter_name)
            
            if tiles_to_process:
                filter_tiles = [t for t in filter_tiles if t in tiles_to_process]
            
            if filter_tiles:
                is_inverse = filter_name in inverse_filters
                update_homogenized_paths_in_images_toml(
                    project_dir=project_dir,
                    filter_name=filter_name,
                    tiles=filter_tiles,
                    target_filter=target_filter,
                    is_inverse=is_inverse,
                    output_prefix=config.psf_homogenization.output_prefix
                )
        except Exception as e:
            logger.warning(f"Failed to update images.toml for {filter_name}: {e}")
    
    # TODO: Update state tracking when implemented
    
    logger.info(f"[bold green]Success:[/bold green] PSF homogenization completed")