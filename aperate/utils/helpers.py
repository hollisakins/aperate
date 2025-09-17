"""Common utility functions for aperate."""

from pathlib import Path
from typing import Optional

from ..core.catalog import find_catalog_in_directory
from ..core.logging import print_error

import numpy as np
import warnings
from astropy.wcs import FITSFixedWarning
warnings.simplefilter('ignore', category=FITSFixedWarning)


def Gaussian(x, a, mu, sig):
    return a * np.exp(-(x-mu)**2/(2*sig**2))


def get_project_catalog(project_dir: Optional[Path] = None) -> Path:
    """Get the catalog path for a project directory."""
    if project_dir is None:
        project_dir = Path.cwd()
    
    catalog_path = find_catalog_in_directory(project_dir)
    if catalog_path is None:
        print_error(f"No catalog.fits found in {project_dir}")
        raise FileNotFoundError(f"No catalog found in {project_dir}")
    
    return catalog_path


def validate_project_directory(project_dir: Path) -> bool:
    """Check if a directory is a valid aperate project."""
    if not project_dir.exists() or not project_dir.is_dir():
        return False
    
    # Check for required files
    required_files = ["config.toml", "catalog.fits"]
    for filename in required_files:
        if not (project_dir / filename).exists():
            return False
    
    # Check for required subdirectories
    required_dirs = ["psf_models", "detection", "photometry"]
    for dirname in required_dirs:
        if not (project_dir / dirname).exists():
            return False
    
    return True


# TODO: Add more utility functions as needed
# This is placeholder - user will add specific utilities as needed


def get_unit_conversion(image_header, output_unit):
    """
    Get unit conversion factor from image native units to desired output units.
    
    Args:
        image_header: FITS header containing unit and calibration information
        output_unit: Desired output unit (string, e.g., 'uJy')
        
    Returns:
        Conversion factor to multiply fluxes by
        
    Raises:
        ValueError: If units are unsupported or header information is missing
    """
    import astropy.units as u
    from astropy.wcs import WCS
    from astropy.constants import c as speed_of_light
    
    try:
        output_unit = u.Unit(output_unit)
    except Exception as e:
        raise ValueError(f"Invalid output unit '{output_unit}': {e}")

    # Check if output unit is a spectral flux density
    try:
        physical_type = output_unit.physical_type
        if 'spectral flux density' not in physical_type:
            raise ValueError(f'Currently only spectral flux density units are supported, got {physical_type}')
    except Exception as e:
        raise ValueError(f"Could not determine physical type of output unit '{output_unit}': {e}")

    # Get current unit from header
    if 'BUNIT' not in image_header:
        raise ValueError("Image header missing required 'BUNIT' key")
    
    current_unit = image_header['BUNIT']

    try:
        match current_unit:
            case 'MJy/sr': # typical units for JWST images 
                conversion = 1e6*u.Jy/u.sr
                try:
                    wcs = WCS(image_header)
                    pixel_scale = wcs.proj_plane_pixel_scales()[0]
                    conversion *= (pixel_scale**2).to(u.sr)
                    conversion = conversion.to(output_unit).value
                except Exception as e:
                    raise ValueError(f"Failed to compute pixel scale from WCS: {e}")

            case 'ELECTRONS/S': # typical units for HST images
                # Check for required HST calibration keywords
                required_keys = ['PHOTFLAM', 'PHOTPLAM']
                missing_keys = [key for key in required_keys if key not in image_header]
                if missing_keys:
                    raise ValueError(f"HST image header missing required keys: {missing_keys}")
                
                try:
                    photflam = image_header['PHOTFLAM'] * u.erg/u.cm**2/u.angstrom
                    photplam = image_header['PHOTPLAM'] * u.angstrom
                    conversion = ((1/u.s) * photflam * photplam**2 / speed_of_light).to(output_unit).value
                except Exception as e:
                    raise ValueError(f"Failed to compute HST unit conversion: {e}")
            
            case 'uJy/pixel':
                conversion = (1*u.uJy).to(output_unit).value

            case _:
                raise ValueError(f"Unsupported image units: '{current_unit}'")
        
        return conversion
        
    except Exception as e:
        # Re-raise with more context if it's not already a ValueError
        if isinstance(e, ValueError):
            raise
        else:
            raise ValueError(f"Unit conversion failed for {current_unit} -> {output_unit}: {e}")