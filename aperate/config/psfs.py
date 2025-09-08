"""PSFs.toml file handling for aperate."""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# Handle tomli/toml imports
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        raise ImportError("tomli package required for Python < 3.11")

try:
    import toml
except ImportError:
    raise ImportError("toml package required for writing TOML files")

from ..core.logging import get_logger


@dataclass
class PSFsConfig:
    """Container for PSF metadata."""
    master_psf: bool
    target_filter: str
    fwhm: Dict[str, float]  # filter -> FWHM in pixels
    inverse_filters: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for TOML serialization."""
        return {
            'psfs': {
                'master_psf': self.master_psf,
                'target_filter': self.target_filter,
                'fwhm': self.fwhm,
                'inverse_filters': {
                    'filters': self.inverse_filters
                }
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PSFsConfig':
        """Create from dictionary loaded from TOML."""
        psfs_data = data.get('psfs', {})
        return cls(
            master_psf=psfs_data.get('master_psf', True),
            target_filter=psfs_data.get('target_filter', ''),
            fwhm=psfs_data.get('fwhm', {}),
            inverse_filters=psfs_data.get('inverse_filters', {}).get('filters', [])
        )


def create_psfs_toml(
    project_dir: Path,
    master_psf: bool,
    target_filter: str,
    fwhm_measurements: Dict[str, float],
    inverse_filters: List[str]
) -> None:
    """
    Create a new psfs.toml file.
    
    Args:
        project_dir: Project directory
        master_psf: Whether using master PSFs
        target_filter: Target filter for homogenization
        fwhm_measurements: Dictionary of filter -> FWHM (pixels)
        inverse_filters: List of filters requiring inverse homogenization
    """
    logger = get_logger()
    
    config = PSFsConfig(
        master_psf=master_psf,
        target_filter=target_filter,
        fwhm=fwhm_measurements,
        inverse_filters=inverse_filters
    )
    
    output_path = project_dir / 'psfs.toml'
    
    try:
        with open(output_path, 'w') as f:
            toml.dump(config.to_dict(), f)
        logger.info(f"Created psfs.toml: {output_path}")
    except Exception as e:
        logger.error(f"Failed to create psfs.toml: {e}")
        raise


def load_psfs_config(project_dir: Path) -> Optional[PSFsConfig]:
    """
    Load PSF configuration from project directory.
    
    Args:
        project_dir: Project directory
        
    Returns:
        PSFsConfig object or None if file doesn't exist
    """
    logger = get_logger()
    psfs_path = project_dir / 'psfs.toml'
    
    if not psfs_path.exists():
        logger.debug("psfs.toml not found")
        return None
    
    try:
        with open(psfs_path, 'rb') as f:
            data = tomllib.load(f)
        return PSFsConfig.from_dict(data)
    except Exception as e:
        logger.error(f"Failed to load psfs.toml: {e}")
        raise


def update_psf_fwhm(
    project_dir: Path,
    filter_name: str,
    fwhm_pixels: float
) -> None:
    """
    Update FWHM measurement for a specific filter.
    
    Args:
        project_dir: Project directory
        filter_name: Filter to update
        fwhm_pixels: FWHM measurement in pixels
    """
    logger = get_logger()
    
    # Load existing config or create new
    config = load_psfs_config(project_dir)
    if config is None:
        logger.warning("No psfs.toml found, cannot update FWHM")
        return
    
    # Update FWHM
    config.fwhm[filter_name] = fwhm_pixels
    
    # Save back
    output_path = project_dir / 'psfs.toml'
    try:
        with open(output_path, 'w') as f:
            toml.dump(config.to_dict(), f)
        logger.debug(f"Updated FWHM for {filter_name}: {fwhm_pixels:.2f} pixels")
    except Exception as e:
        logger.error(f"Failed to update psfs.toml: {e}")
        raise


def update_psfs_toml_with_missing_fwhms(
    project_dir: Path,
    missing_filters: List[str],
    new_fwhm_measurements: Dict[str, float],
    target_filter: str,
    master_psf: bool
) -> None:
    """
    Update psfs.toml with new FWHM measurements, preserving existing data.
    
    Args:
        project_dir: Project directory
        missing_filters: List of filters that needed measurement
        new_fwhm_measurements: New FWHM measurements for missing filters
        target_filter: Target filter for homogenization
        master_psf: Whether using master PSFs
    """
    from ..commands.homogenize import determine_inverse_filters  # Import here to avoid circular dependency
    logger = get_logger()
    
    # Load existing config or create base structure
    existing_config = load_psfs_config(project_dir)
    
    if existing_config:
        # Update existing config
        logger.info("Updating existing psfs.toml with new FWHM measurements")
        
        # Merge FWHM measurements
        combined_fwhm = existing_config.fwhm.copy()
        combined_fwhm.update(new_fwhm_measurements)
        
        # Update target filter and master_psf settings
        updated_target_filter = target_filter
        updated_master_psf = master_psf
        
        # Log what's being updated
        updated_filters = list(new_fwhm_measurements.keys())
        logger.info(f"Adding FWHM measurements for: {', '.join(updated_filters)}")
        
    else:
        # Create new config
        logger.info("Creating new psfs.toml with FWHM measurements")
        combined_fwhm = new_fwhm_measurements.copy()
        updated_target_filter = target_filter
        updated_master_psf = master_psf
    
    # Recalculate inverse filters with complete FWHM dataset
    inverse_filters = determine_inverse_filters(combined_fwhm, updated_target_filter)
    
    # Create updated config
    updated_config = PSFsConfig(
        master_psf=updated_master_psf,
        target_filter=updated_target_filter,
        fwhm=combined_fwhm,
        inverse_filters=inverse_filters
    )
    
    # Save updated config
    output_path = project_dir / 'psfs.toml'
    try:
        with open(output_path, 'w') as f:
            toml.dump(updated_config.to_dict(), f)
        
        if existing_config:
            logger.info(f"Updated psfs.toml with {len(new_fwhm_measurements)} new FWHM measurements")
        else:
            logger.info(f"Created psfs.toml with {len(combined_fwhm)} FWHM measurements")
            
    except Exception as e:
        logger.error(f"Failed to update psfs.toml: {e}")
        raise