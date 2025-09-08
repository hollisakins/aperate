"""Images.toml file handling for aperate."""

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

import numpy as np
from astropy.io import fits

from ..core.logging import get_logger, print_error


@dataclass
class ImageFiles:
    """Container for image files for a specific filter/tile combination."""
    filter: str
    tile: str
    extensions: Dict[str, str]  # extension -> absolute_path
    
    def get_full_path(self, extension: str) -> Optional[Path]:
        """Get full path for a specific extension."""
        if extension not in self.extensions:
            return None
        
        # All paths are now absolute, so return directly
        return Path(self.extensions[extension])
    
    def has_extension(self, extension: str) -> bool:
        """Check if extension is available."""
        return extension in self.extensions
    
    def get_science_path(self) -> Optional[Path]:
        """Get path to science image."""
        return self.get_full_path('sci')
    
    def get_error_path(self) -> Optional[Path]:
        """Get path to error image."""
        return self.get_full_path('err')
    
    def get_weight_path(self) -> Optional[Path]:
        """Get path to weight image."""
        return self.get_full_path('wht')
    
    def get_rms_path(self) -> Optional[Path]:
        """Get path to RMS image."""
        return self.get_full_path('rms')
    
    def get_psf_path(self) -> Optional[Path]:
        """Get path to PSF file."""
        return self.get_full_path('psf')
    
    def get_homogenized_path(self, target_filter: str) -> Optional[Path]:
        """Get path to homogenized file for a specific target filter."""
        hom_ext = f'hom-{target_filter}'
        return self.get_full_path(hom_ext)
    
    # NOTE: Detection products are now stored per-tile, not per filter-tile
    # Use ImagesConfig.get_tile_detection_image_path() and get_tile_segmap_path() instead
    def get_detection_image_path(self, detection_type: str = 'ivw') -> Optional[Path]:
        """DEPRECATED: Detection products are now per-tile. Use ImagesConfig.get_tile_detection_image_path()."""
        # Try simple key first, then complex key for backward compatibility
        if 'det' in self.extensions:
            return self.get_full_path('det')
        det_ext = f'det-{detection_type}'
        return self.get_full_path(det_ext)
    
    def get_segmap_path(self, detection_type: str = 'ivw', detection_scheme: str = 'hot+cold') -> Optional[Path]:
        """DEPRECATED: Detection products are now per-tile. Use ImagesConfig.get_tile_segmap_path()."""
        # Try simple key first, then complex key for backward compatibility
        if 'segmap' in self.extensions:
            return self.get_full_path('segmap')
        segmap_ext = f'segmap-{detection_type}-{detection_scheme}'
        return self.get_full_path(segmap_ext)


class ImagesConfig:
    """Manager for images.toml configuration."""
    
    def __init__(self, images_path: Path):
        self.images_path = images_path
        self.logger = get_logger()
        self._data = None
        self._load()
    
    def _load(self) -> None:
        """Load images.toml file."""
        if not self.images_path.exists():
            raise FileNotFoundError(f"Images file not found: {self.images_path}")
        
        try:
            with open(self.images_path, "rb") as f:
                self._data = tomllib.load(f)
        except Exception as e:
            print_error(f"Failed to parse images.toml: {e}")
            raise
        
        # Validate structure
        if 'images' not in self._data:
            raise ValueError("images.toml must contain an 'images' section")
        
        if 'files' not in self._data['images']:
            raise ValueError("images.toml must contain an 'images.files' section")
    
    # base_path and psf_base_path properties removed - images.toml now uses absolute paths
    
    def get_filters(self) -> List[str]:
        """Get list of available filters."""
        return list(self._data['images']['files'].keys())
    
    def get_tiles_for_filter(self, filter_name: str) -> List[str]:
        """Get list of tiles available for a filter."""
        filter_data = self._data['images']['files'].get(filter_name, {})
        return list(filter_data.keys())
    
    def get_image_files(self, filter_name: str, tile: str) -> Optional[ImageFiles]:
        """Get ImageFiles object for a specific filter/tile combination."""
        filter_data = self._data['images']['files'].get(filter_name, {})
        tile_data = filter_data.get(tile, {})
        
        if not tile_data:
            return None
        
        return ImageFiles(
            filter=filter_name,
            tile=tile,
            extensions=tile_data
        )
    
    def get_all_image_files(self) -> List[ImageFiles]:
        """Get all ImageFiles objects."""
        all_files = []
        for filter_name in self.get_filters():
            for tile in self.get_tiles_for_filter(filter_name):
                image_files = self.get_image_files(filter_name, tile)
                if image_files:
                    all_files.append(image_files)
        return all_files
    
    def filter_tile_exists(self, filter_name: str, tile: str) -> bool:
        """Check if a filter/tile combination exists."""
        return self.get_image_files(filter_name, tile) is not None
    
    def get_tile_detection_image_path(self, tile: str) -> Optional[Path]:
        """Get path to detection image for a specific tile."""
        detection_data = self._data.get('images', {}).get('detection', {}).get(tile, {})
        
        # Try new simple key first
        det_path = detection_data.get('det')
        if det_path:
            return Path(det_path)
        
        # Backward compatibility: try old complex keys
        for key in detection_data.keys():
            if key.startswith('det-'):
                return Path(detection_data[key])
        
        # Backward compatibility: try to find in filter data (old structure)
        for filter_name in self.get_filters():
            image_files = self.get_image_files(filter_name, tile)
            if image_files:
                for ext in image_files.extensions.keys():
                    if ext.startswith('det-'):
                        return Path(image_files.extensions[ext])
        
        return None
    
    def get_tile_segmap_path(self, tile: str) -> Optional[Path]:
        """Get path to segmentation map for a specific tile."""
        detection_data = self._data.get('images', {}).get('detection', {}).get(tile, {})
        
        # Try new simple key first
        segmap_path = detection_data.get('segmap')
        if segmap_path:
            return Path(segmap_path)
        
        # Backward compatibility: try old complex keys
        for key in detection_data.keys():
            if key.startswith('segmap-'):
                return Path(detection_data[key])
        
        # Backward compatibility: try to find in filter data (old structure)
        for filter_name in self.get_filters():
            image_files = self.get_image_files(filter_name, tile)
            if image_files:
                for ext in image_files.extensions.keys():
                    if ext.startswith('segmap-'):
                        return Path(image_files.extensions[ext])
        
        return None
    
    def get_tiles_with_detection_products(self) -> List[str]:
        """Get list of tiles that have detection products."""
        detection_data = self._data.get('images', {}).get('detection', {})
        tiles_with_detection = list(detection_data.keys())
        
        # Also check old structure for backward compatibility
        for filter_name in self.get_filters():
            for tile in self.get_tiles_for_filter(filter_name):
                image_files = self.get_image_files(filter_name, tile)
                if image_files:
                    # Check if any detection products exist in old structure
                    for ext in image_files.extensions.keys():
                        if ext.startswith('det-') or ext.startswith('segmap-'):
                            if tile not in tiles_with_detection:
                                tiles_with_detection.append(tile)
                            break
        
        return sorted(set(tiles_with_detection))
    
    def get_psf_path(self, filter_name: str, tile: str) -> Optional[Path]:
        """Get path to PSF file for a specific filter/tile."""
        filter_data = self._data['images']['files'].get(filter_name, {})
        tile_data = filter_data.get(tile, {})
        psf_path = tile_data.get('psf')
        
        if psf_path:
            # Paths are now absolute
            return Path(psf_path)
        return None
    
    def add_derived_file(self, filter_name: str, tile: str, extension: str, absolute_path: str) -> None:
        """Add a derived file (e.g., PSF-homogenized) to the configuration."""
        # This would update the in-memory structure and could be saved back
        # For now, we'll implement this as needed
        # Note: absolute_path should be absolute since images.toml now uses absolute paths
        raise NotImplementedError("Adding derived files not yet implemented")
    
    def validate_files_exist(self) -> Dict[str, List[str]]:
        """Validate that all referenced files actually exist."""
        missing_files = {'filters': [], 'files': []}
        
        for image_files in self.get_all_image_files():
            for extension, absolute_path in image_files.extensions.items():
                full_path = Path(absolute_path)
                if not full_path.exists():
                    missing_files['files'].append(str(full_path))
        
        return missing_files


def create_images_toml(images_data: Dict[str, Any], output_path: Path) -> None:
    """Create a new images.toml file."""
    logger = get_logger()
    
    try:
        with open(output_path, 'w') as f:
            toml.dump(images_data, f)
        logger.info(f"Created images.toml: {output_path}")
    except Exception as e:
        print_error(f"Failed to create images.toml: {e}")
        raise


def load_images_config(project_dir: Path) -> ImagesConfig:
    """Load images configuration from a project directory."""
    images_path = project_dir / 'images.toml'
    return ImagesConfig(images_path)


def find_images_config(project_dir: Optional[Path] = None) -> Optional[ImagesConfig]:
    """Find and load images config, returning None if not found."""
    if project_dir is None:
        project_dir = Path.cwd()
    
    images_path = project_dir / 'images.toml'
    if images_path.exists():
        return ImagesConfig(images_path)
    return None


def _construct_psf_path(filter_name: str, tile: str, master_psf: bool) -> str:
    """Construct relative PSF path (relative to psf_base_path)."""
    if master_psf:
        return f"psfs/psf_{filter_name}_master.fits"
    else:
        return f"psfs/psf_{filter_name}_{tile}.fits"


def _get_full_psf_path(project_dir: Path, psf_base_path: str, psf_rel_path: str) -> Path:
    """Convert relative PSF path to absolute path."""
    if psf_base_path == '.':
        # PSF base path is project directory
        return project_dir / psf_rel_path
    else:
        # PSF base path is absolute or relative to some other location
        return Path(psf_base_path) / psf_rel_path


def update_psf_paths_in_images_toml(
    project_dir: Path,
    filter_name: str, 
    tiles: List[str],
    master_psf: bool
) -> None:
    """
    Update images.toml with PSF file paths for a specific filter.
    
    Args:
        project_dir: Project directory containing images.toml
        filter_name: Filter being processed (e.g., 'f814w')
        tiles: List of tiles for this filter (e.g., ['B1', 'B2'])
        master_psf: Whether to use master PSF (True) or individual PSFs (False)
    """
    logger = get_logger()
    images_path = project_dir / "images.toml"
    
    if not images_path.exists():
        logger.error(f"images.toml not found at {images_path}")
        return
    
    # Load current TOML
    with open(images_path, 'rb') as f:
        images_data = tomllib.load(f)
    
    # Use psf_base_path from images.toml if it exists (for backward compatibility)
    # Otherwise default to project directory - only used for path construction, not storage
    psf_base_path = images_data['images'].get('psf_base_path', '.')
    logger.info(f"Updating PSF paths for {filter_name}")
    
    # Update each tile
    for tile in tiles:
        psf_rel_path = _construct_psf_path(filter_name, tile, master_psf)
        psf_full_path = _get_full_psf_path(project_dir, psf_base_path, psf_rel_path)
        
        # Only add PSF path if file actually exists
        if psf_full_path.exists():
            # Ensure the nested structure exists
            if filter_name not in images_data['images']['files']:
                images_data['images']['files'][filter_name] = {}
            if tile not in images_data['images']['files'][filter_name]:
                images_data['images']['files'][filter_name][tile] = {}
            
            # Add PSF path (absolute path)
            images_data['images']['files'][filter_name][tile]['psf'] = str(psf_full_path)
            logger.info(f"  {filter_name}.{tile}: psf = {psf_full_path}")
        else:
            logger.warning(f"  {filter_name}.{tile}: PSF file not found at {psf_full_path}")
    
    # Write updated TOML
    with open(images_path, 'w') as f:
        toml.dump(images_data, f)
    
    logger.info(f"Updated images.toml with PSF paths for {filter_name}")


def update_homogenized_paths_in_images_toml(
    project_dir: Path,
    filter_name: str,
    tiles: List[str],
    target_filter: str,
    is_inverse: bool,
    output_prefix: str = ''
) -> None:
    """
    Update images.toml with homogenized file paths.
    
    For inverse filters, the homogenized path points to the homogenized
    target image, not the filter's own image.
    
    Args:
        project_dir: Project directory
        filter_name: Filter being processed
        tiles: List of tiles for this filter
        target_filter: Target filter for homogenization
        is_inverse: Whether this is an inverse filter
    """
    logger = get_logger()
    images_path = project_dir / "images.toml"
    
    if not images_path.exists():
        logger.error(f"images.toml not found at {images_path}")
        return
    
    # Load current TOML
    with open(images_path, 'rb') as f:
        images_data = tomllib.load(f)
    
    # Prefix is only used for constructing the filename, not the key
    if output_prefix:
        prefix = f"{output_prefix}-"
    else:
        prefix = ""
    
    logger.info(f"Updating homogenized paths for {filter_name} (inverse={is_inverse}, prefix='{output_prefix}')")
    
    # Update each tile
    for tile in tiles:
        # Ensure nested structure exists
        if filter_name not in images_data['images']['files']:
            images_data['images']['files'][filter_name] = {}
        if tile not in images_data['images']['files'][filter_name]:
            images_data['images']['files'][filter_name][tile] = {}
        
        # Get the appropriate source image info
        if is_inverse:
            # For inverse filters, we homogenized the target image
            source_filter = target_filter
        else:
            # Normal case: we homogenized this filter's image
            source_filter = filter_name
        
        # Get source image path to construct homogenized path
        source_data = images_data['images']['files'].get(source_filter, {}).get(tile, {})
        sci_path = source_data.get('sci') or source_data.get('drz')
        
        if not sci_path:
            logger.warning(f"  {filter_name}.{tile}: No science image found")
            continue
        
        # Construct homogenized filename - must match what homogenize_filter_tile creates
        sci_name = Path(sci_path).stem
        
        # Determine the actual filename extension used (matches homogenize_filter_tile logic)
        # Filename includes prefix
        if is_inverse:
            actual_hom_ext = f"{prefix}hom-{filter_name}"  # Inverse: file named hom-filter
        else:
            actual_hom_ext = f"{prefix}hom-{target_filter}"  # Forward: file named hom-target
        
        # Key for images.toml should always be hom-{target_filter}
        # For both forward and inverse, photometry looks for hom-{target_filter}
        hom_key = f"hom-{target_filter}"
        
        if "_sci" in sci_name:
            hom_name = sci_name.replace("_sci", f"_{actual_hom_ext}")
        elif "_drz" in sci_name:
            hom_name = sci_name.replace("_drz", f"_{actual_hom_ext}")
        elif "_mock" in sci_name:
            # Handle mock files - try to replace the exact prefix first
            if output_prefix and f"_{output_prefix}" in sci_name:
                # Replace exact prefix match (e.g., _mock1)
                hom_name = sci_name.replace(f"_{output_prefix}", f"_{actual_hom_ext}")
            else:
                # Fallback: replace generic _mock
                hom_name = sci_name.replace("_mock", f"_{actual_hom_ext}")
        else:
            hom_name = f"{sci_name}_{actual_hom_ext}"
        
        # Construct absolute homogenized path (sci_path is now absolute)
        hom_full_path = Path(sci_path).parent / f"{hom_name}.fits"
        if hom_full_path.exists():
            # Add homogenized path (absolute path)
            # Use hom_key (without prefix) for the key, but the path contains the file with prefix
            images_data['images']['files'][filter_name][tile][hom_key] = str(hom_full_path)
            logger.info(f"  {filter_name}.{tile}: {hom_key} = {hom_full_path}")
        else:
            logger.warning(f"  {filter_name}.{tile}: Homogenized file not found at {hom_full_path}")
    
    # Write updated TOML
    with open(images_path, 'w') as f:
        toml.dump(images_data, f)
    
    logger.info(f"Updated images.toml with homogenized paths for {filter_name}")


def update_detection_paths_in_images_toml(
    project_dir: Path,
    tiles: List[str],
    detection_type: str,
    detection_scheme: str
) -> None:
    """
    Update images.toml with detection image and segmentation map paths using simplified keys.
    
    Args:
        project_dir: Project directory containing images.toml
        tiles: List of tiles to update
        detection_type: Detection image type (e.g., 'ivw', 'chisq')
        detection_scheme: Detection scheme (e.g., 'hot+cold', 'single')
    """
    logger = get_logger()
    images_path = project_dir / "images.toml"
    
    if not images_path.exists():
        logger.error(f"images.toml not found at {images_path}")
        return
    
    # Load current TOML
    with open(images_path, 'rb') as f:
        images_data = tomllib.load(f)
    
    detection_dir = project_dir / "detection_images"
    
    logger.info(f"Updating detection paths for {len(tiles)} tiles")
    
    # Ensure detection section exists
    if 'detection' not in images_data['images']:
        images_data['images']['detection'] = {}
    
    # Update each tile in the detection section
    for tile in tiles:
        detection_image_path = detection_dir / f"detection_image_{detection_type}_{tile}.fits"
        segmap_path = detection_dir / f"segmap_{detection_type}_{detection_scheme}_{tile}.fits"
        
        # Ensure tile section exists in detection
        if tile not in images_data['images']['detection']:
            images_data['images']['detection'][tile] = {}
        
        # Add detection image path if it exists (using simple 'det' key)
        if detection_image_path.exists():
            images_data['images']['detection'][tile]['det'] = str(detection_image_path)
            logger.info(f"  {tile}: det = {detection_image_path}")
        else:
            logger.warning(f"  {tile}: Detection image not found at {detection_image_path}")
        
        # Add segmentation map path if it exists (using simple 'segmap' key)
        if segmap_path.exists():
            images_data['images']['detection'][tile]['segmap'] = str(segmap_path)
            logger.info(f"  {tile}: segmap = {segmap_path}")
        else:
            logger.warning(f"  {tile}: Segmentation map not found at {segmap_path}")
    
    # Write updated TOML
    with open(images_path, 'w') as f:
        toml.dump(images_data, f)
    
    logger.info(f"Updated images.toml with detection paths for {len(tiles)} tiles")


def check_detection_paths_completeness(
    project_dir: Path,
    tiles: List[str],
    detection_type: str,
    detection_scheme: str
) -> bool:
    """
    Check if detection paths are complete in images.toml using simplified keys.
    
    Args:
        project_dir: Project directory containing images.toml
        tiles: List of tiles to check
        detection_type: Detection image type (e.g., 'ivw', 'chisq')
        detection_scheme: Detection scheme (e.g., 'hot+cold', 'single')
        
    Returns:
        True if paths need updating, False if complete
    """
    logger = get_logger()
    images_path = project_dir / "images.toml"
    
    if not images_path.exists():
        return True  # Need to update if images.toml doesn't exist
    
    try:
        with open(images_path, 'rb') as f:
            images_data = tomllib.load(f)
        
        detection_dir = project_dir / "detection_images"
        detection_data = images_data.get('images', {}).get('detection', {})
        
        # Check if detection paths are missing for any tile
        for tile in tiles:
            detection_image_path = detection_dir / f"detection_image_{detection_type}_{tile}.fits"
            segmap_path = detection_dir / f"segmap_{detection_type}_{detection_scheme}_{tile}.fits"
            
            tile_detection_data = detection_data.get(tile, {})
            
            # Check if detection image path is missing but file exists
            if detection_image_path.exists():
                # Check for simple key first, then complex keys for backward compatibility
                if ('det' not in tile_detection_data and 
                    not any(k.startswith('det-') for k in tile_detection_data.keys())):
                    logger.debug(f"    Missing detection image path for tile {tile}")
                    return True
            
            # Check if segmap path is missing but file exists
            if segmap_path.exists():
                # Check for simple key first, then complex keys for backward compatibility
                if ('segmap' not in tile_detection_data and 
                    not any(k.startswith('segmap-') for k in tile_detection_data.keys())):
                    logger.debug(f"    Missing segmap path for tile {tile}")
                    return True
        
        return False  # All detection paths are present
        
    except Exception as e:
        logger.warning(f"    Could not check detection paths: {e}")
        return True  # Assume needs update if we can't read it


def migrate_detection_paths_to_per_tile_structure(project_dir: Path) -> bool:
    """
    Migrate detection paths from old structures to new simplified per-tile structure.
    
    Handles migration from:
    1. Old per-filter-tile structure with complex keys (det-ivw, segmap-ivw-hot+cold)
    2. New per-tile structure with complex keys → simple keys (det, segmap)
    
    Args:
        project_dir: Project directory containing images.toml
        
    Returns:
        True if migration was performed, False if not needed or failed
    """
    logger = get_logger()
    images_path = project_dir / "images.toml"
    
    if not images_path.exists():
        logger.warning(f"images.toml not found at {images_path}")
        return False
    
    try:
        # Load current TOML
        with open(images_path, 'rb') as f:
            images_data = tomllib.load(f)
        
        files_data = images_data.get('images', {}).get('files', {})
        detection_data = images_data.get('images', {}).get('detection', {})
        detection_products = {}
        migration_needed = False
        
        # Step 1: Migrate from old per-filter-tile structure
        for filter_name, filter_data in files_data.items():
            for tile, tile_data in filter_data.items():
                for ext_name, ext_path in list(tile_data.items()):
                    if ext_name.startswith('det-') or ext_name.startswith('segmap-'):
                        # Add to per-tile structure with simple key
                        if tile not in detection_products:
                            detection_products[tile] = {}
                        
                        # Convert to simple key
                        if ext_name.startswith('det-'):
                            detection_products[tile]['det'] = ext_path
                        elif ext_name.startswith('segmap-'):
                            detection_products[tile]['segmap'] = ext_path
                        
                        migration_needed = True
        
        # Step 2: Migrate existing per-tile structure with complex keys to simple keys
        for tile, tile_products in detection_data.items():
            if tile not in detection_products:
                detection_products[tile] = {}
            
            for ext_name, ext_path in tile_products.items():
                if ext_name.startswith('det-') and 'det' not in detection_products[tile]:
                    detection_products[tile]['det'] = ext_path
                    migration_needed = True
                elif ext_name.startswith('segmap-') and 'segmap' not in detection_products[tile]:
                    detection_products[tile]['segmap'] = ext_path
                    migration_needed = True
                elif ext_name in ['det', 'segmap']:
                    # Already using simple keys
                    detection_products[tile][ext_name] = ext_path
        
        if not migration_needed:
            logger.info("Detection paths already use simplified per-tile structure")
            return False
        
        # Update images.toml with new structure
        images_data['images']['detection'] = detection_products
        
        # Remove detection products from filter-tile sections
        for filter_name, filter_data in files_data.items():
            for tile, tile_data in filter_data.items():
                keys_to_remove = []
                for ext_name in tile_data.keys():
                    if ext_name.startswith('det-') or ext_name.startswith('segmap-'):
                        keys_to_remove.append(ext_name)
                
                for key in keys_to_remove:
                    del tile_data[key]
        
        # Write updated TOML
        with open(images_path, 'w') as f:
            toml.dump(images_data, f)
        
        logger.info(f"Migrated {len(detection_products)} tiles to simplified per-tile detection structure")
        for tile, products in detection_products.items():
            for ext_name in products.keys():
                logger.info(f"  {tile}: {ext_name}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to migrate detection paths: {e}")
        return False


def load_segmentation_map(images_config, tile: str) -> Optional[np.ndarray]:
    """
    Load segmentation map for a tile.
    
    Args:
        images_config: ImagesConfig object
        tile: Tile name
        
    Returns:
        Segmentation map array or None if not found
    """
    logger = get_logger()
    segmap_path = images_config.get_tile_segmap_path(tile)
    
    if not segmap_path or not segmap_path.exists():
        logger.warning(f"Segmentation map not found for tile {tile}")
        return None
        
    try:
        with fits.open(segmap_path) as hdul:
            segmap = hdul[0].data
        logger.debug(f"    Loaded segmentation map from {segmap_path}")
        return segmap
    except Exception as e:
        logger.error(f"    Failed to load segmentation map {segmap_path}: {str(e)}")
        return None