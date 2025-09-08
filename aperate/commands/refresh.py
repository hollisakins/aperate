"""Refresh aperate project configuration and image discovery."""

import click
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Handle tomli import for different Python versions
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

from ..config.parser import load_config
from ..config.images import create_images_toml
from ..core.discovery import ImageDiscovery, generate_images_toml
from ..core.logging import get_logger


def validate_project_directory(project_dir: Path) -> None:
    """Validate that we're in a valid aperate project directory."""
    logger = get_logger()
    
    required_files = ['config.toml', 'images.toml']
    missing_files = []
    
    for filename in required_files:
        filepath = project_dir / filename
        if not filepath.exists():
            missing_files.append(filename)
    
    if missing_files:
        logger.error(f"Not a valid aperate project directory. Missing files: {', '.join(missing_files)}")
        raise click.ClickException(f"Missing required files: {', '.join(missing_files)}")
    
    logger.debug("Project directory validation passed")


def backup_images_toml(project_dir: Path) -> Path:
    """Create a timestamped backup of the existing images.toml file."""
    logger = get_logger()
    
    images_path = project_dir / "images.toml"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = project_dir / f"images.toml.backup_{timestamp}"
    
    # Copy the file
    backup_path.write_text(images_path.read_text())
    logger.info(f"Backed up existing images.toml to: {backup_path.name}")
    
    return backup_path


def load_existing_images_toml(project_dir: Path) -> Dict[str, Any]:
    """Load the existing images.toml file."""
    images_path = project_dir / "images.toml"
    
    with open(images_path, "rb") as f:
        return tomllib.load(f)


def merge_preserve_derived_files(
    old_images: Dict[str, Any], 
    new_images_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge old and new images data, preserving derived files (PSF, homogenized).
    
    Strategy:
    1. Start with new discovery results (base images: sci, err, wht, rms)
    2. Add back any derived files (psf, hom-*) from old data that still exist on disk
    """
    logger = get_logger()
    
    # Start with new discovery results
    merged = new_images_data.copy()
    
    # Get old files data
    old_files = old_images.get('images', {}).get('files', {})
    new_files = merged['images']['files']
    
    derived_extensions = set()
    preserved_count = 0
    
    # For each filter/tile in old data, check for derived files
    for filter_name, filter_data in old_files.items():
        if filter_name not in new_files:
            continue
            
        for tile, tile_data in filter_data.items():
            if tile not in new_files[filter_name]:
                continue
                
            # Check each extension in old data
            for extension, file_path in tile_data.items():
                # Skip base extensions - these come from new discovery
                if extension in ['sci', 'err', 'wht', 'rms']:
                    continue
                
                # This is a derived file (psf, hom-*, etc.)
                derived_extensions.add(extension)
                
                # Check if the file still exists on disk
                # Handle both absolute paths (new format) and relative paths (old format)
                if Path(file_path).is_absolute():
                    # New format: absolute path
                    full_path = Path(file_path)
                else:
                    # Old format: relative path with base_path
                    if old_images.get('images', {}).get('base_path'):
                        base_path = Path(old_images['images']['base_path'])
                        full_path = base_path / file_path
                    else:
                        logger.debug(f"Skipped derived file without base_path: {file_path}")
                        continue
                
                if full_path.exists():
                    # Preserve this derived file (convert to absolute path)
                    if filter_name not in merged['images']['files']:
                        merged['images']['files'][filter_name] = {}
                    if tile not in merged['images']['files'][filter_name]:
                        merged['images']['files'][filter_name][tile] = {}
                    
                    merged['images']['files'][filter_name][tile][extension] = str(full_path)
                    preserved_count += 1
                    logger.debug(f"Preserved {filter_name}.{tile}.{extension}: {full_path}")
                else:
                    logger.debug(f"Skipped missing derived file: {full_path}")
    
    # No need to preserve base_path or psf_base_path - we now use absolute paths
    
    if derived_extensions:
        ext_list = sorted(derived_extensions)
        logger.info(f"Preserved {preserved_count} derived files: {', '.join(ext_list)}")
    
    return merged


@click.command("refresh")
@click.option(
    "--yes", "-y",
    is_flag=True,
    help="Skip confirmation prompts"
)
@click.option(
    "--backup/--no-backup",
    default=True,
    help="Create backup of existing images.toml (default: True)"
)
@click.option(
    "--config", 
    type=click.Path(exists=True, path_type=Path),
    help="Use different config file (default: config.toml)"
)
def refresh_cmd(yes, backup, config):
    """Refresh project configuration and regenerate images.toml."""
    logger = get_logger()
    
    try:
        # Use current directory as project directory
        project_dir = Path.cwd()
        
        # Validate we're in a project directory
        validate_project_directory(project_dir)
        logger.info(f"Refreshing project in: {project_dir}")
        
        # Determine config file to use
        if config is None:
            config = project_dir / "config.toml"
            if not config.exists():
                raise click.ClickException("config.toml not found in current directory")
        
        logger.info(f"Using config file: {config}")
        
        # Load and validate config
        logger.info("Loading and validating configuration...")
        updated_config = load_config(config)
        logger.info("Configuration validation successful")
        
        # Load existing images.toml to preserve derived files
        logger.info("Loading existing images.toml...")
        old_images = load_existing_images_toml(project_dir)
        
        # Run image discovery with updated config
        logger.info(f"[bold]Running image discovery...[/bold]")
        discovery = ImageDiscovery(updated_config)
        result = discovery.discover_images()
        
        # Present results to user
        discovery.present_discovery_results(result)
        
        # Get user confirmation unless --yes
        if not yes:
            logger.info(f"[yellow]This will regenerate images.toml while preserving derived files (PSF, homogenized)[/yellow]")
            if backup:
                logger.info(f"[yellow]Existing images.toml will be backed up[/yellow]")
            
            response = input("Proceed with refresh? [Y/n]: ").strip().lower()
            if response not in ('', 'y', 'yes'):
                logger.info("[yellow]Refresh cancelled[/yellow]")
                return
        
        # Create backup if requested
        if backup:
            backup_path = backup_images_toml(project_dir)
        
        # Generate new images data from discovery
        logger.info("Generating new images.toml data...")
        new_images_data = generate_images_toml(updated_config, result)
        
        # Merge with old data to preserve derived files
        logger.info("Merging with existing derived files...")
        merged_images_data = merge_preserve_derived_files(old_images, new_images_data)
        
        # Write updated images.toml
        images_path = project_dir / "images.toml"
        create_images_toml(merged_images_data, images_path)
        
        # Success message
        stats = result.get_summary_stats()
        logger.info(f"[bold green]Success:[/bold green] Refreshed images.toml")
        logger.info(f"Found: {stats['total_found']}/{stats['total_expected']} files")
        logger.info(f"Filter/tile combinations: {stats['filter_tile_combinations']}")
        
        if backup:
            logger.info(f"Previous version saved as: {backup_path.name}")
        
    except Exception as e:
        logger.error(f"[bold red]Error:[/bold red] Failed to refresh project: {e}")
        raise click.ClickException(str(e))