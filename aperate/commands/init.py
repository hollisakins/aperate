"""Initialize aperate project from config file."""

import click
from pathlib import Path

from ..config.parser import load_config, get_catalog_name_from_config
from ..config.images import create_images_toml
from ..core.catalog import create_project_directory, AperateCatalog
from ..core.discovery import ImageDiscovery, generate_images_toml
from ..core.logging import get_logger


@click.command("init")
@click.argument("config_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--yes", "-y",
    is_flag=True,
    help="Skip image discovery confirmation prompt"
)
def init_cmd(config_file, yes):
    """Initialize a new aperate project from a config file."""
    logger = get_logger()
    
    try:
        # Extract catalog name first to validate config
        catalog_name = get_catalog_name_from_config(config_file)
        logger.info(f"Initializing project: {catalog_name}")
        
        # Validate full config
        config = load_config(config_file)
        logger.info("Config validation successful")
        
        # Run image discovery
        logger.info(f"[bold]Discovering images for project '{catalog_name}'...[/bold]")
        discovery = ImageDiscovery(config)
        result = discovery.discover_images()
        
        # Present results to user
        discovery.present_discovery_results(result)
        
        # Get user confirmation
        if not discovery.prompt_user_confirmation(result, auto_yes=yes):
            logger.info("[yellow]Project initialization cancelled[/yellow]")
            return
        
        # Create project directory structure
        project_dir = create_project_directory(catalog_name, config_file)
        logger.info(f"Created project directory: {project_dir}")
        logger.info(f"Copied config file: {project_dir}/config.toml")
        
        # Generate and save images.toml
        images_data = generate_images_toml(config, result)
        images_path = project_dir / "images.toml"
        create_images_toml(images_data, images_path)
        
        # # Create empty catalog
        # catalog_path = project_dir / "catalog.fits"
        # catalog = AperateCatalog(catalog_path)
        # logger.info(f"Created catalog file: {catalog_path}")
        
        # # TODO: Replace with actual catalog creation
        # # This is placeholder - user will provide actual implementation
        # try:
        #     # For now, we'll create a placeholder that uses the config
        #     # The actual implementation will use config data to set up the catalog structure
        #     catalog.create_empty_catalog({'name': config.name, 'filters': config.filters, 'tiles': config.tiles})
        # except NotImplementedError:
        #     # Create a simple placeholder file for now
        #     import warnings
        #     warnings.warn("Catalog creation not yet implemented - creating placeholder", UserWarning)
        #     catalog_path.touch()
        
        logger.info(f"[bold green]Success:[/bold green] Project '{catalog_name}' initialized successfully")
        
        
    except Exception as e:
        logger.error(f"[bold red]Error:[/bold red] Failed to initialize project: {e}")
        raise click.ClickException(str(e))