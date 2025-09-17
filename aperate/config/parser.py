"""TOML configuration file parsing and validation."""

import os
import sys
from pathlib import Path
from typing import Dict, Any
from astropy.io import fits

# Handle tomli import for different Python versions
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        raise ImportError("tomli package required for Python < 3.11")

from ..core.logging import get_logger, print_error
from .schema import AperateConfig


def load_config(config_path: Path) -> AperateConfig:
    """Load and validate configuration from TOML file."""
    logger = get_logger()
    
    if not config_path.exists():
        print_error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    logger.info(f"Loading config from {config_path}")
    
    try:
        with open(config_path, "rb") as f:
            config_data = tomllib.load(f)
    except Exception as e:
        print_error(f"Failed to parse TOML config: {e}")
        raise
    
    try:
        config = AperateConfig.from_dict(config_data)
        config.validate()
        logger.info("Config validation successful")
        
        # Apply performance settings (environment variable takes precedence)
        apply_performance_config(config)
        
        return config
    except Exception as e:
        print_error(f"Config validation failed: {e}")
        raise


def apply_performance_config(config: AperateConfig) -> None:
    """
    Apply performance configuration settings.
    Environment variable APERATE_MEMMAP takes precedence over config file.
    """
    logger = get_logger()
    
    # Check environment variable first (takes precedence)
    memmap_env = os.environ.get('APERATE_MEMMAP')
    if memmap_env is not None:
        # Environment variable overrides config
        if memmap_env == '0':
            fits.Conf.use_memmap = False
            logger.debug("Memory mapping disabled via APERATE_MEMMAP=0")
        else:
            fits.Conf.use_memmap = True
            logger.debug(f"Memory mapping enabled via APERATE_MEMMAP={memmap_env}")
    else:
        # Use config file setting
        fits.Conf.use_memmap = config.performance.memmap
        if not config.performance.memmap:
            logger.debug("Memory mapping disabled via config.toml")
        else:
            logger.debug("Memory mapping enabled via config.toml")


def validate_config_file(config_path: Path) -> bool:
    """Validate a config file without creating a config object."""
    try:
        load_config(config_path)
        return True
    except Exception:
        return False


def get_catalog_name_from_config(config_path: Path) -> str:
    """Extract catalog name from config file without full validation."""
    try:
        with open(config_path, "rb") as f:
            config_data = tomllib.load(f)
        
        if "name" not in config_data:
            raise ValueError("Config must contain a 'name' field")
        
        return config_data["name"]
    except Exception as e:
        print_error(f"Failed to extract catalog name: {e}")
        raise