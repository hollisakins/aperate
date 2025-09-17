"""Main Click CLI entry point for aperate."""

import os
import click
from pathlib import Path
from astropy.io import fits

from .core.logging import setup_logging, get_logger, print_error, print_success
from .commands import init, refresh, psf, homogenize, detect, photometry, postprocess


@click.group()
@click.option(
    "--verbose", "-v", 
    count=True, 
    help="Increase verbosity (use -v, -vv, or -vvv)"
)
@click.option(
    "--quiet", "-q", 
    is_flag=True, 
    help="Suppress output except for errors"
)
@click.option(
    "--log-file", 
    type=click.Path(), 
    help="Write logs to file"
)
@click.pass_context
def main(ctx, verbose, quiet, log_file):
    """aperate: Create catalogs from JWST+HST images."""
    
    # Configure memory mapping based on environment variable
    memmap_env = os.environ.get('APERATE_MEMMAP', '1')
    if memmap_env == '0':
        fits.Conf.use_memmap = False
        # Will log after logger is set up
        memmap_status = "disabled"
    else:
        fits.Conf.use_memmap = True
        memmap_status = "enabled"
    
    # Determine log level
    if quiet:
        level = "ERROR"
    elif verbose == 0:
        level = "INFO"
    elif verbose == 1:
        level = "DEBUG"
    else:
        level = "DEBUG"
    
    # Setup logging
    log_file_path = Path(log_file) if log_file else None
    setup_logging(level=level, log_file=log_file_path, quiet=quiet)
    
    # Log memmap status if verbose
    if verbose > 0:
        logger = get_logger()
        logger.debug(f"Memory mapping {memmap_status} (APERATE_MEMMAP={memmap_env})")
    
    # Store context for subcommands
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet


# Add subcommands
main.add_command(init.init_cmd)
main.add_command(refresh.refresh_cmd)
main.add_command(psf.psf_cmd)
main.add_command(homogenize.homogenize_cmd)
main.add_command(detect.detect_cmd)
main.add_command(photometry.photometry_cmd)
main.add_command(postprocess.postprocess_cmd)


if __name__ == "__main__":
    main()