"""Main Click CLI entry point for aperate."""

import click
from pathlib import Path

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