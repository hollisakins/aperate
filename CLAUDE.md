# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
**aperate** is a Python package for creating astronomical catalogs from JWST+HST images. The goal is to keep the codebase simple and robust, migrating functionality from a previous project while avoiding bloat.

## Architecture
The package is built around a Click CLI with a project-based workflow:

1. `aperate init config.toml` - Initialize project directory from TOML config
2. `aperate psf` - PSF generation by stacking stars
3. `aperate homogenize` - PSF homogenization to a target filter  
4. `aperate detect` - Source detection and detection image creation
5. `aperate photometry` - Aperture photometry measurements
6. `aperate postprocess` - PSF corrections and uncertainty calibration

### Project Initialization Workflow
1. User creates `config.toml` with data source templates and pipeline parameters
2. `aperate init config.toml` runs image discovery:
   - Resolves templates for all filter/tile/extension combinations
   - Checks which files exist on disk
   - Presents filter/tile matrix to user for confirmation
   - Creates project directory with validated file inventory
3. Generates `images.toml` with explicit file paths (single source of truth)
4. Creates empty catalog.fits with metadata

### Project Structure
- Each project is a directory containing config.toml, images.toml, and catalog.fits
- Config validation and image discovery occur during `aperate init`
- images.toml provides explicit file paths (no template resolution needed later)
- FITS catalog serves as pipeline state tracking
- Commands run from within project directory
- Separate modules for each command to avoid monolithic code

### Directory Layout
```
catalog_name/
├── config.toml      # User configuration (data templates + pipeline params)
├── images.toml      # Explicit file inventory (generated from discovery)
├── catalog.fits     # Main catalog + pipeline state
├── psf_models/      # Intermediate products
├── detection/
└── photometry/
```

### Image Discovery System
- Config.toml contains data source templates with `{filter}`, `{tile}`, `{ext}` placeholders
- Extension mapping standardizes names (e.g., 'drz' → 'sci', 'i2d' → 'sci')
- Discovery process validates all expected files exist before proceeding
- Missing files are clearly reported with filter/tile breakdown
- User can fix issues and re-run discovery as needed

## Technology Stack
- Python package with Click CLI framework
- Core dependencies: astropy, sep, photutils
- Separate module per command for clean organization
- TOML config files with validation
- FITS format for catalogs and metadata

## Image extensions
Standard extension names after mapping: `sci`, `err`, `wht`, `rms`, `exp`. `exp` is an optional
exposure map: PRIMARY + `EXPTIME` (float32, seconds) + `NEXP` (int16 exposure count), same grid as
`sci`, zero where `sci` is NaN. Read planes by HDU name (`EXP_NEXP_HDU` in `aperate/config/images.py`);
the first image HDU is EXPTIME. Photometry records the local mean NEXP per filter (`nexp_<filter>`),
and postprocess stratifies the empirical noise model by exposure count when the extension exists.

## Conventions
- `config.toml` is validated by the pydantic schema in `aperate/config/schema.py` (unknown keys are
  rejected). `aperate/example_config.toml` must stay valid against it.
- Pipeline state lives in FITS headers: per-filter corrections (`KRONCORR`, `APERCORR`) in the tile
  catalogs, step state (`PSFCORR`, `ERRCAL`, `EXTCORR`) in the merged catalog. Steps are idempotent and
  consult these records before applying.
- `merge_tile_catalogs` detaches each tile's header metadata before stacking and writes one summary
  (`TILES`, `NTILES`, union of corrections). Never `vstack` tables that carry list-valued `meta`
  sequentially: astropy concatenates the lists at every step and the header grows exponentially.
- Noise model (`aperate/core/noise_curves.py`): sigma = sigma1 * c_s(sqrt N) * S(x), with one curve per
  depth stratum (and per exposure-count group when `exp` exists), fitted on random apertures with a
  hold-out validation fit. Keep `tests/test_noise_curves*.py` passing when changing it.

## Development
- Install: `pip install -e .` (editable). After pulling, check `python -c "import aperate; print(aperate.__file__)"`
  points at the source tree.
- Tests: `pytest` (synthetic images only, no external data; runs in under a minute).
- Lint/format: `ruff` and `black`, settings in `pyproject.toml`.
- CLI: `aperate <subcommand>` from the project directory.
