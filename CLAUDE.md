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

## Development Notes
**IMPORTANT**: Much of the core functionality is being migrated from another project. Claude Code will typically handle package structure, CLI setup, and integration - the user will provide the core astronomical processing algorithms.

## Development Setup
Project setup pending. Will need:
- pyproject.toml for package configuration
- requirements.txt for dependencies
- Testing framework (pytest)
- Code formatting (black, ruff)

## Common Commands
To be established once package structure is created:
- Package installation: `pip install -e .`
- Running tests: `pytest`
- Code formatting and linting
- CLI usage: `aperate <subcommand>` (from project directory)