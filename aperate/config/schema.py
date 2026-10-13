"""Configuration schema for aperate projects (config.toml).

Every section is a pydantic model with ``extra='forbid'`` (an unknown key or a misspelled
section is an error instead of a silent fall-back to defaults) and ``strict=True`` (TOML types
must match the field: ``'0.3'`` is not a number, ``1`` is not ``true``). Field-level rules
(positive, ordered, allowed values) live next to the fields; rules that involve several
sections live in :meth:`AperateConfig.check_consistency`. Review finding F02: before this
schema 21 of 22 plausible mistakes in the fiducial config loaded without a message.

The public API used by the rest of the package is unchanged: ``AperateConfig.from_dict(data)``
builds a config from the parsed TOML and raises ``ValueError`` with one line per problem, and
attribute access (``config.photometry.aperture.diameters``) works as before.
"""

from typing import Any, Dict, List, Literal, Optional

import astropy.units as u
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from ..core.logging import get_logger

# Sections of [steps] and their attribute names on AperateConfig
STEP_SECTIONS = ('psf_generation', 'psf_homogenization', 'detection', 'photometry', 'postprocess')


class _Section(BaseModel):
    """Base class: unknown keys rejected, TOML types taken literally."""
    model_config = ConfigDict(extra='forbid', strict=True)


def _pair(name: str, value: List[float], ordered: bool = False, positive: bool = True) -> List[float]:
    if len(value) != 2:
        raise ValueError(f'{name} must have exactly two values [a, b]')
    if positive and not all(v > 0 for v in value):
        raise ValueError(f'{name} values must be > 0')
    if ordered and not value[0] < value[1]:
        raise ValueError(f'{name} must be ordered [min, max], got {value}')
    return value


# ---------------------------------------------------------------------------------------------
# [data]

class DataSource(_Section):
    """One instrument / data set: which filters it provides and how its files are named."""
    filters: List[str]
    extensions: List[str]
    filename: str
    extension_mapping: Dict[str, str] = Field(default_factory=dict)
    base_path: Optional[str] = None
    required_extensions: Optional[List[str]] = None

    @field_validator('filters', 'extensions')
    @classmethod
    def _non_empty(cls, v, info):
        if not v:
            raise ValueError(f'{info.field_name} must not be empty')
        return v

    @field_validator('filename')
    @classmethod
    def _template(cls, v):
        if '{ext}' not in v:
            raise ValueError("filename template must contain the '{ext}' placeholder")
        return v

    @model_validator(mode='after')
    def _required_subset(self):
        if self.required_extensions:
            unknown = set(self.required_extensions) - set(self.extensions)
            if unknown:
                raise ValueError(f'required_extensions {sorted(unknown)} are not in extensions {self.extensions}')
        unknown = set(self.extension_mapping) - set(self.extensions)
        if unknown:
            raise ValueError(f'extension_mapping keys {sorted(unknown)} are not in extensions {self.extensions}')
        standard = [self.extension_mapping.get(e, e) for e in self.extensions]
        dup = sorted({s for s in standard if standard.count(s) > 1})
        if dup:
            raise ValueError(f'extension_mapping maps two extensions to the same name: {dup}')
        return self

    def get_required_extensions(self) -> List[str]:
        """Get list of required extensions, defaulting to first extension."""
        if self.required_extensions:
            return self.required_extensions
        return [self.extensions[0]] if self.extensions else []

    def get_effective_base_path(self, global_base_path: str) -> str:
        """Get effective base path, using source-specific or global fallback."""
        return self.base_path if self.base_path is not None else global_base_path


# ---------------------------------------------------------------------------------------------
# [steps.psf_generation]

class FilterParams(_Section):
    """Star-selection FWHM window (pixels) for one filter."""
    fwhm_min: float = 0.0
    fwhm_max: float = 10.0

    @model_validator(mode='after')
    def _ordered(self):
        if not self.fwhm_min < self.fwhm_max:
            raise ValueError(f'fwhm_min ({self.fwhm_min}) must be < fwhm_max ({self.fwhm_max})')
        return self


class PSFGenerationConfig(_Section):
    """PSF generation step configuration."""
    master_psf: bool = True
    checkplots: bool = False
    min_snr: float = 75
    max_snr: float = 1000
    max_ellip: float = 0.1
    psf_size: int = 151
    az_average: bool = True
    filter_params: Dict[str, FilterParams] = Field(default_factory=dict)

    @model_validator(mode='after')
    def _ranges(self):
        if not 0 < self.min_snr < self.max_snr:
            raise ValueError(f'need 0 < min_snr < max_snr, got {self.min_snr}, {self.max_snr}')
        if not 0 <= self.max_ellip < 1:
            raise ValueError(f'max_ellip must be in [0, 1), got {self.max_ellip}')
        if self.psf_size < 3 or self.psf_size % 2 == 0:
            raise ValueError(f'psf_size must be an odd integer >= 3, got {self.psf_size}')
        return self


# ---------------------------------------------------------------------------------------------
# [steps.psf_homogenization]

class PSFHomogenizationConfig(_Section):
    """PSF homogenization step configuration."""
    target_filter: str = 'f444w'
    # Accepted for backward compatibility; inverse filters are defined in psfs.toml.
    inverse_filters: List[str] = Field(default_factory=list)
    reg_fact: float = 3e-3
    output_prefix: str = ''  # Optional prefix for homogenized files (e.g., 'mock' for simulations)

    @field_validator('reg_fact')
    @classmethod
    def _positive(cls, v):
        if v <= 0:
            raise ValueError('reg_fact must be > 0')
        return v


# ---------------------------------------------------------------------------------------------
# [steps.detection]

class BuildDetectionImageConfig(_Section):
    """Detection image building configuration."""
    type: Literal['ivw', 'chisq'] = 'ivw'
    filters: List[str] = Field(default_factory=list)
    homogenized: bool = True
    sigma_upper: float = 1.0  # Upper sigma threshold for robust pixel distribution fitting
    maxiters: int = 5  # Maximum iterations for robust pixel distribution fitting

    @model_validator(mode='after')
    def _ranges(self):
        if self.sigma_upper <= 0:
            raise ValueError('sigma_upper must be > 0')
        if self.maxiters < 1:
            raise ValueError('maxiters must be >= 1')
        return self


class DetectionParams(_Section):
    """sep.extract parameters for one detection pass (defaults are the 'hot' values)."""
    thresh: float = 2.0
    minarea: int = 8
    deblend_nthresh: int = 32
    deblend_cont: float = 0.001
    kernel_type: Literal['gaussian', 'tophat'] = 'gaussian'
    kernel_params: Dict[str, Any] = Field(default_factory=lambda: {'fwhm': 3, 'size': 9})
    filter_type: Literal['matched', 'conv'] = 'matched'
    clean: bool = True
    clean_param: float = 0.5

    @model_validator(mode='after')
    def _ranges(self):
        if self.thresh <= 0:
            raise ValueError('thresh must be > 0')
        if self.minarea < 1:
            raise ValueError('minarea must be >= 1')
        if self.deblend_nthresh < 1:
            raise ValueError('deblend_nthresh must be >= 1')
        if not 0 <= self.deblend_cont <= 1:
            raise ValueError('deblend_cont must be in [0, 1]')
        if self.clean_param <= 0:
            raise ValueError('clean_param must be > 0')
        expected = {'gaussian': {'fwhm', 'size'}, 'tophat': {'radius'}}[self.kernel_type]
        if not expected <= set(self.kernel_params):
            raise ValueError(f"kernel_params for kernel_type='{self.kernel_type}' needs {sorted(expected)}, got {sorted(self.kernel_params)}")
        return self


class ColdDetectionParams(DetectionParams):
    """Same parameters with the 'cold' (bright, extended) defaults."""
    thresh: float = 3.0
    minarea: int = 25
    deblend_cont: float = 0.01
    kernel_type: Literal['gaussian', 'tophat'] = 'tophat'
    kernel_params: Dict[str, Any] = Field(default_factory=lambda: {'radius': 4.5})
    clean_param: float = 2.0


class SourceDetectionConfig(_Section):
    """Source detection configuration."""
    scheme: Literal['single', 'hot+cold'] = 'hot+cold'
    windowed_positions: bool = True
    save_segmap: bool = True
    plot: bool = False  # Enable detection plotting
    cold_mask_dilate_size: int = 5

    # Top-level detection parameters (used for 'single' scheme)
    kernel_type: Literal['gaussian', 'tophat'] = 'gaussian'
    kernel_params: Dict[str, Any] = Field(default_factory=dict)
    thresh: float = 1.5
    minarea: int = 5
    deblend_nthresh: int = 32
    deblend_cont: float = 0.005
    filter_type: Literal['matched', 'conv'] = 'matched'
    clean: bool = True
    clean_param: float = 1.0

    # Nested parameters for hot+cold scheme
    cold: Optional[ColdDetectionParams] = None
    hot: Optional[DetectionParams] = None

    @model_validator(mode='after')
    def _scheme_sections(self):
        if self.scheme == 'hot+cold' and (self.cold is None or self.hot is None):
            raise ValueError("scheme 'hot+cold' needs both [steps.detection.source_detection.cold] and [...hot] sections")
        if self.cold_mask_dilate_size < 0:
            raise ValueError('cold_mask_dilate_size must be >= 0')
        return self


class DetectionConfig(_Section):
    """Combined detection step configuration."""
    build_detection_image: BuildDetectionImageConfig = Field(default_factory=BuildDetectionImageConfig)
    source_detection: SourceDetectionConfig = Field(default_factory=SourceDetectionConfig)


# ---------------------------------------------------------------------------------------------
# [steps.photometry]

class AperturePhotometryConfig(_Section):
    """Aperture photometry configuration."""
    run_native: bool = True
    run_psf_homogenized: bool = True
    diameters: List[float] = Field(default_factory=list)  # arcsec
    aper_corr: bool = False
    aper_corr_filter: str = 'detection'
    # Values outside aper_corr_bounds are flagged (flag_aper_corr bit 2), not clipped.
    aper_corr_bounds: List[float] = Field(default_factory=lambda: [1.0, 5.0])
    aper_corr_params: List[float] = Field(default_factory=lambda: [2.5, 3.5])
    # Minimum S/N of the fixed-aperture flux on the aper_corr image for a correction to be
    # applied; below it the corrected flux is NaN (the ratio of two noisy fluxes is not a
    # correction).
    aper_corr_min_snr: float = 3.0

    @model_validator(mode='after')
    def _ranges(self):
        if any(d <= 0 for d in self.diameters):
            raise ValueError(f'diameters must be > 0 arcsec, got {self.diameters}')
        if any(b <= a for a, b in zip(self.diameters[:-1], self.diameters[1:])):
            raise ValueError(f'diameters must be strictly increasing, got {self.diameters}')
        _pair('aper_corr_bounds', self.aper_corr_bounds, ordered=True)
        _pair('aper_corr_params', self.aper_corr_params)
        if self.aper_corr_min_snr < 0:
            raise ValueError('aper_corr_min_snr must be >= 0')
        return self


class AutoPhotometryConfig(_Section):
    """Auto (Kron) photometry configuration."""
    run: bool = True
    kron_params: List[float] = Field(default_factory=lambda: [1.1, 1.6])   # [k, min circle radius in px]
    kron_corr: bool = True
    kron_corr_filter: str = 'f277w'
    kron_corr_bounds: List[float] = Field(default_factory=lambda: [1.0, 3.0])
    kron_corr_params: List[float] = Field(default_factory=lambda: [2.5, 3.5])

    @model_validator(mode='after')
    def _ranges(self):
        _pair('kron_params', self.kron_params)
        _pair('kron_corr_bounds', self.kron_corr_bounds, ordered=True)
        _pair('kron_corr_params', self.kron_corr_params)
        return self


class PhotometryConfig(_Section):
    """Photometry step configuration."""
    compute_rhalf: bool = True
    aperture: AperturePhotometryConfig = Field(default_factory=AperturePhotometryConfig)
    auto: AutoPhotometryConfig = Field(default_factory=AutoPhotometryConfig)


# ---------------------------------------------------------------------------------------------
# [steps.postprocess]

class MergeTilesConfig(_Section):
    """Tile merging configuration."""
    run: bool = True
    matching_radius: float = 0.1   # arcsec
    edge_mask: int = 300           # pixels

    @model_validator(mode='after')
    def _ranges(self):
        if self.matching_radius <= 0:
            raise ValueError('matching_radius must be > 0')
        if self.edge_mask < 0:
            raise ValueError('edge_mask must be >= 0')
        return self


class RandomAperturesConfig(_Section):
    """Random apertures (error calibration) configuration.

    The error model is sigma = sigma1 * c_s(sqrt N) * S(x): one empirical curve c_s per depth
    stratum s (depth_strata quantile bins of the depth proxy, plus tail bins), times a per-pixel
    scale S. depth_strata = 1 is the single-curve model of earlier versions.
    """
    run: bool = True
    min_radius: float = 0.05   # arcsec
    max_radius: float = 1.50   # arcsec
    num_radii: int = 40
    num_apertures_per_sq_arcmin: int = 100
    depth_strata: int = 8
    # depth proxy: 'wht' (weight map, used only to order pixels by depth), 'rms', or 'empirical'
    # (noise map measured from the source-masked image); 'auto' = wht if present, else rms, else empirical
    depth_proxy: Literal['auto', 'wht', 'rms', 'empirical'] = 'auto'
    # per-pixel scale: 'rms' map, 'wht' (1/sqrt(wht)) or 'empirical'; 'auto' = rms, else wht, else empirical
    scale_map: Literal['auto', 'rms', 'wht', 'empirical'] = 'auto'
    mask_sources: bool = True          # apertures avoid the segmentation map
    mask_dilate: int = 3               # pixels added around sources and invalid pixels
    holdout_fraction: float = 0.2      # apertures kept back for the validation fit only; the production fit uses all
    check_weight_map: bool = True      # compare the weight map with the empirical noise before trusting it
    parallel: int = 1                  # tiles sampled in parallel (processes)
    # with an 'nexp' image extension (number of exposures per pixel): up to nexp_groups groups of
    # exposure count, each with nexp_strata strata of depth per exposure; nexp_groups = 1 ignores the map
    nexp_groups: int = 5
    nexp_strata: int = 4

    @model_validator(mode='after')
    def _ranges(self):
        if self.nexp_groups < 1 or self.nexp_strata < 1:
            raise ValueError('nexp_groups and nexp_strata must be >= 1')
        if not 0 < self.min_radius < self.max_radius:
            raise ValueError(f'need 0 < min_radius < max_radius, got {self.min_radius}, {self.max_radius}')
        if self.num_radii < 2:
            raise ValueError('num_radii must be >= 2')
        if self.num_apertures_per_sq_arcmin < 1:
            raise ValueError('num_apertures_per_sq_arcmin must be >= 1')
        if self.depth_strata < 1:
            raise ValueError('depth_strata must be >= 1')
        if self.mask_dilate < 0:
            raise ValueError('mask_dilate must be >= 0')
        if not 0 <= self.holdout_fraction < 0.5:
            raise ValueError('holdout_fraction must be in [0, 0.5)')
        if self.parallel < 1:
            raise ValueError('parallel must be >= 1')
        return self


class PSFCorrectionsConfig(_Section):
    """PSF corrections configuration."""
    run: bool = True


class ExtinctionCorrectionConfig(_Section):
    """Galactic extinction correction configuration."""
    run: bool = False
    dust_map: str = ''  # Path to dust map FITS file (relative to the project directory)
    pivot_wavelengths: Dict[str, float] = Field(default_factory=dict)  # Filter -> wavelength in Angstroms
    max_correction_factor: float = 10.0  # Safety limit to prevent extreme corrections

    @model_validator(mode='after')
    def _ranges(self):
        if self.run and not self.dust_map:
            raise ValueError('dust_map is required when extinction_correction.run = true')
        if self.max_correction_factor < 1:
            raise ValueError('max_correction_factor must be >= 1')
        if any(w <= 0 for w in self.pivot_wavelengths.values()):
            raise ValueError('pivot_wavelengths must be > 0 (Angstrom)')
        return self


class PostprocessConfig(_Section):
    """Postprocessing step configuration."""
    merge_tiles: MergeTilesConfig = Field(default_factory=MergeTilesConfig)
    random_apertures: RandomAperturesConfig = Field(default_factory=RandomAperturesConfig)
    psf_corrections: PSFCorrectionsConfig = Field(default_factory=PSFCorrectionsConfig)
    extinction_correction: ExtinctionCorrectionConfig = Field(default_factory=ExtinctionCorrectionConfig)


class PerformanceConfig(_Section):
    """Performance and memory configuration."""
    memmap: bool = True  # Enable memory mapping for FITS files (reduces memory, increases I/O)


# ---------------------------------------------------------------------------------------------
# Top level

class AperateConfig(_Section):
    """Main configuration class for aperate projects."""

    # Required fields
    name: str
    tiles: List[str]
    filters: List[str]

    # Optional fields with defaults
    flux_unit: str = 'uJy'
    plot: bool = True          # accepted for backward compatibility; not used by any step
    checkplots: bool = True    # accepted for backward compatibility; not used by any step

    # Data sources ([data] in the TOML: base_path plus one table per source)
    data_sources: Dict[str, DataSource] = Field(default_factory=dict)
    base_path: str = ''

    # Performance settings
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)

    # Pipeline steps ([steps.<name>] in the TOML)
    psf_generation: PSFGenerationConfig = Field(default_factory=PSFGenerationConfig)
    psf_homogenization: PSFHomogenizationConfig = Field(default_factory=PSFHomogenizationConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    photometry: PhotometryConfig = Field(default_factory=PhotometryConfig)
    postprocess: PostprocessConfig = Field(default_factory=PostprocessConfig)

    @field_validator('name')
    @classmethod
    def _name(cls, v):
        if not v.strip():
            raise ValueError('name must not be empty')
        return v

    @field_validator('tiles', 'filters')
    @classmethod
    def _unique_non_empty(cls, v, info):
        if not v:
            raise ValueError(f'at least one entry is required in {info.field_name}')
        dup = sorted({x for x in v if v.count(x) > 1})
        if dup:
            raise ValueError(f'duplicate entries in {info.field_name}: {dup}')
        return v

    @field_validator('flux_unit')
    @classmethod
    def _flux_unit(cls, v):
        try:
            unit = u.Unit(v)
        except Exception as e:
            raise ValueError(f'flux_unit {v!r} is not a unit astropy understands: {e}')
        if not unit.is_equivalent(u.Jy):
            raise ValueError(f'flux_unit {v!r} is not a spectral flux density (e.g. uJy, nJy, erg/s/cm2/Hz)')
        return v

    @field_validator('data_sources')
    @classmethod
    def _sources(cls, v):
        if not v:
            raise ValueError('at least one [data.<source>] table is required')
        return v

    # ---- construction from the parsed TOML -------------------------------------------------

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AperateConfig":
        """Create config from the dictionary loaded from config.toml.

        Raises ValueError with one line per problem, naming the key in TOML terms.
        """
        if not isinstance(data, dict):
            raise ValueError('config must be a table')
        d = dict(data)
        problems = []

        data_cfg = d.pop('data', {})
        if not isinstance(data_cfg, dict):
            problems.append('data: must be a table ([data] with base_path and one [data.<source>] per instrument)')
            data_cfg = {}
        data_cfg = dict(data_cfg)
        d['base_path'] = data_cfg.pop('base_path', '')
        d['data_sources'] = data_cfg

        steps = d.pop('steps', {})
        if not isinstance(steps, dict):
            problems.append('steps: must be a table')
            steps = {}
        steps = dict(steps)
        for key in STEP_SECTIONS:
            if key in steps:
                d[key] = steps.pop(key)
        for key in steps:
            problems.append(f"steps.{key}: unknown section (expected one of {', '.join(STEP_SECTIONS)})")
        for key in ('base_path', 'data_sources', *STEP_SECTIONS):
            if key in data:
                problems.append(f'{key}: must be given under [data] / [steps], not at the top level')

        try:
            config = cls.model_validate(d)
        except ValidationError as e:
            problems.extend(_format_errors(e))
            config = None
        if problems:
            raise ValueError('invalid config.toml:\n  - ' + '\n  - '.join(problems))
        config.check_consistency()
        return config

    def validate(self) -> None:
        """Cross-section checks (kept for backward compatibility; from_dict already ran them)."""
        self.check_consistency()

    def check_consistency(self) -> None:
        """Rules that involve more than one section. Raises ValueError listing every violation."""
        logger = get_logger()
        problems = []
        filters = set(self.filters)

        source_filters = set()
        for source in self.data_sources.values():
            source_filters.update(source.filters)
        missing = sorted(filters - source_filters)
        if missing:
            problems.append(f'filters not provided by any [data.<source>]: {missing}')
        unused = sorted(source_filters - filters)
        if unused:
            logger.warning(f'data sources provide filters that are not in `filters` and will be ignored: {unused}')

        unknown = sorted(set(self.detection.build_detection_image.filters) - filters)
        if unknown:
            problems.append(f'steps.detection.build_detection_image.filters not in `filters`: {unknown}')

        if self.psf_homogenization.target_filter not in filters:
            problems.append(f"steps.psf_homogenization.target_filter '{self.psf_homogenization.target_filter}' is not in `filters`")
        unknown = sorted(set(self.psf_homogenization.inverse_filters) - filters)
        if unknown:
            problems.append(f'steps.psf_homogenization.inverse_filters not in `filters`: {unknown}')

        for key, value in (('aperture.aper_corr_filter', self.photometry.aperture.aper_corr_filter),
                           ('auto.kron_corr_filter', self.photometry.auto.kron_corr_filter)):
            if value != 'detection' and value not in filters:
                problems.append(f"steps.photometry.{key} '{value}' must be 'detection' or one of `filters`")
        if self.photometry.aperture.run_psf_homogenized and self.photometry.aperture.aper_corr and not self.photometry.aperture.diameters:
            problems.append('steps.photometry.aperture.aper_corr = true but no diameters are given')
        if self.photometry.aperture.aper_corr and not self.photometry.aperture.run_psf_homogenized:
            logger.warning('steps.photometry.aperture.aper_corr = true has no effect: it applies to PSF-homogenized '
                           'aperture photometry and run_psf_homogenized = false')
        no_params = sorted(filters - set(self.psf_generation.filter_params))
        if no_params:
            logger.warning(f'steps.psf_generation.filter_params missing for {no_params}: the PSF stage would select '
                           f'stars with the default FWHM window (0-10 px)')

        ext = self.postprocess.extinction_correction
        if ext.run:
            missing = sorted(filters - set(ext.pivot_wavelengths))
            if missing:
                problems.append(f'steps.postprocess.extinction_correction.pivot_wavelengths missing for: {missing}')

        # Entries for filters that are not used are harmless (one table can serve several projects)
        unknown = sorted(set(self.psf_generation.filter_params) - filters)
        if unknown:
            logger.debug(f'steps.psf_generation.filter_params given for filters not in `filters` (ignored): {unknown}')
        unknown = sorted(set(ext.pivot_wavelengths) - filters)
        if unknown:
            logger.debug(f'steps.postprocess.extinction_correction.pivot_wavelengths given for filters not in `filters` (ignored): {unknown}')

        if problems:
            raise ValueError('invalid config.toml:\n  - ' + '\n  - '.join(problems))

    def to_dict(self) -> Dict[str, Any]:
        """Config as a plain dictionary (attribute layout, not the TOML layout)."""
        return self.model_dump()


def _format_errors(e: ValidationError) -> List[str]:
    """Render pydantic errors as 'toml.key.path: message (got ...)' lines."""
    lines = []
    for err in e.errors():
        parts = [str(p) for p in err['loc']]
        if parts and parts[0] == 'data_sources':
            parts[0] = 'data'
        elif parts and parts[0] in STEP_SECTIONS:
            parts.insert(0, 'steps')
        path = '.'.join(parts) or '(top level)'
        msg = err['msg']
        if msg.startswith('Value error, '):
            msg = msg[len('Value error, '):]
        got = err.get('input')
        if err['type'] not in ('missing', 'extra_forbidden', 'value_error') and got is not None:
            got_repr = repr(got)
            if len(got_repr) > 60:
                got_repr = got_repr[:57] + '...'
            msg += f' (got {got_repr})'
        lines.append(f'{path}: {msg}')
    return lines
