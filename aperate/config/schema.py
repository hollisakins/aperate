"""Configuration schema definitions for aperate."""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FilterParams:
    """Filter-specific parameters."""
    fwhm_min: float
    fwhm_max: float


@dataclass
class DataSource:
    """Configuration for a data source (instrument)."""
    filters: List[str]
    extensions: List[str]
    extension_mapping: Dict[str, str]
    filename: str
    base_path: Optional[str] = None
    required_extensions: Optional[List[str]] = None
    
    def get_required_extensions(self) -> List[str]:
        """Get list of required extensions, defaulting to first extension."""
        if self.required_extensions:
            return self.required_extensions
        return [self.extensions[0]] if self.extensions else []
    
    def get_effective_base_path(self, global_base_path: str) -> str:
        """Get effective base path, using source-specific or global fallback."""
        return self.base_path if self.base_path is not None else global_base_path


@dataclass
class DetectionParams:
    """Source detection parameters."""
    thresh: float
    minarea: int
    deblend_nthresh: int
    deblend_cont: float
    kernel_type: str
    kernel_params: Dict[str, Any]
    filter_type: str
    clean: bool
    clean_param: float


@dataclass
class PSFGenerationConfig:
    """PSF generation step configuration."""
    master_psf: bool = True
    checkplots: bool = False
    min_snr: float = 75
    max_snr: float = 1000
    max_ellip: float = 0.1
    psf_size: int = 151
    az_average: bool = True
    filter_params: Dict[str, FilterParams] = field(default_factory=dict)


@dataclass
class PSFHomogenizationConfig:
    """PSF homogenization step configuration."""
    target_filter: str
    inverse_filters: List[str] = field(default_factory=list)
    reg_fact: float = 3e-3
    output_prefix: str = ''  # Optional prefix for homogenized files (e.g., 'mock' for simulations)


@dataclass
class BuildDetectionImageConfig:
    """Detection image building configuration."""
    type: str = 'ivw'  # 'ivw' or 'chisq'
    filters: List[str] = field(default_factory=list)
    homogenized: bool = True
    sigma_upper: float = 1.0  # Upper sigma threshold for robust pixel distribution fitting
    maxiters: int = 5  # Maximum iterations for robust pixel distribution fitting


@dataclass
class SourceDetectionConfig:
    """Source detection configuration."""
    scheme: str = 'hot+cold'  # 'single' or 'hot+cold'
    windowed_positions: bool = True
    save_segmap: bool = True
    plot: bool = False  # Enable detection plotting
    cold_mask_dilate_size: int = 5
    
    # Top-level detection parameters (used for 'single' scheme)
    kernel_type: str = 'gaussian'
    kernel_params: Dict[str, Any] = field(default_factory=dict)  
    thresh: float = 1.5
    minarea: int = 5
    deblend_nthresh: int = 32
    deblend_cont: float = 0.005
    filter_type: str = 'matched'
    clean: bool = True
    clean_param: float = 1.0
    
    # Nested parameters for hot+cold scheme
    cold: Optional[DetectionParams] = None
    hot: Optional[DetectionParams] = None


@dataclass
class DetectionConfig:
    """Combined detection step configuration."""
    build_detection_image: BuildDetectionImageConfig = field(default_factory=BuildDetectionImageConfig)
    source_detection: SourceDetectionConfig = field(default_factory=SourceDetectionConfig)


@dataclass
class AperturePhotometryConfig:
    """Aperture photometry configuration."""
    run_native: bool = True
    run_psf_homogenized: bool = True
    diameters: List[float] = field(default_factory=list)
    aper_corr: bool = False
    aper_corr_filter: str = 'detection'
    aper_corr_bounds: List[float] = field(default_factory=lambda: [1.0, 5.0])
    aper_corr_params: List[float] = field(default_factory=lambda: [2.5, 3.5])


@dataclass
class AutoPhotometryConfig:
    """Auto photometry configuration."""
    run: bool = True
    kron_params: List[float] = field(default_factory=lambda: [1.1, 1.6])
    kron_corr: bool = True
    kron_corr_filter: str = 'f277w'
    kron_corr_bounds: List[float] = field(default_factory=lambda: [1.0, 3.0])
    kron_corr_params: List[float] = field(default_factory=lambda: [2.5, 3.5])


@dataclass
class PhotometryConfig:
    """Photometry step configuration."""
    compute_rhalf: bool = True
    aperture: AperturePhotometryConfig = field(default_factory=AperturePhotometryConfig)
    auto: AutoPhotometryConfig = field(default_factory=AutoPhotometryConfig)


@dataclass
class MergeTilesConfig:
    """Tile merging configuration."""
    run: bool = True
    matching_radius: float = 0.1
    edge_mask: int = 300


@dataclass
class RandomAperturesConfig:
    """Random apertures configuration."""
    run: bool = True
    min_radius: float = 0.05
    max_radius: float = 1.50
    num_radii: int = 40
    num_apertures_per_sq_arcmin: int = 100


@dataclass
class PSFCorrectionsConfig:
    """PSF corrections configuration."""
    run: bool = True


@dataclass
class ExtinctionCorrectionConfig:
    """Galactic extinction correction configuration."""
    run: bool = False
    dust_map: str = ''  # Path to dust map FITS file
    pivot_wavelengths: Dict[str, float] = field(default_factory=dict)  # Filter -> wavelength in Angstroms
    max_correction_factor: float = 10.0  # Safety limit to prevent extreme corrections


@dataclass
class PerformanceConfig:
    """Performance and memory configuration."""
    memmap: bool = True  # Enable memory mapping for FITS files (reduces memory, increases I/O)


@dataclass
class PostprocessConfig:
    """Postprocessing step configuration."""
    merge_tiles: MergeTilesConfig = field(default_factory=MergeTilesConfig)
    random_apertures: RandomAperturesConfig = field(default_factory=RandomAperturesConfig)
    psf_corrections: PSFCorrectionsConfig = field(default_factory=PSFCorrectionsConfig)
    extinction_correction: ExtinctionCorrectionConfig = field(default_factory=ExtinctionCorrectionConfig)


@dataclass
class AperateConfig:
    """Main configuration class for aperate projects."""
    
    # Required fields
    name: str
    tiles: List[str]
    filters: List[str]
    
    # Optional fields with defaults
    flux_unit: str = 'uJy'
    plot: bool = True
    checkplots: bool = True
    
    # Data sources
    data_sources: Dict[str, DataSource] = field(default_factory=dict)
    base_path: str = ""
    
    # Performance settings
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # Pipeline steps
    psf_generation: PSFGenerationConfig = field(default_factory=PSFGenerationConfig)
    psf_homogenization: PSFHomogenizationConfig = field(default_factory=PSFHomogenizationConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    photometry: PhotometryConfig = field(default_factory=PhotometryConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AperateConfig":
        """Create config from dictionary (loaded from TOML)."""
        # Extract main fields
        name = data.get('name')
        if not name:
            raise ValueError("Config must contain a 'name' field")
        
        tiles = data.get('tiles', [])
        filters = data.get('filters', [])
        flux_unit = data.get('flux_unit', 'uJy')
        plot = data.get('plot', True)
        checkplots = data.get('checkplots', True)
        
        # Extract data configuration
        data_config = data.get('data', {})
        base_path = data_config.get('base_path', "")
        
        # Parse data sources
        data_sources = {}
        for source_name, source_data in data_config.items():
            if source_name == 'base_path':
                continue
            data_sources[source_name] = cls._parse_data_source(source_data)
        
        # Parse pipeline steps
        steps = data.get('steps', {})
        
        # Parse PSF generation
        psf_gen_data = steps.get('psf_generation', {})
        psf_generation = cls._parse_psf_generation(psf_gen_data)
        
        # Parse PSF homogenization
        psf_hom_data = steps.get('psf_homogenization', {})
        psf_homogenization = PSFHomogenizationConfig(
            target_filter=psf_hom_data.get('target_filter', 'f444w'),
            inverse_filters=psf_hom_data.get('inverse_filters', []),
            reg_fact=psf_hom_data.get('reg_fact', 3e-3),
            output_prefix=psf_hom_data.get('output_prefix', '')
        )
        
        # Parse detection
        detection_data = steps.get('detection', {})
        detection = cls._parse_detection(detection_data)
        
        # Parse photometry
        phot_data = steps.get('photometry', {})
        photometry = cls._parse_photometry(phot_data)
        
        # Parse postprocess
        post_data = steps.get('postprocess', {})
        postprocess = cls._parse_postprocess(post_data)
        
        # Parse performance config
        performance_data = data.get('performance', {})
        performance = PerformanceConfig(
            memmap=performance_data.get('memmap', True)
        )
        
        return cls(
            name=name,
            tiles=tiles,
            filters=filters,
            flux_unit=flux_unit,
            plot=plot,
            checkplots=checkplots,
            data_sources=data_sources,
            base_path=base_path,
            performance=performance,
            psf_generation=psf_generation,
            psf_homogenization=psf_homogenization,
            detection=detection,
            photometry=photometry,
            postprocess=postprocess
        )
    
    @staticmethod
    def _parse_data_source(data: Dict[str, Any]) -> DataSource:
        """Parse a data source configuration."""
        return DataSource(
            filters=data.get('filters', []),
            extensions=data.get('extensions', []),
            extension_mapping=data.get('extension_mapping', {}),
            filename=data.get('filename', ''),
            base_path=data.get('base_path'),
            required_extensions=data.get('required_extensions')
        )
    
    @staticmethod
    def _parse_psf_generation(data: Dict[str, Any]) -> PSFGenerationConfig:
        """Parse PSF generation configuration."""
        filter_params = {}
        filter_data = data.get('filter_params', {})
        for filter_name, params in filter_data.items():
            filter_params[filter_name] = FilterParams(
                fwhm_min=params.get('fwhm_min', 0),
                fwhm_max=params.get('fwhm_max', 10)
            )
        
        return PSFGenerationConfig(
            master_psf=data.get('master_psf', True),
            checkplots=data.get('checkplots', False),
            min_snr=data.get('min_snr', 75),
            max_snr=data.get('max_snr', 1000),
            max_ellip=data.get('max_ellip', 0.1),
            psf_size=data.get('psf_size', 151),
            az_average=data.get('az_average', True),
            filter_params=filter_params
        )
    
    @staticmethod
    def _parse_detection(data: Dict[str, Any]) -> DetectionConfig:
        """Parse detection configuration."""
        # Parse build_detection_image section
        build_img_data = data.get('build_detection_image', {})
        build_detection_image = BuildDetectionImageConfig(
            type=build_img_data.get('type', 'ivw'),
            filters=build_img_data.get('filters', []),
            homogenized=build_img_data.get('homogenized', True)
        )
        
        # Parse source_detection section
        source_det_data = data.get('source_detection', {})
        cold_data = source_det_data.get('cold', {})
        hot_data = source_det_data.get('hot', {})
        
        cold = DetectionParams(
            thresh=cold_data.get('thresh', 3.0),
            minarea=cold_data.get('minarea', 25),
            deblend_nthresh=cold_data.get('deblend_nthresh', 32),
            deblend_cont=cold_data.get('deblend_cont', 0.01),
            kernel_type=cold_data.get('kernel_type', 'tophat'),
            kernel_params=cold_data.get('kernel_params', {'radius':4.5}),
            filter_type=cold_data.get('filter_type', 'matched'),
            clean=cold_data.get('clean', True),
            clean_param=cold_data.get('clean_param', 2.0)
        ) if cold_data else None
        
        hot = DetectionParams(
            thresh=hot_data.get('thresh', 2.0),
            minarea=hot_data.get('minarea', 8),
            deblend_nthresh=hot_data.get('deblend_nthresh', 32),
            deblend_cont=hot_data.get('deblend_cont', 0.001),
            kernel_type=hot_data.get('kernel_type', 'gaussian'),
            kernel_params=hot_data.get('kernel_params', {'fwhm':3, 'size':9}),
            filter_type=hot_data.get('filter_type', 'matched'),
            clean=hot_data.get('clean', True),
            clean_param=hot_data.get('clean_param', 0.5)
        ) if hot_data else None
        
        source_detection = SourceDetectionConfig(
            scheme=source_det_data.get('scheme', 'hot+cold'),
            windowed_positions=source_det_data.get('windowed_positions', True),
            save_segmap=source_det_data.get('save_segmap', True),
            plot=source_det_data.get('plot', False),
            cold_mask_dilate_size=source_det_data.get('cold_mask_dilate_size', 5),
            cold=cold,
            hot=hot
        )
        
        return DetectionConfig(
            build_detection_image=build_detection_image,
            source_detection=source_detection
        )
    
    @staticmethod
    def _parse_photometry(data: Dict[str, Any]) -> PhotometryConfig:
        """Parse photometry configuration."""
        aper_data = data.get('aperture', {})
        auto_data = data.get('auto', {})
        
        aperture = AperturePhotometryConfig(
            run_native=aper_data.get('run_native', True),
            run_psf_homogenized=aper_data.get('run_psf_homogenized', True),
            diameters=aper_data.get('diameters', []),
            aper_corr=aper_data.get('aper_corr', False),
            aper_corr_filter=aper_data.get('aper_corr_filter', 'detection'),
            aper_corr_bounds=aper_data.get('aper_corr_bounds', [1.0, 5.0]),
            aper_corr_params=aper_data.get('aper_corr_params', [2.5, 3.5])
        )
        
        auto = AutoPhotometryConfig(
            run=auto_data.get('run', True),
            kron_params=auto_data.get('kron_params', [1.1, 1.6]),
            kron_corr=auto_data.get('kron_corr', True),
            kron_corr_filter=auto_data.get('kron_corr_filter', 'f277w'),
            kron_corr_bounds=auto_data.get('kron_corr_bounds', [1.0, 3.0]),
            kron_corr_params=auto_data.get('kron_corr_params', [2.5, 3.5])
        )
        
        return PhotometryConfig(
            compute_rhalf=data.get('compute_rhalf', True),
            aperture=aperture,
            auto=auto
        )
    
    @staticmethod
    def _parse_postprocess(data: Dict[str, Any]) -> PostprocessConfig:
        """Parse postprocess configuration."""
        merge_data = data.get('merge_tiles', {})
        random_data = data.get('random_apertures', {})
        psf_data = data.get('psf_corrections', {})
        extinction_data = data.get('extinction_correction', {})
        
        merge_tiles = MergeTilesConfig(
            run=merge_data.get('run', True),
            matching_radius=merge_data.get('matching_radius', 0.1),
            edge_mask=merge_data.get('edge_mask', 300)
        )
        
        random_apertures = RandomAperturesConfig(
            run=random_data.get('run', True),
            min_radius=random_data.get('min_radius', 0.05),
            max_radius=random_data.get('max_radius', 1.50),
            num_radii=random_data.get('num_radii', 40),
            num_apertures_per_sq_arcmin=random_data.get('num_apertures_per_sq_arcmin', 100)
        )
        
        psf_corrections = PSFCorrectionsConfig(
            run=psf_data.get('run', True)
        )
        
        extinction_correction = ExtinctionCorrectionConfig(
            run=extinction_data.get('run', False),
            dust_map=extinction_data.get('dust_map', ''),
            pivot_wavelengths=extinction_data.get('pivot_wavelengths', {}),
            max_correction_factor=extinction_data.get('max_correction_factor', 10.0)
        )
        
        return PostprocessConfig(
            merge_tiles=merge_tiles,
            random_apertures=random_apertures,
            psf_corrections=psf_corrections,
            extinction_correction=extinction_correction
        )
    
    def validate(self) -> None:
        """Validate configuration values."""
        if not self.name:
            raise ValueError("Config name cannot be empty")
        
        if not self.tiles:
            raise ValueError("At least one tile must be specified")
        
        if not self.filters:
            raise ValueError("At least one filter must be specified")
        
        if not self.data_sources:
            raise ValueError("At least one data source must be specified")
        
        # Validate that all filters are covered by data sources
        all_source_filters = set()
        for source in self.data_sources.values():
            all_source_filters.update(source.filters)
        
        missing_filters = set(self.filters) - all_source_filters
        if missing_filters:
            raise ValueError(f"Filters not covered by any data source: {missing_filters}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        # This could be implemented for saving modified configs
        # For now, it's not needed for the core functionality
        raise NotImplementedError("Config serialization not yet needed")