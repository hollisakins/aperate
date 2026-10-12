"""Config validation (review finding F02): plausible mistakes must be rejected with a message
that names the offending key; a correct config must load."""
import copy

import pytest

from aperate.config.schema import AperateConfig


def valid():
    return {
        'name': 'proj', 'tiles': ['A1'], 'filters': ['f150w', 'f277w', 'f444w', 'f770w'], 'flux_unit': 'uJy',
        'data': {
            'base_path': '/data',
            'nircam': {'filters': ['f150w', 'f277w', 'f444w'], 'extensions': ['sci', 'err', 'wht'],
                       'filename': '{filter}/mosaic_{filter}_{tile}_{ext}.fits'},
            'miri': {'filters': ['f770w'], 'extensions': ['sci', 'err'], 'filename': '{filter}/m_{filter}_{tile}_{ext}.fits'},
        },
        'steps': {
            'psf_generation': {'psf_size': 151, 'filter_params': {'f444w': {'fwhm_min': 5.0, 'fwhm_max': 7.0}}},
            'psf_homogenization': {'target_filter': 'f444w'},
            'detection': {
                'build_detection_image': {'type': 'ivw', 'filters': ['f277w', 'f444w']},
                'source_detection': {'scheme': 'hot+cold', 'windowed_positions': False,
                                     'cold': {'thresh': 3.0, 'kernel_type': 'tophat', 'kernel_params': {'radius': 4.5}},
                                     'hot': {'thresh': 2.0, 'kernel_type': 'gaussian', 'kernel_params': {'fwhm': 3, 'size': 9}}},
            },
            'photometry': {'aperture': {'diameters': [0.1, 0.3, 1.0], 'aper_corr': True, 'aper_corr_filter': 'detection'},
                           'auto': {'kron_params': [1.1, 1.6], 'kron_corr_filter': 'detection'}},
            'postprocess': {'random_apertures': {'min_radius': 0.05, 'max_radius': 1.5},
                            'extinction_correction': {'run': True, 'dust_map': 'dust.fits',
                                                      'pivot_wavelengths': {'f150w': 15007.0, 'f277w': 27617.0, 'f444w': 44043.0, 'f770w': 76393.0}}},
        },
    }


def test_valid_config_loads():
    cfg = AperateConfig.from_dict(valid())
    assert cfg.photometry.aperture.diameters == [0.1, 0.3, 1.0]
    assert cfg.detection.source_detection.cold.minarea == 25      # cold defaults
    assert cfg.detection.source_detection.hot.minarea == 8        # hot defaults
    assert cfg.data_sources['nircam'].get_effective_base_path('/data') == '/data'
    assert cfg.base_path == '/data'


def _set(d, path, value):
    *keys, last = path.split('.')
    for k in keys:
        d = d[k]
    d[last] = value


def _rename(d, path, new_key):
    *keys, last = path.split('.')
    for k in keys:
        d = d[k]
    d[new_key] = d.pop(last)


MISTAKES = [
    # (description, mutation, text expected in the message)
    ('key typo', lambda d: _rename(d, 'steps.photometry.aperture.aper_corr', 'aper_cor'), 'aper_cor'),
    ('section typo', lambda d: _rename(d, 'steps.photometry', 'photometri'), 'photometri'),
    ('string for list', lambda d: _set(d, 'steps.photometry.aperture.diameters', '0.3'), 'diameters'),
    ('negative diameter', lambda d: _set(d, 'steps.photometry.aperture.diameters', [-0.1, 0.3]), 'diameters'),
    ('unsorted diameters', lambda d: _set(d, 'steps.photometry.aperture.diameters', [0.3, 0.1]), 'increasing'),
    ('detection filter unknown', lambda d: _set(d, 'steps.detection.build_detection_image.filters', ['f277w', 'f999w']), 'f999w'),
    ('target filter unknown', lambda d: _set(d, 'steps.psf_homogenization.target_filter', 'f444'), 'target_filter'),
    ('bad flux unit', lambda d: _set(d, 'flux_unit', 'mag'), 'flux_unit'),
    ('bad scheme', lambda d: _set(d, 'steps.detection.source_detection.scheme', 'hotcold'), 'scheme'),
    ('bad detection type', lambda d: _set(d, 'steps.detection.build_detection_image.type', 'chi2'), 'type'),
    ('reversed bounds', lambda d: _set(d, 'steps.photometry.aperture.aper_corr_bounds', [5.0, 1.0]), 'aper_corr_bounds'),
    ('radius order', lambda d: _set(d, 'steps.postprocess.random_apertures.min_radius', 2.0), 'min_radius'),
    ('missing pivot wavelength', lambda d: d['steps']['postprocess']['extinction_correction']['pivot_wavelengths'].pop('f770w'), 'f770w'),
    ('string for bool', lambda d: _set(d, 'performance', {'memmap': 'no'}), 'memmap'),
    ('filter without data source', lambda d: d['filters'].append('f090w'), 'f090w'),
    ('even psf size', lambda d: _set(d, 'steps.psf_generation.psf_size', 150), 'psf_size'),
    ('kron params wrong length', lambda d: _set(d, 'steps.photometry.auto.kron_params', [1.1]), 'kron_params'),
    ('aper_corr_filter unknown', lambda d: _set(d, 'steps.photometry.aperture.aper_corr_filter', 'detect'), 'aper_corr_filter'),
    ('hot+cold without hot', lambda d: d['steps']['detection']['source_detection'].pop('hot'), 'hot'),
    ('unknown steps section', lambda d: d['steps'].__setitem__('photometry_extra', {}), 'photometry_extra'),
    ('duplicate filter', lambda d: d['filters'].append('f444w'), 'duplicate'),
    ('kernel params mismatch', lambda d: _set(d, 'steps.detection.source_detection.hot.kernel_params', {'radius': 3}), 'kernel_params'),
    ('two extensions mapped to sci', lambda d: _set(d, 'data.nircam.extension_mapping', {'sci': 'sci', 'err': 'sci'}), 'extension_mapping'),
    ('extension_mapping for unknown ext', lambda d: _set(d, 'data.nircam.extension_mapping', {'drz': 'sci'}), 'extension_mapping'),
]


@pytest.mark.parametrize('desc,mutate,expect', MISTAKES, ids=[m[0] for m in MISTAKES])
def test_mistake_is_rejected_with_named_key(desc, mutate, expect):
    d = copy.deepcopy(valid())
    mutate(d)
    with pytest.raises(ValueError) as exc:
        AperateConfig.from_dict(d)
    assert expect in str(exc.value), str(exc.value)


def test_messages_use_toml_paths():
    d = copy.deepcopy(valid())
    _set(d, 'steps.photometry.aperture.diameters', '0.3')
    with pytest.raises(ValueError) as exc:
        AperateConfig.from_dict(d)
    assert 'steps.photometry.aperture.diameters' in str(exc.value)
