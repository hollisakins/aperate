"""Merging tile catalogs must not multiply their header records (repeated KRONCORR/APERCORR
cards, COMMENT/HISTORY): astropy's vstack concatenates list-valued metadata of every input, and
a sequential merge that keeps the running catalog among the inputs doubles them at each step."""
import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from aperate.commands import postprocess as pp

FILTERS = [f'f{i:03d}w' for i in range(19)]


def _project(tmp_path, tiles, rows=60, kron=None, bands=None):
    """Tile catalogs on a shared sky grid (tiles overlap by a few sources) and a detection image."""
    rng = np.random.default_rng(0)
    (tmp_path / 'catalogs').mkdir()
    det = tmp_path / 'det.fits'
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(np.zeros((4000, 4000), np.float32))]).writeto(det)
    for n, tile in enumerate(tiles):
        ra = 150.0 + 0.01 * n + rng.random(rows) * 0.01; dec = 2.0 + rng.random(rows) * 0.01
        # five sources shared with the next tile (rows 0-4 here, rows 5-9 there): exercises the matching
        ra[:5] = 150.0 + 0.01 * (n + 1) + 1e-4 * np.arange(5); dec[:5] = 2.005
        ra[5:10] = 150.0 + 0.01 * n + 1e-4 * np.arange(5); dec[5:10] = 2.005
        t = Table({'id': np.arange(rows), 'ra': ra, 'dec': dec,
                   'x': rng.uniform(500, 3500, rows), 'y': rng.uniform(500, 3500, rows)})
        for f in (bands or {}).get(tile, FILTERS):
            t[f'f_auto_{f}'] = rng.random(rows); t[f'f_aper_hom_{f}'] = rng.random(rows)
        t.meta['KRONCORR'] = list((kron or {}).get(tile, (bands or {}).get(tile, FILTERS)))
        t.meta['APERCORR'] = list((bands or {}).get(tile, FILTERS))
        t.meta['comments'] = ['made by the test', f'tile {tile}']
        t.meta['WINDOWED'] = False
        t.write(tmp_path / 'catalogs' / f'catalog_p_{tile}.fits')
    return det


class _Images:
    def __init__(self, det):
        self.det = det

    def get_tile_detection_image_path(self, tile):
        return self.det


def _merge(tmp_path, tiles, monkeypatch, det):
    monkeypatch.setattr(pp, 'load_images_config', lambda project_dir: _Images(det))
    out = tmp_path / f'merged_{len(tiles)}.fits'
    cat = pp.merge_tile_catalogs(tmp_path, 'p', list(tiles), 0.5, 100, out, True)
    return cat, fits.getheader(out, 1)


def test_merged_header_is_bounded_and_records_the_corrections(tmp_path, monkeypatch):
    tiles = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6']
    det = _project(tmp_path, tiles)
    tile_cards = len(fits.getheader(tmp_path / 'catalogs' / 'catalog_p_A1.fits', 1))
    cat3, h3 = _merge(tmp_path, tiles[:3], monkeypatch, det)
    cat6, h6 = _merge(tmp_path, tiles, monkeypatch, det)
    assert len(cat3) == 3 * 60 - 2 * 5 and len(cat6) == 6 * 60 - 5 * 5       # the shared sources merged
    # one card per record: the only growth with the number of tiles is the per-tile COMMENT
    assert len(h6) == len(h3) + 3 and len(h6) < tile_cards
    for key in ('KRONCORR', 'APERCORR'):
        assert list(h6.keys()).count(key) == 1
        assert h6[key] == ','.join(FILTERS)
    assert h6['NTILES'] == 6 and h6['TILES'] == ','.join(tiles)
    assert list(h6['COMMENT']) == ['made by the test'] + [f'tile {t}' for t in tiles]
    assert h6['WINDOWED'] is False
    # the merged table carries the same summary in memory (postprocess steps read catalog.meta)
    assert cat6.meta['KRONCORR'] == ','.join(FILTERS) and cat6.meta['NTILES'] == 6


def test_a_tile_with_uncorrected_columns_is_reported_and_the_union_recorded(tmp_path, monkeypatch, caplog):
    tiles = ['B1', 'B2', 'B3', 'B4']
    det = _project(tmp_path, tiles, kron={'B3': FILTERS[:17]})       # B3 has all columns, two filters uncorrected
    with caplog.at_level('WARNING'):
        cat, h = _merge(tmp_path, tiles, monkeypatch, det)
    assert h['KRONCORR'] == ','.join(FILTERS)
    assert any('KRONCORR' in r.message and 'B3' in r.message and FILTERS[17] in r.message and FILTERS[18] in r.message
               and 'B2' not in r.message for r in caplog.records)
    assert not any('APERCORR' in r.message for r in caplog.records)


def test_tiles_covering_different_filters_do_not_warn(tmp_path, monkeypatch, caplog):
    tiles = ['D1', 'D2', 'D3']
    det = _project(tmp_path, tiles, bands={'D2': FILTERS[:12]})       # D2 simply has fewer filters (like cosmos A10)
    with caplog.at_level('WARNING'):
        cat, h = _merge(tmp_path, tiles, monkeypatch, det)
    assert h['KRONCORR'] == ','.join(FILTERS) and h['APERCORR'] == ','.join(FILTERS)
    assert not any('KRONCORR' in r.message or 'APERCORR' in r.message for r in caplog.records)


def test_running_catalog_carries_no_metadata_during_the_merge(tmp_path, monkeypatch):
    """Bounded by construction: the pieces handed to vstack have empty meta at every step."""
    tiles = ['C1', 'C2', 'C3', 'C4', 'C5']
    det = _project(tmp_path, tiles)
    seen = []
    real = pp.vstack

    def spy(tables, **kw):
        seen.append([dict(t.meta) for t in tables])
        return real(tables, **kw)
    monkeypatch.setattr(pp, 'vstack', spy)
    _merge(tmp_path, tiles, monkeypatch, det)
    assert len(seen) == 4 and all(m == {} for step in seen for m in step)
