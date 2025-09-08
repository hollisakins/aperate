"""Image discovery and template resolution for aperate."""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict

# Handle tomli import for different Python versions
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        raise ImportError("tomli package required for Python < 3.11")

from rich.table import Table
from rich.console import Console

from ..config.schema import AperateConfig, DataSource
from .logging import get_logger, print_error, print_success


@dataclass
class FileInfo:
    """Information about a discovered file."""
    filter: str
    tile: str
    extension: str  # standardized name (sci, err, wht)
    filepath: str
    exists: bool


@dataclass
class DiscoveryResult:
    """Results of image discovery process."""
    files: List[FileInfo] = field(default_factory=list)
    missing_files: List[FileInfo] = field(default_factory=list)
    
    def get_filter_tile_matrix(self) -> Dict[str, Dict[str, Set[str]]]:
        """Get matrix of available extensions by filter and tile."""
        matrix = defaultdict(lambda: defaultdict(set))
        
        for file_info in self.files:
            if file_info.exists:
                matrix[file_info.filter][file_info.tile].add(file_info.extension)
        
        return dict(matrix)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        total_expected = len(self.files) + len(self.missing_files)
        total_found = len([f for f in self.files if f.exists])
        
        # Count complete sets (filter/tile combinations with all required extensions)
        filter_tile_extensions = defaultdict(set)
        for file_info in self.files:
            if file_info.exists:
                filter_tile_extensions[(file_info.filter, file_info.tile)].add(file_info.extension)
        
        return {
            'total_expected': total_expected,
            'total_found': total_found,
            'total_missing': total_expected - total_found,
            'filter_tile_combinations': len(filter_tile_extensions)
        }
    
    def analyze_missing_by_completeness(self) -> Dict[str, Dict[str, Any]]:
        """Analyze missing files by completeness - completely missing vs partial."""
        # Get matrix of found files
        found_matrix = self.get_filter_tile_matrix()
        
        # Group missing files by filter/tile
        missing_by_filter_tile = defaultdict(lambda: defaultdict(set))
        for file_info in self.missing_files:
            missing_by_filter_tile[file_info.filter][file_info.tile].add(file_info.extension)
        
        result = {}
        
        for filter_name in missing_by_filter_tile:
            completely_missing = []
            partially_complete = {}
            
            for tile, missing_extensions in missing_by_filter_tile[filter_name].items():
                # Check if this tile has any found files
                found_extensions = found_matrix.get(filter_name, {}).get(tile, set())
                
                if not found_extensions:
                    # No files found for this tile - completely missing
                    completely_missing.append(tile)
                else:
                    # Some files found - partially complete
                    partially_complete[tile] = sorted(missing_extensions)
            
            if completely_missing or partially_complete:
                result[filter_name] = {
                    'completely_missing': sorted(completely_missing),
                    'partially_complete': partially_complete
                }
        
        return result


class ImageDiscovery:
    """Handle image discovery and validation."""
    
    def __init__(self, config: AperateConfig):
        self.config = config
        self.logger = get_logger()
    
    def discover_images(self) -> DiscoveryResult:
        """Discover all images based on config templates."""
        self.logger.info("Starting image discovery...")
        
        result = DiscoveryResult()
        
        for source_name, source in self.config.data_sources.items():
            self.logger.debug(f"Processing data source: {source_name}")
            
            # Only process filters that are in the main config filter list
            source_filters = [f for f in source.filters if f in self.config.filters]
            
            for filter_name in source_filters:
                for tile in self.config.tiles:
                    for extension in source.extensions:
                        # Resolve template
                        filepath = self._resolve_template(source, filter_name, tile, extension)
                        effective_base_path = source.get_effective_base_path(self.config.base_path)
                        full_path = Path(effective_base_path) / filepath
                        
                        # Map to standardized extension name
                        std_extension = source.extension_mapping.get(extension, extension)
                        
                        # Check if file exists
                        exists = full_path.exists()
                        
                        file_info = FileInfo(
                            filter=filter_name,
                            tile=tile,
                            extension=std_extension,
                            filepath=str(full_path),
                            exists=exists
                        )
                        
                        if exists:
                            result.files.append(file_info)
                        else:
                            result.missing_files.append(file_info)
                            self.logger.debug(f"Missing: {full_path}")
        
        self.logger.info(f"Discovery complete: {len(result.files)} files found, {len(result.missing_files)} missing")
        return result
    
    def _resolve_template(self, source: DataSource, filter_name: str, tile: str, extension: str) -> str:
        """Resolve a filename template with actual values."""
        template = source.filename
        
        # Replace template variables
        resolved = template.format(
            filter=filter_name,
            tile=tile,
            ext=extension
        )
        
        return resolved
    
    def present_discovery_results(self, result: DiscoveryResult) -> None:
        """Present discovery results in a formatted table."""
        matrix = result.get_filter_tile_matrix()
        stats = result.get_summary_stats()
        
        # Create summary
        self.logger.info(f"[bold]Image Discovery Results[/bold]")
        self.logger.info(f"Found: {stats['total_found']}/{stats['total_expected']} files")
        self.logger.info(f"Filter/Tile combinations: {stats['filter_tile_combinations']}")
        
        if stats['total_missing'] > 0:
            self.logger.info(f"[yellow]Missing: {stats['total_missing']} files[/yellow]")
        
        # Create detailed table
        table = Table(title="Filter/Tile Coverage")
        table.add_column("Filter", style="cyan")
        
        # Add tile columns
        for tile in sorted(self.config.tiles):
            table.add_column(tile, justify="center")
        
        # Add rows for each filter
        use_compact_format = len(self.config.tiles) > 5
        
        for filter_name in sorted(self.config.filters):
            row = [filter_name]
            for tile in sorted(self.config.tiles):
                extensions = matrix.get(filter_name, {}).get(tile, set())
                if extensions:
                    if use_compact_format:
                        # Compact format: S/E/W/R with colors
                        compact_parts = []
                        if 'sci' in extensions:
                            compact_parts.append("[green]S[/green]")
                        if 'err' in extensions:
                            compact_parts.append("[blue]E[/blue]")
                        if 'wht' in extensions:
                            compact_parts.append("[cyan]W[/cyan]")
                        if 'rms' in extensions:
                            compact_parts.append("[magenta]R[/magenta]")
                        row.append("/".join(compact_parts))
                    else:
                        # Full format for small tables
                        ext_str = ",".join(sorted(extensions))
                        if 'sci' in extensions:
                            row.append(f"[green]{ext_str}[/green]")
                        else:
                            row.append(f"[yellow]{ext_str}[/yellow]")
                else:
                    row.append("[red]✗[/red]")
            table.add_row(*row)
        
        # Print table directly (Rich tables need direct console printing)
        console = Console()
        console.print(table)
        
        # Show missing files with better categorization
        if result.missing_files:
            self.logger.info(f"[yellow]Missing files ({len(result.missing_files)}):[/yellow]")
            missing_analysis = result.analyze_missing_by_completeness()
            
            for filter_name, analysis in missing_analysis.items():
                self.logger.info(f"  [cyan]{filter_name}[/cyan]:")
                
                # Show completely missing tiles first (higher priority)
                if analysis['completely_missing']:
                    tiles_str = ', '.join(analysis['completely_missing'])
                    self.logger.info(f"    [red]✗[/red] Missing tiles: {tiles_str} (no files found)")
                
                # Show partially complete tiles
                if analysis['partially_complete']:
                    partial_parts = []
                    for tile, missing_exts in analysis['partially_complete'].items():
                        exts_str = ','.join(missing_exts)
                        partial_parts.append(f"{tile} (missing: {exts_str})")
                    
                    partial_str = ', '.join(partial_parts)
                    self.logger.info(f"    [yellow]⚠[/yellow] Incomplete tiles: {partial_str}")
    
    def prompt_user_confirmation(self, result: DiscoveryResult, auto_yes: bool = False) -> bool:
        """Prompt user to confirm discovery results."""
        if auto_yes:
            return True
        
        stats = result.get_summary_stats()
        
        if stats['total_missing'] == 0:
            self.logger.info(f"[green]All {stats['total_found']} expected files found![/green]")
            response = input("Proceed with catalog initialization? [Y/n]: ").strip().lower()
            return response in ('', 'y', 'yes')
        else:
            self.logger.info(f"[yellow]Warning: {stats['total_missing']} files missing[/yellow]")
            self.logger.info("You may want to:")
            self.logger.info("  1. Check file paths and fix any issues")
            self.logger.info("  2. Update config.toml if needed")
            self.logger.info("  3. Proceed anyway (missing files will be skipped)")
            
            while True:
                response = input("Proceed anyway? [y/N/details]: ").strip().lower()
                if response in ('n', 'no', ''):
                    return False
                elif response in ('y', 'yes'):
                    return True
                elif response == 'details':
                    self._show_missing_details(result)
                else:
                    self.logger.info("Please enter 'y', 'n', or 'details'")
    
    def _show_missing_details(self, result: DiscoveryResult) -> None:
        """Show detailed list of missing files."""
        self.logger.info("[yellow]Detailed missing file list:[/yellow]")
        missing_analysis = result.analyze_missing_by_completeness()
        
        for filter_name, analysis in missing_analysis.items():
            self.logger.info(f"  [cyan]{filter_name}[/cyan]:")
            
            # Show completely missing tiles with full paths
            if analysis['completely_missing']:
                self.logger.info(f"    [red]Completely missing tiles:[/red]")
                for tile in analysis['completely_missing']:
                    # Find any missing file for this tile to show example path
                    for file_info in result.missing_files:
                        if file_info.filter == filter_name and file_info.tile == tile:
                            # filepath is now absolute, so we can use it directly
                            full_path = Path(file_info.filepath)
                            parent_dir = full_path.parent
                            self.logger.info(f"      {tile}: {parent_dir}/* (all extensions)")
                            break
            
            # Show partially complete tiles with specific missing files
            if analysis['partially_complete']:
                self.logger.info(f"    [yellow]Partially complete tiles:[/yellow]")
                for tile, missing_exts in analysis['partially_complete'].items():
                    for file_info in result.missing_files:
                        if (file_info.filter == filter_name and 
                            file_info.tile == tile and 
                            file_info.extension in missing_exts):
                            # filepath is now absolute, so we can use it directly
                            self.logger.info(f"      {tile}.{file_info.extension}: {file_info.filepath}")
            
            self.logger.info("")  # Empty line between filters


def generate_images_toml(config: AperateConfig, result: DiscoveryResult) -> Dict[str, Any]:
    """Generate images.toml content from discovery results."""
    images_data = {
        'images': {
            'files': {}
        }
    }
    
    # Group files by filter and tile
    files_by_filter_tile = defaultdict(dict)
    for file_info in result.files:
        if file_info.exists:
            key = f"{file_info.filter}.{file_info.tile}"
            files_by_filter_tile[key][file_info.extension] = file_info.filepath
    
    # Convert to nested dictionary format
    for filter_tile, extensions in files_by_filter_tile.items():
        filter_name, tile = filter_tile.split('.', 1)
        
        if filter_name not in images_data['images']['files']:
            images_data['images']['files'][filter_name] = {}
        
        images_data['images']['files'][filter_name][tile] = extensions
    
    return images_data


def save_images_toml(images_data: Dict[str, Any], output_path: Path) -> None:
    """Save images.toml file."""
    import toml
    
    with open(output_path, 'w') as f:
        toml.dump(images_data, f)
    
    get_logger().info(f"Generated images.toml: {output_path}")


def load_images_toml(images_path: Path) -> Dict[str, Any]:
    """Load images.toml file."""
    if not images_path.exists():
        raise FileNotFoundError(f"Images file not found: {images_path}")
    
    try:
        with open(images_path, "rb") as f:
            return tomllib.load(f)
    except Exception as e:
        raise ValueError(f"Failed to parse images.toml: {e}")