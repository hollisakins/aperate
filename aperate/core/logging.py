"""Logging setup with rich integration for aperate."""

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn


class AperateLogger:
    """Centralized logging setup for aperate with rich integration."""
    
    def __init__(self):
        self.console = Console()
        self._logger = None
        self._progress = None
    
    def setup_logging(
        self, 
        level: str = "INFO", 
        log_file: Optional[Path] = None,
        quiet: bool = False
    ) -> logging.Logger:
        """Set up logging with rich console output and optional file logging."""
        
        # Create logger
        logger = logging.getLogger("aperate")
        logger.setLevel(getattr(logging, level.upper()))
        
        # Clear any existing handlers
        logger.handlers.clear()
        
        if not quiet:
            # Rich console handler
            rich_handler = RichHandler(
                console=self.console,
                show_time=True,
                show_path=False,
                markup=True,
                omit_repeated_times=False,
            )
            rich_handler.setLevel(getattr(logging, level.upper()))
            logger.addHandler(rich_handler)
        
        # File handler if requested
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        self._logger = logger
        return logger
    
    def get_logger(self) -> logging.Logger:
        """Get the configured logger instance."""
        if self._logger is None:
            return self.setup_logging()
        return self._logger
    
    def create_progress(self, description: str = "Processing...") -> Progress:
        """Create a rich progress bar for long operations."""
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True
        )
        self._progress = progress
        return progress
    
    def print(self, *args, **kwargs):
        """Print using rich console."""
        self.console.print(*args, **kwargs)
    
    def print_error(self, message: str):
        """Print error message in red."""
        self.console.print(f"[bold red]Error:[/bold red] {message}")
    
    def print_warning(self, message: str):
        """Print warning message in yellow."""
        self.console.print(f"[bold yellow]Warning:[/bold yellow] {message}")
    
    def print_success(self, message: str):
        """Print success message in green."""
        self.console.print(f"[bold green]Success:[/bold green] {message}")


# Global logger instance
_aperate_logger = AperateLogger()

def get_logger() -> logging.Logger:
    """Get the global aperate logger."""
    return _aperate_logger.get_logger()

def setup_logging(**kwargs) -> logging.Logger:
    """Set up logging with the given configuration."""
    return _aperate_logger.setup_logging(**kwargs)

def get_console() -> Console:
    """Get the global rich console."""
    return _aperate_logger.console

def print_error(message: str):
    """Print error message."""
    _aperate_logger.print_error(message)

def print_warning(message: str):
    """Print warning message."""
    _aperate_logger.print_warning(message)

def print_success(message: str):
    """Print success message."""
    _aperate_logger.print_success(message)