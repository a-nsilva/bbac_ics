#!/usr/bin/env python3
"""
BBAC ICS Framework - Centralized Logger
Unified logging for ROS 2 and Python modules.
"""
import logging
import sys
from pathlib import Path
from typing import Optional

try:
    import rclpy
    from rclpy.logging import get_logger as get_ros_logger
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False


def setup_logger(
    name: str = 'bbac',
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    log_to_console: bool = True,
    log_format: Optional[str] = None,
    use_ros: bool = False
) -> logging.Logger:
    """
    Setup and configure logger.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file path
        log_to_console: Log to console
        log_format: Custom format
        use_ros: Use ROS 2 logging if available
        
    Returns:
        Configured logger
    """
    # Use ROS logger if requested and available
    if use_ros and ROS_AVAILABLE:
        try:
            return get_ros_logger(name)
        except Exception:
            pass  # Fallback to standard logging
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    
    # Format
    if log_format is None:
        log_format = '[%(asctime)s] [%(name)s] %(levelname)s: %(message)s'
    
    formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str, use_ros: bool = False) -> logging.Logger:
    """
    Get logger with BBAC configuration.
    
    Args:
        name: Logger name
        use_ros: Attempt to use ROS logger
        
    Returns:
        Logger instance
    """
    if use_ros and ROS_AVAILABLE:
        try:
            return get_ros_logger(name)
        except Exception:
            pass
    
    return logging.getLogger(name)


def configure_framework_logging(
    level: int = logging.INFO,
    log_dir: Optional[Path] = None
):
    """
    Configure logging for entire framework.
    
    Args:
        level: Logging level
        log_dir: Log directory
    """
    log_file = None
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / 'bbac.log'
    
    setup_logger(
        name='bbac',
        level=level,
        log_file=log_file,
        log_to_console=True
    )


__all__ = ['setup_logger', 'get_logger', 'configure_framework_logging']
