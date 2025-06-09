#!/usr/bin/env python3
"""
Standardized Logging Configuration for Multi-Agent System
Provides consistent logging across all modules with proper formatting
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from datetime import datetime
try:
    from .config import Config
except ImportError:
    from config import Config

class LoggingSetup:
    """Centralized logging setup and configuration"""
    
    _configured = False
    
    @classmethod
    def setup_logging(cls, 
                     log_level: str = None,
                     log_file: str = None,
                     enable_console: bool = True,
                     enable_file: bool = True) -> None:
        """
        Set up standardized logging configuration
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Path to log file (optional)
            enable_console: Whether to enable console logging
            enable_file: Whether to enable file logging
        """
        if cls._configured:
            return
            
        # Get log level from config
        log_level = log_level or Config.LOG_LEVEL
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
        
        # Create root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(numeric_level)
        
        # Clear any existing handlers
        root_logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(numeric_level)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # File handler
        if enable_file:
            if not log_file:
                # Default log file location
                log_dir = Config.WORKSPACE_DIR / "logs"
                log_dir.mkdir(exist_ok=True)
                log_file = log_dir / f"agents_{datetime.now().strftime('%Y%m%d')}.log"
            
            # Rotating file handler (10MB max, 5 backups)
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        # Set specific logger levels for noisy libraries
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger('aioredis').setLevel(logging.WARNING)
        
        cls._configured = True
        
        # Log initialization message
        logger = logging.getLogger(__name__)
        logger.info(f"Logging configured - Level: {log_level}, Console: {enable_console}, File: {enable_file}")
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get a logger with the specified name
        Ensures logging is configured before returning logger
        """
        if not cls._configured:
            cls.setup_logging()
        return logging.getLogger(name)
    
    @classmethod
    def log_system_info(cls):
        """Log system information for debugging"""
        logger = cls.get_logger(__name__)
        logger.info("=== Multi-Agent System Startup ===")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Working directory: {Path.cwd()}")
        logger.info(f"Config - Redis URL: {Config.REDIS_URL}")
        logger.info(f"Config - Model Path: {Config.MODEL_PATH}")
        logger.info(f"Config - Workspace: {Config.WORKSPACE_DIR}")
        logger.info("=== Configuration Complete ===")

# Convenience function for getting configured loggers
def get_logger(name: str) -> logging.Logger:
    """Get a properly configured logger"""
    return LoggingSetup.get_logger(name)