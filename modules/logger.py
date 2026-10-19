import logging
import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def setup_logger(name: str = "apex", level: str = None) -> logging.Logger:
    """
    Configure and return a standardized logger for the APEX RAG pipeline.
    """
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO").upper()
    
    numeric_level = getattr(logging, level, logging.INFO)
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)
    
    # Avoid duplicate handlers if setup_logger is called repeatedly
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(numeric_level)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        
    return logger

def get_logger(module_name: str) -> logging.Logger:
    """Helper to get a child logger for a specific module."""
    root_logger = setup_logger()
    return logging.getLogger(f"apex.{module_name}")
