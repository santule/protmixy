"""
Logging configuration for protein evolution experiments.

Provides a configured logger instance for tracking experiment progress
and debugging.
"""
import logging
import sys

# Create logger
logger = logging.getLogger('protein_evolution')
logger.setLevel(logging.INFO)

# Create console handler with formatting
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(console_handler)

# Prevent propagation to root logger
logger.propagate = False
