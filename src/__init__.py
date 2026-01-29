"""
Volatility Surface Constructor Package

A robust Python application for constructing, visualizing, and analyzing
three-dimensional volatility surfaces from live options market data.
"""

__version__ = "0.1.0"

# Set up logging configuration
import logging
import sys
from pathlib import Path

# Create logs directory if it doesn't exist
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "volatility_surface.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)
logger.info("Volatility Surface package initialized")
