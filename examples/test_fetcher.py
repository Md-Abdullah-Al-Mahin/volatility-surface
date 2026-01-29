"""
Example script to test SmartOptionsDataFetcher

This script demonstrates how to use the data fetcher and can be used
to verify the implementation works correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_fetcher import SmartOptionsDataFetcher
import logging

# Set up logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Test the data fetcher with SPY."""
    print("Testing SmartOptionsDataFetcher...")
    print("=" * 50)
    
    # Create fetcher instance
    fetcher = SmartOptionsDataFetcher(
        throttle_seconds=0.5,
        cache_valid_hours=24
    )
    
    # Fetch SPY options data
    print("\nFetching SPY options data...")
    try:
        options_data = fetcher.fetch_options_data("SPY")
        
        print(f"\nSuccess! Fetched {len(options_data)} expiration dates:")
        for exp_date, df in list(options_data.items())[:5]:  # Show first 5
            print(f"  {exp_date}: {len(df)} contracts")
        
        # Show sample data structure
        if options_data:
            first_exp = list(options_data.keys())[0]
            first_df = options_data[first_exp]
            print(f"\nSample data from {first_exp}:")
            print(first_df.head())
            print(f"\nColumns: {list(first_df.columns)}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    print("\n" + "=" * 50)
    print("Test completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
