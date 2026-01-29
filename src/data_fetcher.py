"""
Module 1: SmartOptionsDataFetcher

Interface with yfinance to retrieve raw options chain data for all available expirations.
"""

import logging
import time
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class SmartOptionsDataFetcher:
    """
    Fetches options chain data from yfinance with caching and error handling.
    
    Attributes:
        cache_dir: Directory path for caching data
        throttle_seconds: Seconds to wait between API requests
        cache_valid_hours: Hours before cache is considered stale
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        throttle_seconds: float = 0.5,
        cache_valid_hours: int = 24
    ):
        """
        Initialize the data fetcher.
        
        Args:
            cache_dir: Directory for caching data. Defaults to 'data/cache'
            throttle_seconds: Seconds to wait between API requests (default: 0.5)
            cache_valid_hours: Hours before cache is considered stale (default: 24)
        """
        if cache_dir is None:
            # Default to project's data/cache directory
            project_root = Path(__file__).parent.parent
            cache_dir = project_root / "data" / "cache"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.throttle_seconds = throttle_seconds
        self.cache_valid_hours = cache_valid_hours
        
        logger.info(
            f"Initialized SmartOptionsDataFetcher with cache_dir={self.cache_dir}, "
            f"throttle={self.throttle_seconds}s, valid_hours={self.cache_valid_hours}"
        )
    
    def _get_cache_path(self, ticker: str, exp_date: str) -> Path:
        """
        Generate cache file path for a ticker and expiration date.
        
        Args:
            ticker: Ticker symbol
            exp_date: Expiration date string (YYYY-MM-DD)
            
        Returns:
            Path to cache file
        """
        # Sanitize ticker and exp_date for filename
        safe_ticker = ticker.replace("^", "").upper()
        safe_exp = exp_date.replace("-", "_")
        filename = f"{safe_ticker}_{safe_exp}.parquet"
        return self.cache_dir / filename
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """
        Check if cache file exists and is still valid.
        
        Args:
            cache_path: Path to cache file
            
        Returns:
            True if cache is valid, False otherwise
        """
        if not cache_path.exists():
            return False
        
        # Check if cache is within valid time window
        cache_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age_hours = (datetime.now() - cache_time).total_seconds() / 3600
        
        return age_hours < self.cache_valid_hours
    
    def _delete_cache_file(self, cache_path: Path) -> None:
        """
        Delete a cache file if it exists. Used when cache is invalid or corrupt.
        
        Args:
            cache_path: Path to cache file
        """
        if cache_path.exists():
            try:
                cache_path.unlink()
                logger.debug(f"Deleted invalid/stale cache file {cache_path}")
            except OSError as e:
                logger.warning(f"Could not delete cache file {cache_path}: {e}")

    def _load_from_cache(self, ticker: str, exp_date: str) -> Optional[pd.DataFrame]:
        """
        Load options data from cache if available and valid.
        If cache exists but is invalid (stale or corrupt), the file is deleted.
        
        Args:
            ticker: Ticker symbol
            exp_date: Expiration date string
            
        Returns:
            DataFrame if cache exists and is valid, None otherwise
        """
        cache_path = self._get_cache_path(ticker, exp_date)
        
        if self._is_cache_valid(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                logger.debug(f"Loaded {ticker} {exp_date} from cache")
                return df
            except Exception as e:
                logger.warning(f"Failed to load cache for {ticker} {exp_date}: {e}")
                self._delete_cache_file(cache_path)
                return None
        
        # Cache exists but is stale - delete it so we don't keep invalid files
        self._delete_cache_file(cache_path)
        return None
    
    def _save_to_cache(self, ticker: str, exp_date: str, data: pd.DataFrame) -> None:
        """
        Save options data to cache.
        
        Args:
            ticker: Ticker symbol
            exp_date: Expiration date string
            data: DataFrame to cache
        """
        cache_path = self._get_cache_path(ticker, exp_date)
        
        try:
            data.to_parquet(cache_path, index=False)
            logger.debug(f"Cached {ticker} {exp_date} to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache for {ticker} {exp_date}: {e}")
    
    def _fetch_single_expiration(
        self,
        ticker_obj: yf.Ticker,
        exp_date: str,
        ticker: str
    ) -> Optional[pd.DataFrame]:
        """
        Fetch options data for a single expiration date.
        
        Args:
            ticker_obj: yfinance Ticker object
            exp_date: Expiration date string
            ticker: Ticker symbol (for logging/caching)
            
        Returns:
            DataFrame with combined calls and puts, or None if fetch fails
        """
        # Check cache first
        cached_data = self._load_from_cache(ticker, exp_date)
        if cached_data is not None:
            return cached_data
        
        try:
            # Fetch option chain
            option_chain = ticker_obj.option_chain(exp_date)
            
            # Get calls and puts
            calls = option_chain.calls.copy()
            puts = option_chain.puts.copy()
            
            # Add option type column
            calls['optionType'] = 'call'
            puts['optionType'] = 'put'
            
            # Add expiration date column
            calls['expirationDate'] = exp_date
            puts['expirationDate'] = exp_date
            
            # Combine calls and puts
            combined = pd.concat([calls, puts], ignore_index=True)
            
            # Ensure we have the required columns
            required_columns = [
                'contractSymbol', 'strike', 'lastPrice', 'bid', 'ask',
                'volume', 'openInterest', 'impliedVolatility', 'inTheMoney'
            ]
            
            # Add missing columns with NaN if they don't exist
            for col in required_columns:
                if col not in combined.columns:
                    combined[col] = None
            
            # Select and reorder columns
            columns_to_keep = required_columns + ['optionType', 'expirationDate']
            available_columns = [col for col in columns_to_keep if col in combined.columns]
            combined = combined[available_columns]
            
            # Cache the data
            self._save_to_cache(ticker, exp_date, combined)
            
            logger.info(
                f"Fetched {ticker} {exp_date}: {len(calls)} calls, {len(puts)} puts"
            )
            
            return combined
            
        except Exception as e:
            logger.error(f"Failed to fetch {ticker} {exp_date}: {e}")
            return None
    
    def fetch_options_data(
        self,
        ticker: str,
        date_filter: Optional[date] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch options chain data for all available expirations.
        
        Args:
            ticker: Ticker symbol (e.g., 'SPY', 'AAPL')
            date_filter: Optional date to filter expirations (not yet implemented)
            
        Returns:
            Dictionary mapping expiration dates (strings) to DataFrames
            
        Raises:
            ValueError: If ticker symbol is invalid
            ConnectionError: If network request fails repeatedly
        """
        logger.info(f"Fetching options data for {ticker}")
        
        try:
            # Create yfinance Ticker object
            ticker_obj = yf.Ticker(ticker)
            
            # Get available expiration dates
            try:
                expiration_dates = ticker_obj.options
            except Exception as e:
                logger.error(f"Failed to get expiration dates for {ticker}: {e}")
                raise ValueError(f"Invalid ticker symbol or unable to fetch data: {ticker}")
            
            if not expiration_dates:
                logger.warning(f"No expiration dates found for {ticker}")
                return {}
            
            logger.info(f"Found {len(expiration_dates)} expiration dates for {ticker}")
            
            # Fetch data for each expiration
            options_data = {}
            failed_fetches = []
            
            for i, exp_date in enumerate(expiration_dates):
                # Throttle requests (except for first one)
                if i > 0:
                    time.sleep(self.throttle_seconds)
                
                df = self._fetch_single_expiration(ticker_obj, exp_date, ticker)
                
                if df is not None and not df.empty:
                    options_data[exp_date] = df
                else:
                    failed_fetches.append(exp_date)
            
            if failed_fetches:
                logger.warning(
                    f"Failed to fetch {len(failed_fetches)} expirations: {failed_fetches}"
                )
            
            if not options_data:
                raise ConnectionError(
                    f"Failed to fetch any options data for {ticker}. "
                    "Check network connection and ticker symbol."
                )
            
            total_contracts = sum(len(df) for df in options_data.values())
            logger.info(
                f"Successfully fetched {len(options_data)} expirations "
                f"with {total_contracts} total contracts for {ticker}"
            )
            
            return options_data
            
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching data for {ticker}: {e}")
            raise ConnectionError(f"Failed to fetch options data: {e}")
    
    def clear_cache(self, ticker: Optional[str] = None) -> int:
        """
        Clear cached data.
        
        Args:
            ticker: If provided, only clear cache for this ticker.
                   If None, clear all cache.
                   
        Returns:
            Number of files deleted
        """
        deleted_count = 0
        
        if ticker:
            # Clear cache for specific ticker
            safe_ticker = ticker.replace("^", "").upper()
            pattern = f"{safe_ticker}_*.parquet"
            for cache_file in self.cache_dir.glob(pattern):
                cache_file.unlink()
                deleted_count += 1
            logger.info(f"Cleared {deleted_count} cache files for {ticker}")
        else:
            # Clear all cache
            for cache_file in self.cache_dir.glob("*.parquet"):
                cache_file.unlink()
                deleted_count += 1
            logger.info(f"Cleared all {deleted_count} cache files")
        
        return deleted_count
