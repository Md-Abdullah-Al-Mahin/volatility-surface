"""
Unit tests for Module 1: SmartOptionsDataFetcher
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from datetime import datetime, date
from unittest.mock import Mock, patch, MagicMock

from src.data_fetcher import SmartOptionsDataFetcher


class TestSmartOptionsDataFetcher:
    """Test suite for SmartOptionsDataFetcher."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def fetcher(self, temp_cache_dir):
        """Create a SmartOptionsDataFetcher instance with temp cache."""
        return SmartOptionsDataFetcher(
            cache_dir=temp_cache_dir,
            throttle_seconds=0.1,  # Faster for testing
            cache_valid_hours=24
        )
    
    def test_init_default_cache_dir(self):
        """Test initialization with default cache directory."""
        fetcher = SmartOptionsDataFetcher()
        assert fetcher.cache_dir.exists()
        assert fetcher.cache_dir.name == "cache"
        assert fetcher.throttle_seconds == 0.5
        assert fetcher.cache_valid_hours == 24
    
    def test_init_custom_parameters(self, temp_cache_dir):
        """Test initialization with custom parameters."""
        fetcher = SmartOptionsDataFetcher(
            cache_dir=temp_cache_dir,
            throttle_seconds=1.0,
            cache_valid_hours=12
        )
        assert fetcher.cache_dir == Path(temp_cache_dir)
        assert fetcher.throttle_seconds == 1.0
        assert fetcher.cache_valid_hours == 12
    
    def test_get_cache_path(self, fetcher):
        """Test cache path generation."""
        path = fetcher._get_cache_path("SPY", "2024-01-15")
        assert path.name == "SPY_2024_01_15.parquet"
        assert path.parent == fetcher.cache_dir
        
        # Test with special characters
        path = fetcher._get_cache_path("^SPX", "2024-01-15")
        assert path.name == "SPX_2024_01_15.parquet"
    
    def test_is_cache_valid_no_file(self, fetcher):
        """Test cache validation when file doesn't exist."""
        cache_path = fetcher._get_cache_path("SPY", "2024-01-15")
        assert not fetcher._is_cache_valid(cache_path)
    
    def test_is_cache_valid_stale_file(self, fetcher):
        """Test cache validation with stale file."""
        cache_path = fetcher._get_cache_path("SPY", "2024-01-15")
        
        # Create a file and make it old
        cache_path.touch()
        old_time = datetime.now().timestamp() - (25 * 3600)  # 25 hours ago
        import os
        os.utime(cache_path, (old_time, old_time))
        
        assert not fetcher._is_cache_valid(cache_path)

    def test_load_from_cache_deletes_stale_file(self, fetcher):
        """Test that loading when cache is stale deletes the stale cache file."""
        cache_path = fetcher._get_cache_path("SPY", "2024-01-15")
        cache_path.touch()
        old_time = datetime.now().timestamp() - (25 * 3600)  # 25 hours ago
        os.utime(cache_path, (old_time, old_time))
        assert cache_path.exists()

        result = fetcher._load_from_cache("SPY", "2024-01-15")
        assert result is None
        assert not cache_path.exists(), "Stale cache file should be deleted"

    def test_load_from_cache_deletes_corrupt_file(self, fetcher):
        """Test that loading when cache file is corrupt deletes the corrupt file."""
        cache_path = fetcher._get_cache_path("SPY", "2024-01-15")
        cache_path.touch()  # Fresh but invalid content
        cache_path.write_text("not valid parquet")
        assert cache_path.exists()

        result = fetcher._load_from_cache("SPY", "2024-01-15")
        assert result is None
        assert not cache_path.exists(), "Corrupt cache file should be deleted"
    
    def test_is_cache_valid_fresh_file(self, fetcher):
        """Test cache validation with fresh file."""
        cache_path = fetcher._get_cache_path("SPY", "2024-01-15")
        cache_path.touch()
        
        assert fetcher._is_cache_valid(cache_path)
    
    def test_save_and_load_cache(self, fetcher):
        """Test saving and loading from cache."""
        ticker = "SPY"
        exp_date = "2024-01-15"
        
        # Create test data
        test_data = pd.DataFrame({
            'contractSymbol': ['SPY240115C00100000', 'SPY240115P00100000'],
            'strike': [100.0, 100.0],
            'lastPrice': [50.0, 1.0],
            'optionType': ['call', 'put'],
            'expirationDate': [exp_date, exp_date]
        })
        
        # Save to cache
        fetcher._save_to_cache(ticker, exp_date, test_data)
        
        # Verify file exists
        cache_path = fetcher._get_cache_path(ticker, exp_date)
        assert cache_path.exists()
        
        # Load from cache
        loaded_data = fetcher._load_from_cache(ticker, exp_date)
        
        assert loaded_data is not None
        assert len(loaded_data) == len(test_data)
        pd.testing.assert_frame_equal(loaded_data, test_data)
    
    @patch('src.data_fetcher.yf.Ticker')
    def test_fetch_single_expiration_success(self, mock_ticker_class, fetcher):
        """Test successful fetch of single expiration."""
        # Mock yfinance Ticker
        mock_ticker = Mock()
        mock_option_chain = Mock()
        
        # Create mock DataFrames for calls and puts
        mock_calls = pd.DataFrame({
            'contractSymbol': ['SPY240115C00100000'],
            'strike': [100.0],
            'lastPrice': [50.0],
            'bid': [49.5],
            'ask': [50.5],
            'volume': [1000],
            'openInterest': [5000],
            'impliedVolatility': [0.20],
            'inTheMoney': [True]
        })
        
        mock_puts = pd.DataFrame({
            'contractSymbol': ['SPY240115P00100000'],
            'strike': [100.0],
            'lastPrice': [1.0],
            'bid': [0.95],
            'ask': [1.05],
            'volume': [500],
            'openInterest': [2000],
            'impliedVolatility': [0.18],
            'inTheMoney': [False]
        })
        
        mock_option_chain.calls = mock_calls
        mock_option_chain.puts = mock_puts
        mock_ticker.option_chain.return_value = mock_option_chain
        mock_ticker_class.return_value = mock_ticker
        
        # Fetch data
        result = fetcher._fetch_single_expiration(mock_ticker, "2024-01-15", "SPY")
        
        assert result is not None
        assert len(result) == 2  # One call + one put
        assert 'optionType' in result.columns
        assert 'expirationDate' in result.columns
        assert set(result['optionType'].unique()) == {'call', 'put'}
        
        # Verify cache was created
        cache_path = fetcher._get_cache_path("SPY", "2024-01-15")
        assert cache_path.exists()
    
    @patch('src.data_fetcher.yf.Ticker')
    def test_fetch_single_expiration_failure(self, mock_ticker_class, fetcher):
        """Test handling of fetch failure."""
        mock_ticker = Mock()
        mock_ticker.option_chain.side_effect = Exception("Network error")
        mock_ticker_class.return_value = mock_ticker
        
        result = fetcher._fetch_single_expiration(mock_ticker, "2024-01-15", "SPY")
        
        assert result is None
    
    @patch('src.data_fetcher.yf.Ticker')
    def test_fetch_options_data_success(self, mock_ticker_class, fetcher):
        """Test successful fetch of all expirations."""
        # Mock yfinance Ticker
        mock_ticker = Mock()
        mock_ticker.options = ["2024-01-15", "2024-02-15", "2024-03-15"]
        
        # Mock option chain
        mock_option_chain = Mock()
        mock_calls = pd.DataFrame({
            'contractSymbol': ['SPY240115C00100000'],
            'strike': [100.0],
            'lastPrice': [50.0],
            'bid': [49.5],
            'ask': [50.5],
            'volume': [1000],
            'openInterest': [5000],
            'impliedVolatility': [0.20],
            'inTheMoney': [True]
        })
        mock_puts = pd.DataFrame({
            'contractSymbol': ['SPY240115P00100000'],
            'strike': [100.0],
            'lastPrice': [1.0],
            'bid': [0.95],
            'ask': [1.05],
            'volume': [500],
            'openInterest': [2000],
            'impliedVolatility': [0.18],
            'inTheMoney': [False]
        })
        mock_option_chain.calls = mock_calls
        mock_option_chain.puts = mock_puts
        mock_ticker.option_chain.return_value = mock_option_chain
        mock_ticker_class.return_value = mock_ticker
        
        # Fetch data
        result = fetcher.fetch_options_data("SPY")
        
        assert isinstance(result, dict)
        assert len(result) == 3
        assert "2024-01-15" in result
        assert "2024-02-15" in result
        assert "2024-03-15" in result
        
        # Verify each DataFrame has expected structure
        for exp_date, df in result.items():
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            assert 'contractSymbol' in df.columns
            assert 'strike' in df.columns
            assert 'expirationDate' in df.columns
    
    @patch('src.data_fetcher.yf.Ticker')
    def test_fetch_options_data_invalid_ticker(self, mock_ticker_class, fetcher):
        """Test handling of invalid ticker symbol."""
        mock_ticker = Mock()
        mock_ticker.options = []  # No options available
        mock_ticker_class.return_value = mock_ticker
        
        result = fetcher.fetch_options_data("INVALID")
        assert result == {}
    
    @patch('src.data_fetcher.yf.Ticker')
    def test_fetch_options_data_network_error(self, mock_ticker_class, fetcher):
        """Test handling of network errors."""
        mock_ticker_class.side_effect = Exception("Network error")
        
        with pytest.raises(ConnectionError):
            fetcher.fetch_options_data("SPY")
    
    def test_clear_cache_specific_ticker(self, fetcher):
        """Test clearing cache for specific ticker."""
        # Create some cache files
        test_data = pd.DataFrame({'test': [1, 2, 3]})
        fetcher._save_to_cache("SPY", "2024-01-15", test_data)
        fetcher._save_to_cache("SPY", "2024-02-15", test_data)
        fetcher._save_to_cache("AAPL", "2024-01-15", test_data)
        
        # Clear SPY cache only
        deleted = fetcher.clear_cache("SPY")
        
        assert deleted == 2
        
        # Verify SPY cache is gone but AAPL remains
        spy_path1 = fetcher._get_cache_path("SPY", "2024-01-15")
        spy_path2 = fetcher._get_cache_path("SPY", "2024-02-15")
        aapl_path = fetcher._get_cache_path("AAPL", "2024-01-15")
        
        assert not spy_path1.exists()
        assert not spy_path2.exists()
        assert aapl_path.exists()
    
    def test_clear_cache_all(self, fetcher):
        """Test clearing all cache."""
        # Create some cache files
        test_data = pd.DataFrame({'test': [1, 2, 3]})
        fetcher._save_to_cache("SPY", "2024-01-15", test_data)
        fetcher._save_to_cache("AAPL", "2024-01-15", test_data)
        fetcher._save_to_cache("QQQ", "2024-01-15", test_data)
        
        # Clear all cache
        deleted = fetcher.clear_cache()
        
        assert deleted == 3
        
        # Verify all cache is gone
        for ticker in ["SPY", "AAPL", "QQQ"]:
            cache_path = fetcher._get_cache_path(ticker, "2024-01-15")
            assert not cache_path.exists()


# Integration test (requires network connection - can be skipped)
@pytest.mark.integration
class TestSmartOptionsDataFetcherIntegration:
    """Integration tests that require network connection."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def fetcher(self, temp_cache_dir):
        """Create a SmartOptionsDataFetcher instance."""
        return SmartOptionsDataFetcher(
            cache_dir=temp_cache_dir,
            throttle_seconds=0.5
        )
    
    @pytest.mark.skipif(
        not pytest.config.getoption("--run-integration", default=False),
        reason="Integration tests require --run-integration flag"
    )
    def test_fetch_real_data_spy(self, fetcher):
        """Test fetching real SPY data (requires network)."""
        result = fetcher.fetch_options_data("SPY")
        
        assert isinstance(result, dict)
        assert len(result) > 0
        
        # Verify structure of first expiration
        first_exp = list(result.keys())[0]
        df = result[first_exp]
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'contractSymbol' in df.columns
        assert 'strike' in df.columns
        assert 'expirationDate' in df.columns
        assert 'optionType' in df.columns
    
    @pytest.mark.skipif(
        not pytest.config.getoption("--run-integration", default=False),
        reason="Integration tests require --run-integration flag"
    )
    def test_fetch_real_data_aapl(self, fetcher):
        """Test fetching real AAPL data (requires network)."""
        result = fetcher.fetch_options_data("AAPL")
        
        assert isinstance(result, dict)
        assert len(result) > 0
