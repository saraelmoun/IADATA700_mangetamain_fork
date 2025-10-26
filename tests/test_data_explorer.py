"""
Tests for the DataExplorer module.

This test suite covers:
- Initialization with DataFrame or DataLoader
- Data property behavior and lazy loading
- Reload functionality
- Error handling for invalid configurations
- Integration with DataLoader
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.data_explorer import DataExplorer
from core.data_loader import DataLoader


class TestDataExplorer:
    """Test suite for DataExplorer."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Recipe A', 'Recipe B', 'Recipe C', 'Recipe D', 'Recipe E'],
            'minutes': [30, 45, 15, 60, 120],
            'n_steps': [5, 8, 3, 10, 15],
            'rating': [4.5, 4.0, 3.5, 4.8, 4.2]
        })
    
    @pytest.fixture
    def temp_csv_file(self, sample_data):
        """Create a temporary CSV file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            yield Path(f.name)
        # Cleanup
        Path(f.name).unlink(missing_ok=True)
    
    @pytest.fixture
    def sample_loader(self, temp_csv_file):
        """Create a DataLoader for testing."""
        return DataLoader(temp_csv_file)
    
    # ==================== INITIALIZATION TESTS ====================
    
    def test_initialization_with_dataframe(self, sample_data):
        """Test DataExplorer initialization with DataFrame."""
        explorer = DataExplorer(df=sample_data)
        
        assert explorer._df is sample_data
        assert explorer.loader is None
    
    def test_initialization_with_loader(self, sample_loader):
        """Test DataExplorer initialization with DataLoader."""
        explorer = DataExplorer(loader=sample_loader)
        
        assert explorer._df is None
        assert explorer.loader is sample_loader
    
    def test_initialization_with_both_df_and_loader(self, sample_data, sample_loader):
        """Test DataExplorer initialization with both DataFrame and DataLoader."""
        explorer = DataExplorer(df=sample_data, loader=sample_loader)
        
        assert explorer._df is sample_data
        assert explorer.loader is sample_loader
    
    def test_initialization_without_arguments(self):
        """Test that initialization fails without DataFrame or DataLoader."""
        with pytest.raises(ValueError) as exc_info:
            DataExplorer()
        
        assert "Provide either a DataFrame or a DataLoader" in str(exc_info.value)
    
    def test_initialization_with_none_arguments(self):
        """Test that initialization fails with explicit None arguments."""
        with pytest.raises(ValueError) as exc_info:
            DataExplorer(df=None, loader=None)
        
        assert "Provide either a DataFrame or a DataLoader" in str(exc_info.value)
    
    # ==================== DATA PROPERTY TESTS ====================
    
    def test_df_property_with_existing_dataframe(self, sample_data):
        """Test df property when DataFrame is already provided."""
        explorer = DataExplorer(df=sample_data)
        
        result_df = explorer.df
        assert result_df is sample_data
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 5
    
    def test_df_property_with_loader_lazy_loading(self, sample_loader, sample_data):
        """Test df property with DataLoader (lazy loading)."""
        explorer = DataExplorer(loader=sample_loader)
        
        # Initially no data loaded
        assert explorer._df is None
        
        # Access df property should trigger loading
        result_df = explorer.df
        
        # Now data should be loaded and cached
        assert explorer._df is not None
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 5
        
        # Second access should return cached data
        result_df2 = explorer.df
        assert result_df is result_df2
    
    def test_df_property_without_loader_raises_error(self):
        """Test df property raises error when no data and no loader."""
        # This should not be possible due to __init__ validation, but test anyway
        explorer = DataExplorer.__new__(DataExplorer)  # Bypass __init__
        explorer._df = None
        explorer.loader = None
        
        with pytest.raises(RuntimeError) as exc_info:
            _ = explorer.df
        
        assert "No data available and no loader configured" in str(exc_info.value)
    
    # ==================== RELOAD FUNCTIONALITY TESTS ====================
    
    def test_reload_with_loader(self, sample_loader):
        """Test reload functionality with DataLoader."""
        explorer = DataExplorer(loader=sample_loader)
        
        # Initial load via df property
        df1 = explorer.df
        assert explorer._df is not None
        original_id = id(explorer._df)
        
        # Reload should create new DataFrame object
        df2 = explorer.reload()
        new_id = id(explorer._df)
        
        assert original_id != new_id  # Different objects
        assert explorer._df is df2
        pd.testing.assert_frame_equal(df1, df2)  # Same content
    
    def test_reload_without_force(self, sample_loader):
        """Test reload without force parameter."""
        explorer = DataExplorer(loader=sample_loader)
        
        # Initial load
        df1 = explorer.df
        
        # Reload with force=False should still reload because that's the method's behavior
        df2 = explorer.reload(force=False)
        
        # The actual behavior might be that it still reloads, let's test what it actually does
        # Since reload calls loader.load_data(force=force), and force=False means use cache
        # But if we haven't changed the file, the content should be the same
        pd.testing.assert_frame_equal(df1, df2)  # Content should be same
        assert explorer._df is df2  # Current data reference should be updated
    
    def test_reload_without_loader_raises_error(self, sample_data):
        """Test reload raises error when no loader is configured."""
        explorer = DataExplorer(df=sample_data)
        
        with pytest.raises(RuntimeError) as exc_info:
            explorer.reload()
        
        assert "Cannot reload without a DataLoader" in str(exc_info.value)
    
    # ==================== INTEGRATION TESTS ====================
    
    def test_integration_with_data_loader(self, temp_csv_file, sample_data):
        """Test complete integration with DataLoader."""
        # Create loader
        loader = DataLoader(temp_csv_file)
        
        # Create explorer
        explorer = DataExplorer(loader=loader)
        
        # Access data through explorer
        df = explorer.df
        
        # Verify data is loaded correctly
        assert len(df) == len(sample_data)
        assert 'id' in df.columns
        assert 'name' in df.columns
        
        # Verify loader's cache is also updated
        assert loader._df is not None
        
        # Test reload
        df_reloaded = explorer.reload()
        assert df is not df_reloaded
        pd.testing.assert_frame_equal(df, df_reloaded)
    
    def test_data_consistency_across_operations(self, sample_data):
        """Test data consistency across different operations."""
        explorer = DataExplorer(df=sample_data)
        
        # Multiple accesses should return same object
        df1 = explorer.df
        df2 = explorer.df
        assert df1 is df2
        
        # Data should match original
        pd.testing.assert_frame_equal(df1, sample_data)
    
    # ==================== EDGE CASES ====================
    
    def test_empty_dataframe(self):
        """Test DataExplorer with empty DataFrame."""
        empty_df = pd.DataFrame()
        explorer = DataExplorer(df=empty_df)
        
        result_df = explorer.df
        assert len(result_df) == 0
        assert result_df is empty_df
    
    def test_dataframe_with_missing_values(self):
        """Test DataExplorer with DataFrame containing missing values."""
        df_with_nan = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [1.0, np.nan, 3.0],
            'text': ['a', None, 'c']
        })
        
        explorer = DataExplorer(df=df_with_nan)
        result_df = explorer.df
        
        assert len(result_df) == 3
        assert pd.isna(result_df.iloc[1]['value'])
        assert pd.isna(result_df.iloc[1]['text'])
    
    def test_large_dataframe_handling(self):
        """Test DataExplorer with larger DataFrame."""
        large_df = pd.DataFrame({
            'id': range(1000),
            'value': np.random.randn(1000),
            'category': (['A', 'B', 'C'] * 334)[:1000]  # Ensure exactly 1000 elements
        })
        
        explorer = DataExplorer(df=large_df)
        result_df = explorer.df
        
        assert len(result_df) == 1000
        assert result_df is large_df
    
    # ==================== ERROR HANDLING TESTS ====================
    
    def test_invalid_dataframe_type(self):
        """Test behavior when non-DataFrame is passed as df."""
        # DataExplorer accepts any object as df, validation happens at usage
        explorer = DataExplorer(df="not_a_dataframe")
        
        # The df property just returns whatever was stored
        result = explorer.df
        assert result == "not_a_dataframe"
        
        # The error would occur when trying to use pandas operations on this object
    
    def test_invalid_loader_type(self):
        """Test behavior when invalid loader type is passed."""
        # This test checks that our validation focuses on the existence of df/loader
        # rather than their types (which is validated at usage time)
        explorer = DataExplorer(loader="not_a_loader")
        
        # Should work initially
        assert explorer.loader == "not_a_loader"
        
        # But fail when trying to use the invalid loader
        with pytest.raises(AttributeError):
            _ = explorer.df
    
    # ==================== WORKFLOW TESTS ====================
    
    def test_typical_workflow_with_dataframe(self, sample_data):
        """Test typical usage workflow with DataFrame."""
        # 1. Initialize with DataFrame
        explorer = DataExplorer(df=sample_data)
        
        # 2. Access data multiple times
        df1 = explorer.df
        df2 = explorer.df
        assert df1 is df2
        
        # 3. Verify data properties
        assert len(df1) == 5
        assert list(df1.columns) == ['id', 'name', 'minutes', 'n_steps', 'rating']
    
    def test_typical_workflow_with_loader(self, sample_loader):
        """Test typical usage workflow with DataLoader."""
        # 1. Initialize with DataLoader
        explorer = DataExplorer(loader=sample_loader)
        
        # 2. Access data (triggers loading)
        df1 = explorer.df
        assert len(df1) == 5
        
        # 3. Reload data
        df2 = explorer.reload()
        assert df1 is not df2
        pd.testing.assert_frame_equal(df1, df2)
        
        # 4. Access again (should return reloaded data)
        df3 = explorer.df
        assert df3 is df2