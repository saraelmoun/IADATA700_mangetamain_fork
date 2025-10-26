"""
Tests for the DataLoader module.

This test suite covers:
- Data loading from different file formats (CSV, Parquet)
- Error handling for missing files and unsupported formats
- Cache behavior and force reload
- Preprocessing functionality
- Path handling and validation
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.data_loader import DataLoader


class TestDataLoader:
    """Test suite for DataLoader."""
    
    @pytest.fixture
    def sample_csv_data(self):
        """Create sample data for CSV testing."""
        return pd.DataFrame({
            'Recipe ID': [1, 2, 3, 4, 5],
            'Recipe Name': ['Pasta', 'Pizza', 'Salad', 'Soup', 'Cake'],
            'Minutes ': [30, 45, 15, 60, 120],  # Note the trailing space
            'N Steps': [5, 8, 3, 10, 15],
            'N Ingredients': [8, 12, 5, 15, 20]
        })
    
    @pytest.fixture
    def temp_csv_file(self, sample_csv_data):
        """Create a temporary CSV file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_csv_data.to_csv(f.name, index=False)
            yield Path(f.name)
        # Cleanup
        Path(f.name).unlink(missing_ok=True)
    
    @pytest.fixture
    def temp_parquet_file(self, sample_csv_data):
        """Create a temporary Parquet file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            sample_csv_data.to_parquet(f.name, index=False)
            yield Path(f.name)
        # Cleanup
        Path(f.name).unlink(missing_ok=True)
    
    # ==================== INITIALIZATION TESTS ====================
    
    def test_initialization_with_string_path(self, temp_csv_file):
        """Test DataLoader initialization with string path."""
        loader = DataLoader(str(temp_csv_file))
        assert loader.data_path == temp_csv_file
        assert loader.cache is True  # default
        assert loader._df is None
    
    def test_initialization_with_path_object(self, temp_csv_file):
        """Test DataLoader initialization with Path object."""
        loader = DataLoader(temp_csv_file)
        assert loader.data_path == temp_csv_file
        assert loader.cache is True
    
    def test_initialization_with_cache_disabled(self, temp_csv_file):
        """Test DataLoader initialization with cache disabled."""
        loader = DataLoader(temp_csv_file, cache=False)
        assert loader.cache is False
    
    # ==================== FILE LOADING TESTS ====================
    
    def test_load_csv_file(self, temp_csv_file, sample_csv_data):
        """Test loading CSV file."""
        loader = DataLoader(temp_csv_file)
        df = loader.load_data()
        
        # Check data shape
        assert df.shape == sample_csv_data.shape
        
        # Check that columns were preprocessed (lowercase, spaces removed)
        expected_columns = ['recipe_id', 'recipe_name', 'minutes', 'n_steps', 'n_ingredients']
        assert list(df.columns) == expected_columns
        
        # Check data content (first row)
        assert df.iloc[0]['recipe_id'] == 1
        assert df.iloc[0]['recipe_name'] == 'Pasta'
        assert df.iloc[0]['minutes'] == 30
    
    def test_load_parquet_file(self, temp_parquet_file, sample_csv_data):
        """Test loading Parquet file."""
        loader = DataLoader(temp_parquet_file)
        df = loader.load_data()
        
        # Check data shape
        assert df.shape == sample_csv_data.shape
        
        # Check that columns were preprocessed
        expected_columns = ['recipe_id', 'recipe_name', 'minutes', 'n_steps', 'n_ingredients']
        assert list(df.columns) == expected_columns
    
    def test_file_not_found_error(self):
        """Test error handling for non-existent file."""
        loader = DataLoader("/non/existent/file.csv")
        
        with pytest.raises(FileNotFoundError) as exc_info:
            loader.load_data()
        
        assert "Data file not found" in str(exc_info.value)
    
    def test_unsupported_file_format_error(self):
        """Test error handling for unsupported file format."""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
            loader = DataLoader(f.name)
            
            with pytest.raises(ValueError) as exc_info:
                loader.load_data()
            
            assert "Unsupported file format" in str(exc_info.value)
            assert ".xlsx" in str(exc_info.value)
        
        # Cleanup
        Path(f.name).unlink(missing_ok=True)
    
    # ==================== CACHE BEHAVIOR TESTS ====================
    
    def test_cache_behavior_default(self, temp_csv_file):
        """Test that data is cached by default."""
        loader = DataLoader(temp_csv_file)
        
        # First load
        df1 = loader.load_data()
        assert loader._df is not None
        
        # Second load should return cached data
        df2 = loader.load_data()
        assert df1 is df2  # Same object reference
    
    def test_force_reload(self, temp_csv_file):
        """Test force reload functionality."""
        loader = DataLoader(temp_csv_file)
        
        # First load
        df1 = loader.load_data()
        original_id = id(df1)
        
        # Force reload
        df2 = loader.load_data(force=True)
        new_id = id(df2)
        
        # Should be different objects but same content
        assert original_id != new_id
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_get_data_loads_if_needed(self, temp_csv_file):
        """Test get_data method loads data if not already loaded."""
        loader = DataLoader(temp_csv_file)
        assert loader._df is None
        
        df = loader.get_data()
        assert loader._df is not None
        assert df is loader._df
    
    def test_get_data_returns_cached(self, temp_csv_file):
        """Test get_data method returns cached data."""
        loader = DataLoader(temp_csv_file)
        
        # Load data first
        df1 = loader.load_data()
        
        # get_data should return the same cached data
        df2 = loader.get_data()
        assert df1 is df2
    
    # ==================== PREPROCESSING TESTS ====================
    
    def test_column_preprocessing(self, temp_csv_file):
        """Test column name preprocessing."""
        loader = DataLoader(temp_csv_file)
        df = loader.load_data()
        
        # Original columns: ['Recipe ID', 'Recipe Name', 'Minutes ', 'N Steps', 'N Ingredients']
        # Should become: ['recipe_id', 'recipe_name', 'minutes', 'n_steps', 'n_ingredients']
        expected_columns = ['recipe_id', 'recipe_name', 'minutes', 'n_steps', 'n_ingredients']
        assert list(df.columns) == expected_columns
    
    def test_preprocess_method_direct(self):
        """Test preprocess method directly."""
        loader = DataLoader("/dummy/path.csv")  # Path doesn't matter for this test
        
        # Create test dataframe with problematic column names
        test_df = pd.DataFrame({
            'Column With Spaces': [1, 2, 3],
            'UPPERCASE': [4, 5, 6],
            ' Leading Space': [7, 8, 9],
            'Trailing Space ': [10, 11, 12]
        })
        
        processed_df = loader.preprocess(test_df)
        
        expected_columns = ['column_with_spaces', 'uppercase', 'leading_space', 'trailing_space']
        assert list(processed_df.columns) == expected_columns
        
        # Check that data is preserved
        assert processed_df.shape == test_df.shape
        assert processed_df.iloc[0, 0] == 1  # First value should be preserved
    
    def test_preprocess_preserves_data(self, sample_csv_data):
        """Test that preprocessing preserves original data values."""
        loader = DataLoader("/dummy/path.csv")
        processed_df = loader.preprocess(sample_csv_data)
        
        # Data values should be unchanged
        assert processed_df.iloc[0, 0] == sample_csv_data.iloc[0, 0]  # Recipe ID
        assert processed_df.iloc[0, 1] == sample_csv_data.iloc[0, 1]  # Recipe Name
        assert processed_df.iloc[0, 2] == sample_csv_data.iloc[0, 2]  # Minutes
    
    # ==================== EDGE CASES AND ERROR HANDLING ====================
    
    def test_empty_csv_file(self):
        """Test handling of empty CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Write just headers
            f.write("col1,col2\n")
            f.flush()
            
            loader = DataLoader(f.name)
            df = loader.load_data()
            
            assert len(df) == 0
            assert list(df.columns) == ['col1', 'col2']
        
        # Cleanup
        Path(f.name).unlink(missing_ok=True)
    
    def test_csv_with_special_characters(self):
        """Test CSV with special characters in data."""
        test_data = pd.DataFrame({
            'Name': ['Café', 'Naïve', 'Résumé'],
            'Description': ['Test with "quotes"', "Test with 'apostrophes'", 'Test with, commas']
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            test_data.to_csv(f.name, index=False)
            
            loader = DataLoader(f.name)
            df = loader.load_data()
            
            assert len(df) == 3
            assert df.iloc[0]['name'] == 'Café'
            assert df.iloc[1]['name'] == 'Naïve'
        
        # Cleanup
        Path(f.name).unlink(missing_ok=True)
    
    def test_path_conversion(self):
        """Test that string paths are properly converted to Path objects."""
        loader = DataLoader("/some/path/file.csv")
        assert isinstance(loader.data_path, Path)
        assert str(loader.data_path) == "/some/path/file.csv"
    
    # ==================== INTEGRATION TESTS ====================
    
    def test_full_workflow(self, temp_csv_file, sample_csv_data):
        """Test complete workflow from initialization to data access."""
        # Initialize loader
        loader = DataLoader(temp_csv_file, cache=True)
        
        # Check initial state
        assert loader._df is None
        
        # Load data
        df1 = loader.load_data()
        assert df1 is not None
        assert loader._df is not None
        
        # Access via get_data
        df2 = loader.get_data()
        assert df1 is df2
        
        # Force reload
        df3 = loader.load_data(force=True)
        assert df1 is not df3  # Different objects
        pd.testing.assert_frame_equal(df1, df3)  # Same content
        
        # Verify final state
        assert loader._df is df3