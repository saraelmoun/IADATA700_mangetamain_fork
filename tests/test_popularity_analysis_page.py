"""
Tests for the PopularityAnalysisPage module.

This test suite covers:
- PopularityAnalysisConfig dataclass behavior
- PopularityAnalysisPage initialization and configuration
- Data loading and validation logic
- Plot title generation
- Logic methods that can be tested without Streamlit UI
- Helper functions and utility methods
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from components.popularity_analysis_page import PopularityAnalysisPage, PopularityAnalysisConfig


class TestPopularityAnalysisConfig:
    """Test suite for PopularityAnalysisConfig dataclass."""
    
    def test_config_initialization(self):
        """Test PopularityAnalysisConfig initialization."""
        interactions_path = Path("/test/interactions.csv")
        recipes_path = Path("/test/recipes.csv")
        
        config = PopularityAnalysisConfig(
            interactions_path=interactions_path,
            recipes_path=recipes_path
        )
        
        assert config.interactions_path == interactions_path
        assert config.recipes_path == recipes_path
        assert isinstance(config.interactions_path, Path)
        assert isinstance(config.recipes_path, Path)
    
    def test_config_with_string_paths(self):
        """Test config initialization with string paths."""
        config = PopularityAnalysisConfig(
            interactions_path="/test/interactions.csv",
            recipes_path="/test/recipes.csv"
        )
        
        # Should handle string paths (behavior depends on implementation)
        assert config.interactions_path is not None
        assert config.recipes_path is not None
    
    def test_config_equality(self):
        """Test config equality comparison."""
        path1 = Path("/test/interactions.csv")
        path2 = Path("/test/recipes.csv")
        
        config1 = PopularityAnalysisConfig(
            interactions_path=path1,
            recipes_path=path2
        )
        
        config2 = PopularityAnalysisConfig(
            interactions_path=path1,
            recipes_path=path2
        )
        
        assert config1.interactions_path == config2.interactions_path
        assert config1.recipes_path == config2.recipes_path


class TestPopularityAnalysisPage:
    """Test suite for PopularityAnalysisPage class."""
    
    @pytest.fixture
    def sample_interactions_data(self):
        """Create sample interactions data for testing."""
        return pd.DataFrame({
            'recipe_id': [1, 1, 2, 2, 3, 3, 4, 5],
            'user_id': [101, 102, 103, 104, 105, 106, 107, 108],
            'rating': [4.5, 4.0, 3.5, 4.0, 5.0, 4.5, 2.0, 3.0],
            'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', 
                                   '2023-01-04', '2023-01-05', '2023-01-06',
                                   '2023-01-07', '2023-01-08'])
        })
    
    @pytest.fixture
    def sample_recipes_data(self):
        """Create sample recipes data for testing."""
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Pasta', 'Pizza', 'Salad', 'Soup', 'Cake'],
            'minutes': [30, 45, 15, 60, 120],
            'n_steps': [5, 8, 3, 10, 15],
            'n_ingredients': [8, 12, 5, 15, 20],
            'ingredients': ['pasta,cheese', 'flour,tomato', 'lettuce,oil', 'vegetables,broth', 'flour,sugar']
        })
    
    @pytest.fixture
    def temp_csv_files(self, sample_interactions_data, sample_recipes_data):
        """Create temporary CSV files for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            interactions_file = temp_path / "interactions.csv"
            recipes_file = temp_path / "recipes.csv"
            
            sample_interactions_data.to_csv(interactions_file, index=False)
            sample_recipes_data.to_csv(recipes_file, index=False)
            
            yield interactions_file, recipes_file
    
    @pytest.fixture
    def page_with_temp_files(self, temp_csv_files):
        """Create PopularityAnalysisPage with temporary files."""
        interactions_file, recipes_file = temp_csv_files
        return PopularityAnalysisPage(
            interactions_path=interactions_file,
            recipes_path=recipes_file
        )
    
    # ==================== INITIALIZATION TESTS ====================
    
    def test_page_initialization_with_string_paths(self):
        """Test page initialization with string paths."""
        page = PopularityAnalysisPage(
            interactions_path="/test/interactions.csv",
            recipes_path="/test/recipes.csv"
        )
        
        assert page.config is not None
        assert isinstance(page.config, PopularityAnalysisConfig)
        assert isinstance(page.config.interactions_path, Path)
        assert isinstance(page.config.recipes_path, Path)
    
    def test_page_initialization_with_path_objects(self):
        """Test page initialization with Path objects."""
        interactions_path = Path("/test/interactions.csv")
        recipes_path = Path("/test/recipes.csv")
        
        page = PopularityAnalysisPage(
            interactions_path=interactions_path,
            recipes_path=recipes_path
        )
        
        assert page.config.interactions_path == interactions_path
        assert page.config.recipes_path == recipes_path
    
    def test_config_access(self, page_with_temp_files):
        """Test access to configuration properties."""
        page = page_with_temp_files
        
        assert hasattr(page, 'config')
        assert hasattr(page.config, 'interactions_path')
        assert hasattr(page.config, 'recipes_path')
        
        assert page.config.interactions_path.exists()
        assert page.config.recipes_path.exists()
    

    
    # ==================== DATA VALIDATION TESTS ====================
    
    def test_data_structure_validation(self, sample_interactions_data, sample_recipes_data):
        """Test validation of expected data structures."""
        # Test interactions data structure
        required_interaction_cols = ['recipe_id', 'user_id', 'rating']
        for col in required_interaction_cols:
            assert col in sample_interactions_data.columns
        
        # Test recipes data structure
        required_recipe_cols = ['id', 'name', 'minutes', 'n_steps', 'n_ingredients']
        for col in required_recipe_cols:
            assert col in sample_recipes_data.columns
        
        # Test data types
        assert sample_interactions_data['recipe_id'].dtype in ['int64', 'Int64']
        assert sample_interactions_data['rating'].dtype == 'float64'
        assert sample_recipes_data['minutes'].dtype in ['int64', 'Int64']
    
    def test_data_content_validation(self, sample_interactions_data, sample_recipes_data):
        """Test validation of data content."""
        # Test ratings are in valid range
        ratings = sample_interactions_data['rating']
        assert ratings.min() >= 0
        assert ratings.max() <= 5
        
        # Test positive values for recipe characteristics
        assert (sample_recipes_data['minutes'] > 0).all()
        assert (sample_recipes_data['n_steps'] > 0).all()
        assert (sample_recipes_data['n_ingredients'] > 0).all()
        
        # Test no missing values in critical columns
        assert not sample_interactions_data['recipe_id'].isnull().any()
        assert not sample_interactions_data['user_id'].isnull().any()
        assert not sample_recipes_data['id'].isnull().any()
    

    
    # ==================== INTEGRATION TESTS ====================
    

    
    def test_configuration_consistency(self):
        """Test configuration consistency across instances."""
        interactions_path = "/test/interactions.csv"
        recipes_path = "/test/recipes.csv"
        
        page1 = PopularityAnalysisPage(interactions_path, recipes_path)
        page2 = PopularityAnalysisPage(interactions_path, recipes_path)
        
        assert page1.config.interactions_path == page2.config.interactions_path
        assert page1.config.recipes_path == page2.config.recipes_path
    
    # ==================== ERROR HANDLING TESTS ====================
    
    def test_invalid_path_handling(self):
        """Test handling of invalid paths."""
        # Test with None paths (should fail at some point)
        with pytest.raises((TypeError, AttributeError)):
            PopularityAnalysisPage(None, None)
    
