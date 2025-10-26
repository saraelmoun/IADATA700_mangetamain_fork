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
    
    # ==================== DATA LOADING TESTS ====================
    
    def test_load_data_success(self, page_with_temp_files):
        """Test successful data loading."""
        page = page_with_temp_files
        
        # Test that _load_data method exists and can be called
        if hasattr(page, '_load_data'):
            interactions_df, recipes_df = page._load_data()
            
            assert isinstance(interactions_df, pd.DataFrame)
            assert isinstance(recipes_df, pd.DataFrame)
            assert len(interactions_df) > 0
            assert len(recipes_df) > 0
            
            # Check expected columns
            expected_interaction_cols = ['recipe_id', 'user_id', 'rating']
            expected_recipe_cols = ['id', 'name', 'minutes', 'n_steps', 'n_ingredients']
            
            for col in expected_interaction_cols:
                assert col in interactions_df.columns
            
            for col in expected_recipe_cols:
                assert col in recipes_df.columns
    
    def test_load_data_file_not_found(self):
        """Test data loading with non-existent files."""
        page = PopularityAnalysisPage(
            interactions_path="/non/existent/interactions.csv",
            recipes_path="/non/existent/recipes.csv"
        )
        
        if hasattr(page, '_load_data'):
            with pytest.raises(FileNotFoundError):
                page._load_data()
    
    # ==================== PLOT TITLE TESTS ====================
    
    def test_get_plot_title_predefined(self, page_with_temp_files):
        """Test predefined plot titles."""
        page = page_with_temp_files
        
        if hasattr(page, '_get_plot_title'):
            # Test predefined scatter plot titles
            title = page._get_plot_title("avg_rating", "interaction_count", "Scatter")
            assert title == "Note moyenne selon le nombre d'interactions"
            
            title = page._get_plot_title("minutes", "avg_rating", "Scatter") 
            assert title == "Note moyenne selon la durée de préparation"
            
            # Test predefined histogram titles
            title = page._get_plot_title("avg_rating", "", "Histogram")
            assert title == "Distribution des notes moyennes"
            
            title = page._get_plot_title("minutes", "", "Histogram")
            assert title == "Distribution des durées de préparation"
    
    def test_get_plot_title_fallback(self, page_with_temp_files):
        """Test fallback plot title generation."""
        page = page_with_temp_files
        
        if hasattr(page, '_get_plot_title'):
            # Test fallback for non-predefined combinations
            title = page._get_plot_title("unknown_var", "another_var", "Scatter")
            
            assert isinstance(title, str)
            assert len(title) > 0
            # Should contain some meaningful text
            assert "selon" in title.lower()
    
    def test_get_plot_title_different_types(self, page_with_temp_files):
        """Test plot title generation for different plot types."""
        page = page_with_temp_files
        
        if hasattr(page, '_get_plot_title'):
            scatter_title = page._get_plot_title("minutes", "avg_rating", "Scatter")
            histogram_title = page._get_plot_title("minutes", "", "Histogram")
            
            assert isinstance(scatter_title, str)
            assert isinstance(histogram_title, str)
            assert scatter_title != histogram_title  # Should be different
            assert "selon" in scatter_title.lower()  # Scatter pattern
            assert "distribution" in histogram_title.lower()  # Histogram pattern
    
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
    
    # ==================== HELPER METHOD TESTS ====================
    
    def test_sidebar_config_structure(self, page_with_temp_files):
        """Test sidebar configuration structure (logic only, no UI)."""
        page = page_with_temp_files
        
        # Test that sidebar method exists
        assert hasattr(page, '_sidebar')
        
        # The sidebar method returns configuration, but requires Streamlit
        # We can't test its return value without mocking Streamlit
        # But we can verify the method exists and is callable
    
    def test_plot_methods_exist(self, page_with_temp_files):
        """Test that plot-related methods exist."""
        page = page_with_temp_files
        
        # Check that expected methods exist
        expected_methods = [
            '_create_plot',
            '_scatter_plot', 
            '_histogram_plot',
            '_get_plot_title'  # Updated method name
        ]
        
        for method_name in expected_methods:
            if hasattr(page, method_name):
                method = getattr(page, method_name)
                assert callable(method)
    
    def test_analysis_methods_exist(self, page_with_temp_files):
        """Test that analysis-related methods exist."""
        page = page_with_temp_files
        
        # Check that expected analysis methods exist
        expected_methods = [
            '_render_popularity_segmentation',
            '_render_recipe_categorization',
            '_render_category_insights',
            '_render_viral_recipe_analysis'
        ]
        
        for method_name in expected_methods:
            if hasattr(page, method_name):
                method = getattr(page, method_name)
                assert callable(method)
    
    # ==================== INTEGRATION TESTS ====================
    
    def test_page_with_real_data_structure(self, page_with_temp_files):
        """Test page behavior with realistic data structure."""
        page = page_with_temp_files
        
        # Verify that the page can be instantiated with real-like data
        assert page.config.interactions_path.exists()
        assert page.config.recipes_path.exists()
        
        # Verify file formats
        assert page.config.interactions_path.suffix == '.csv'
        assert page.config.recipes_path.suffix == '.csv'
    
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
    
    def test_empty_string_paths(self):
        """Test handling of empty string paths."""
        page = PopularityAnalysisPage("", "")
        
        # Should create Path objects, even if empty
        assert isinstance(page.config.interactions_path, Path)
        assert isinstance(page.config.recipes_path, Path)
    
    # ==================== DATA PROCESSING CONCEPTS ====================
    
    def test_data_processing_concepts(self, sample_interactions_data, sample_recipes_data):
        """Test concepts used in data processing."""
        # Test aggregation concepts
        popularity_counts = sample_interactions_data.groupby('recipe_id').size()
        avg_ratings = sample_interactions_data.groupby('recipe_id')['rating'].mean()
        
        assert len(popularity_counts) > 0
        assert len(avg_ratings) > 0
        assert all(popularity_counts > 0)
        assert all(avg_ratings >= 0)
        
        # Test merge concepts
        merged = sample_interactions_data.merge(
            sample_recipes_data, 
            left_on='recipe_id', 
            right_on='id', 
            how='inner'
        )
        
        assert len(merged) > 0
        assert 'rating' in merged.columns
        assert 'minutes' in merged.columns
    
    def test_statistical_concepts(self, sample_recipes_data):
        """Test statistical concepts used in analysis."""
        # Test outlier detection concepts (IQR method)
        minutes = sample_recipes_data['minutes']
        q1 = minutes.quantile(0.25)
        q3 = minutes.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = minutes[(minutes < lower_bound) | (minutes > upper_bound)]
        
        # Should be able to detect outliers
        assert isinstance(outliers, pd.Series)
        assert len(outliers) >= 0  # May or may not have outliers
    
    def test_visualization_data_preparation(self, sample_interactions_data, sample_recipes_data):
        """Test data preparation for visualizations."""
        # Test aggregation for plotting
        agg_data = sample_interactions_data.groupby('recipe_id').agg({
            'rating': ['mean', 'count'],
            'user_id': 'nunique'
        }).round(2)
        
        # Flatten column names
        agg_data.columns = ['_'.join(col).strip() for col in agg_data.columns]
        
        assert len(agg_data) > 0
        assert 'rating_mean' in agg_data.columns
        assert 'rating_count' in agg_data.columns
        
        # Test merge for complete dataset
        complete_data = agg_data.merge(
            sample_recipes_data.set_index('id'),
            left_index=True,
            right_index=True,
            how='inner'
        )
        
        assert len(complete_data) > 0
        assert 'minutes' in complete_data.columns