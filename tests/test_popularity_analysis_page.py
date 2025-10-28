"""
Tests for PopularityAnalysisPage - New Test Suite
==================================================

Fresh test implementation focusing on simplicity and real functionality.
This test suite replaces the previous overly complex mock-based approach
with direct testing of actual methods and behaviors.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import warnings
from unittest.mock import Mock, MagicMock, patch

# Suppress warnings during testing
warnings.filterwarnings("ignore")

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from components.popularity_analysis_page import PopularityAnalysisPage, PopularityAnalysisConfig


class TestPopularityAnalysisConfig:
    """Test the configuration dataclass."""
    
    def test_config_creation(self):
        """Test basic config creation."""
        config = PopularityAnalysisConfig(
            interactions_path=Path("interactions.csv"),
            recipes_path=Path("recipes.csv")
        )
        assert config.interactions_path == Path("interactions.csv")
        assert config.recipes_path == Path("recipes.csv")
    
    def test_config_path_conversion(self):
        """Test that config handles string paths correctly."""
        config = PopularityAnalysisConfig(
            interactions_path="interactions.csv",
            recipes_path="recipes.csv"
        )
        # Note: PopularityAnalysisConfig is a simple dataclass
        # Path conversion happens in PopularityAnalysisPage.__init__
        assert config.interactions_path == "interactions.csv"
        assert config.recipes_path == "recipes.csv"


class TestPopularityAnalysisPage:
    """Test the main PopularityAnalysisPage class."""
    
    @pytest.fixture
    def sample_interactions_data(self):
        """Generate realistic interactions sample data."""
        np.random.seed(42)
        n_interactions = 1000
        
        data = {
            'user_id': np.random.randint(1, 101, n_interactions),
            'recipe_id': np.random.randint(1, 51, n_interactions),
            'rating': np.random.choice([1, 2, 3, 4, 5], n_interactions, p=[0.05, 0.1, 0.2, 0.35, 0.3]),
            'date': pd.date_range('2023-01-01', periods=n_interactions, freq='H')
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_recipes_data(self):
        """Generate realistic recipes sample data."""
        np.random.seed(42)
        n_recipes = 50
        
        data = {
            'id': range(1, n_recipes + 1),
            'name': [f"Recipe {i}" for i in range(1, n_recipes + 1)],
            'minutes': np.random.randint(5, 180, n_recipes),
            'n_steps': np.random.randint(1, 20, n_recipes),
            'n_ingredients': np.random.randint(2, 15, n_recipes)
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def temp_csv_files(self, sample_interactions_data, sample_recipes_data):
        """Create temporary CSV files for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            interactions_path = Path(tmpdir) / "interactions.csv"
            recipes_path = Path(tmpdir) / "recipes.csv"
            
            sample_interactions_data.to_csv(interactions_path, index=False)
            sample_recipes_data.to_csv(recipes_path, index=False)
            
            yield interactions_path, recipes_path
    
    @pytest.fixture
    def page_instance(self, temp_csv_files):
        """Create a PopularityAnalysisPage instance for testing."""
        interactions_path, recipes_path = temp_csv_files
        return PopularityAnalysisPage(interactions_path, recipes_path)
    
    def test_initialization(self, temp_csv_files):
        """Test basic page initialization."""
        interactions_path, recipes_path = temp_csv_files
        page = PopularityAnalysisPage(interactions_path, recipes_path)
        
        assert isinstance(page.config, PopularityAnalysisConfig)
        assert page.config.interactions_path == interactions_path
        assert page.config.recipes_path == recipes_path
        assert page.logger is not None
    
    def test_initialization_with_string_paths(self, temp_csv_files):
        """Test initialization with string paths."""
        interactions_path, recipes_path = temp_csv_files
        page = PopularityAnalysisPage(str(interactions_path), str(recipes_path))
        
        assert isinstance(page.config.interactions_path, Path)
        assert isinstance(page.config.recipes_path, Path)
    
    def test_load_data(self, page_instance):
        """Test data loading functionality."""
        interactions_df, recipes_df = page_instance._load_data()
        
        # Check data types
        assert isinstance(interactions_df, pd.DataFrame)
        assert isinstance(recipes_df, pd.DataFrame)
        
        # Check expected columns
        expected_interactions_cols = ['user_id', 'recipe_id', 'rating', 'date']
        expected_recipes_cols = ['id', 'name', 'minutes', 'n_steps', 'n_ingredients']
        
        for col in expected_interactions_cols:
            assert col in interactions_df.columns
        
        for col in expected_recipes_cols:
            assert col in recipes_df.columns
        
        # Check data integrity
        assert len(interactions_df) > 0
        assert len(recipes_df) > 0
    
    def test_get_plot_title(self, page_instance):
        """Test plot title generation."""
        # Test scatter plot titles
        title = page_instance._get_plot_title("avg_rating", "interaction_count", "Scatter")
        assert "Note moyenne selon le nombre d'interactions" == title
        
        title = page_instance._get_plot_title("minutes", "avg_rating", "Scatter")
        assert "Note moyenne selon la durée de préparation" == title
        
        # Test histogram titles
        title = page_instance._get_plot_title("avg_rating", "", "Histogram")
        assert "Distribution des notes moyennes" == title
        
        title = page_instance._get_plot_title("minutes", "", "Histogram")
        assert "Distribution des durées de préparation" == title
    
    def test_create_plot_scatter(self, page_instance):
        """Test scatter plot creation."""
        # Create test data
        data = pd.DataFrame({
            'x_var': np.random.normal(5, 2, 100),
            'y_var': np.random.normal(10, 3, 100),
            'size_var': np.random.uniform(1, 5, 100)
        })
        
        # Test scatter plot without size
        fig = page_instance._create_plot(
            data, x='x_var', y='y_var', plot_type='Scatter', alpha=0.6
        )
        
        assert isinstance(fig, plt.Figure)
        axes = fig.get_axes()
        assert len(axes) == 1
        
        # Close the figure to prevent memory issues
        plt.close(fig)
    
    def test_create_plot_histogram(self, page_instance):
        """Test histogram plot creation."""
        data = pd.DataFrame({
            'x_var': np.random.normal(5, 2, 100),
            'y_var': np.random.normal(10, 3, 100)
        })
        
        # Test histogram
        fig = page_instance._create_plot(
            data, x='x_var', y='y_var', plot_type='Histogram', n_bins=20
        )
        
        assert isinstance(fig, plt.Figure)
        axes = fig.get_axes()
        assert len(axes) == 1
        
        plt.close(fig)
    
    def test_sidebar_default_values(self, page_instance):
        """Test sidebar method returns expected structure."""
        # Mock streamlit sidebar components
        with patch('streamlit.sidebar.selectbox') as mock_selectbox, \
             patch('streamlit.sidebar.slider') as mock_slider, \
             patch('streamlit.sidebar.markdown'):
            
            # Set default return values
            mock_selectbox.return_value = "Scatter"
            mock_slider.side_effect = [0.6, 10.0]  # alpha, outlier_threshold
            
            params = page_instance._sidebar()
            
            expected_keys = ['plot_type', 'n_bins', 'bin_agg', 'alpha', 'outlier_threshold']
            assert all(key in params for key in expected_keys)
            assert params['plot_type'] == "Scatter"
            assert params['alpha'] == 0.6
            assert params['outlier_threshold'] == 10.0
    
    def test_formal_language_validation(self, page_instance):
        """Test that generated titles use formal language."""
        # Test various title combinations
        title1 = page_instance._get_plot_title("avg_rating", "interaction_count", "Scatter")
        title2 = page_instance._get_plot_title("minutes", "", "Histogram")
        
        # Should not contain informal language (whole words only)
        informal_words = [" tu ", " vous ", " on ", " votre ", " ton ", " ta "]
        
        for title in [title1, title2]:
            title_with_spaces = f" {title} "  # Add spaces for whole word matching
            for word in informal_words:
                assert word.lower() not in title_with_spaces.lower()
            
            # Should use formal vocabulary
            assert any(formal_word in title.lower() for formal_word in ["distribution", "selon", "moyenne"])


class TestIntegration:
    """Integration tests for the complete workflow."""
    
    def test_full_workflow_integration(self):
        """Test that the complete workflow can run end-to-end."""
        # Create minimal test data
        interactions_data = pd.DataFrame({
            'user_id': [1, 2, 3],
            'recipe_id': [1, 1, 2], 
            'rating': [4.0, 5.0, 3.0],
            'date': pd.date_range('2023-01-01', periods=3)
        })
        
        recipes_data = pd.DataFrame({
            'id': [1, 2],
            'name': ['Test Recipe 1', 'Test Recipe 2'],
            'minutes': [30, 45],
            'n_steps': [5, 8],
            'n_ingredients': [6, 10]
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            interactions_path = Path(tmpdir) / "interactions.csv"
            recipes_path = Path(tmpdir) / "recipes.csv"
            
            interactions_data.to_csv(interactions_path, index=False)
            recipes_data.to_csv(recipes_path, index=False)
            
            # Initialize page
            page = PopularityAnalysisPage(interactions_path, recipes_path)
            
            # Test data loading
            loaded_interactions, loaded_recipes = page._load_data()
            
            assert len(loaded_interactions) == 3
            assert len(loaded_recipes) == 2
            assert all(col in loaded_interactions.columns for col in ['user_id', 'recipe_id', 'rating'])
            assert all(col in loaded_recipes.columns for col in ['id', 'name', 'minutes'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])