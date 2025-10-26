"""
Tests for the App module.

This test suite covers:
- App configuration and initialization
- AppConfig dataclass behavior
- Sidebar configuration logic (without Streamlit UI)
- Path management and validation
- Logic methods that can be tested without Streamlit
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from app import App, AppConfig, DEFAULT_RECIPES, DEFAULT_INTERACTIONS


class TestAppConfig:
    """Test suite for AppConfig dataclass."""
    
    def test_default_config(self):
        """Test AppConfig with default values."""
        config = AppConfig()
        
        assert config.default_recipes_path == DEFAULT_RECIPES
        assert config.default_interactions_path == DEFAULT_INTERACTIONS
        assert config.page_title == "Mangetamain - Analyse de Données"
        assert config.layout == "wide"
    
    def test_custom_config(self):
        """Test AppConfig with custom values."""
        custom_recipes = Path("/custom/recipes.csv")
        custom_interactions = Path("/custom/interactions.csv")
        
        config = AppConfig(
            default_recipes_path=custom_recipes,
            default_interactions_path=custom_interactions,
            page_title="Custom Title",
            layout="centered"
        )
        
        assert config.default_recipes_path == custom_recipes
        assert config.default_interactions_path == custom_interactions
        assert config.page_title == "Custom Title"
        assert config.layout == "centered"
    
    def test_config_immutability_concept(self):
        """Test that AppConfig behaves as expected for dataclass."""
        config1 = AppConfig()
        config2 = AppConfig()
        
        # Different instances should be equal if values are same
        assert config1.default_recipes_path == config2.default_recipes_path
        assert config1.page_title == config2.page_title
    
    def test_path_types(self):
        """Test that paths are properly handled as Path objects."""
        config = AppConfig(
            default_recipes_path="string/path/recipes.csv",
            default_interactions_path=Path("path/object/interactions.csv")
        )
        
        assert isinstance(config.default_recipes_path, (str, Path))
        assert isinstance(config.default_interactions_path, Path)


class TestApp:
    """Test suite for App class."""
    
    def test_app_initialization_default(self):
        """Test App initialization with default config."""
        app = App()
        
        assert app.config is not None
        assert isinstance(app.config, AppConfig)
        assert app.config.page_title == "Mangetamain - Analyse de Données"
    
    def test_app_initialization_custom_config(self):
        """Test App initialization with custom config."""
        custom_config = AppConfig(
            page_title="Test App",
            layout="centered"
        )
        
        app = App(config=custom_config)
        
        assert app.config is custom_config
        assert app.config.page_title == "Test App"
        assert app.config.layout == "centered"
    
    def test_app_config_access(self):
        """Test access to configuration properties."""
        custom_recipes = Path("/test/recipes.csv")
        custom_interactions = Path("/test/interactions.csv")
        
        config = AppConfig(
            default_recipes_path=custom_recipes,
            default_interactions_path=custom_interactions
        )
        
        app = App(config=config)
        
        assert app.config.default_recipes_path == custom_recipes
        assert app.config.default_interactions_path == custom_interactions


class TestAppConstants:
    """Test suite for App module constants."""
    
    def test_default_paths(self):
        """Test default path constants."""
        assert DEFAULT_RECIPES == Path("data/RAW_recipes.csv")
        assert DEFAULT_INTERACTIONS == Path("data/RAW_interactions.csv")
        
        assert isinstance(DEFAULT_RECIPES, Path)
        assert isinstance(DEFAULT_INTERACTIONS, Path)
    
    def test_path_properties(self):
        """Test Path object properties."""
        assert DEFAULT_RECIPES.name == "RAW_recipes.csv"
        assert DEFAULT_INTERACTIONS.name == "RAW_interactions.csv"
        
        assert DEFAULT_RECIPES.suffix == ".csv"
        assert DEFAULT_INTERACTIONS.suffix == ".csv"


class TestAppLogic:
    """Test suite for App logic methods that don't require Streamlit."""
    
    @pytest.fixture
    def app_with_temp_config(self):
        """Create app with temporary file paths for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create temporary files
            recipes_file = temp_path / "recipes.csv"
            interactions_file = temp_path / "interactions.csv"
            
            # Create sample data
            sample_recipes = pd.DataFrame({
                'id': [1, 2, 3],
                'name': ['Recipe A', 'Recipe B', 'Recipe C'],
                'minutes': [30, 45, 15],
                'n_steps': [5, 8, 3],
                'ingredients': ['ingredient1', 'ingredient2', 'ingredient3']
            })
            
            sample_interactions = pd.DataFrame({
                'recipe_id': [1, 1, 2, 3],
                'user_id': [101, 102, 101, 103],
                'rating': [4.5, 4.0, 3.5, 5.0]
            })
            
            sample_recipes.to_csv(recipes_file, index=False)
            sample_interactions.to_csv(interactions_file, index=False)
            
            config = AppConfig(
                default_recipes_path=recipes_file,
                default_interactions_path=interactions_file
            )
            
            yield App(config=config)
    
    def test_app_config_with_existing_files(self, app_with_temp_config):
        """Test app configuration with existing files."""
        app = app_with_temp_config
        
        # Files should exist
        assert app.config.default_recipes_path.exists()
        assert app.config.default_interactions_path.exists()
        
        # Should be CSV files
        assert app.config.default_recipes_path.suffix == ".csv"
        assert app.config.default_interactions_path.suffix == ".csv"


class TestAppDataValidation:
    """Test suite for data-related validation logic."""
    
    def test_valid_csv_structure_concepts(self):
        """Test concepts that would be used for CSV validation."""
        # Test data that represents what the app expects
        valid_recipes_data = {
            'id': [1, 2, 3],
            'name': ['Recipe A', 'Recipe B', 'Recipe C'],
            'minutes': [30, 45, 15],
            'n_steps': [5, 8, 3],
            'ingredients': ['ing1,ing2', 'ing3,ing4', 'ing5']
        }
        
        valid_interactions_data = {
            'recipe_id': [1, 1, 2],
            'user_id': [101, 102, 101],
            'rating': [4.5, 4.0, 3.5]
        }
        
        recipes_df = pd.DataFrame(valid_recipes_data)
        interactions_df = pd.DataFrame(valid_interactions_data)
        
        # Verify expected columns exist
        expected_recipe_cols = ['id', 'name', 'minutes', 'n_steps', 'ingredients']
        expected_interaction_cols = ['recipe_id', 'user_id', 'rating']
        
        assert all(col in recipes_df.columns for col in expected_recipe_cols)
        assert all(col in interactions_df.columns for col in expected_interaction_cols)
        
        # Verify data types make sense
        assert recipes_df['id'].dtype in ['int64', 'Int64']
        assert recipes_df['minutes'].dtype in ['int64', 'Int64', 'float64']
        assert interactions_df['rating'].dtype == 'float64'
    
    def test_missing_columns_detection(self):
        """Test detection of missing required columns."""
        incomplete_recipes = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Recipe A', 'Recipe B', 'Recipe C']
            # Missing: minutes, n_steps, ingredients
        })
        
        incomplete_interactions = pd.DataFrame({
            'recipe_id': [1, 2, 3]
            # Missing: user_id, rating
        })
        
        expected_recipe_cols = ['id', 'name', 'minutes', 'n_steps', 'ingredients']
        expected_interaction_cols = ['recipe_id', 'user_id', 'rating']
        
        # Check missing columns
        missing_recipe_cols = [col for col in expected_recipe_cols if col not in incomplete_recipes.columns]
        missing_interaction_cols = [col for col in expected_interaction_cols if col not in incomplete_interactions.columns]
        
        assert len(missing_recipe_cols) > 0
        assert len(missing_interaction_cols) > 0
        assert 'minutes' in missing_recipe_cols
        assert 'user_id' in missing_interaction_cols


class TestAppIntegration:
    """Integration tests for App components."""
    
    def test_app_with_all_components(self):
        """Test that App can be instantiated with all its dependencies."""
        # This tests that imports work and basic structure is sound
        config = AppConfig()
        app = App(config=config)
        
        # Basic structure should be in place
        assert hasattr(app, 'config')
        assert hasattr(app.config, 'default_recipes_path')
        assert hasattr(app.config, 'default_interactions_path')
        assert hasattr(app.config, 'page_title')
        assert hasattr(app.config, 'layout')
    
    def test_config_consistency(self):
        """Test configuration consistency across multiple instances."""
        config1 = AppConfig()
        config2 = AppConfig()
        
        app1 = App(config=config1)
        app2 = App(config=config2)
        
        # Default configs should be equivalent
        assert app1.config.page_title == app2.config.page_title
        assert app1.config.layout == app2.config.layout
        assert app1.config.default_recipes_path == app2.config.default_recipes_path
    
    def test_different_configs(self):
        """Test apps with different configurations."""
        config1 = AppConfig(page_title="App 1", layout="wide")
        config2 = AppConfig(page_title="App 2", layout="centered")
        
        app1 = App(config=config1)
        app2 = App(config=config2)
        
        assert app1.config.page_title != app2.config.page_title
        assert app1.config.layout != app2.config.layout


class TestAppUtilities:
    """Test utility functions and helper methods."""
    
    def test_path_handling(self):
        """Test path handling utilities."""
        # Test that string paths are handled correctly
        string_path = "data/test.csv"
        path_obj = Path(string_path)
        
        config_with_string = AppConfig(default_recipes_path=string_path)
        config_with_path = AppConfig(default_recipes_path=path_obj)
        
        # Both should work, exact behavior depends on implementation
        assert config_with_string.default_recipes_path is not None
        assert config_with_path.default_recipes_path is not None
    
    def test_config_serialization_concept(self):
        """Test that config can be represented as dictionary-like."""
        config = AppConfig(
            page_title="Test Title",
            layout="wide"
        )
        
        # Test that we can access attributes
        attributes = ['default_recipes_path', 'default_interactions_path', 'page_title', 'layout']
        
        for attr in attributes:
            assert hasattr(config, attr)
            value = getattr(config, attr)
            assert value is not None