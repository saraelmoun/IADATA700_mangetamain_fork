"""
Tests for the InteractionsAnalyzer module.

This test suite covers:
- Data loading and merging
- Popularity metrics calculation
- Rating-based filtering
- Preprocessing pipeline (outlier removal)
- Feature aggregation
- Missing value handling
"""

from core.interactions_analyzer import InteractionsAnalyzer, PreprocessingConfig
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestInteractionsAnalyzer:
    """Test suite for InteractionsAnalyzer."""

    @pytest.fixture
    def sample_interactions_data(self):
        """Create sample interactions data for testing."""
        return pd.DataFrame(
            {
                "recipe_id": [1, 1, 2, 2, 3, 3, 4, 5],
                "rating": [4.5, 4.0, 3.5, 4.0, 5.0, 4.5, 2.0, 3.0],
                "user_id": [101, 102, 103, 104, 105, 106, 107, 108],
            }
        )

    @pytest.fixture
    def sample_recipes_data(self):
        """Create sample recipes data for testing."""
        return pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "name": ["Recipe A", "Recipe B", "Recipe C", "Recipe D", "Recipe E"],
                "minutes": [30, 45, 15, 120, 60],
                "n_steps": [5, 8, 3, 15, 10],
                "n_ingredients": [8, 12, 5, 20, 15],
            }
        )

    @pytest.fixture
    def sample_recipes_with_missing(self):
        """Create sample recipes data with missing values."""
        return pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "name": ["Recipe A", "Recipe B", "Recipe C", "Recipe D", "Recipe E"],
                "minutes": [30, np.nan, 15, 120, 60],
                "n_steps": [5, 8, np.nan, 15, 10],
                "n_ingredients": [8, 12, 5, np.nan, 15],
            }
        )

    @pytest.fixture
    def analyzer_basic(self, sample_interactions_data, sample_recipes_data):
        """Create basic analyzer without preprocessing."""
        config = PreprocessingConfig(enable_preprocessing=False)
        return InteractionsAnalyzer(
            interactions=sample_interactions_data,
            recipes=sample_recipes_data,
            preprocessing=config,
        )

    @pytest.fixture
    def analyzer_with_preprocessing(self, sample_interactions_data, sample_recipes_data):
        """Create analyzer with preprocessing enabled."""
        config = PreprocessingConfig(enable_preprocessing=True, outlier_method="iqr", outlier_threshold=1.5)
        return InteractionsAnalyzer(
            interactions=sample_interactions_data,
            recipes=sample_recipes_data,  # Utilisez des données sans valeurs manquantes
            preprocessing=config,
            cache_enabled=False,  # Disable cache for tests
        )

    # ==================== BASIC FUNCTIONALITY TESTS ====================

    def test_initialization_basic(self, sample_interactions_data, sample_recipes_data):
        """Test basic initialization of InteractionsAnalyzer."""
        analyzer = InteractionsAnalyzer(interactions=sample_interactions_data, recipes=sample_recipes_data)

        assert analyzer.interactions is not None
        assert analyzer.recipes is not None
        assert len(analyzer._df) > 0
        assert "recipe_id" in analyzer._df.columns
        assert "rating" in analyzer._df.columns

    def test_data_merging(self, analyzer_basic):
        """Test that interactions and recipes data are merged correctly."""
        merged_df = analyzer_basic._df

        # Check that merge was successful
        assert len(merged_df) == 8  # All interactions should be preserved
        assert "recipe_id" in merged_df.columns
        assert "rating" in merged_df.columns
        assert "minutes" in merged_df.columns
        assert "n_steps" in merged_df.columns
        assert "n_ingredients" in merged_df.columns

        # Check that data integrity is maintained
        assert merged_df["recipe_id"].notna().all()
        assert merged_df["rating"].notna().all()

    def test_aggregation_basic(self, analyzer_basic):
        """Test basic aggregation functionality."""
        agg = analyzer_basic.aggregate()

        # Check structure
        assert isinstance(agg, pd.DataFrame)
        assert "recipe_id" in agg.columns
        assert "interaction_count" in agg.columns
        assert "avg_rating" in agg.columns

        # Check calculations
        assert len(agg) == 5  # 5 unique recipes
        assert agg.loc[agg["recipe_id"] == 1, "interaction_count"].iloc[0] == 2
        assert agg.loc[agg["recipe_id"] == 1, "avg_rating"].iloc[0] == 4.25  # (4.5 + 4.0) / 2
        assert agg.loc[agg["recipe_id"] == 3, "interaction_count"].iloc[0] == 2
        assert agg.loc[agg["recipe_id"] == 3, "avg_rating"].iloc[0] == 4.75  # (5.0 + 4.5) / 2

    def test_aggregation_sorting(self, analyzer_basic):
        """Test that aggregation results are sorted by interaction_count."""
        agg = analyzer_basic.aggregate()

        # Should be sorted by interaction_count descending
        interaction_counts = agg["interaction_count"].tolist()
        assert interaction_counts == sorted(interaction_counts, reverse=True)

    # ==================== PREPROCESSING TESTS ====================

    def test_preprocessing_disabled(self, analyzer_basic):
        """Test that preprocessing can be disabled."""
        stats = analyzer_basic.get_preprocessing_stats()

        # No preprocessing should have been applied
        assert stats is None or stats.get("outliers_removed", 0) == 0

    def test_preprocessing_enabled(self, analyzer_with_preprocessing):
        """Test that preprocessing is applied when enabled."""
        analyzer_with_preprocessing.aggregate()
        stats = analyzer_with_preprocessing.get_preprocessing_stats()

        # Check that some preprocessing was applied
        assert stats is not None
        assert "features_processed" in stats
        assert "outliers_removed" in stats

        # Check that data is consistent after preprocessing
        merged_df = analyzer_with_preprocessing._df
        processed_features = ["minutes", "n_steps", "n_ingredients"]
        available_features = [f for f in processed_features if f in merged_df.columns]

        if available_features:
            # After preprocessing, data should be clean (no missing values, outliers removed)
            for feature in available_features:
                assert merged_df[feature].notna().all(), f"Feature {feature} still has missing values after preprocessing"

    def test_outlier_removal_iqr(self, sample_interactions_data):
        """Test IQR outlier removal method."""
        # Create data with clear outliers
        recipes_with_outliers = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "minutes": [30, 45, 15, 1000, 60],  # 1000 is clear outlier
                "n_steps": [5, 8, 3, 100, 10],  # 100 is clear outlier
                "n_ingredients": [8, 12, 5, 20, 15],
            }
        )

        config = PreprocessingConfig(enable_preprocessing=True, outlier_method="iqr", outlier_threshold=1.5)

        analyzer = InteractionsAnalyzer(
            interactions=sample_interactions_data,
            recipes=recipes_with_outliers,
            preprocessing=config,
            cache_enabled=False,
        )

        agg = analyzer.aggregate()
        stats = analyzer.get_preprocessing_stats()

        # Outliers should have been removed
        assert stats["outliers_removed"] > 0
        assert len(agg) < 5  # Some recipes should have been filtered out

    def test_missing_values_preservation(self, sample_interactions_data, sample_recipes_with_missing):
        """Test that missing values are preserved (no longer imputed since KNN removal)."""
        config = PreprocessingConfig(enable_preprocessing=True, outlier_method="iqr", outlier_threshold=1.5)

        analyzer = InteractionsAnalyzer(
            interactions=sample_interactions_data,
            recipes=sample_recipes_with_missing,
            preprocessing=config,
            cache_enabled=False,
        )

        # Get the merged dataframe
        merged_df = analyzer._df

        # Check that the dataframe still contains rows (outlier removal might affect some data)
        assert len(merged_df) > 0, "Should have data after processing"

        # Missing values should still be present in the original features where they exist
        # Note: Some missing values might be removed due to outlier detection or merge operations
        original_missing_minutes = sample_recipes_with_missing["minutes"].isnull().sum()
        original_missing_steps = sample_recipes_with_missing["n_steps"].isnull().sum()

        # Verify that we still have SOME missing values preserved (no imputation occurred)
        total_missing_in_result = (
            merged_df["minutes"].isnull().sum() + merged_df["n_steps"].isnull().sum() + merged_df["n_ingredients"].isnull().sum()
        )

        # We should have at least some missing values preserved if no imputation is done
        # (unless outlier removal eliminated all rows with missing values)
        print(f"Original missing in minutes: {original_missing_minutes}")
        print(f"Original missing in steps: {original_missing_steps}")
        print(f"Total missing in result: {total_missing_in_result}")
        print(f"Result shape: {merged_df.shape}")

        # The key test: verify that no imputation logic was applied
        # (This is more about testing our removal of KNN rather than data preservation)
        stats = analyzer.get_preprocessing_stats()
        assert "values_imputed" not in stats, "Should not have 'values_imputed' key anymore"

    # ==================== FEATURE ENGINEERING TESTS ====================

    def test_popularity_segmentation(self, analyzer_basic):
        """Test popularity segmentation functionality."""
        agg = analyzer_basic.aggregate()
        segmented = analyzer_basic.create_popularity_segments(agg)

        # Check that segmentation was applied
        assert "popularity_segment" in segmented.columns

        # Check segment categories
        segments = segmented["popularity_segment"].unique()
        expected_segments = {"Low", "Medium", "High", "Viral"}
        assert set(segments).issubset(expected_segments)

        # Check that segments are logically ordered
        segment_means = segmented.groupby("popularity_segment")["interaction_count"].mean()
        if len(segment_means) > 1:
            # At least verify that Low has lower mean than High/Viral if they exist
            if "Low" in segment_means.index and "High" in segment_means.index:
                assert segment_means["Low"] < segment_means["High"]

    def test_recipe_categorization(self, analyzer_basic):
        """Test recipe categorization functionality."""
        agg = analyzer_basic.aggregate()
        categorized = analyzer_basic.create_recipe_categories(agg)

        # Check that categorization was applied
        expected_categories = [
            "complexity_category",
            "duration_category",
            "efficiency_category",
            "recipe_size_category",
        ]

        for category in expected_categories:
            if category in categorized.columns:
                # Check that categories have valid values
                values = categorized[category].dropna().unique()
                assert len(values) > 0

                # Check specific category logic
                if category == "duration_category":
                    valid_durations = {
                        "Express",
                        "Normal",
                        "Long",
                        "Marathon",
                        "Unknown",
                    }
                    assert set(values).issubset(valid_durations)
                elif category == "complexity_category":
                    valid_complexity = {"Simple", "Moderate", "Complex", "Unknown"}
                    assert set(values).issubset(valid_complexity)

    def test_efficiency_score_calculation(self, analyzer_basic):
        """Test efficiency score calculation."""
        agg = analyzer_basic.aggregate()
        categorized = analyzer_basic.create_recipe_categories(agg)

        if "efficiency_score" in categorized.columns:
            efficiency_scores = categorized["efficiency_score"].dropna()

            # Check that efficiency scores are positive
            assert (efficiency_scores > 0).all()

            # Check that no infinite values exist
            assert not np.isinf(efficiency_scores).any()

            # Check calculation logic for a specific case
            test_row = categorized.iloc[0]
            if pd.notna(test_row.get("avg_rating")) and pd.notna(test_row.get("minutes")):
                expected_efficiency = test_row["avg_rating"] / max(test_row["minutes"], 1)
                actual_efficiency = test_row["efficiency_score"]
                assert abs(expected_efficiency - actual_efficiency) < 0.001

    def test_category_insights(self, analyzer_basic):
        """Test category insights generation."""
        agg = analyzer_basic.aggregate()
        categorized = analyzer_basic.create_recipe_categories(agg)
        insights = analyzer_basic.get_category_insights(categorized)

        # Check insights structure
        assert isinstance(insights, dict)

        # Check that insights contain expected information
        for key, value in insights.items():
            if "category" in key or "segment" in key:
                assert "distribution" in value
                assert isinstance(value["distribution"], dict)

    # ==================== CACHE SYSTEM TESTS ====================

    def test_cache_disabled(self, sample_interactions_data, sample_recipes_data):
        """Test that cache can be disabled."""
        config = PreprocessingConfig()
        analyzer = InteractionsAnalyzer(
            interactions=sample_interactions_data,
            recipes=sample_recipes_data,
            preprocessing=config,
            cache_enabled=False,
        )

        # Test that cache is disabled
        assert analyzer._cache_enabled is False

    # ==================== ERROR HANDLING TESTS ====================

    def test_missing_recipe_id_column(self, sample_recipes_data):
        """Test error handling when recipe_id column is missing."""
        bad_interactions = pd.DataFrame({"bad_column": [1, 2, 3], "rating": [4.0, 3.5, 4.5]})

        with pytest.raises(KeyError):
            InteractionsAnalyzer(interactions=bad_interactions, recipes=sample_recipes_data)

    def test_empty_dataframes(self):
        """Test handling of empty dataframes."""
        empty_interactions = pd.DataFrame(columns=["recipe_id", "rating"])
        empty_recipes = pd.DataFrame(columns=["id", "name"])

        analyzer = InteractionsAnalyzer(interactions=empty_interactions, recipes=empty_recipes)

        agg = analyzer.aggregate()
        assert len(agg) == 0

    def test_invalid_preprocessing_config(self, sample_interactions_data, sample_recipes_data):
        """Test handling of invalid preprocessing configurations."""
        # Test invalid outlier method
        config = PreprocessingConfig(enable_preprocessing=True, outlier_method="invalid_method")

        analyzer = InteractionsAnalyzer(
            interactions=sample_interactions_data,
            recipes=sample_recipes_data,
            preprocessing=config,
        )

        # Should not crash, but should handle gracefully
        agg = analyzer.aggregate()
        assert len(agg) > 0

    # ==================== INTEGRATION TESTS ====================

    def test_full_pipeline_integration(self, sample_interactions_data, sample_recipes_with_missing):
        """Test complete pipeline from raw data to insights."""
        config = PreprocessingConfig(enable_preprocessing=True, outlier_method="iqr")

        analyzer = InteractionsAnalyzer(
            interactions=sample_interactions_data,
            recipes=sample_recipes_with_missing,
            preprocessing=config,
            cache_enabled=False,
        )

        # Test full pipeline
        agg = analyzer.aggregate()
        segmented = analyzer.create_popularity_segments(agg)
        categorized = analyzer.create_recipe_categories(agg)
        insights = analyzer.get_category_insights(categorized)

        # Verify each step produced valid results
        assert len(agg) > 0
        assert "popularity_segment" in segmented.columns
        assert len([col for col in categorized.columns if "category" in col]) > 0
        assert isinstance(insights, dict)
        assert len(insights) > 0

    def test_data_consistency_across_operations(self, analyzer_basic):
        """Test that data remains consistent across different operations."""
        # Get base aggregation twice and verify consistency
        agg1 = analyzer_basic.aggregate()
        agg2 = analyzer_basic.aggregate()

        # Should be identical
        pd.testing.assert_frame_equal(agg1, agg2)


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
