from __future__ import annotations

"""Core analytics for recipe interactions (popularity & rating relationships).

This module was (re)created after the previous refactor removed the original
implementation inadvertently. It provides a focused analytical surface used by
Streamlit pages:
 - Popularity vs Rating
 - Rating vs structural features (minutes, n_steps, n_ingredients)
 - Popularity vs structural features

Design goals:
 - Pure computation (no Streamlit / plotting here)
 - Accept pre-loaded DataFrames (recipes, interactions) or a unified df
 - Graceful handling of missing columns
 - Light, dependency‑minimal (pandas only)

Public class: InteractionsAnalyzer
Backward compatibility alias: InteractionsExplorer
"""
from dataclasses import dataclass, field
from typing import Optional, Iterable
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import pickle
import hashlib
from pathlib import Path


RECIPE_ID_COL = "recipe_id"
RATING_COL = "rating"


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing options."""
    enable_preprocessing: bool = True
    outlier_method: str = "iqr"  # "iqr", "zscore", "none"
    outlier_threshold: float = 1.5  # IQR multiplier or Z-score threshold
    knn_neighbors: int = 5
    knn_features: list = field(default_factory=lambda: ["minutes", "n_steps", "n_ingredients", "rating"])
    enable_cache: bool = True
    cache_dir: str = "cache"
    
    def get_hash(self) -> str:
        """Generate hash for cache validation."""
        config_str = f"{self.enable_preprocessing}_{self.outlier_method}_{self.outlier_threshold}_{self.knn_neighbors}_{sorted(self.knn_features)}"
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


@dataclass
class InteractionsAnalyzer:
    """Compute relational aggregates between interactions & recipe metadata.

    Parameters
    ----------
    interactions : pd.DataFrame
        Raw interactions dataframe (expects at least recipe_id; may include rating)
    recipes : pd.DataFrame | None
        Recipes dataframe providing features (minutes, n_steps, ingredients...)
    merged : pd.DataFrame | None
        Pre‑merged dataframe (bypasses merge step if provided). If given it
        supersedes interactions/recipes arguments.
    """

    interactions: Optional[pd.DataFrame] = None
    recipes: Optional[pd.DataFrame] = None
    merged: Optional[pd.DataFrame] = None
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    
    def __post_init__(self) -> None:
        # Try to load from cache first
        if self.preprocessing.enable_cache:
            cached_df = self._load_from_cache()
            if cached_df is not None:
                self._df = cached_df
                self.preprocessing_stats = {"loaded_from_cache": True}
                return
                
        if self.merged is not None:
            self._df = self._standardize_cols(self.merged.copy())
        else:
            if self.interactions is None:
                raise ValueError("Provide either 'merged' or 'interactions' DataFrame")
            inter = self._standardize_cols(self.interactions.copy())
            if self.recipes is not None:
                rec = self._standardize_cols(self.recipes.copy())
                # Handle common alternate primary key naming ('id' -> 'recipe_id')
                if RECIPE_ID_COL not in rec.columns and 'id' in rec.columns:
                    rec = rec.rename(columns={'id': RECIPE_ID_COL})
                # prefer left join to keep only interactions that occurred
                if RECIPE_ID_COL in rec.columns:
                    self._df = inter.merge(rec, on=RECIPE_ID_COL, how="left", suffixes=("", "_r"))
                else:
                    self._df = inter
            else:
                self._df = inter

        # Derive n_ingredients if ingredients list present and column absent
        if "n_ingredients" not in self._df.columns:
            ingredient_col = self._detect_ingredients_column(self._df.columns)
            if ingredient_col:
                self._df["n_ingredients"] = self._df[ingredient_col].apply(self._safe_count_ingredients)

        # Apply preprocessing if enabled
        if self.preprocessing.enable_preprocessing:
            self._df, self.preprocessing_stats = self._preprocess_data(self._df)
        
        # Save to cache if enabled
        if self.preprocessing.enable_cache:
            self._save_to_cache(self._df)

    # ------------------ Internal helpers ------------------ #
    @staticmethod
    def _standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        return df

    @staticmethod
    def _detect_ingredients_column(cols: Iterable[str]) -> Optional[str]:
        for candidate in ["ingredients", "recipe_ingredient_parts", "ingredients_list"]:
            if candidate in cols:
                return candidate
        return None

    @staticmethod
    def _safe_count_ingredients(val) -> Optional[int]:
        if pd.isna(val):
            return None
        # Expect a stringified list or already a list
        try:
            if isinstance(val, list):
                return len(val)
            text = str(val)
            if text.startswith("[") and text.endswith("]"):
                # crude split on commas (avoid ast for speed/safety here)
                inside = text[1:-1].strip()
                if not inside:
                    return 0
                return sum(1 for _ in inside.split(","))
            # fallback: count semicolons/commas
            return len([p for p in text.split(",") if p.strip()])
        except Exception:
            return None

    # ------------------ Preprocessing methods ------------------ #
    def _preprocess_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """Apply preprocessing: outlier removal and KNN imputation.
        
        Returns:
            Tuple of (processed_dataframe, statistics_dict)
        """
        df_processed = df.copy()
        stats = {
            "original_rows": len(df),
            "outliers_removed": 0,
            "values_imputed": 0,
            "features_processed": []
        }
        
        # Get numerical features that exist in the dataframe
        available_features = [f for f in self.preprocessing.knn_features if f in df_processed.columns]
        stats["features_processed"] = available_features
        
        if not available_features:
            return df_processed, stats
        
        # 1. KNN imputation for missing values (first to have complete data)
        if any(df_processed[available_features].isnull().sum() > 0):
            df_processed, imputed_count = self._knn_imputation(df_processed, available_features)
            stats["values_imputed"] = imputed_count

        # 2. Outlier detection and removal (after imputation on complete data)
        if self.preprocessing.outlier_method != "none":
            df_processed, outliers_count = self._remove_outliers(df_processed, available_features)
            stats["outliers_removed"] = outliers_count
            
        stats["final_rows"] = len(df_processed)
        return df_processed, stats
    
    def _remove_outliers(self, df: pd.DataFrame, features: list) -> tuple[pd.DataFrame, int]:
        """Remove outliers using IQR or Z-score method."""
        outlier_mask = pd.Series(False, index=df.index)
        
        for feature in features:
            if feature not in df.columns:
                continue
                
            values = df[feature].dropna()
            if len(values) == 0:
                continue
                
            if self.preprocessing.outlier_method == "iqr":
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.preprocessing.outlier_threshold * IQR
                upper_bound = Q3 + self.preprocessing.outlier_threshold * IQR
                feature_outliers = (df[feature] < lower_bound) | (df[feature] > upper_bound)
                
            elif self.preprocessing.outlier_method == "zscore":
                z_scores = np.abs((df[feature] - values.mean()) / values.std())
                feature_outliers = z_scores > self.preprocessing.outlier_threshold
                
            else:
                continue
                
            outlier_mask |= feature_outliers.fillna(False)
        
        # Remove outliers
        df_clean = df[~outlier_mask].copy()
        outliers_removed = outlier_mask.sum()
        
        return df_clean, outliers_removed
    
    def _knn_imputation(self, df: pd.DataFrame, features: list) -> tuple[pd.DataFrame, int]:
        """Apply KNN imputation to missing values."""
        df_imputed = df.copy()
        
        # Count missing values before imputation
        missing_before = df[features].isnull().sum().sum()
        
        if missing_before == 0:
            return df_imputed, 0
        
        # Create imputer
        imputer = KNNImputer(n_neighbors=self.preprocessing.knn_neighbors)
        
        # Apply imputation only to the specified features
        feature_data = df[features].values
        imputed_data = imputer.fit_transform(feature_data)
        
        # Replace the values in the dataframe
        df_imputed[features] = imputed_data
        
        # Count missing values after imputation
        missing_after = df_imputed[features].isnull().sum().sum()
        values_imputed = missing_before - missing_after
        
        return df_imputed, values_imputed
    
    def get_preprocessing_stats(self) -> Optional[dict]:
        """Get preprocessing statistics if available."""
        return getattr(self, 'preprocessing_stats', None)
    
    # ------------------ Cache management ------------------ #
    def _get_cache_key(self) -> str:
        """Generate cache key based on data and preprocessing config."""
        # Create hash from interactions and recipes data shapes + config
        interactions_hash = hashlib.md5(str(len(self.interactions) if self.interactions is not None else 0).encode()).hexdigest()[:8]
        recipes_hash = hashlib.md5(str(len(self.recipes) if self.recipes is not None else 0).encode()).hexdigest()[:8]
        config_hash = self.preprocessing.get_hash()
        return f"interactions_{interactions_hash}_recipes_{recipes_hash}_config_{config_hash}"
    
    def _get_cache_path(self) -> Path:
        """Get cache file path."""
        cache_dir = Path(self.preprocessing.cache_dir)
        cache_dir.mkdir(exist_ok=True)
        cache_key = self._get_cache_key()
        return cache_dir / f"{cache_key}.pkl"
    
    def _load_from_cache(self) -> Optional[pd.DataFrame]:
        """Load preprocessed data from cache if available and valid."""
        try:
            cache_path = self._get_cache_path()
            if not cache_path.exists():
                return None
            
            # Check if cache is recent (less than 1 hour old)
            import time
            cache_age = time.time() - cache_path.stat().st_mtime
            if cache_age > 3600:  # 1 hour
                return None
            
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                
            # Validate cache structure
            if isinstance(cached_data, dict) and 'dataframe' in cached_data:
                return cached_data['dataframe']
            return None
            
        except Exception as e:
            # If any error, just don't use cache
            return None
    
    def _save_to_cache(self, df: pd.DataFrame) -> None:
        """Save preprocessed data to cache."""
        try:
            cache_path = self._get_cache_path()
            cache_data = {
                'dataframe': df,
                'preprocessing_config': self.preprocessing,
                'timestamp': pd.Timestamp.now(),
                'stats': getattr(self, 'preprocessing_stats', {})
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
                
        except Exception as e:
            # Silently fail - caching is optional
            pass
    
    def clear_cache(self) -> bool:
        """Clear all cache files for this analyzer."""
        try:
            cache_dir = Path(self.preprocessing.cache_dir)
            if cache_dir.exists():
                for cache_file in cache_dir.glob("*.pkl"):
                    cache_file.unlink()
                return True
        except Exception:
            pass
        return False
    
    def get_cache_info(self) -> dict:
        """Get information about cache status."""
        cache_path = self._get_cache_path()
        cache_dir = Path(self.preprocessing.cache_dir)
        
        info = {
            "cache_enabled": self.preprocessing.enable_cache,
            "cache_dir": str(cache_dir),
            "cache_key": self._get_cache_key(),
            "cache_exists": cache_path.exists(),
            "cache_files_count": len(list(cache_dir.glob("*.pkl"))) if cache_dir.exists() else 0
        }
        
        if cache_path.exists():
            import time
            stat = cache_path.stat()
            info.update({
                "cache_size_mb": stat.st_size / (1024*1024),
                "cache_age_minutes": (time.time() - stat.st_mtime) / 60,
                "cache_modified": pd.Timestamp.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            })
        
        return info

    # ------------------ Aggregations ------------------ #
    def aggregate(self) -> pd.DataFrame:
        """Return core aggregate per recipe.

        Metrics:
          - interaction_count
          - avg_rating (if ratings provided)
          - minutes / n_steps / n_ingredients (if present)
        """
        if RECIPE_ID_COL not in self._df.columns:
            raise KeyError(f"'{RECIPE_ID_COL}' column required for aggregation")

        grp = self._df.groupby(RECIPE_ID_COL)
        base = grp.size().rename("interaction_count")

        frames = [base]
        if RATING_COL in self._df.columns:
            frames.append(grp[RATING_COL].mean().rename("avg_rating"))
        for feature in ["minutes", "n_steps", "n_ingredients"]:
            if feature in self._df.columns:
                frames.append(grp[feature].mean().rename(feature))
        agg = pd.concat(frames, axis=1).reset_index()
        return agg.sort_values("interaction_count", ascending=False)

    # ------------------ Relationship helpers ------------------ #
    def _filter_min(self, df: pd.DataFrame, min_interactions: int) -> pd.DataFrame:
        if min_interactions <= 1:
            return df
        return df[df["interaction_count"] >= min_interactions]

    def popularity_vs_rating(self, min_interactions: int = 1) -> pd.DataFrame:
        agg = self.aggregate()
        if "avg_rating" not in agg.columns:
            raise ValueError("Ratings not available; cannot compute popularity_vs_rating")
        return self._filter_min(agg[[RECIPE_ID_COL, "interaction_count", "avg_rating"]], min_interactions)

    def rating_vs_feature(self, feature: str, min_interactions: int = 1) -> pd.DataFrame:
        if feature not in {"minutes", "n_steps", "n_ingredients"}:
            raise ValueError("Unsupported feature; choose among 'minutes', 'n_steps', 'n_ingredients'")
        agg = self.aggregate()
        needed = {feature, "avg_rating"}
        if not needed.issubset(agg.columns):
            missing = needed - set(agg.columns)
            raise ValueError(f"Missing columns for analysis: {missing}")
        subset = agg[[RECIPE_ID_COL, "interaction_count", feature, "avg_rating"]]
        subset = subset.dropna(subset=[feature, "avg_rating"])
        return self._filter_min(subset, min_interactions)

    def popularity_vs_feature(self, feature: str, min_interactions: int = 1) -> pd.DataFrame:
        if feature not in {"minutes", "n_steps", "n_ingredients"}:
            raise ValueError("Unsupported feature; choose among 'minutes', 'n_steps', 'n_ingredients'")
        agg = self.aggregate()
        needed = {feature}
        if not needed.issubset(agg.columns):
            raise ValueError(f"Missing feature column: {feature}")
        subset = agg[[RECIPE_ID_COL, "interaction_count", feature]]
        subset = subset.dropna(subset=[feature])
        return self._filter_min(subset, min_interactions)
    
    # ------------------ Feature Engineering Methods ------------------ #
    
    def create_popularity_segments(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """Create popularity segments based on interaction_count."""
        if df is None:
            df = self.aggregate()
        
        df = df.copy()
        
        # Calculate percentiles for intelligent thresholds
        percentiles = df['interaction_count'].quantile([0.25, 0.50, 0.75, 0.90, 0.95])
        
        # Define segments with data-driven thresholds
        def assign_popularity_segment(interaction_count):
            if interaction_count <= percentiles[0.25]:
                return "Low"
            elif interaction_count <= percentiles[0.75]:
                return "Medium" 
            elif interaction_count <= percentiles[0.95]:
                return "High"
            else:
                return "Viral"
        
        df['popularity_segment'] = df['interaction_count'].apply(assign_popularity_segment)
        
        # Add segment statistics
        segment_stats = df.groupby('popularity_segment').agg({
            'interaction_count': ['count', 'mean', 'min', 'max'],
            'avg_rating': ['mean', 'std']
        }).round(2)
        
        # Store segment info for reporting
        self._popularity_segments_info = {
            'thresholds': {
                'low_max': percentiles[0.25],
                'medium_max': percentiles[0.75], 
                'high_max': percentiles[0.95]
            },
            'stats': segment_stats
        }
        
        return df
    
    def create_recipe_categories(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """Create sophisticated categorization based on recipe characteristics."""
        if df is None:
            df = self.aggregate()
            
        df = df.copy()
        
        # 1. Complexity categorization (based on steps + ingredients)
        if 'n_steps' in df.columns and 'n_ingredients' in df.columns:
            df['complexity_score'] = df['n_steps'] + df['n_ingredients'] * 0.5
            complexity_percentiles = df['complexity_score'].quantile([0.33, 0.67])
            
            def assign_complexity(score):
                if pd.isna(score):
                    return "Unknown"
                elif score <= complexity_percentiles[0.33]:
                    return "Simple"
                elif score <= complexity_percentiles[0.67]:
                    return "Moderate"
                else:
                    return "Complex"
            
            df['complexity_category'] = df['complexity_score'].apply(assign_complexity)
        
        # 2. Duration categorization
        if 'minutes' in df.columns:
            def assign_duration(minutes):
                if pd.isna(minutes):
                    return "Unknown"
                elif minutes <= 15:
                    return "Express"
                elif minutes <= 45:
                    return "Normal"
                elif minutes <= 120:
                    return "Long"
                else:
                    return "Marathon"
            
            df['duration_category'] = df['minutes'].apply(assign_duration)
        
        # 3. Efficiency score (rating per minute)
        if 'avg_rating' in df.columns and 'minutes' in df.columns:
            # Handle edge cases: use max(minutes, 1) to avoid division by 0 while keeping logic correct
            df['efficiency_score'] = df['avg_rating'] / df['minutes'].clip(lower=1)
            efficiency_percentiles = df['efficiency_score'].quantile([0.25, 0.75])
            
            def assign_efficiency(score):
                if pd.isna(score):
                    return "Unknown"
                elif score <= efficiency_percentiles[0.25]:
                    return "Low Efficiency"
                elif score <= efficiency_percentiles[0.75]:
                    return "Medium Efficiency"
                else:
                    return "High Efficiency"
            
            df['efficiency_category'] = df['efficiency_score'].apply(assign_efficiency)
        
        # 4. Recipe size categorization (based on ingredients)
        if 'n_ingredients' in df.columns:
            def assign_recipe_size(n_ingredients):
                if pd.isna(n_ingredients):
                    return "Unknown"
                elif n_ingredients <= 5:
                    return "Minimal"
                elif n_ingredients <= 10:
                    return "Standard"
                elif n_ingredients <= 15:
                    return "Rich"
                else:
                    return "Elaborate"
            
            df['recipe_size_category'] = df['n_ingredients'].apply(assign_recipe_size)
        
        return df
    
    def get_category_insights(self, df: pd.DataFrame = None) -> dict:
        """Get insights and statistics about the created categories."""
        if df is None:
            df = self.aggregate()
            df = self.create_recipe_categories(df)
        
        insights = {}
        
        # Popularity segments analysis
        if 'popularity_segment' in df.columns:
            insights['popularity_segments'] = {
                'distribution': df['popularity_segment'].value_counts().to_dict(),
                'avg_rating_by_segment': df.groupby('popularity_segment')['avg_rating'].mean().round(2).to_dict(),
                'thresholds': getattr(self, '_popularity_segments_info', {}).get('thresholds', {})
            }
        
        # Category correlations
        categorical_cols = [col for col in df.columns if 'category' in col or 'segment' in col]
        
        for cat_col in categorical_cols:
            if cat_col in df.columns:
                insights[cat_col] = {
                    'distribution': df[cat_col].value_counts().to_dict(),
                    'avg_rating_by_category': df.groupby(cat_col)['avg_rating'].mean().round(2).to_dict()
                }
        
        return insights


# Backward compatibility alias (older code may import InteractionsExplorer)
InteractionsExplorer = InteractionsAnalyzer
