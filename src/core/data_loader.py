from __future__ import annotations

"""Data loading and preprocessing utilities.

Responsibilities:
- Locate raw data sources (CSV, Parquet, SQL, API, etc.)
- Load into pandas DataFrame
- Apply basic preprocessing (renaming columns, type casting, filtering)
- Provide caching hooks
"""
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from .logger import get_logger


class DataLoader:
    def __init__(
        self,
        data_path: Union[
            str,
            Path,
        ],
        cache: bool = True,
    ) -> None:
        self.data_path = Path(data_path)
        self.cache = cache
        self._df: Optional[pd.DataFrame] = None
        self.logger = get_logger()
        self.logger.debug(f"DataLoader initialized for {self.data_path}")

    def load_data(
        self,
        force: bool = False,
    ) -> pd.DataFrame:
        """Load data from the configured path.

        Parameters
        ----------
        force: bool
            If True, bypass cache even if already loaded.
        """
        if self._df is not None and not force:
            self.logger.debug("Returning cached data")
            return self._df
        if not self.data_path.exists():
            self.logger.error(f"Data file not found: {self.data_path}")
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        # Basic format dispatch (extend as needed)
        self.logger.info(f"Loading data from {self.data_path}")

        if self.data_path.suffix == ".csv":
            self.logger.debug("Loading CSV file")
            df = pd.read_csv(self.data_path)
        elif self.data_path.suffix in {
            ".parquet",
            ".pq",
        }:
            self.logger.debug("Loading Parquet file")
            df = pd.read_parquet(self.data_path)
        else:
            self.logger.error(f"Unsupported file format: {self.data_path.suffix}")
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")

        self._df = self.preprocess(df)

        self.logger.info(
            f"Data loaded successfully: {self._df.shape} rows/cols, "
            f"{self._df.memory_usage(deep=True).sum() / 1024**2:.1f} MB"
        )

        return self._df

    def preprocess(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Apply light preprocessing steps. Override/customize as needed."""
        self.logger.debug(f"Preprocessing data: {df.shape}")

        # Example: standardize column names
        df = df.copy()
        original_cols = list(df.columns)
        df.columns = [
            c.strip()
            .lower()
            .replace(
                " ",
                "_",
            )
            for c in df.columns
        ]

        if original_cols != list(df.columns):
            self.logger.debug("Column names standardized")

        return df

    def get_data(
        self,
    ) -> pd.DataFrame:
        if self._df is None:
            return self.load_data()
        return self._df
