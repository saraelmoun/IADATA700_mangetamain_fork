from __future__ import (
    annotations,
)

"""Data loading and preprocessing utilities.

Responsibilities:
- Locate raw data sources (CSV, Parquet, SQL, API, etc.)
- Load into pandas DataFrame
- Apply basic preprocessing (renaming columns, type casting, filtering)
- Provide caching hooks
"""
from pathlib import (
    Path,
)
from typing import (
    Optional,
    Union,
)
import pandas as pd


class DataLoader:
    def __init__(
        self,
        data_path: Union[
            str,
            Path,
        ],
        cache: bool = True,
    ) -> None:
        self.data_path = Path(
            data_path
        )
        self.cache = cache
        self._df: Optional[
            pd.DataFrame
        ] = None

    def load_data(
        self,
        force: bool = False,
    ) -> (
        pd.DataFrame
    ):
        """Load data from the configured path.

        Parameters
        ----------
        force: bool
            If True, bypass cache even if already loaded.
        """
        if (
            self._df
            is not None
            and not force
        ):
            return (
                self._df
            )
        if (
            not self.data_path.exists()
        ):
            raise FileNotFoundError(
                f"Data file not found: {self.data_path}"
            )

        # Basic format dispatch (extend as needed)
        if (
            self.data_path.suffix
            == ".csv"
        ):
            df = pd.read_csv(
                self.data_path
            )
        elif (
            self.data_path.suffix
            in {
                ".parquet",
                ".pq",
            }
        ):
            df = pd.read_parquet(
                self.data_path
            )
        else:
            raise ValueError(
                f"Unsupported file format: {self.data_path.suffix}"
            )

        self._df = self.preprocess(
            df
        )
        return (
            self._df
        )

    def preprocess(
        self,
        df: pd.DataFrame,
    ) -> (
        pd.DataFrame
    ):
        """Apply light preprocessing steps. Override/customize as needed."""
        # Example: standardize column names
        df = (
            df.copy()
        )
        df.columns = [
            c.strip()
            .lower()
            .replace(
                " ",
                "_",
            )
            for c in df.columns
        ]
        return df

    def get_data(
        self,
    ) -> (
        pd.DataFrame
    ):
        if (
            self._df
            is None
        ):
            return (
                self.load_data()
            )
        return (
            self._df
        )
