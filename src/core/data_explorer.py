from __future__ import (
    annotations,
)

"""Explorateur minimal : accès aux données + rechargement.

Anciennes fonctions de statistiques et de visualisation retirées pour simplification.
"""
from typing import (
    Optional,
)
import pandas as pd
from .data_loader import (
    DataLoader,
)
from .logger import get_logger


class DataExplorer:
    def __init__(
        self,
        df: Optional[
            pd.DataFrame
        ] = None,
        loader: Optional[
            DataLoader
        ] = None,
    ) -> None:
        # Initialize logger first
        self.logger = get_logger()
        
        if (
            df
            is None
            and loader
            is None
        ):
            self.logger.error("No DataFrame or DataLoader provided")
            raise ValueError(
                "Provide either a DataFrame or a DataLoader"
            )
        self._df = df
        self.loader = loader
        
        # Log initialization details only if objects are valid
        if df is not None and hasattr(df, 'shape'):
            self.logger.debug(f"DataExplorer initialized with DataFrame: {df.shape}")
        elif loader is not None and hasattr(loader, 'data_path'):
            self.logger.debug(f"DataExplorer initialized with DataLoader: {loader.data_path}")
        else:
            self.logger.debug("DataExplorer initialized with basic objects")

    # ---------- Data Access ---------- #
    @property
    def df(
        self,
    ) -> (
        pd.DataFrame
    ):
        if (
            self._df
            is None
        ):
            if (
                self.loader
                is None
            ):
                # Handle case where logger might not be initialized (e.g., in tests)
                if hasattr(self, 'logger'):
                    self.logger.error("No data available and no loader configured")
                raise RuntimeError(
                    "No data available and no loader configured"
                )
            if hasattr(self, 'logger'):
                self.logger.debug("Loading data via DataLoader")
            self._df = (
                self.loader.load_data()
            )
            if hasattr(self, 'logger'):
                self.logger.info(f"Data exploration ready: {self._df.shape}")
        return (
            self._df
        )

    def reload(
        self,
        force: bool = True,
    ) -> (
        pd.DataFrame
    ):
        if (
            self.loader
            is None
        ):
            if hasattr(self, 'logger'):
                self.logger.error("Cannot reload without a DataLoader")
            raise RuntimeError(
                "Cannot reload without a DataLoader"
            )
        if hasattr(self, 'logger'):
            self.logger.info(f"Reloading data (force={force})")
        self._df = self.loader.load_data(
            force=force
        )
        if hasattr(self, 'logger'):
            self.logger.info(f"Data reloaded: {self._df.shape}")
        return (
            self._df
        )

    # Plus de méthodes analytiques ici — extension future possible.
