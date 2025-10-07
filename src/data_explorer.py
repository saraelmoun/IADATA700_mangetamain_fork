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
from data_loader import (
    DataLoader,
)


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
        if (
            df
            is None
            and loader
            is None
        ):
            raise ValueError(
                "Provide either a DataFrame or a DataLoader"
            )
        self._df = df
        self.loader = loader

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
                raise RuntimeError(
                    "No data available and no loader configured"
                )
            self._df = (
                self.loader.load_data()
            )
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
            raise RuntimeError(
                "Cannot reload without a DataLoader"
            )
        self._df = self.loader.load_data(
            force=force
        )
        return (
            self._df
        )

    # Plus de méthodes analytiques ici — extension future possible.
