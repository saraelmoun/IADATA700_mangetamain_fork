from __future__ import (
    annotations,
)

"""Explorer recettes (diversité ingrédients & moyenne)."""
from typing import (
    Optional,
)
import re
import pandas as pd
from data_explorer import (
    DataExplorer,
)
from data_loader import (
    DataLoader,
)


class RecipesExplorer(
    DataExplorer
):
    """Fonctionnalités minimales autour des ingrédients."""

    def __init__(
        self,
        df: Optional[
            pd.DataFrame
        ] = None,
        loader: Optional[
            DataLoader
        ] = None,
    ):
        super().__init__(
            df=df,
            loader=loader,
        )

    def ingredient_diversity(
        self,
    ) -> int:
        col = None
        for candidate in [
            "ingredients",
            "ingredient_list",
            "raw_ingredients",
        ]:
            if (
                candidate
                in self.df.columns
            ):
                col = candidate
                break
        if (
            col
            is None
        ):
            raise KeyError(
                "No ingredient column found (expected one of: ingredients, ingredient_list, raw_ingredients)"
            )
        # Split on commas or pipes; remove empty tokens
        exploded = (
            self.df[
                col
            ]
            .dropna()
            .astype(
                str
            )
            .apply(
                lambda s: re.split(
                    r"[|,]",
                    s,
                )
            )
            .explode()
            .astype(
                str
            )
            .str.strip()
            .replace(
                "",
                pd.NA,
            )
            .dropna()
            .str.lower()
        )
        return (
            exploded.nunique()
        )

    def avg_ingredients_per_recipe(
        self,
    ) -> float:
        if (
            "ingredients"
            not in self.df.columns
        ):
            return float(
                "nan"
            )
        counts = (
            self.df[
                "ingredients"
            ]
            .dropna()
            .astype(
                str
            )
            .apply(
                lambda s: [
                    t.strip()
                    for t in re.split(
                        r"[|,]",
                        s,
                    )
                    if t.strip()
                ]
            )
        )
        counts = counts.apply(
            len
        )
        return (
            float(
                counts.mean()
            )
            if not counts.empty
            else float(
                "nan"
            )
        )
