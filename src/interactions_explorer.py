from __future__ import (
    annotations,
)

"""Explorer minimal pour les interactions (popularité vs rating uniquement)."""
from typing import (
    Optional,
)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data_explorer import (
    DataExplorer,
)
from data_loader import (
    DataLoader,
)


class InteractionsExplorer(
    DataExplorer
):
    """Fonctionnalités réduites: analyse popularité vs note moyenne."""

    def __init__(self, df: Optional[pd.DataFrame] = None, loader: Optional[DataLoader] = None):  # type: ignore[name-defined]
        super().__init__(
            df=df,
            loader=loader,
        )

    # --- Popularité --- #
    def popularity_vs_rating(
        self,
    ) -> (
        pd.DataFrame
    ):
        """Return a DataFrame with recipe_id, interaction_count, avg_rating.

        Assumes columns 'recipe_id' and 'rating' (rating optional: if absent only counts returned).
        """
        if (
            "recipe_id"
            not in self.df.columns
        ):
            raise KeyError(
                "Column 'recipe_id' required for popularity analysis"
            )
        grp = self.df.groupby(
            "recipe_id"
        )
        counts = grp.size().rename(
            "interaction_count"
        )
        if (
            "rating"
            in self.df.columns
        ):
            avg_rating = (
                grp[
                    "rating"
                ]
                .mean()
                .rename(
                    "avg_rating"
                )
            )
            out = pd.concat(
                [
                    counts,
                    avg_rating,
                ],
                axis=1,
            ).reset_index()
        else:
            out = (
                counts.reset_index()
            )
        return out.sort_values(
            "interaction_count",
            ascending=False,
        )

    def plot_popularity_vs_rating(
        self,
        min_interactions: int = 1,
    ):
        """Scatter plot of average rating vs interaction count (popularity).

        Parameters
        ----------
        min_interactions : int
            Filter out recipes with fewer interaction_count than this threshold.
        """
        data = (
            self.popularity_vs_rating()
        )
        if (
            "avg_rating"
            not in data.columns
        ):
            raise ValueError(
                "No 'rating' column available to compute average ratings."
            )
        data_f = data[
            data[
                "interaction_count"
            ]
            >= min_interactions
        ]
        if (
            data_f.empty
        ):
            raise ValueError(
                "No data after filtering by min_interactions"
            )
        (
            fig,
            ax,
        ) = plt.subplots(
            figsize=(
                6,
                4,
            )
        )
        sns.scatterplot(
            data=data_f,
            x="interaction_count",
            y="avg_rating",
            ax=ax,
        )
        ax.set_xlabel(
            "Interactions (popularité)"
        )
        ax.set_ylabel(
            "Note moyenne"
        )
        ax.set_title(
            "Popularité vs Note moyenne"
        )
        return fig
