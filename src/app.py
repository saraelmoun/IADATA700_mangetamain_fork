from __future__ import (
    annotations,
)

from pathlib import (
    Path,
)
from dataclasses import (
    dataclass,
)
import streamlit as st

from data_loader import (
    DataLoader,
)
from data_explorer import (
    DataExplorer,
)
from recipes_explorer import (
    RecipesExplorer,
)
from interactions_explorer import (
    InteractionsExplorer,
)

DEFAULT_RECIPES = Path(
    "data/RAW_recipes.csv"
)
DEFAULT_INTERACTIONS = Path(
    "data/RAW_interactions.csv"
)


@dataclass
class AppConfig:
    default_recipes_path: Path = DEFAULT_RECIPES
    default_interactions_path: Path = DEFAULT_INTERACTIONS
    page_title: str = "Data Explorer"
    layout: str = "wide"


class App:
    """High-level orchestrator for the Streamlit data exploration UI."""

    def __init__(
        self,
        config: (
            AppConfig
            | None
        ) = None,
    ):
        self.config = (
            config
            or AppConfig()
        )

    # Cached loader separated to allow instance usage
    @st.cache_data(
        show_spinner=False
    )
    def _load_dataframe(
        _self,
        path: Path,
    ):  # noqa: D401 - internal
        loader = DataLoader(
            path
        )
        return (
            loader.load_data()
        )

    # ---------- UI Sections ---------- #
    def _sidebar(
        self,
    ):
        """Render sidebar with two logical tabs (simulated via radio)
        and return selection.

        Streamlit ne supporte pas encore des tabs directement dans la sidebar, on simule
        donc deux onglets avec un composant radio horizontal. Les champs affichés
        dépendent de l'onglet actif.

        Returns
        -------
        dict: {"active": str, "path": Path, "refresh": bool}
        """
        st.sidebar.header(
            "Navigation"
        )
        page = st.sidebar.selectbox(
            "Page",
            [
                "Display",
                "Analysis1",
                "Analysis2",
            ],
            key="page_select_box",
        )

        if (
            page
            in {
                "Analysis1",
                "Analysis2",
            }
        ):
            st.sidebar.markdown(
                f"### {page}"
            )
            if (
                page
                == "Analysis1"
            ):
                st.sidebar.caption(
                    "Analyse dédiée aux recettes (RAW_recipes)."
                )
            else:
                st.sidebar.caption(
                    "Analyse combinée recettes + interactions."
                )
            return {
                "page": page
            }

        st.sidebar.markdown(
            "### Display configuration"
        )
        active = st.sidebar.radio(
            "Dataset",
            [
                "recettes",
                "interactions",
            ],
            horizontal=True,
            key="active_explorer_radio",
        )

        # On mémorise les deux chemins indépendamment dans session_state
        # Migration simple si anciens noms en minuscules utilisés
        if (
            "recipes_path"
            in st.session_state
            and "raw_recipes.csv"
            in st.session_state[
                "recipes_path"
            ].lower()
        ):
            st.session_state[
                "recipes_path"
            ] = str(
                self.config.default_recipes_path
            )
        if (
            "interactions_path"
            in st.session_state
            and "raw_interactions.csv"
            in st.session_state[
                "interactions_path"
            ].lower()
        ):
            st.session_state[
                "interactions_path"
            ] = str(
                self.config.default_interactions_path
            )
        if (
            "recipes_path"
            not in st.session_state
        ):
            st.session_state[
                "recipes_path"
            ] = str(
                self.config.default_recipes_path
            )
        if (
            "interactions_path"
            not in st.session_state
        ):
            st.session_state[
                "interactions_path"
            ] = str(
                self.config.default_interactions_path
            )

        if (
            active
            == "recettes"
        ):
            st.session_state[
                "recipes_path"
            ] = st.sidebar.text_input(
                "Fichier recettes",
                st.session_state[
                    "recipes_path"
                ],
                key="recipes_path_input",
            )
            refresh = st.sidebar.button(
                "Recharger recettes",
                key="reload_recettes",
            )
            current_path = Path(
                st.session_state[
                    "recipes_path"
                ]
            )
        else:
            st.session_state[
                "interactions_path"
            ] = st.sidebar.text_input(
                "Fichier interactions",
                st.session_state[
                    "interactions_path"
                ],
                key="interactions_path_input",
            )
            refresh = st.sidebar.button(
                "Recharger interactions",
                key="reload_interactions",
            )
            current_path = Path(
                st.session_state[
                    "interactions_path"
                ]
            )

        return {
            "page": page,
            "active": active,
            "path": current_path,
            "refresh": refresh,
        }

    def _render_summary(
        self,
        explorer: DataExplorer,
    ):
        # Simplified mode: no summary computation for now
        return None

    def _render_visualisations(
        self,
        explorer: DataExplorer,
        summary,
    ):
        # Disabled in simplified view
        return

    def _render_correlations(
        self,
        explorer: DataExplorer,
    ):
        # Disabled in simplified view
        return

    # ---------- Public API ---------- #
    def run(
        self,
    ):  # noqa: D401 - Streamlit entry
        st.set_page_config(
            page_title=self.config.page_title,
            layout=self.config.layout,
        )
        st.title(
            "🔎 Data Explorer"
        )

        selection = (
            self._sidebar()
        )
        page = selection.get(
            "page"
        )

        # Pages d'analyse (vides pour le moment)
        if (
            page
            == "Analysis1"
        ):
            st.subheader(
                "Analysis 1 - Recettes"
            )
            # Charger le dataset recettes
            loader_r = DataLoader(
                self.config.default_recipes_path
            )
            try:
                df_r = (
                    loader_r.load_data()
                )
            except Exception as e:
                st.error(
                    f"Impossible de charger les recettes: {e}"
                )
                st.stop()

            st.caption(
                "Aperçu recettes (5 premières lignes)"
            )
            st.dataframe(
                df_r.head(
                    5
                )
            )

            explorer_r = RecipesExplorer(
                df=df_r
            )
            (
                col1,
                col2,
            ) = st.columns(
                2
            )
            with col1:
                try:
                    diversity = (
                        explorer_r.ingredient_diversity()
                    )
                    st.metric(
                        "Diversité d'ingrédients",
                        f"{diversity}",
                    )
                except Exception as e:
                    st.warning(
                        f"Diversité indisponible: {e}"
                    )
            with col2:
                try:
                    avg_ing = (
                        explorer_r.avg_ingredients_per_recipe()
                    )
                    if (
                        avg_ing
                        == avg_ing
                    ):  # check not NaN
                        st.metric(
                            "Ingrédients moyens / recette",
                            f"{avg_ing:.2f}",
                        )
                    else:
                        st.metric(
                            "Ingrédients moyens / recette",
                            "N/A",
                        )
                except Exception as e:
                    st.warning(
                        f"Moyenne indisponible: {e}"
                    )
            st.stop()
        elif (
            page
            == "Analysis2"
        ):
            st.subheader(
                "Analysis 2 - Recettes + Interactions"
            )
            # Charger datasets nécessaires
            loader_r = DataLoader(
                self.config.default_recipes_path
            )
            loader_i = DataLoader(
                self.config.default_interactions_path
            )
            df_r = None
            df_i = None
            try:
                df_r = (
                    loader_r.load_data()
                )
            except Exception as e:
                st.warning(
                    f"Recettes indisponible: {e}"
                )
            try:
                df_i = (
                    loader_i.load_data()
                )
            except Exception as e:
                st.warning(
                    f"Interactions indisponible: {e}"
                )

            (
                col_r,
                col_i,
            ) = st.columns(
                2
            )
            with col_r:
                if (
                    df_r
                    is not None
                ):
                    st.caption(
                        "Aperçu recettes"
                    )
                    st.dataframe(
                        df_r.head(
                            5
                        )
                    )
            with col_i:
                if (
                    df_i
                    is not None
                ):
                    st.caption(
                        "Aperçu interactions"
                    )
                    st.dataframe(
                        df_i.head(
                            5
                        )
                    )

            # Analyse popularité vs note (interactions)
            if (
                df_i
                is not None
            ):
                st.markdown(
                    "---"
                )
                st.subheader(
                    "Popularité vs Note moyenne (Interactions)"
                )
                interactions_explorer = InteractionsExplorer(
                    df=df_i
                )
                min_int = st.slider(
                    "Filtrer recettes avec au moins N interactions",
                    min_value=1,
                    max_value=50,
                    value=1,
                    step=1,
                    key="min_interactions_slider_analysis2",
                )
                try:
                    fig = interactions_explorer.plot_popularity_vs_rating(
                        min_interactions=min_int
                    )
                    st.pyplot(
                        fig
                    )
                    with st.expander(
                        "Données agrégées"
                    ):
                        st.dataframe(
                            interactions_explorer.popularity_vs_rating().head(
                                50
                            )
                        )
                except Exception as e:
                    st.info(
                        f"Impossible d'afficher le graphique: {e}"
                    )
            else:
                st.info(
                    "Dataset interactions requis pour la visualisation."
                )

            st.stop()

        data_path = selection[
            "path"
        ]
        refresh = selection[
            "refresh"
        ]
        dataset_type = selection[
            "active"
        ]

        loader = DataLoader(
            data_path
        )
        uploaded_df = None
        try:
            loader.load_data(
                force=refresh
            )
        except FileNotFoundError:
            st.warning(
                f"Fichier introuvable: {data_path}. Vous pouvez en téléverser un ci-dessous."
            )
            uploaded = st.file_uploader(
                "Déposer un fichier CSV",
                type=[
                    "csv"
                ],
                key="uploader",
            )
            if (
                uploaded
                is not None
            ):
                import pandas as pd  # local import to avoid overhead otherwise

                try:
                    tmp_df = pd.read_csv(
                        uploaded
                    )
                    uploaded_df = tmp_df
                    # Inject uploaded df into loader instance
                    loader._df = tmp_df  # type: ignore[attr-defined]
                    st.success(
                        "Fichier chargé depuis l'upload."
                    )
                except Exception as e:
                    st.error(
                        f"Erreur lecture CSV uploadé: {e}"
                    )
                    st.stop()
            else:
                st.stop()
        except Exception as e:  # pragma: no cover - UI path
            st.error(
                f"Erreur chargement données: {e}"
            )
            st.stop()

        # Choix de l'explorer spécialisé (on passe le loader, DataFrame lazy via property)
        if (
            dataset_type
            == "recettes"
        ):
            explorer: DataExplorer = RecipesExplorer(
                loader=loader,
                df=uploaded_df,
            )
        elif (
            dataset_type
            == "interactions"
        ):
            explorer = InteractionsExplorer(
                loader=loader,
                df=uploaded_df,
            )
        else:  # fallback de sécurité
            explorer = DataExplorer(
                loader=loader
            )

        st.subheader(
            "Aperçu (10 premières lignes)"
        )
        st.dataframe(
            explorer.df.head(
                10
            )
        )

        # Plus d'analyse spécifique interactions ici (déplacé vers Analysis2)


def main():  # Retained for direct execution via streamlit run src/app.py
    App().run()


if (
    __name__
    == "__main__"
):  # pragma: no cover
    main()
