from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import streamlit as st

from core.data_loader import DataLoader
from core.data_explorer import DataExplorer
from components.ingredients_clustering_page import IngredientsClusteringPage


DEFAULT_RECIPES = Path("data/RAW_recipes.csv")
DEFAULT_INTERACTIONS = Path("data/RAW_interactions.csv")


@dataclass
class AppConfig:
    default_recipes_path: Path = DEFAULT_RECIPES
    default_interactions_path: Path = DEFAULT_INTERACTIONS
    page_title: str = "Mangetamain - Analyse de Données"
    layout: str = "wide"


class App:
    """Application Streamlit pour l'analyse de données de recettes."""

    def __init__(self, config: AppConfig | None = None):
        self.config = config or AppConfig()

    def _sidebar(self) -> dict:
        """Configuration de la sidebar avec sélection des pages et datasets."""
        st.sidebar.header("Navigation")
        
        # Sélection de la page
        page = st.sidebar.selectbox(
            "Page",
            ["Home", "Analyse de clustering des ingrédients"],
            key="page_select_box",
        )

        if page == "Analyse de clustering des ingrédients":
            st.sidebar.markdown(f"### {page}")
            st.sidebar.caption("Clustering d'ingrédients basé sur la co-occurrence.")
            return {"page": page}

        # Configuration pour la page Home
        st.sidebar.markdown("### Configuration des données")
        
        # Sélection du dataset
        dataset_type = st.sidebar.radio(
            "Type de dataset",
            ["recettes", "interactions"],
            key="dataset_type",
        )

        # Chemin par défaut selon le type
        if dataset_type == "recettes":
            default_path = self.config.default_recipes_path
            st.sidebar.caption("Analyse dédiée aux recettes (RAW_recipes).")
        else:
            default_path = self.config.default_interactions_path
            st.sidebar.caption("Analyse des interactions utilisateur-recette.")

        # Options de rechargement
        refresh = st.sidebar.checkbox(
            "Forcer le rechargement", 
            value=False, 
            key="force_refresh"
        )

        return {
            "page": page,
            "path": default_path,
            "refresh": refresh,
            "active": dataset_type,
        }

    def run(self):
        """Point d'entrée principal de l'application."""
        st.set_page_config(
            page_title=self.config.page_title,
            layout=self.config.layout,
        )
        
        # Gestion du titre dynamique
        page = st.session_state.get("page_select_box", "Home")
        
        if page == "Analyse de clustering des ingrédients":
            st.title("🍳 Analyse de clustering des ingrédients")
        else:
            st.title("🏠 Home - Data Explorer")

        selection = self._sidebar()
        page = selection.get("page")

        # Logique des pages
        if page == "Analyse de clustering des ingrédients":
            clustering_page = IngredientsClusteringPage(
                str(self.config.default_recipes_path)
            )
            clustering_page.run()
            return

        # Page Home - Affichage des données avec exploration
        self._render_home_page(selection)

    def _render_home_page(self, selection: dict):
        """Rendu de la page d'accueil avec exploration des données."""
        data_path = selection["path"]
        refresh = selection["refresh"]
        dataset_type = selection["active"]

        loader = DataLoader(data_path)
        uploaded_df = None
        
        try:
            loader.load_data(force=refresh)
        except FileNotFoundError:
            st.warning(f"Fichier introuvable: {data_path}. Vous pouvez en téléverser un ci-dessous.")
            uploaded = st.file_uploader("Déposer un fichier CSV", type=["csv"], key="uploader")
            if uploaded is not None:
                import pandas as pd
                try:
                    tmp_df = pd.read_csv(uploaded)
                    uploaded_df = tmp_df
                    loader._df = tmp_df  # type: ignore[attr-defined]
                    st.success("Fichier chargé depuis l'upload.")
                except Exception as e:
                    st.error(f"Erreur lecture CSV uploadé: {e}")
                    return
            else:
                return
        except Exception as e:
            st.error(f"Erreur chargement données: {e}")
            return

        # Explorer de base pour tous les types de données
        explorer = DataExplorer(loader=loader)

        st.subheader("📋 Aperçu des données (10 premières lignes)")
        st.dataframe(explorer.df.head(10))

        # Affichage des informations de base
        st.subheader("📊 Informations sur le dataset")
        with st.expander("Informations générales", expanded=True):
            df = explorer.df
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Nombre de lignes", f"{len(df):,}")
                st.metric("Nombre de colonnes", len(df.columns))
            with col2:
                st.metric("Taille mémoire", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
                st.metric("Valeurs manquantes", f"{df.isnull().sum().sum():,}")
                
        with st.expander("Types de données"):
            st.dataframe(df.dtypes.to_frame("Type"))
            
        with st.expander("Analyse des colonnes clés"):
            # Analyse spécifique aux recettes si les colonnes existent
            if 'ingredients' in df.columns:
                st.write("🥘 **Ingrédients** :")
                # Compter les recettes avec ingrédients valides
                valid_ingredients = df['ingredients'].notna().sum()
                st.write(f"- Recettes avec ingrédients : {valid_ingredients:,}")
                
            if 'name' in df.columns:
                st.write("📝 **Noms de recettes** :")
                unique_names = df['name'].nunique()
                st.write(f"- Recettes uniques : {unique_names:,}")
                
            if 'minutes' in df.columns:
                st.write("⏱️ **Temps de préparation** :")
                avg_minutes = df['minutes'].mean()
                st.write(f"- Temps moyen : {avg_minutes:.1f} minutes")
                
            if 'n_steps' in df.columns:
                st.write("📋 **Étapes de préparation** :")
                avg_steps = df['n_steps'].mean()
                st.write(f"- Nombre moyen d'étapes : {avg_steps:.1f}")


def main():
    """Point d'entrée pour l'exécution directe via streamlit run."""
    App().run()


if __name__ == "__main__":
    main()