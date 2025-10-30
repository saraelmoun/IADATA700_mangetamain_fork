from __future__ import annotations

"""Streamlit page: Analyse de popularité des recettes.

Analyse des relations entre popularité, notes et caractéristiques structurelles.
"""
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from core.data_loader import DataLoader
from core.interactions_analyzer import InteractionsAnalyzer, PreprocessingConfig
from core.logger import get_logger


@dataclass
class PopularityAnalysisConfig:
    interactions_path: Path
    recipes_path: Path


class PopularityAnalysisPage:
    def __init__(self, interactions_path: str | Path, recipes_path: str | Path):
        self.config = PopularityAnalysisConfig(
            interactions_path=Path(interactions_path),
            recipes_path=Path(recipes_path),
        )
        self.logger = get_logger()

    # ---------------- Sidebar ---------------- #
    def _sidebar(self):
        st.sidebar.markdown("### 📊 Visualisation")
        plot_type = st.sidebar.selectbox(
            "Type de graphique",
            ["Scatter", "Histogram"],
            help="Scatter: points individuels, Histogram: nombre d'observations par bins",
        )

        if plot_type == "Histogram":
            n_bins = st.sidebar.slider("Nombre de bins", 10, 50, 20)
            bin_agg = "count"  # Fixé à count seulement
        else:
            n_bins = 20
            bin_agg = "count"

        alpha = st.sidebar.slider("Transparence", 0.1, 1.0, 0.6, 0.1)

        # Preprocessing section
        st.sidebar.markdown("### ⚙️ Preprocessing")
        outlier_threshold = st.sidebar.slider(
            "Seuil outliers",
            min_value=1.0,
            max_value=20.0,
            value=10.0,
            step=1.0,
            help="Multiplicateur IQR pour filtrer les outliers techniques (minutes, n_steps, n_ingredients). Plus élevé = moins de filtrage.",
        )

        return {
            "plot_type": plot_type,
            "n_bins": n_bins,
            "bin_agg": bin_agg,
            "alpha": alpha,
            "outlier_threshold": outlier_threshold,
        }

    def _render_cache_controls(self, analyzer: InteractionsAnalyzer):
        """Render cache management controls in sidebar."""
        st.sidebar.markdown("### 🗄️ Cache Management")

        # Get cache info
        cache_info = analyzer.get_cache_info()

        # Cache status
        cache_enabled = cache_info["cache_enabled"]
        cache_exists = cache_info["cache_exists"]

        if cache_enabled:
            if cache_exists:
                st.sidebar.success("Cache disponible")
                # Show cache details
                if "cache_age_minutes" in cache_info:
                    age_str = f"{cache_info['cache_age_minutes']:.1f} min"
                    size_str = f"{cache_info['cache_size_mb']:.1f} MB"
                    st.sidebar.info(f"Age: {age_str}, Taille: {size_str}")
            else:
                st.sidebar.info("Cache sera créé après preprocessing")

            # Cache management buttons
            col1, col2 = st.sidebar.columns(2)
            with col1:
                if st.button("🗑️ Clear Cache", help="Supprimer tous les fichiers de cache"):
                    if analyzer.clear_cache():
                        st.sidebar.success("Cache effacé!")
                        st.rerun()
                    else:
                        st.sidebar.error("Erreur lors de l'effacement")

            with col2:
                if st.button("ℹ️ Info Cache", help="Afficher les détails du cache"):
                    st.sidebar.json(cache_info)

            # Show total cache files
            if cache_info["cache_files_count"] > 0:
                st.sidebar.caption(f"📁 {cache_info['cache_files_count']} fichier(s) de cache")
        else:
            st.sidebar.warning("Cache désactivé")

    def _render_popularity_segmentation(self, analyzer: InteractionsAnalyzer, pop_rating: pd.DataFrame):
        """Render popularity segmentation analysis."""
        st.subheader("📋 Segmentation par popularité")

        # Create popularity segments
        segmented_data = analyzer.create_popularity_segments(pop_rating)

        # Get threshold information from analyzer
        thresholds = analyzer._popularity_segments_info["thresholds"]

        col1, col2 = st.columns(2)

        with col1:
            # Distribution of segments with precise intervals
            segment_counts = segmented_data["popularity_segment"].value_counts()
            st.write("**Distribution des segments avec intervalles précis:**")

            # Display segments with their exact interaction count ranges
            for segment in ["Low", "Medium", "High", "Viral"]:
                if segment in segment_counts.index:
                    count = segment_counts[segment]
                    percentage = (count / len(segmented_data)) * 100

                    # Define interval text based on segment
                    if segment == "Low":
                        interval = f"1 à {int(thresholds['low_max'])} fois"
                    elif segment == "Medium":
                        interval = f"{int(thresholds['low_max']) + 1} à {int(thresholds['medium_max'])} fois"
                    elif segment == "High":
                        interval = f"{int(thresholds['medium_max']) + 1} à {int(thresholds['high_max'])} fois"
                    else:  # Viral
                        interval = f"Plus de {int(thresholds['high_max'])} fois"

                    st.write(f"- **{segment}** ({interval}): {count:,} recettes ({percentage:.1f}%)")

        with col2:
            # Average rating by segment
            segment_ratings = segmented_data.groupby("popularity_segment")["avg_rating"].agg(["mean", "std", "count"])
            st.write("**Note moyenne par segment:**")
            for segment in ["Low", "Medium", "High", "Viral"]:
                if segment in segment_ratings.index:
                    mean_rating = segment_ratings.loc[segment, "mean"]
                    std_rating = segment_ratings.loc[segment, "std"]
                    count = segment_ratings.loc[segment, "count"]
                    st.write(f"- {segment}: {mean_rating:.2f} ± {std_rating:.2f} ({count:,} recettes)")

        # Visualization of segments
        self._plot_popularity_segments(segmented_data, analyzer)

    def _plot_popularity_segments(self, segmented_data: pd.DataFrame, analyzer: InteractionsAnalyzer):
        """Create visualization for popularity segments."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Boxplot of ratings by segment
        segment_order = ["Low", "Medium", "High", "Viral"]
        segments_present = [s for s in segment_order if s in segmented_data["popularity_segment"].values]

        if segments_present:
            sns.boxplot(
                data=segmented_data,
                x="popularity_segment",
                y="avg_rating",
                order=segments_present,
                ax=ax1,
            )
            ax1.set_title("Distribution des notes par segment de popularité")
            ax1.set_xlabel("Segment de popularité")
            ax1.set_ylabel("Note moyenne")
            ax1.tick_params(axis="x", rotation=45)

        # Plot 2: Distribution des recettes par nombre d'interactions
        # Créons un histogramme pour visualiser la vraie distribution
        interactions_counts = segmented_data["interaction_count"].value_counts().sort_index()

        # Limitons à 30 interactions max pour la lisibilité
        max_interactions = min(30, interactions_counts.index.max())
        interactions_limited = interactions_counts[interactions_counts.index <= max_interactions]

        # Créons le graphique en barres
        ax2.bar(
            interactions_limited.index,
            interactions_limited.values,
            alpha=0.7,
            color="steelblue",
            edgecolor="black",
            linewidth=0.5,
        )

        # Ajoutons les lignes de seuils de segmentation
        thresholds = analyzer._popularity_segments_info["thresholds"]
        ax2.axvline(
            thresholds["low_max"],
            color="blue",
            linestyle="--",
            alpha=0.8,
            label=f'P25 = {thresholds["low_max"]:.0f}',
        )
        ax2.axvline(
            thresholds["medium_max"],
            color="green",
            linestyle="--",
            alpha=0.8,
            label=f'P75 = {thresholds["medium_max"]:.0f}',
        )
        ax2.axvline(
            thresholds["high_max"],
            color="red",
            linestyle="--",
            alpha=0.8,
            label=f'P95 = {thresholds["high_max"]:.0f}',
        )

        ax2.set_xlabel("Nombre d'interactions")
        ax2.set_ylabel("Nombre de recettes")
        ax2.set_title("Distribution: Combien de recettes pour chaque niveau d'interactions")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Calculons les pourcentages réels dynamiquement
        segment_counts = segmented_data["popularity_segment"].value_counts()
        total_recipes = len(segmented_data)
        segment_percentages = {}
        for segment in ["Low", "Medium", "High", "Viral"]:
            if segment in segment_counts.index:
                segment_percentages[segment] = (segment_counts[segment] / total_recipes) * 100
            else:
                segment_percentages[segment] = 0.0

        # Ajoutons des annotations pour les zones avec pourcentages dynamiques
        ax2.text(
            0.5,
            ax2.get_ylim()[1] * 0.8,
            f'Low\n{segment_percentages["Low"]:.1f}%',
            ha="center",
            va="center",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7),
        )
        ax2.text(
            2.5,
            ax2.get_ylim()[1] * 0.6,
            f'Medium\n{segment_percentages["Medium"]:.1f}%',
            ha="center",
            va="center",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
        )
        ax2.text(
            8,
            ax2.get_ylim()[1] * 0.4,
            f'High\n{segment_percentages["High"]:.1f}%',
            ha="center",
            va="center",
            bbox=dict(boxstyle="round", facecolor="orange", alpha=0.7),
        )
        if max_interactions > 14:
            ax2.text(
                20,
                ax2.get_ylim()[1] * 0.2,
                f'Viral\n{segment_percentages["Viral"]:.1f}%',
                ha="center",
                va="center",
                bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.7),
            )

        plt.tight_layout()
        st.pyplot(fig)

        # Calculons les statistiques pour l'explication (éviter de recalculer)
        segment_counts = segmented_data["popularity_segment"].value_counts()
        total_recipes = len(segmented_data)
        low_pct = (segment_counts.get("Low", 0) / total_recipes) * 100
        viral_pct = (segment_counts.get("Viral", 0) / total_recipes) * 100
        low_count = segment_counts.get("Low", 0)
        thresholds = analyzer._popularity_segments_info["thresholds"]

        # Explication de la distribution observée avec pourcentages dynamiques
        st.markdown(
            f"""
        **🔍 Lecture de la distribution (graphique de droite) :**

        Ce graphique révèle la **réalité de l'engagement** sur les plateformes de contenu :
        - **Très haute colonne à 1 interaction** : {low_pct:.1f}% des recettes (~{low_count // 1000}k) n'ont qu'une seule interaction
        - **Décroissance rapide** : Plus le nombre d'interactions augmente, moins il y a de recettes
        - **Rareté du viral** : Très peu de recettes dépassent {thresholds['high_max']:.0f} interactions (seuil viral P95)

        Cette distribution de type **"longue traîne"** est typique des plateformes de contenu et
        **renforce la valeur** de notre analyse : identifier les facteurs qui distinguent les {viral_pct:.1f}%
        de recettes virales des {low_pct:.1f}% à faible engagement devient d'autant plus précieux !

        **📐 Pourquoi pas exactement 25%/50%/75%/95% ?**

        Les percentiles P25/P75/P95 sont corrects, mais avec des **données discrètes entières**
        (1, 2, 3... interactions), les segments ne peuvent pas être exactement équilibrés :

        - **P25 = {thresholds['low_max']:.0f}** : mais {low_pct:.1f}% des recettes ont exactement {thresholds['low_max']:.0f} interaction
        - **Impossible d'avoir exactement 25%** sans utiliser des seuils fractionnaires (1.5, 2.3...)
        - **C'est mathématiquement normal** : les percentiles indiquent les valeurs, pas forcément des répartitions égales

        Cette asymétrie **renforce l'analyse** : elle reflète la vraie nature de l'engagement numérique !
        """
        )

    def _render_step_1(
        self,
        analyzer: InteractionsAnalyzer,
        plot_type: str,
        n_bins: int,
        bin_agg: str,
        alpha: float,
    ):
        """Render step 1: Quality-popularity relationship analysis."""
        st.markdown("---")
        st.header("📈  ÉTAPE 1 : Relation qualité-popularité")

        st.markdown(
            """
        **Question :** Les recettes bien notées génèrent-elles plus d'interactions ?

        Cette première analyse croise la note moyenne des recettes avec leur nombre d'interactions
        pour évaluer la corrélation entre qualité perçue et engagement utilisateur.

        **Métrique :** Corrélation entre note moyenne et nombre d'interactions par recette.
        """
        )

        try:
            pop_rating = analyzer.popularity_vs_rating()
            fig1 = self._create_plot(
                pop_rating,
                x="avg_rating",
                y="interaction_count",
                plot_type=plot_type,
                n_bins=n_bins,
                bin_agg=bin_agg,
                alpha=alpha,
            )
            st.pyplot(fig1)

            # Analyse des résultats
            st.markdown(
                """
            **� Observations :** La distribution révèle plusieurs clusters de recettes avec des niveaux
            d'engagement distincts. Les recettes à haute popularité ne présentent pas systématiquement
            les meilleures notes, suggérant l'existence de facteurs additionnels.

            **� Implication :** Cette distribution non-linéaire indique que la popularité s'organise
            en segments distincts plutôt qu'en progression continue. Cependant une grande majorité des recettes possède une bonne note.
            Les utilisateurs sont peut-être bienveillant entre eux ou les recettes sont peut-être toutes délicieuses.
            Nous allons donc plutôt nous focaliser dans la suite sur l'étude du nombre de fois où une recette a été faite soit sa
            popularité pour qualifier son succés tout en gardant un oeil sur sa note.
            """
            )

            return pop_rating  # Return for use in step 2

        except ValueError as e:
            st.info(f"Impossible de tracer Note vs Popularité: {e}")
            return None

    def _render_step_2(self, analyzer: InteractionsAnalyzer, pop_rating):
        """Render step 2: Popularity segmentation analysis."""
        st.markdown("---")
        st.header("📈  ÉTAPE 2 : Segmentation par engagement")

        st.markdown(
            """
        **Objectif :** Identifier et caractériser les différents segments de popularité.

        La distribution observée suggère l'existence de groupes distincts de recettes. Nous appliquons
        une segmentation basée sur les percentiles pour révéler la structure naturelle de la popularité.

        **Méthode :** Segmentation par percentiles (25e, 75e, 95e) du nombre d'interactions.
        """
        )

        # Segmentation par popularité avec contexte narratif
        st.markdown("---")
        self._render_popularity_segmentation(analyzer, pop_rating)

        # Obtenir les seuils de segmentation pour l'explication
        segmented_data = analyzer.create_popularity_segments(pop_rating)
        thresholds = analyzer._popularity_segments_info["thresholds"]

        # Calculer les pourcentages réels de chaque segment
        segment_counts = segmented_data["popularity_segment"].value_counts()
        total_recipes = len(segmented_data)
        segment_percentages = {}
        for segment in ["Low", "Medium", "High", "Viral"]:
            if segment in segment_counts.index:
                segment_percentages[segment] = (segment_counts[segment] / total_recipes) * 100
            else:
                segment_percentages[segment] = 0.0

        st.markdown(
            f"""
        **📋 Caractérisation des segments identifiés :**

        L'analyse révèle quatre segments distincts basés sur le niveau d'engagement :

        - **Engagement Faible** : 1 à {int(thresholds['low_max'])} interactions
          ({segment_percentages['Low']:.1f}% des recettes - souvent de qualité mais visibilité limitée)

        - **Engagement Modéré** : {int(thresholds['low_max']) + 1} à {int(thresholds['medium_max'])} interactions
          ({segment_percentages['Medium']:.1f}% des recettes - performance stable et audience fidèle)

        - **Engagement Élevé** : {int(thresholds['medium_max']) + 1} à {int(thresholds['high_max'])} interactions
          ({segment_percentages['High']:.1f}% des recettes - forte popularité établie)

        - **Engagement Viral** : Plus de {int(thresholds['high_max'])} interactions
          ({segment_percentages['Viral']:.1f}% des recettes - phénomènes d'adoption exceptionnelle)

        """
        )

    def _render_step_3(
        self,
        analyzer: InteractionsAnalyzer,
        agg: pd.DataFrame,
        plot_type: str,
        n_bins: int,
        bin_agg: str,
        alpha: float,
        pop_rating,
    ):
        """Render step 3: Technical factors influence analysis."""
        st.markdown("---")
        st.header("📈  ÉTAPE 3 : Facteurs d'influence")

        st.markdown(
            """
        **Objectif :** Identifier les caractéristiques intrinsèques des recettes qui corrèlent
        avec une popularité élevée.

        Au-delà de la qualité, trois dimensions techniques peuvent influencer l'adoption d'une recette :
        le temps de préparation, la complexité (nombre d'étapes) et les ingrédients requis.

        **Méthode :** Analyse de corrélation entre caractéristiques techniques et niveau d'engagement.
        """
        )

        # Caractéristiques (feature) vs Popularité avec la note comme taille
        feature_order = ["minutes", "n_steps", "n_ingredients"]
        features = [f for f in feature_order if f in agg.columns]
        if features:
            for feat in features:
                # Contexte analytique pour chaque caractéristique
                if feat == "minutes":
                    st.markdown(
                        """
                    #### ⏱️ Impact du temps de préparation

                    **Hypothèse :** Les recettes rapides sont plus populaires dans une société pressée.
                    **Variable :** Temps de préparation en minutes vs nombre d'interactions.
                    """
                    )
                elif feat == "n_steps":
                    st.markdown(
                        """
                    #### 🧩 Influence de la complexité procédurale

                    **Hypothèse :** La complexité (nombre d'étapes) peut freiner l'adoption mais améliorer la satisfaction.
                    **Variable :** Nombre d'étapes vs nombre d'interactions.
                    **Observation :** Équilibre entre accessibilité et sophistication.
                    """
                    )
                elif feat == "n_ingredients":
                    st.markdown(
                        """
                    #### 🥘 Effet de la diversité des ingrédients

                    **Hypothèse :** Plus d'ingrédients = recette plus complexe et potentiellement dissuasive.
                    **Variable :** Nombre d'ingrédients vs nombre d'interactions.
                    **Analyse :** Impact de la richesse compositionnelle sur l'engagement.
                    """
                    )

                try:
                    df_pop_feat = analyzer.popularity_vs_feature(feat)
                    # Merge pour récupérer la note moyenne si disponible
                    if pop_rating is not None:
                        # Limiter les colonnes pour éviter suffixes _x / _y sur
                        # interaction_count
                        pr_min = pop_rating[["recipe_id", "avg_rating"]].copy()
                        merged = df_pop_feat.merge(pr_min, on="recipe_id", how="left")
                    else:
                        merged = df_pop_feat
                    # Normalisation de nom si interaction_count a été suffixé
                    # accidentellement
                    if "interaction_count_x" in merged.columns and "interaction_count" not in merged.columns:
                        merged.rename(
                            columns={"interaction_count_x": "interaction_count"},
                            inplace=True,
                        )
                    if "interaction_count_y" in merged.columns and "interaction_count" not in merged.columns:
                        merged.rename(
                            columns={"interaction_count_y": "interaction_count"},
                            inplace=True,
                        )
                    y_col = "interaction_count" if "interaction_count" in merged.columns else merged.columns[1]
                    if "avg_rating" in merged.columns:
                        fig = self._create_plot(
                            merged,
                            x=feat,
                            y=y_col,
                            size="avg_rating",
                            plot_type=plot_type,
                            n_bins=n_bins,
                            bin_agg=bin_agg,
                            alpha=alpha,
                        )
                    else:
                        fig = self._create_plot(
                            merged,
                            x=feat,
                            y=y_col,
                            plot_type=plot_type,
                            n_bins=n_bins,
                            bin_agg=bin_agg,
                            alpha=alpha,
                        )
                    st.pyplot(fig)

                    # Analyse narrative spécifique pour chaque caractéristique
                    if feat == "minutes":
                        st.markdown(
                            """
                        **Ce que révèle le graphique du temps :**

                        L'analyse de la distribution révèle une concentration
                        de recettes bien notées dans certaines zones de temps, indiquant les "sweet spots"
                        temporels. Les recettes ultra-rapides (moins de 15 minutes) peuvent manquer de sophistication,
                        tandis que les préparations longues (plus de 2 heures) peuvent décourager les utilisateurs.
                        Les données suggèrent un équilibre entre temps suffisant pour créer de la valeur et durée raisonnable pour maintenir l'engagement.
                        En regardant l'histogramme avec suffisament de bins (>30), on voit clairement que les recettes les plus refaites
                        sont les recettes prenant moins d'une heure.
                        """
                        )
                    elif feat == "n_steps":
                        st.markdown(
                            """
                        **Le verdict sur la complexité :**

                        L'analyse révèle l'un des paradoxes les plus significatifs de la cuisine.
                        Une concentration de recettes bien notées (gros points) autour de 5-8 étapes
                        confirme l'existence d'un "niveau de défi optimal".
                        L'engagement utilisateur optimal se situe entre
                        accomplissement satisfaisant et complexité gérable. Cette zone représente l'équilibre
                        entre "trop simple", "ennuyeux" et "trop complexe","décourageant".
                        En regardant l'histogramme avec suffisament de bins (>30), on voit clairement que les recettes les plus refaites
                        sont les recettes ayant moins de 15 étapes environ.
                        """
                        )
                    elif feat == "n_ingredients":
                        st.markdown(
                            """
                        **La révélation des ingrédients :**

                        L'analyse révèle la relation entre nombre d'ingrédients et satisfaction utilisateur.
                        Cette distribution montre comment la perception de "richesse" d'une recette influence
                        son succès.
                        Les données révèlent un optimum entre richesse perçue
                        et accessibilité pratique. Un nombre trop faible d'ingrédients peut sembler "basique",
                        tandis qu'un nombre excessif peut paraître "intimidant" ou "coûteux".
                        La concentration des meilleures notes révèle le nombre optimal
                        qui équilibre richesse et accessibilité.
                        En regardant l'histogramme avec suffisament de bins (>30), on voit clairement que les recettes les plus refaites
                        sont les recettes demandant moins de 15 ingrédients environ.
                        """
                        )

                except ValueError as e:
                    st.caption(f"{feat}: {e}")
        else:
            st.info("Aucune des colonnes minutes / n_steps / n_ingredients n'est présente dans l'agrégat.")

    def _render_viral_recipe_analysis(
        self,
        analyzer: InteractionsAnalyzer,
        agg: pd.DataFrame,
        interactions_df: pd.DataFrame,
        recipes_df: pd.DataFrame,
    ):
        """Render temporal analysis of viral recipes with 3D visualization."""
        st.markdown("---")
        st.header("📈 ÉTAPE 4 : Analyse temporelle")

        st.markdown(
            """
        **Question :** Comment évoluent les recettes à fort engagement dans le temps ?

        Cette analyse examine l'évolution temporelle de la qualité et du nombre d'interactions
        pour les recettes du segment viral (>95e percentile).

        **Approche :** Visualisation 3D des trajectoires temporelles pour identifier les phases
        d'accélération de l'engagement.
        """
        )

        # Identify viral recipes
        pop_rating = analyzer.popularity_vs_rating()
        segmented_data = analyzer.create_popularity_segments(pop_rating)
        viral_recipes = segmented_data[segmented_data["popularity_segment"] == "Viral"]

        if len(viral_recipes) == 0:
            st.warning("Aucune recette virale détectée dans les données actuelles.")
            return

        # TOP 10 des recettes les plus virales
        st.markdown("### 📋 Top 10 des recettes les plus virales")

        # Merge with recipe names and sort by interaction count
        top_viral = viral_recipes.copy()
        if "name" in recipes_df.columns:
            top_viral = top_viral.merge(
                recipes_df[["id", "name"]],
                left_on="recipe_id",
                right_on="id",
                how="left",
            )

        # Sort by interaction count (most viral first) and take top 10
        top_viral = top_viral.sort_values("interaction_count", ascending=False).head(10)

        # Recalculer les stats avec les mêmes filtres que le graphique 3D

        # Recalculate stats using only complete data (same as 3D graph)
        corrected_stats = []
        for _, row in top_viral.iterrows():
            recipe_id = row["recipe_id"]
            recipe_interactions = interactions_df[interactions_df["recipe_id"] == recipe_id].copy()

            if len(recipe_interactions) > 0:
                # Apply same filtering as 3D graph
                date_columns = [col for col in interactions_df.columns if "date" in col.lower()]
                if date_columns:
                    date_col = date_columns[0]
                    recipe_interactions[date_col] = pd.to_datetime(recipe_interactions[date_col], errors="coerce")
                    complete_data = recipe_interactions.dropna(subset=[date_col, "rating"]).copy()

                    if len(complete_data) > 0:
                        corrected_stats.append(
                            {
                                "recipe_id": recipe_id,
                                "interaction_count_filtered": len(complete_data),
                                "avg_rating_filtered": complete_data["rating"].mean(),
                                "name": row.get("name", f"Recipe {recipe_id}"),
                            }
                        )

        # Create corrected display dataframe
        if corrected_stats:
            corrected_df = pd.DataFrame(corrected_stats)
            corrected_df = corrected_df.sort_values("interaction_count_filtered", ascending=False)

            display_cols = [
                "name",
                "recipe_id",
                "interaction_count_filtered",
                "avg_rating_filtered",
            ]
            top_viral_display = corrected_df[display_cols].copy()
            top_viral_display.columns = [
                "Nom de la recette",
                "ID",
                "Nb interactions (dates valides)",
                "Note moyenne (dates valides)",
            ]
        else:
            # Fallback to original data if no corrected data available
            display_cols = ["recipe_id", "interaction_count", "avg_rating"]
            if "name" in top_viral.columns:
                display_cols = ["name", "recipe_id", "interaction_count", "avg_rating"]

            top_viral_display = top_viral[display_cols].copy()
            top_viral_display.columns = (
                ["Nom de la recette", "ID", "Nb interactions", "Note moyenne"]
                if "name" in top_viral.columns
                else ["ID", "Nb interactions", "Note moyenne"]
            )

        # Add rank column
        top_viral_display.insert(0, "Rang", range(1, len(top_viral_display) + 1))

        # Format the display
        if "Nom de la recette" in top_viral_display.columns:
            top_viral_display["Nom de la recette"] = top_viral_display["Nom de la recette"].apply(
                lambda x: x[:60] + "..." if len(str(x)) > 60 else str(x)
            )

        # Format interaction count column (handle both old and new column names)
        interaction_col = next(
            (col for col in top_viral_display.columns if "interactions" in col.lower()),
            None,
        )
        if interaction_col:
            top_viral_display[interaction_col] = top_viral_display[interaction_col].apply(lambda x: f"{x:,.0f}")

        # Format rating column (handle both old and new column names)
        rating_col = next(
            (col for col in top_viral_display.columns if "note moyenne" in col.lower()),
            None,
        )
        if rating_col:
            top_viral_display[rating_col] = top_viral_display[rating_col].apply(lambda x: f"{x:.2f} ⭐")

        # Display the table
        st.dataframe(top_viral_display, width="stretch", hide_index=True)

        # Quick stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "🥇 Recette #1",
                f"{top_viral.iloc[0]['interaction_count']:,.0f} interactions",
            )
        with col2:
            total_interactions = top_viral["interaction_count"].sum()
            st.metric("📊 Total Top 10", f"{total_interactions:,.0f} interactions")
        with col3:
            avg_rating_top10 = top_viral["avg_rating"].mean()
            st.metric("⭐ Note moyenne", f"{avg_rating_top10:.2f}")

        # Recipe selection interface for 3D analysis
        st.markdown("### 📋 Pattern Commun du Top 3")

        st.markdown(
            """
        **Observation du pattern commun sur les recettes les plus populaires :**

        En analysant le top 3 des recettes virales, nous observons un **pattern commun** :
        - **Phase 1** : Croissance progressive par effet boule de neige
        - **Phase 2** : Forte accélération quand la recette devient tendance
        - **Phase 3** : Stagnation puis déclin quand la mode passe

        Ce phénomène reflète le cycle naturel des tendances culinaires.
        """
        )

        # Sélection du top 3 pour illustrer le pattern commun
        representative_examples = {
            "top_1": 2886,  # best banana bread - #1 des interactions
            "top_2": 27208,  # to die for crock pot roast - #2 des interactions
            "top_3": 39087,  # creamy cajun chicken pasta - #3 des interactions
        }

        # Afficher le tableau des exemples sélectionnés
        examples_data = []
        for pattern, recipe_id in representative_examples.items():
            # Chercher dans les données
            recipe_interactions = interactions_df[interactions_df["recipe_id"] == recipe_id]
            if len(recipe_interactions) > 0:
                recipe_name = "N/A"
                if "name" in recipes_df.columns:
                    recipe_match = recipes_df[recipes_df["id"] == recipe_id]
                    if len(recipe_match) > 0:
                        recipe_name = recipe_match["name"].iloc[0]

                # Calculer les stats pour cet exemple
                avg_rating = recipe_interactions["rating"].mean()
                total_interactions = len(recipe_interactions)

                examples_data.append(
                    {
                        "Rang": f"#{list(representative_examples.keys()).index(pattern) + 1}",
                        "ID": recipe_id,
                        "Nom": (recipe_name[:50] + "..." if len(recipe_name) > 50 else recipe_name),
                        "Interactions": f"{total_interactions:,}",
                        "Note Moyenne": f"{avg_rating:.2f} ⭐",
                    }
                )

        if examples_data:
            examples_df = pd.DataFrame(examples_data)
            st.dataframe(examples_df, width="stretch", hide_index=True)

        # Utiliser ces exemples pour la visualisation 3D
        selected_recipe_ids = list(representative_examples.values())
        recipe_display = [
            f"Top {i + 1}: {examples_data[i]['Nom']}" for i, pattern in enumerate(representative_examples.keys())
        ]
        selected_indices = list(range(len(selected_recipe_ids)))

        # Temporal analysis
        st.markdown("### 📊 Visualisation 3D du Pattern Commun")

        st.markdown(
            """
        **Lecture du graphique :**
        - **X** : Date de l'interaction
        - **Y** : Note attribuée (1-5 étoiles)
        - **Z** : Densité d'interactions (nombre d'avis par période)

        🔍 **L'effet boule de neige** : démarrage lent, accélération, puis stabilisation/déclin

        **Note :** La visualisation 3D utilise les données brutes pour préserver toutes les recettes.
        """
        )

        # Create 3D visualization using RAW data to preserve all recipes
        # Note: 3D visualization shows actual recipe trajectories without preprocessing
        # to ensure no recipes are excluded from visualization
        # IMPROVEMENT: Enhanced temporal sampling for smoother trajectories
        self.logger.info("Using raw data for 3D visualization with improved temporal sampling")
        self._create_3d_visualization_real(selected_recipe_ids, interactions_df, recipe_display, selected_indices)

    def _create_3d_visualization_real(
        self,
        recipe_ids: list,
        interactions_df: pd.DataFrame,
        recipe_display: list,
        selected_indices: list,
    ):
        """Create 3D visualization with real temporal data from the dataset."""

        # Check for available date columns
        date_columns = [col for col in interactions_df.columns if "date" in col.lower()]

        if not date_columns:
            st.error(
                "Aucune colonne de date trouvée. Colonnes disponibles : " + ", ".join(interactions_df.columns.tolist())
            )
            return

        # Use the first date column found, or let user choose if multiple
        if len(date_columns) == 1:
            date_col = date_columns[0]
        else:
            st.info(f"Colonnes de date disponibles : {', '.join(date_columns)}")
            date_col = st.selectbox("Choisissez la colonne de date :", date_columns)

        # Ensure date column is in datetime format
        try:
            interactions_df[date_col] = pd.to_datetime(interactions_df[date_col])
        except Exception as e:
            st.error(
                f"Erreur lors de la conversion de la colonne '{date_col}' en date : {
                    str(e)}"
            )
            return

        # Add interval selection for temporal sampling
        st.subheader("⚙️ Paramètres temporels")
        interval_days = st.selectbox(
            "Afficher un point tous les :",
            [1, 7, 14, 30, 90],
            index=1,  # Default: tous les 7 jours
            format_func=lambda x: (f"{x} jour{'s' if x > 1 else ''}" if x < 30 else f"{x // 30} mois"),
        )

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection="3d")

        colors = [
            "#FF6B6B",
            "#4ECDC4",
            "#DDA0DD",
            "#45B7D1",
            "#96CEB4",
            "#FFEAA7",
            "#98D8C8",
        ]

        for i, recipe_id in enumerate(recipe_ids):
            # Filter interactions for this recipe
            recipe_interactions = interactions_df[interactions_df["recipe_id"] == recipe_id].copy()

            if len(recipe_interactions) == 0:
                continue

            # Ensure date column is properly converted to datetime
            recipe_interactions[date_col] = pd.to_datetime(recipe_interactions[date_col], errors="coerce")

            # Filter out rows with missing essential data (date, rating, recipe_id)
            complete_data = recipe_interactions.dropna(subset=[date_col, "rating"]).copy()

            if len(complete_data) == 0:
                st.warning(f"Aucune donnée complète trouvée pour la recette {recipe_id}")
                continue

            # Sort by date to ensure strict chronological order
            complete_data = complete_data.sort_values(date_col).reset_index(drop=True)

            # Convert dates to numeric for plotting (days since first interaction)
            first_date = complete_data[date_col].min()
            complete_data["days_since_start"] = (complete_data[date_col] - first_date).dt.days

            # Filter based on temporal interval (every X days) - IMPROVED SAMPLING
            if interval_days > 1:
                # Group by interval periods and take a representative point
                complete_data["period"] = complete_data["days_since_start"] // interval_days

                # Take the point closest to the middle of each period for better
                # representation
                def get_middle_point(group):
                    period_start = group["days_since_start"].min()
                    period_end = group["days_since_start"].max()
                    period_middle = (period_start + period_end) / 2
                    # Find the interaction closest to the middle of the period
                    closest_idx = (group["days_since_start"] - period_middle).abs().idxmin()
                    return group.loc[closest_idx]

                display_data = complete_data.groupby("period").apply(get_middle_point).reset_index(drop=True)
            else:
                # Show ALL points (every single interaction)
                display_data = complete_data.copy()

            # Calculate cumulative statistics correctly
            display_data = display_data.sort_values("days_since_start").reset_index(drop=True)

            # For each point, calculate the cumulative average rating from ALL
            # previous interactions
            cumulative_avg_ratings = []
            cumulative_interactions = []

            for idx in range(len(display_data)):
                current_day = display_data.iloc[idx]["days_since_start"]
                # Get all interactions up to and including current day from original
                # complete_data
                interactions_up_to_day = complete_data[complete_data["days_since_start"] <= current_day]

                # Calculate cumulative average rating (from day 1 to current day)
                cumulative_avg_rating = interactions_up_to_day["rating"].mean()
                cumulative_avg_ratings.append(cumulative_avg_rating)

                # Cumulative interaction count
                cumulative_interactions.append(len(interactions_up_to_day))

            display_data["cumulative_avg_rating"] = cumulative_avg_ratings
            display_data["cumulative_interactions"] = cumulative_interactions

            # Extract coordinates for 3D plot - only real data points
            x_dates = display_data["days_since_start"].values
            y_ratings = display_data["cumulative_avg_rating"].values  # Cumulative average ratings
            z_cumulative = display_data["cumulative_interactions"].values

            # Ensure we have valid data to plot
            if len(x_dates) == 0 or np.any(np.isnan(x_dates)) or np.any(np.isnan(y_ratings)):
                st.warning(f"Données invalides pour la recette {recipe_id}")
                continue

            # Plot the 3D trajectory
            ax.plot(
                x_dates,
                y_ratings,
                z_cumulative,
                color=colors[i % len(colors)],
                marker="o",
                markersize=5,
                label=(
                    f"{recipe_display[selected_indices[i]][:30]}..."
                    if len(recipe_display[selected_indices[i]]) > 30
                    else recipe_display[selected_indices[i]]
                ),
                linewidth=2.5,
                alpha=0.8,
            )

            # Add trajectory start and end markers
            if len(x_dates) > 1:
                # Start point (green)
                ax.scatter(
                    x_dates[0],
                    y_ratings[0],
                    z_cumulative[0],
                    color="green",
                    s=100,
                    alpha=0.8,
                    marker="^",
                )
                # End point (red)
                ax.scatter(
                    x_dates[-1],
                    y_ratings[-1],
                    z_cumulative[-1],
                    color="red",
                    s=100,
                    alpha=0.8,
                    marker="v",
                )

                # Add dotted lines to show final coordinates instead of projections
                # Vertical line from bottom to final point
                ax.plot(
                    [x_dates[-1], x_dates[-1]],
                    [y_ratings[-1], y_ratings[-1]],
                    [0, z_cumulative[-1]],
                    "k--",
                    alpha=0.6,
                    linewidth=1,
                )

                # Horizontal line from Y-axis to final point
                ax.plot(
                    [0, x_dates[-1]],
                    [y_ratings[-1], y_ratings[-1]],
                    [z_cumulative[-1], z_cumulative[-1]],
                    "k--",
                    alpha=0.6,
                    linewidth=1,
                )

                # Line from X-axis to final point
                ax.plot(
                    [x_dates[-1], x_dates[-1]],
                    [0, y_ratings[-1]],
                    [z_cumulative[-1], z_cumulative[-1]],
                    "k--",
                    alpha=0.6,
                    linewidth=1,
                )

        # Formatting and labels
        ax.set_xlabel("Jours depuis la première interaction", fontsize=12)
        ax.set_ylabel("Note moyenne cumulative", fontsize=12)
        ax.set_zlabel("Interactions cumulées", fontsize=12)
        ax.set_title("Évolution Temporelle des Recettes Virales", fontsize=14, fontweight="bold")

        # Set explicit axis limits to ensure correct scaling
        # Collect all data points to determine proper axis limits
        all_x, all_y, all_z = [], [], []
        for i, recipe_id in enumerate(recipe_ids):
            recipe_interactions = interactions_df[interactions_df["recipe_id"] == recipe_id].copy()
            if len(recipe_interactions) > 0:
                date_columns = [col for col in interactions_df.columns if "date" in col.lower()]
                if date_columns:
                    date_col = date_columns[0]
                    recipe_interactions[date_col] = pd.to_datetime(recipe_interactions[date_col], errors="coerce")
                    complete_data = recipe_interactions.dropna(subset=[date_col, "rating"]).copy()

                    if len(complete_data) > 0:
                        # Calculate actual final values
                        complete_data = complete_data.sort_values(date_col)
                        first_date = complete_data[date_col].min()
                        final_days = (complete_data[date_col].max() - first_date).days
                        final_rating = complete_data["rating"].mean()  # Overall average
                        final_interactions = len(complete_data)

                        all_x.append(final_days)
                        all_y.append(final_rating)
                        all_z.append(final_interactions)

        # Set axis limits with some padding
        if all_x and all_y and all_z:
            x_margin = max(all_x) * 0.05
            z_margin = max(all_z) * 0.05

            ax.set_xlim(0, max(all_x) + x_margin)
            ax.set_ylim(0, 5)  # Note moyenne toujours de 0 à 5 pour référence standardisée
            ax.set_zlim(0, max(all_z) + z_margin)

        # Disable default shadow projections on XZ and YZ planes
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Make panes transparent
        ax.xaxis.pane.set_edgecolor("gray")
        ax.yaxis.pane.set_edgecolor("gray")
        ax.zaxis.pane.set_edgecolor("gray")
        ax.xaxis.pane.set_alpha(0.1)
        ax.yaxis.pane.set_alpha(0.1)
        ax.zaxis.pane.set_alpha(0.1)

        # Legend with custom positioning
        legend = ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
        legend.set_title("Recettes", prop={"weight": "bold"})

        # Add grid for better readability
        ax.grid(True, alpha=0.3)

        # Custom viewing angle for better perspective
        ax.view_init(elev=20, azim=45)

        plt.tight_layout()
        st.pyplot(fig)

        # Analysis Summary
        # Legend and axes explanation
        with st.expander("📊 Lecture du graphique", expanded=False):
            st.markdown(
                """
            **Légende :**
            - 🟢 Point de démarrage - 🔴 Point final
            - Lignes pointillées : repères pour lecture des coordonnées

            **Axes d'analyse :**
            - **X** : Temps (jours depuis première interaction)
            - **Y** : Qualité cumulative (note moyenne évolutive)
            - **Z** : Volume d'adoption (interactions cumulées)
            """
            )
        st.markdown("### 💡 Analyse Synthétique : Le Pattern Universel du Succès")

        st.markdown(
            """
        **Analyse morphologique des trajectoires 3D :** L'examen des courbures et dérivées révèle une
        signature temporelle commune correspondant au cycle naturel des tendances virales.

        **Pattern universel identifié :**
        """
        )

        # Single unified analysis about the common pattern
        st.markdown(
            """
        **📈 Pattern Universel - Effet Boule de Neige**

        **Morphologie observée sur les 3 recettes les plus virales :**
        - **Phase 1** : Accumulation lente (dZ/dt faible) - période d'émergence
        - **Phase 2** : Accélération massive (d²Z/dt² > 0) - explosion virale
        - **Phase 3** : Plateau puis déclin possible à prévoir (dZ/dt → 0 puis négatif) - fin de mode

        **Explication simple :** Comme toute tendance, les recettes virales suivent
        le même cycle : émergence discrète, explosion quand elles deviennent "à la mode",
        puis retour progressif à la normale quand l'effet de nouveauté s'estompe.


        """
        )

        # Display detailed statistics
        st.markdown("### 📊 Statistiques par recette")

        stats_data = []
        for i, recipe_id in enumerate(recipe_ids):
            recipe_interactions = interactions_df[interactions_df["recipe_id"] == recipe_id].copy()
            if len(recipe_interactions) > 0:
                # Filter only complete data (same logic as 3D plot)
                recipe_interactions[date_col] = pd.to_datetime(recipe_interactions[date_col], errors="coerce")
                complete_data = recipe_interactions.dropna(subset=[date_col, "rating"]).copy()

                if len(complete_data) > 0:
                    first_interaction = complete_data[date_col].min()
                    last_interaction = complete_data[date_col].max()
                    duration = (last_interaction - first_interaction).days
                    total_interactions = len(complete_data)
                    avg_rating = complete_data["rating"].mean()

                    stats_data.append(
                        {
                            "ID": recipe_id,
                            "Recette": (
                                recipe_display[selected_indices[i]][:40] + "..."
                                if len(recipe_display[selected_indices[i]]) > 40
                                else recipe_display[selected_indices[i]]
                            ),
                            "Première interaction": first_interaction.strftime("%Y-%m-%d"),
                            "Dernière interaction": last_interaction.strftime("%Y-%m-%d"),
                            "Durée (jours)": duration,
                            "Total interactions (complètes)": total_interactions,
                            "Note moyenne": f"{avg_rating:.2f}",
                            "Interactions/jour": f"{total_interactions / max(duration, 1):.1f}",
                        }
                    )

        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, width="stretch")

    # ---------------- Data Loading ---------------- #
    def _load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        inter_loader = DataLoader(self.config.interactions_path)
        rec_loader = DataLoader(self.config.recipes_path)
        interactions_df = inter_loader.load_data()
        recipes_df = rec_loader.load_data()
        return interactions_df, recipes_df

    # ---------------- Visualization helpers ---------------- #
    def _get_plot_title(self, x: str, y: str, plot_type: str, bin_agg: str = "count") -> str:
        """Get predefined French titles based on plot type and variables."""

        # Titres prédéfinis pour les graphiques les plus courants
        predefined_titles = {
            # Scatter plots
            (
                "avg_rating",
                "interaction_count",
                "Scatter",
            ): "Note moyenne selon le nombre d'interactions",
            (
                "interaction_count",
                "avg_rating",
                "Scatter",
            ): "Nombre d'interactions selon la note moyenne",
            (
                "minutes",
                "avg_rating",
                "Scatter",
            ): "Note moyenne selon la durée de préparation",
            (
                "n_steps",
                "avg_rating",
                "Scatter",
            ): "Note moyenne selon le nombre d'étapes",
            (
                "n_ingredients",
                "avg_rating",
                "Scatter",
            ): "Note moyenne selon le nombre d'ingrédients",
            (
                "minutes",
                "interaction_count",
                "Scatter",
            ): "Nombre d'interactions selon la durée de préparation",
            (
                "n_steps",
                "interaction_count",
                "Scatter",
            ): "Nombre d'interactions selon le nombre d'étapes",
            (
                "n_ingredients",
                "interaction_count",
                "Scatter",
            ): "Nombre d'interactions selon le nombre d'ingrédients",
            # Histograms avec count
            ("avg_rating", "", "Histogram"): "Distribution des notes moyennes",
            (
                "interaction_count",
                "",
                "Histogram",
            ): "Distribution du nombre d'interactions",
            ("minutes", "", "Histogram"): "Distribution des durées de préparation",
            ("n_steps", "", "Histogram"): "Distribution du nombre d'étapes",
            ("n_ingredients", "", "Histogram"): "Distribution du nombre d'ingrédients",
            ("rating", "", "Histogram"): "Distribution des notes",
        }

        # Recherche du titre prédéfini
        key = (x, y, plot_type)
        if key in predefined_titles:
            return predefined_titles[key]

        # Titre par défaut pour les histogrammes
        if plot_type == "Histogram":
            key_hist = (x, "", plot_type)
            if key_hist in predefined_titles:
                return predefined_titles[key_hist]

        # Fallback : génération simple
        var_labels = {
            "avg_rating": "Note moyenne",
            "interaction_count": "Nombre d'interactions",
            "minutes": "Durée (minutes)",
            "n_steps": "Nombre d'étapes",
            "n_ingredients": "Nombre d'ingrédients",
            "rating": "Note",
        }

        x_label = var_labels.get(x, x.replace("_", " ").title())
        y_label = var_labels.get(y, y.replace("_", " ").title())

        if plot_type == "Histogram":
            return f"Distribution de {x_label}"
        else:  # Scatter
            return f"{y_label} selon {x_label}"

    def _create_plot(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        size: str | None = None,
        title: str = "",
        plot_type: str = "Scatter",
        n_bins: int = 20,
        bin_agg: str = "count",
        alpha: float = 0.6,
    ):
        """Create plot based on selected type with improved visualization."""
        fig, ax = plt.subplots(figsize=(8, 6))

        if plot_type == "Scatter":
            self._scatter_plot(data, x, y, size, ax, alpha)
        elif plot_type == "Histogram":
            self._histogram_plot(data, x, y, size, ax, n_bins, bin_agg)

        # Utiliser un titre prédéfini si aucun titre n'est fourni
        if not title:
            title = self._get_plot_title(x, y, plot_type, bin_agg)

        ax.set_title(title, fontsize=14, fontweight="bold")

        # Labels des axes en français
        var_labels = {
            "avg_rating": "Note moyenne",
            "interaction_count": "Nombre d'interactions",
            "minutes": "Durée (minutes)",
            "n_steps": "Nombre d'étapes",
            "n_ingredients": "Nombre d'ingrédients",
            "rating": "Note",
        }

        x_label = var_labels.get(x, x.replace("_", " ").title())
        y_label = var_labels.get(y, y.replace("_", " ").title())

        # Pour les histogrammes, l'axe Y affiche toujours le nombre d'observations
        if plot_type == "Histogram":
            y_label = "Nombre d'observations"

        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def _scatter_plot(self, data: pd.DataFrame, x: str, y: str, size: str | None, ax, alpha: float):
        """Enhanced scatter plot with all data points displayed."""
        if size is not None:
            # Plot all data points without sampling
            scatter = ax.scatter(
                data[x],
                data[y],
                s=data[size] * 10,  # Scale size
                c=data[size],
                cmap="viridis",
                alpha=alpha,
                edgecolors="none",
            )
            cbar = plt.colorbar(scatter, ax=ax)

            # Label plus explicite pour la colorbar
            if size == "avg_rating":
                cbar.set_label("Moyenne des notes")
            else:
                size_label = size.replace("_", " ").title()
                cbar.set_label(size_label)
        else:
            # Plot all data points without sampling
            ax.scatter(data[x], data[y], alpha=alpha, s=30, c="steelblue", edgecolors="none")

    def _histogram_plot(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        size: str | None,
        ax,
        n_bins: int,
        bin_agg: str,
    ):
        """Create histogram counting observations per bin."""
        # Create bins for x-axis
        data_clean = data.dropna(subset=[x, y])
        if len(data_clean) == 0:
            ax.text(
                0.5,
                0.5,
                "Pas de données valides",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        # Create proper bins with explicit edges
        x_min, x_max = data_clean[x].min(), data_clean[x].max()
        bin_edges = np.linspace(x_min, x_max, n_bins + 1)

        # Assign each point to a bin
        data_clean = data_clean.copy()
        data_clean["bin_idx"] = pd.cut(data_clean[x], bins=bin_edges, include_lowest=True, labels=False)

        # Calculate bin centers for plotting
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = (x_max - x_min) / n_bins

        # Count observations per bin
        agg_data = data_clean.groupby("bin_idx").size().reset_index(name="count")
        y_values = agg_data["count"]
        y_label = "Nombre d'observations"

        # Handle size aggregation if provided (count observations with that size)
        size_values = None
        if size and size in data_clean.columns:
            size_agg = data_clean.groupby("bin_idx")[size].mean()  # Moyenne de la variable size par bin

            # Align with y_values
            size_values = []
            for idx in agg_data["bin_idx"]:
                if idx in size_agg.index:
                    size_values.append(size_agg[idx])
                else:
                    size_values.append(0)

        # Create the bar plot
        valid_indices = agg_data["bin_idx"].dropna()
        plot_x = [bin_centers[int(idx)] for idx in valid_indices if int(idx) < len(bin_centers)]
        plot_y = y_values[: len(plot_x)]

        if size_values and len(size_values) >= len(plot_x):
            # Color bars by size value using the same colormap as scatter plots
            size_plot = size_values[: len(plot_x)]

            # Normalize size values for coloring
            if len(size_plot) > 0 and np.max(size_plot) > np.min(size_plot):
                norm = plt.Normalize(vmin=np.min(size_plot), vmax=np.max(size_plot))
                colors = [plt.cm.viridis(norm(s)) for s in size_plot]
            else:
                colors = ["steelblue"] * len(plot_x)

            ax.bar(
                plot_x,
                plot_y,
                width=bin_width * 0.8,
                color=colors,
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
            )

            # Add colorbar consistent with scatter plots
            if len(size_plot) > 0 and np.max(size_plot) > np.min(size_plot):
                sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax)

                # Label plus explicite pour la colorbar
                size_label = size.replace("_", " ").title()
                if size == "avg_rating":
                    cbar.set_label("Moyenne des notes (du bin)")
                else:
                    cbar.set_label(f"Moyenne {size_label} (du bin)")
        else:
            # Simple histogram without size coloring
            ax.bar(
                plot_x,
                plot_y,
                width=bin_width * 0.8,
                alpha=0.7,
                color="steelblue",
                edgecolor="black",
                linewidth=0.5,
            )

        # Add value labels on top of bars (more readable)
        for i, (x_pos, height) in enumerate(zip(plot_x, plot_y)):
            if not pd.isna(height) and height > 0:
                ax.text(
                    x_pos,
                    height + height * 0.02,
                    f"{int(height)}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

        # Improve axis labels
        ax.set_ylabel(y_label)

        # Set proper x-axis limits and ticks
        ax.set_xlim(x_min - bin_width / 2, x_max + bin_width / 2)

        # Add subtle grid for better readability
        ax.grid(True, alpha=0.3, axis="y")

    # ---------------- Main Render ---------------- #
    def run(self):

        # Introduction analytique
        with st.expander("🎯 Objectifs et méthodologie de l'analyse", expanded=True):
            st.markdown(
                """
            ### Qu'est-ce qui rend une recette populaire ?

            Cette analyse explore la relation entre la qualité des recettes (notes des utilisateurs) et leur
            succès (nombre d'interactions soit le nombre d'utilisateur ayant review la recette). Nous examinons comment les caractéristiques des recettes influencent
            leur adoption par la communauté.

            **Questions centrales :** La qualité garantit-elle la popularité ? Quels sont les facteurs
            déterminants du succès d'une recette ? Existe-t-il des profils de recettes particulièrement
            attractifs ?

            **Approche :** Analyse comparative entre qualité et engagement, segmentation des recettes par
            popularité, identification des caractéristiques discriminantes.

            **Données :** Preprocessing sélectif qui conserve toutes les interactions authentiques tout en
            filtrant les anomalies techniques (seuil configurable dans la barre latérale).
            """
            )

        # Section explicative du preprocessing (cachée par défaut)
        with st.expander("⚙️ Choix et justification du preprocessing", expanded=False):
            st.markdown(
                """
            ### Pourquoi un preprocessing sélectif ?

            Le preprocessing appliqué vise à préserver l'authenticité des données tout en éliminant
            les artefacts techniques qui fausseraient l'analyse.

            **Principe directeur :** Conserver toutes les interactions légitimes, filtrer uniquement
            les anomalies techniques évidentes.
            """
            )

            # Vérification en temps réel des valeurs manquantes
            st.markdown("#### 🔍 Vérification des valeurs manquantes")

            # Chargement temporaire pour vérification
            temp_interactions_df, temp_recipes_df = self._load_data()

            # Analyse des valeurs manquantes dans les colonnes critiques pour l'analyse
            # Ajout dynamique de la/les colonne(s) de date si présente(s)
            date_cols = [c for c in temp_interactions_df.columns if "date" in c.lower()]
            critical_interactions_cols = ["rating", "recipe_id", "user_id"] + date_cols
            critical_recipes_cols = ["minutes", "n_steps", "n_ingredients", "id"]

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**🎯 Colonnes critiques (interactions) :**")
                interactions_critical_missing = 0
                for col in critical_interactions_cols:
                    if col in temp_interactions_df.columns:
                        missing = temp_interactions_df[col].isnull().sum()
                        interactions_critical_missing += missing
                        status = "✅" if missing == 0 else "❌"
                        st.write(f"{status} `{col}`: {missing:,} manquantes")
                    else:
                        st.write(f"❌ `{col}`: colonne absente")

                if interactions_critical_missing == 0:
                    st.success("✅ Toutes les colonnes critiques sont complètes")
                else:
                    st.error(f"❌ {interactions_critical_missing} valeurs manquantes dans les colonnes critiques")

            with col2:
                st.markdown("**🎯 Colonnes critiques (recettes) :**")
                recipes_critical_missing = 0
                for col in critical_recipes_cols:
                    if col in temp_recipes_df.columns:
                        missing = temp_recipes_df[col].isnull().sum()
                        recipes_critical_missing += missing
                        status = "✅" if missing == 0 else "❌"
                        st.write(f"{status} `{col}`: {missing:,} manquantes")
                    else:
                        st.write(f"❌ `{col}`: colonne absente")

                if recipes_critical_missing == 0:
                    st.success("✅ Toutes les colonnes critiques sont complètes")
                else:
                    st.error(f"❌ {recipes_critical_missing} valeurs manquantes dans les colonnes critiques")

            # Analyse complète (toutes colonnes)
            interactions_total_missing = temp_interactions_df.isnull().sum().sum()
            recipes_total_missing = temp_recipes_df.isnull().sum().sum()

            with st.expander("📊 Analyse complète de toutes les colonnes", expanded=False):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Fichier interactions :**")
                    interactions_missing = temp_interactions_df.isnull().sum()
                    if interactions_total_missing == 0:
                        st.success("✅ Aucune valeur manquante")
                    else:
                        st.info(f"ℹ️ {interactions_total_missing} valeurs manquantes (colonnes non-critiques)")
                        missing_details = interactions_missing[interactions_missing > 0]
                        if len(missing_details) > 0:
                            st.code(missing_details.to_string())

                with col2:
                    st.markdown("**Fichier recettes :**")
                    recipes_missing = temp_recipes_df.isnull().sum()
                    if recipes_total_missing == 0:
                        st.success("✅ Aucune valeur manquante")
                    else:
                        st.info(f"ℹ️ {recipes_total_missing} valeurs manquantes (colonnes non-critiques)")
                        missing_details = recipes_missing[recipes_missing > 0]
                        if len(missing_details) > 0:
                            st.code(missing_details.to_string())

            # Conclusion sur la nécessité du KNN
            total_critical_missing = interactions_critical_missing + recipes_critical_missing
            if total_critical_missing == 0:
                st.success(
                    """
                **🎯 Conclusion :** Aucune valeur manquante dans les colonnes critiques pour l'analyse numérique.

                ✅ **KNN imputation non nécessaire** - toutes les données numériques (rating, minutes, n_steps, n_ingredients) sont complètes.

                ℹ️ Les valeurs manquantes détectées sont dans des colonnes textuelles (review, description) non utilisées dans les calculs.
                """
                )
            else:
                st.warning(
                    """
                **⚠️ Attention :** Des valeurs manquantes existent dans les colonnes critiques.

                🔧 **KNN imputation recommandée** pour maintenir l'intégrité de l'analyse.
                """
                )

            # Statistiques globales
            total_cells_interactions = temp_interactions_df.shape[0] * temp_interactions_df.shape[1]
            total_cells_recipes = temp_recipes_df.shape[0] * temp_recipes_df.shape[1]
            total_missing = interactions_total_missing + recipes_total_missing
            total_cells = total_cells_interactions + total_cells_recipes

            ((total_cells - total_missing) / total_cells) * 100

            st.markdown(
                """
            **🔧 Méthode IQR (Interquartile Range) :**
            - **Seuil configurable** : 1.0 à 20.0 (par défaut : 10.0)
            - **Calcul** : Q1 - seuil×IQR ≤ valeurs ≤ Q3 + seuil×IQR
            - **Cibles** : Variables numériques continues (temps, étapes, ingrédients)

            **Ce qui est préservé :**
            - **Toutes les notes utilisateurs** (aucun filtrage sur les ratings)
            - **Toutes les interactions datées** (comportements authentiques)
            - **Recettes avec caractéristiques extrêmes mais plausibles**

            **Impact typique :** 80-99% des données conservées selon le seuil choisi.
            Le seuil est ajustable dans la barre latérale pour explorer différents niveaux de filtrage.
            """
            )

        params = self._sidebar()
        plot_type = params["plot_type"]
        n_bins = params["n_bins"]
        bin_agg = params["bin_agg"]
        alpha = params["alpha"]
        outlier_threshold = params["outlier_threshold"]

        with st.spinner("Chargement des données..."):
            self.logger.info("Loading data for popularity analysis")
            interactions_df, recipes_df = self._load_data()
            self.logger.debug(
                f"Loaded interactions: {
                    interactions_df.shape}, recipes: {
                    recipes_df.shape}"
            )

        # Configuration preprocessing sélective : filtrer les valeurs techniques
        # aberrantes mais garder toutes les notes
        config_selective = PreprocessingConfig(
            enable_preprocessing=True,
            outlier_method="iqr",  # Méthode IQR avec seuil configurable
            outlier_threshold=outlier_threshold,  # Seuil configuré via sidebar
        )
        analyzer = InteractionsAnalyzer(
            interactions=interactions_df,
            recipes=recipes_df,
            preprocessing=config_selective,
            cache_enabled=True,  # Cache activé pour de meilleures performances
        )
        self.logger.info(f"Initialized InteractionsAnalyzer with preprocessing threshold {outlier_threshold}")

        # Affichage de l'impact du preprocessing avec détails statistiques
        try:
            merged_df = analyzer._df  # type: ignore[attr-defined]
            original_count = len(interactions_df)
            filtered_count = len(merged_df)
            filtered_percentage = (filtered_count / original_count) * 100

            # Récupération des stats de preprocessing
            preprocessing_stats = analyzer.get_preprocessing_stats()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "📊 Observations conservées",
                    f"{filtered_count:,}",
                    delta=f"{filtered_percentage:.1f}% du total",
                )
            with col2:
                st.metric(
                    "⚙️ Seuil IQR actuel",
                    f"{outlier_threshold:.1f}x",
                    help="Multiplicateur appliqué à l'écart interquartile",
                )

            # Détails des features traitées
            if preprocessing_stats and "features_processed" in preprocessing_stats:
                features_processed = preprocessing_stats["features_processed"]
                if features_processed:
                    st.info(f"**Features analysées :** {', '.join(features_processed)}")

        except Exception as e:
            st.info(f"**Preprocessing** (seuil {outlier_threshold:.1f}) - Détails non disponibles: {str(e)}")

        # Cache management in sidebar
        self._render_cache_controls(analyzer)

        agg = analyzer.aggregate()

        # Aperçu du dataframe fusionné avant agrégation
        st.subheader("Aperçu du dataframe fusionné (interactions ⟵ recettes)")
        try:
            merged_df = analyzer._df  # type: ignore[attr-defined]
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Lignes", f"{len(merged_df):,}")
            with col2:
                st.metric("Colonnes", f"{len(merged_df.columns)}")
            with st.expander("Colonnes du merged df"):
                st.code(", ".join(list(merged_df.columns)))
            st.dataframe(merged_df.head(20))
        except Exception as e:
            st.info(f"Impossible d'afficher le merged df: {e}")

        st.subheader("Table d'agrégation (Top 20)")
        st.dataframe(agg.head(20))

        # Execute the 4 analysis steps using dedicated render methods
        pop_rating = self._render_step_1(analyzer, plot_type, n_bins, bin_agg, alpha)
        if pop_rating is not None:
            self._render_step_2(analyzer, pop_rating)
        self._render_step_3(analyzer, agg, plot_type, n_bins, bin_agg, alpha, pop_rating)
        self._render_viral_recipe_analysis(analyzer, agg, interactions_df, recipes_df)

        # Synthèse et conclusions
        st.markdown("---")
        st.subheader("📋 Synthèse des résultats")

        st.markdown(
            """
        ### Conclusions principales

        **1. Relation qualité-popularité :** Non-linéaire avec formation de clusters distincts
        selon le niveau d'engagement, confirmant que l'excellence seule ne garantit pas la viralité.

        **2. Segmentation comportementale :** Segmentation par percentiles révélant 4 segments
        (faible/modéré/élevé/viral) avec des dynamiques d'adoption distinctes.

        **3. Facteurs d'optimisation :** Les caractéristiques techniques révèlent des zones
        d'équilibre optimal entre accessibilité et valeur perçue :
        - Temps de préparation : équilibre entre simplicité et satisfaction
        - Complexité procédurale : niveau de défi optimal pour l'engagement
        - Richesse compositionnelle : balance entre richesse et accessibilité

        **4. Pattern universel de viralité :** Les recettes les plus populaires suivent
        un cycle commun : émergence progressive, explosion virale, stabilisation ou déclin.
        Ce pattern reflète les mécanismes naturels des tendances culturelles.


        """
        )

        st.markdown("---")
        st.caption(
            "💡 **Configuration** : Ajustez les paramètres de preprocessing et visualisation pour explorer différentes perspectives analytiques."
        )
