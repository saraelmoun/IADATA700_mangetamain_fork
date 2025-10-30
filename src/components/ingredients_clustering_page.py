from __future__ import annotations

"""Streamlit page: Analyse de co-occurrence et clustering d'ingrédients.

Note: La User Story est affichée dans l'interface Streamlit (méthode run()),
pas dans cette docstring, conformément au pattern de la page 1.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.data_loader import DataLoader
from core.ingredients_analyzer import IngredientsAnalyzer
from core.logger import get_logger


@dataclass
class IngredientsClusteringConfig:
    """Configuration pour l'analyse de clustering d'ingrédients.

    Attributes:
        recipes_path: Chemin vers le fichier CSV contenant les recettes.
        n_ingredients: Nombre d'ingrédients les plus fréquents à analyser.
        n_clusters: Nombre de clusters à créer avec K-means.
        tsne_perplexity: Paramètre de perplexité pour la visualisation t-SNE.
    """

    recipes_path: Path
    n_ingredients: int = 50
    n_clusters: int = 5
    tsne_perplexity: int = 30


class IngredientsClusteringPage:
    """Page Streamlit pour l'analyse de clustering des ingrédients.

    Cette classe gère l'interface utilisateur et la logique de présentation
    pour l'analyse de co-occurrence et le clustering d'ingrédients basé sur
    leurs patterns d'apparition dans les recettes.

    Attributes:
        default_recipes_path: Chemin par défaut vers le fichier de recettes.
        logger: Instance du logger pour le suivi des opérations.
    """

    def __init__(self, default_recipes_path: str = "data/RAW_recipes.csv") -> None:
        """Initialise la page de clustering d'ingrédients.

        Args:
            default_recipes_path: Chemin par défaut vers le fichier CSV des recettes.
                Doit contenir une colonne avec les listes d'ingrédients.

        Raises:
            ValueError: Si le chemin fourni est invalide ou vide.
        """
        if not default_recipes_path:
            raise ValueError("Le chemin du fichier de recettes ne peut pas être vide")

        self.default_recipes_path = default_recipes_path
        self.logger = get_logger()
        self.logger.info("Initializing IngredientsClusteringPage")

    @st.cache_data
    def _load_and_prepare_data(_self) -> Optional[pd.DataFrame]:
        """Charge automatiquement le dataset au démarrage.

        Cette méthode est mise en cache par Streamlit pour éviter de recharger
        les données à chaque interaction utilisateur.

        Returns:
            DataFrame contenant les recettes si le chargement réussit, None sinon.
            Le DataFrame contient au minimum une colonne d'ingrédients.

        Raises:
            Exception: Affiche une erreur Streamlit mais ne propage pas l'exception.
        """
        try:
            data_loader = DataLoader(_self.default_recipes_path)
            data = data_loader.load_data()
            return data
        except Exception as e:
            st.error(f"Erreur lors du chargement des données : {e}")
            return None

    def _render_cache_controls(self, analyzer: IngredientsAnalyzer) -> None:
        """Affiche les contrôles de gestion du cache dans la sidebar.

        Permet à l'utilisateur de visualiser l'état du cache et de le supprimer
        si nécessaire. Affiche des métriques sur l'âge, la taille et le nombre
        de fichiers en cache.

        Args:
            analyzer: Instance de l'analyseur d'ingrédients dont on gère le cache.
        """
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Cache Management")

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
                st.sidebar.info("Cache sera créé après traitement")

            # Cache management buttons
            col1, col2 = st.sidebar.columns(2)
            with col1:
                if st.button(
                    "🗑️ Clear Cache",
                    help="Supprimer tous les fichiers de cache",
                    key="clear_ingredients_cache",
                ):
                    from core.cache_manager import get_cache_manager

                    cache_manager = get_cache_manager()
                    deleted_files = cache_manager.clear(analyzer_name="ingredients")
                    if deleted_files > 0:
                        st.sidebar.success(f"Cache effacé! ({deleted_files} fichiers)")
                        st.rerun()
                    else:
                        st.sidebar.info("Aucun fichier de cache à supprimer")

            with col2:
                if st.button(
                    "ℹ️ Info Cache",
                    help="Afficher les détails du cache",
                    key="info_ingredients_cache",
                ):
                    st.sidebar.json(cache_info)

            # Show total cache files
            if cache_info["cache_files_count"] > 0:
                st.sidebar.caption(f"📁 {cache_info['cache_files_count']} fichier(s) de cache")
        else:
            st.sidebar.warning("Cache désactivé")

    def render_sidebar(self) -> dict[str, int | bool]:
        """Affiche la sidebar avec les paramètres de clustering.

        Crée une interface interactive dans la sidebar permettant à l'utilisateur
        de configurer les paramètres de l'analyse de clustering:
        - Nombre d'ingrédients à analyser
        - Nombre de clusters à créer
        - Paramètres de visualisation t-SNE

        Returns:
            Dictionnaire contenant les paramètres sélectionnés par l'utilisateur:
                - n_ingredients: Nombre d'ingrédients les plus fréquents (10-200)
                - n_clusters: Nombre de groupes à créer (2-20)
                - tsne_perplexity: Paramètre de densité pour t-SNE (5-50)
                - analyze_button: True si le bouton d'analyse a été cliqué
        """
        st.sidebar.header("🔧 Paramètres de Clustering")

        # Paramètres de clustering
        n_ingredients = st.sidebar.slider(
            "Nombre d'ingrédients à analyser",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            help="Nombre d'ingrédients les plus fréquents à inclure dans l'analyse",
        )

        n_clusters = st.sidebar.slider(
            "Nombre de clusters",
            min_value=2,
            max_value=20,
            value=5,
            step=1,
            help="Nombre de groupes d'ingrédients à créer",
        )

        # Paramètres t-SNE
        st.sidebar.subheader("🎨 Paramètres Visualisation")
        tsne_perplexity = st.sidebar.slider(
            "Perplexité t-SNE",
            min_value=5,
            max_value=50,
            value=30,
            step=5,
            help="Contrôle la densité des groupes dans la visualisation",
        )

        # Bouton d'analyse dans la sidebar
        analyze_button = st.sidebar.button("🚀 Lancer l'analyse", type="primary")

        return {
            "n_ingredients": n_ingredients,
            "n_clusters": n_clusters,
            "tsne_perplexity": tsne_perplexity,
            "analyze_button": analyze_button,
        }

    def render_cooccurrence_analysis(self, ingredient_names: list[str], ingredients_matrix: pd.DataFrame) -> None:
        """Affiche l'analyse de co-occurrence interactive.

        Permet à l'utilisateur de sélectionner deux ingrédients et visualise
        leur score de co-occurrence (nombre de recettes où ils apparaissent ensemble).
        Affiche également des statistiques contextuelles pour interpréter le score.

        Args:
            ingredient_names: Liste des noms d'ingrédients disponibles pour la sélection.
            ingredients_matrix: Matrice de co-occurrence (DataFrame symétrique) où
                matrix[ing1, ing2] = nombre de recettes contenant ing1 ET ing2.

        Raises:
            ValueError: Si les ingrédients sélectionnés ne sont pas dans la matrice.
            IndexError: Si un accès invalide à la matrice est tenté.
        """
        st.subheader("🔍 Analyse de Co-occurrence")

        # Création de trois colonnes pour les menus déroulants
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            ingredient1 = st.selectbox(
                "Premier ingrédient",
                options=ingredient_names,
                index=0,
                key="ingredient1",
            )

        with col2:
            ingredient2 = st.selectbox(
                "Deuxième ingrédient",
                options=ingredient_names,
                index=1 if len(ingredient_names) > 1 else 0,
                key="ingredient2",
            )

        # Afficher le score de co-occurrence
        if ingredient1 and ingredient2 and ingredient1 != ingredient2:
            try:
                # Récupérer le score de co-occurrence
                cooccurrence_score = ingredients_matrix.at[ingredient1, ingredient2]

                # Calculer les statistiques de la matrice
                matrix_values = ingredients_matrix.values
                matrix_values_flat = matrix_values[matrix_values > 0]  # Seulement les valeurs non-nulles

                if len(matrix_values_flat) > 0:
                    max_score = np.max(matrix_values_flat)
                    avg_score = np.mean(matrix_values_flat)
                    median_score = np.median(matrix_values_flat)
                else:
                    max_score = avg_score = median_score = 0

                # Affichage des métriques
                col_metric1, col_metric2, col_metric3 = st.columns(3)

                with col_metric1:
                    st.metric(
                        label="Score de co-occurrence",
                        value=f"{cooccurrence_score:.0f}",
                        help=f"Nombre de recettes contenant '{ingredient1}' ET '{ingredient2}'",
                    )

                with col_metric2:
                    if max_score > 0:
                        percentile = (cooccurrence_score / max_score) * 100
                        st.metric(
                            label="Percentile",
                            value=f"{percentile:.1f}%",
                            help="Position par rapport au score maximum",
                        )

                with col_metric3:
                    if avg_score > 0:
                        ratio_avg = cooccurrence_score / avg_score
                        st.metric(
                            label="Ratio vs Moyenne",
                            value=f"{ratio_avg:.1f}x",
                            help=f"Ratio par rapport à la moyenne ({avg_score:.1f})",
                        )

                # Barre de progression visuelle
                if max_score > 0:
                    normalized_score = cooccurrence_score / max_score
                    st.progress(normalized_score)

                    # Interprétation du score
                    if cooccurrence_score >= median_score * 2:
                        st.success("🔥 Combinaison très fréquente!")
                    elif cooccurrence_score >= median_score:
                        st.info("✅ Combinaison courante")
                    elif cooccurrence_score > 0:
                        st.warning("⚠️ Combinaison rare")
                    else:
                        st.error("❌ Aucune co-occurrence trouvée")

            except (ValueError, IndexError, KeyError):
                st.warning("Erreur lors du calcul du score de co-occurrence")

    def render_clusters(self, clusters: np.ndarray, ingredient_names: list[str], n_clusters: int) -> None:
        """Affiche les clusters d'ingrédients de manière organisée.

        Présente chaque cluster dans un expander séparé avec une couleur distinctive.
        Les ingrédients sont affichés en colonnes pour une meilleure lisibilité.

        Args:
            clusters: Array numpy contenant les labels de cluster pour chaque ingrédient.
                Taille = len(ingredient_names), valeurs de 0 à n_clusters-1.
            ingredient_names: Liste ordonnée des noms d'ingrédients correspondant
                aux indices dans l'array clusters.
            n_clusters: Nombre total de clusters créés (pour l'itération).

        Example:
            >>> clusters = np.array([0, 1, 0, 2, 1])
            >>> names = ['salt', 'sugar', 'pepper', 'flour', 'honey']
            >>> page.render_clusters(clusters, names, 3)
            # Affiche 3 expanders avec les ingrédients regroupés
        """
        st.subheader("🎯 Clusters d'Ingrédients")

        # Affichage par cluster avec couleurs
        colors = ["🔴", "🟠", "🟡", "🟢", "🔵", "🟣", "⚫", "⚪", "🟤", "🔘"]

        for cluster_id in range(n_clusters):
            cluster_ingredients = [ingredient_names[i] for i, cluster in enumerate(clusters) if cluster == cluster_id]

            color_emoji = colors[cluster_id % len(colors)]

            with st.expander(
                f"{color_emoji} Cluster {cluster_id + 1} ({len(cluster_ingredients)} ingrédients)",
                expanded=True,
            ):
                # Affichage en colonnes pour une meilleure lisibilité
                cols = st.columns(4)
                for i, ingredient in enumerate(cluster_ingredients):
                    cols[i % 4].write(f"• **{ingredient}**")

    def render_tsne_visualization(self, analyzer: IngredientsAnalyzer, clusters: np.ndarray, tsne_perplexity: int) -> None:
        """Affiche la visualisation t-SNE 2D des clusters d'ingrédients.

        Génère et affiche un graphique interactif Plotly montrant les ingrédients
        dans un espace 2D obtenu par réduction de dimensionnalité t-SNE. Les points
        sont colorés selon leur cluster et peuvent être régénérés avec de nouveaux
        paramètres.

        Args:
            analyzer: Instance de IngredientsAnalyzer utilisée pour générer la
                visualisation t-SNE à partir de la matrice de co-occurrence.
            clusters: Array numpy des labels de cluster pour chaque ingrédient.
            tsne_perplexity: Paramètre de perplexité pour t-SNE (5-50).
                Contrôle la densité des groupes dans la visualisation.
                Valeurs faibles = focus local, valeurs élevées = structure globale.

        Notes:
            La visualisation est mise en cache dans st.session_state pour éviter
            de la recalculer à chaque interaction. Un bouton permet de forcer la
            régénération avec de nouveaux paramètres aléatoires.
        """
        col_title, col_button = st.columns([3, 1])
        with col_title:
            st.subheader("🎨 Visualisation t-SNE 2D des Clusters")
        with col_button:
            regenerate_tsne = st.button(
                "🔄 Régénérer t-SNE",
                help="Regénérer la visualisation avec de nouveaux paramètres",
            )

        # Générer t-SNE au premier lancement ou si demandé
        should_generate_tsne = "tsne_data" not in st.session_state or regenerate_tsne

        if should_generate_tsne:
            with st.spinner("Génération de la visualisation t-SNE..."):
                tsne_data = analyzer.generate_tsne_visualization(clusters, perplexity=tsne_perplexity)
                st.session_state["tsne_data"] = tsne_data
        else:
            tsne_data = st.session_state["tsne_data"]

        if "error" not in tsne_data:
            # Créer le graphique de dispersion avec Plotly
            fig_tsne = go.Figure()

            # Palette de couleurs hexadécimales pour t-SNE
            tsne_colors = [
                "#FF6B6B",
                "#4ECDC4",
                "#45B7D1",
                "#96CEB4",
                "#FFEAA7",
                "#DDA0DD",
                "#98D8C8",
                "#F7DC6F",
                "#BB8FCE",
                "#85C1E9",
            ]

            n_clusters = tsne_data["n_clusters"]

            # Ajouter les points par cluster pour avoir des couleurs distinctes
            for cluster_id in range(n_clusters):
                # Filtrer les données pour ce cluster
                cluster_mask = [label == cluster_id for label in tsne_data["cluster_labels"]]
                cluster_x = [x for i, x in enumerate(tsne_data["x_coords"]) if cluster_mask[i]]
                cluster_y = [y for i, y in enumerate(tsne_data["y_coords"]) if cluster_mask[i]]
                cluster_names = [name for i, name in enumerate(tsne_data["ingredient_names"]) if cluster_mask[i]]

                color = tsne_colors[cluster_id % len(tsne_colors)]

                fig_tsne.add_trace(
                    go.Scatter(
                        x=cluster_x,
                        y=cluster_y,
                        mode="markers+text",
                        marker=dict(
                            size=12,
                            color=color,
                            line=dict(width=2, color="white"),
                            opacity=0.8,
                        ),
                        text=cluster_names,
                        textposition="top center",
                        textfont=dict(size=10),
                        name=f"Cluster {cluster_id + 1}",
                        hovertemplate="<b>%{text}</b><br>Cluster: "
                        + f"{cluster_id + 1}<br>"
                        + "Coordonnées: (%{x:.2f}, %{y:.2f})<extra></extra>",
                    )
                )

            # Mise en forme du graphique
            fig_tsne.update_layout(
                title="Visualisation t-SNE des Ingrédients par Cluster",
                xaxis_title="Dimension t-SNE 1",
                yaxis_title="Dimension t-SNE 2",
                showlegend=True,
                height=600,
                hovermode="closest",
                plot_bgcolor="rgba(245,245,245,0.8)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )

            # Afficher le graphique
            st.plotly_chart(fig_tsne, use_container_width=True)

            # Informations sur t-SNE
            with st.expander("ℹ️ À propos de la visualisation t-SNE / Diagnostics"):
                st.markdown(
                    """
                **t-SNE (t-Distributed Stochastic Neighbor Embedding)** est une technique de réduction de dimensionnalité
                qui permet de visualiser des données haute-dimensionnelles en 2D.

                **Dans ce contexte :**
                - Chaque point représente un **ingrédient**
                - La position est basée sur les **profils de co-occurrence** avec les autres ingrédients
                - Les couleurs correspondent aux **clusters K-means**
                - Les ingrédients proches ont des **patterns de co-occurrence similaires**

                **Interprétation :**
                - Points regroupés = ingrédients utilisés dans des contextes similaires
                - Clusters colorés = groupes détectés par l'algorithme K-means
                - Distance = mesure de similarité des profils culinaires
                """
                )

                method = tsne_data.get("tsne_params", {}).get("method", "tsne")
                st.markdown(
                    f"""
                **Paramètres & Méthode :**
                - Méthode effective : `{method}`
                - Perplexité (après ajustement) : {tsne_data['tsne_params']['perplexity']}
                - Itérations max : {tsne_data['tsne_params']['max_iter']}
                - Seed aléatoire : {tsne_data['tsne_params']['random_state']}
                - Ingrédients (n_samples) : {len(tsne_data['ingredient_names'])}
                """
                )

                if method != "tsne":
                    if method == "fallback_circle":
                        st.warning(
                            "Fallback circle layout utilisé car t-SNE instable (trop peu d'ingrédients ou matrice dégénérée)."
                        )
                    elif method == "fallback_svd":
                        st.info("Projection issue de la décomposition SVD (approximation PCA) suite à un échec t-SNE.")

                # Afficher quelques stats basiques sur la dispersion
                try:
                    xs = tsne_data["x_coords"]
                    ys = tsne_data["y_coords"]
                    spread_x = max(xs) - min(xs)
                    spread_y = max(ys) - min(ys)
                    st.caption(f"Dispersion: Δx={spread_x:.2f}, Δy={spread_y:.2f} (échelle relative des clusters)")
                except Exception:
                    pass
        else:
            st.error("Erreur lors de la génération de la visualisation t-SNE")
            with st.expander("🛠 Détails de l'erreur"):
                st.json(tsne_data)
                st.markdown(
                    """
                **Causes possibles :**
                - Perplexité trop élevée par rapport au nombre d'ingrédients (doit être < n_samples)
                - Matrice de co-occurrence vide ou dégénérée (toutes valeurs nulles)
                - Incohérence entre le nombre de labels de clusters et la liste d'ingrédients
                - Conflit de cache sur des anciennes données

                **Actions suggérées :**
                1. Réduire le nombre d'ingrédients ou ajuster la perplexité
                2. Vider le cache (bouton Clear Cache) puis relancer
                3. Vérifier que l'étape de clustering a bien été effectuée
                """
                )

    def render_sidebar_statistics(self, clusters: Optional[np.ndarray], ingredient_names: Optional[list[str]]) -> None:
        """Affiche les statistiques de clustering dans la sidebar.

        Présente des métriques récapitulatives et un graphique de répartition
        des ingrédients par cluster. N'affiche rien si les données ne sont pas
        disponibles.

        Args:
            clusters: Array numpy des labels de cluster, ou None si l'analyse
                n'a pas encore été effectuée.
            ingredient_names: Liste des noms d'ingrédients, ou None si l'analyse
                n'a pas encore été effectuée.

        Notes:
            Cette méthode vérifie que les deux paramètres sont non-None avant
            d'afficher les statistiques. Le graphique utilise Plotly pour une
            visualisation interactive.
        """
        if clusters is not None and ingredient_names is not None:
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 📊 Statistiques")

            # Comptage par cluster
            cluster_counts = pd.Series(clusters).value_counts().sort_index()

            st.sidebar.metric("Total ingrédients", len(ingredient_names))
            st.sidebar.metric("Nombre de clusters", len(cluster_counts))

            # Graphique horizontal des proportions par cluster
            st.sidebar.markdown("**Répartition par cluster:**")

            # Créer le graphique avec Plotly
            colors = [
                "#FF6B6B",
                "#4ECDC4",
                "#45B7D1",
                "#96CEB4",
                "#FFEAA7",
                "#DDA0DD",
                "#98D8C8",
                "#F7DC6F",
                "#BB8FCE",
                "#85C1E9",
            ]

            fig = go.Figure()

            for i, count in enumerate(cluster_counts):
                percentage = (count / len(ingredient_names)) * 100
                color = colors[i % len(colors)]

                fig.add_trace(
                    go.Bar(
                        x=[count],
                        y=[f"Cluster {i + 1}"],
                        orientation="h",
                        name=f"Cluster {i + 1}",
                        marker_color=color,
                        text=f"{count} ({percentage:.1f}%)",
                        textposition="outside",
                        showlegend=False,
                    )
                )

            fig.update_layout(
                title="",
                xaxis_title="Nombre d'ingrédients",
                yaxis_title="",
                height=min(400, len(cluster_counts) * 40 + 100),
                margin=dict(l=10, r=10, t=10, b=10),
                font=dict(size=10),
            )

            st.sidebar.plotly_chart(fig, use_container_width=True)

    # ---------------- Étapes de l'analyse ---------------- #

    def _render_step_1_preprocessing(self, analyzer: IngredientsAnalyzer) -> None:
        """Affiche l'étape 1 : Prétraitement NLP des ingrédients.

        Args:
            analyzer: Instance de l'analyseur contenant les résultats du preprocessing.
        """
        st.markdown("---")
        st.header("📈 ÉTAPE 1 : Prétraitement NLP des ingrédients")

        st.markdown(
            """
        **Question :** Comment normaliser et regrouper les variantes d'un même ingrédient ?

        Les recettes utilisent des descriptions variées pour un même ingrédient (ex: "sel", "gros sel",
        "sel de mer", "sel fin"). Le prétraitement NLP vise à identifier et regrouper ces variantes
        pour créer une représentation cohérente.

        **Métrique :** Taux de réduction du nombre d'ingrédients uniques après normalisation.
        """
        )

        # Afficher le résumé du preprocessing
        if hasattr(analyzer, "ingredient_groups") and analyzer.ingredient_groups:
            with st.expander("🔍 Détails du prétraitement", expanded=True):
                # Récupérer les statistiques de traitement
                summary = analyzer.get_processing_summary()

                if "error" not in summary:
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Ingrédients bruts uniques",
                            f"{summary['normalization']['total_unique_raw']:,}",
                            help="Nombre d'ingrédients uniques avant normalisation",
                        )

                    with col2:
                        st.metric(
                            "Après normalisation",
                            f"{summary['normalization']['total_normalized']:,}",
                            delta=f"-{summary['normalization']['reduction_ratio']}%",
                            help="Nombre d'ingrédients après regroupement des variantes",
                        )

                    with col3:
                        st.metric(
                            "Groupes créés",
                            f"{summary['grouping']['groups_with_multiple_items']}",
                            help="Nombre de groupes contenant plusieurs variantes",
                        )

                    # Exemples de normalisation
                    st.markdown("**🧪 Exemples de normalisation :**")
                    test_ingredients = [
                        "large eggs",
                        "fresh ground black pepper",
                        "unsalted butter",
                        "red onions",
                        "whole milk",
                        "extra virgin olive oil",
                    ]
                    for ing in test_ingredients:
                        normalized = analyzer.normalize_ingredient(ing)
                        st.write(f"• `{ing}` → `{normalized}`")

                    # Exemples de regroupements
                    multi_groups = [g for g in analyzer.ingredient_groups if len(g) > 1]
                    if multi_groups:
                        st.markdown("**🔗 Exemples de regroupements d'ingrédients similaires :**")
                        for i, group in enumerate(multi_groups[:5]):
                            members_display = " | ".join(group[:5])
                            if len(group) > 5:
                                members_display += f" (+ {len(group) - 5} autres)"
                            st.write(f"**Groupe {i + 1}:** {members_display}")

        st.markdown(
            """
        **💡 Observations :** Le prétraitement NLP réduit significativement la redondance en identifiant
        les variantes linguistiques d'un même ingrédient. Cette étape est cruciale pour obtenir une
        matrice de co-occurrence fiable.

        **🎯 Implication :** La normalisation permet de concentrer l'analyse sur les véritables patterns
        culinaires plutôt que sur les variations de nomenclature.
        """
        )

    def _render_step_2_cooccurrence(self, ingredient_names: list[str], ingredients_matrix: pd.DataFrame) -> None:
        """Affiche l'étape 2 : Création de la matrice de co-occurrence.

        Args:
            ingredient_names: Liste des noms d'ingrédients.
            ingredients_matrix: Matrice de co-occurrence.
        """
        st.markdown("---")
        st.header("📈 ÉTAPE 2 : Matrice de co-occurrence")

        st.markdown(
            """
        **Objectif :** Quantifier la fréquence d'apparition conjointe de chaque paire d'ingrédients.

        La matrice de co-occurrence capture l'information fondamentale : combien de fois deux
        ingrédients apparaissent ensemble dans les recettes. Cette matrice symétrique constitue
        la base de notre analyse de similarité.

        **Méthode :** Pour chaque recette, toutes les paires d'ingrédients présents sont comptabilisées.
        """
        )

        # Statistiques de la matrice
        total_cooccurrences = int(ingredients_matrix.values.sum() / 2)
        non_zero_pairs = int((ingredients_matrix.values > 0).sum() / 2)
        matrix_size = len(ingredient_names)
        max_possible_pairs = matrix_size * (matrix_size - 1) / 2
        sparsity = (1 - non_zero_pairs / max_possible_pairs) * 100

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Dimension matrice", f"{matrix_size}×{matrix_size}")
        with col2:
            st.metric("Co-occurrences totales", f"{total_cooccurrences:,}")
        with col3:
            st.metric("Paires non-nulles", f"{non_zero_pairs:,}")
        with col4:
            st.metric(
                "Sparsité",
                f"{sparsity:.1f}%",
                help="Pourcentage de paires sans co-occurrence",
            )

        st.markdown("---")

        # Analyse interactive de co-occurrence
        self.render_cooccurrence_analysis(ingredient_names, ingredients_matrix)

        st.markdown(
            """
        **📊 Ce que révèle la matrice :**

        La distribution des co-occurrences n'est pas uniforme. Certaines paires d'ingrédients
        apparaissent ensemble dans des milliers de recettes, révélant des associations culinaires
        fortes.

        """
        )

    def _render_step_3_clustering(self, clusters: np.ndarray, ingredient_names: list[str], n_clusters: int) -> None:
        """Affiche l'étape 3 : Clustering K-means.

        Args:
            clusters: Array des labels de cluster.
            ingredient_names: Liste des noms d'ingrédients.
            n_clusters: Nombre de clusters créés.
        """
        st.markdown("---")
        st.header("📈 ÉTAPE 3 : Clustering K-means")

        st.markdown(
            f"""
        **Objectif :** Regrouper automatiquement les ingrédients en {n_clusters} familles distinctes.

        L'algorithme K-means partitionne les ingrédients en fonction de leurs profils de co-occurrence.
        Deux ingrédients dans le même cluster partagent des contextes d'utilisation similaires, même
        s'ils ne co-occurrent pas directement.

        **Méthode :** K-means avec k={n_clusters}, distance euclidienne sur les vecteurs de co-occurrence.
        """
        )

        # Statistiques des clusters
        cluster_counts = pd.Series(clusters).value_counts().sort_index()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nombre de clusters", n_clusters)
        with col2:
            avg_size = len(ingredient_names) / n_clusters
            st.metric("Taille moyenne", f"{avg_size:.1f} ingrédients")
        with col3:
            largest_cluster_size = cluster_counts.max()
            st.metric("Plus grand cluster", f"{largest_cluster_size} ingrédients")

        st.markdown("---")

        # Affichage des clusters
        self.render_clusters(clusters, ingredient_names, n_clusters)

        st.markdown(
            f"""
        **🎯 Interprétation des clusters :**

        Les clusters arrivent à révéler des "famille culinaire" d'ingrédients. Ils peuvent être :
        - **Ingrédients pour patisserie**
        - **Produits de recettes salés**

        **Limite méthodologique** : Le choix de k={n_clusters} est paramétrique. Différentes valeurs
        de k révèlent des structures à différentes granularités.
        De plus, les clusters ont tendance à ne pas être de la même taille car une masse d'ingrédient à faible co-occurence se regrouppent ensemble.
        """
        )

    def _render_step_4_visualization(self, analyzer: IngredientsAnalyzer, clusters: np.ndarray, tsne_perplexity: int) -> None:
        """Affiche l'étape 4 : Visualisation t-SNE 2D.

        Args:
            analyzer: Instance de l'analyseur.
            clusters: Array des labels de cluster.
            tsne_perplexity: Paramètre de perplexité pour t-SNE.
        """
        st.markdown("---")
        st.header("📈 ÉTAPE 4 : Visualisation t-SNE 2D")

        st.markdown(
            """
        **Objectif :** Projeter l'espace haute-dimensionnalité des co-occurrences en 2D pour exploration visuelle.

        La matrice de co-occurrence est un espace à n dimensions (une par ingrédient). t-SNE
        (t-Distributed Stochastic Neighbor Embedding) réduit cette dimensionnalité à 2D tout en
        préservant les proximités locales.

        **Méthode :** t-SNE avec perplexité={}, optimisation par descente de gradient.
        """.format(
                tsne_perplexity
            )
        )

        # Visualisation t-SNE
        self.render_tsne_visualization(analyzer, clusters, tsne_perplexity)

        st.markdown(
            """
        **🔍 Lecture de la visualisation :**

        - **Proximité spatiale** : Les ingrédients proches dans l'espace 2D ont des profils de
          co-occurrence similaires (utilisés dans des contextes culinaires similaires)
        - **Couleurs** : Chaque couleur représente un cluster K-means. La cohésion spatiale des
          couleurs valide la qualité du clustering
        - **Groupes isolés** : Les clusters bien séparés géographiquement indiquent des familles
          culinaires distinctes

        **💡 Insights visuels :**

        La visualisation révèle souvent une structure non-linéaire de l'espace culinaire. Certains
        ingrédients "pont" peuvent se situer entre plusieurs clusters, reflétant leur polyvalence
        (ex: l'huile d'olive utilisée dans de multiples contextes, ou l'eau).

        **Validation du clustering** : Si les couleurs (clusters K-means) forment des groupes
        visuellement cohérents dans l'espace t-SNE, cela confirme que le clustering a capturé
        des structures réelles plutôt qu'artificielles.

        **Limite de t-SNE** : La représentation 2D est approximative. Les distances absolues ne
        sont pas strictement préservées, seules les proximités relatives comptent. Différentes
        exécutions peuvent donner des configurations légèrement différentes (non-déterminisme).
        """
        )

    def _render_conclusion(self, ingredient_names: list[str], clusters: np.ndarray, n_clusters: int) -> None:
        """Affiche la conclusion de l'analyse.

        Args:
            ingredient_names: Liste des noms d'ingrédients.
            clusters: Array des labels de cluster.
            n_clusters: Nombre de clusters créés.
        """
        st.markdown("---")
        st.subheader("📋 Conclusion de l'analyse")

        # Calculer quelques statistiques finales
        cluster_counts = pd.Series(clusters).value_counts()
        largest_cluster = cluster_counts.max()
        smallest_cluster = cluster_counts.min()

        st.markdown(
            f"""
        ### Synthèse des résultats

        **1. Prétraitement NLP réussi :** La normalisation automatique a permis de réduire
        significativement la redondance des variantes d'ingrédients, créant une base solide
        pour l'analyse.

        **2. Structure révélée par la co-occurrence :** L'analyse de {len(ingredient_names)}
        ingrédients a révélé des patterns clairs d'association culinaire, confirmant que la
        cuisine n'est pas aléatoire.

        **3. Clustering cohérent :** L'algorithme K-means a identifié {n_clusters} familles
        d'ingrédients distinctes, avec des tailles variant de {smallest_cluster} à {largest_cluster}
        ingrédients. Ces clusters essaye de capturer des insight sur le co-usage des ingrédients.

        **4. Validation visuelle :** La projection t-SNE montre la structure des clusters et
        l'organisation de l'espace culinaire.

        ### Applications pratiques

        Ces résultats peuvent être utilisés pour :
        - **Systèmes de recommandation** : Suggérer des ingrédients complémentaires lors de la
          création de recettes
        - **Analyse nutritionnelle** : Identifier les associations alimentaires courantes pour
          des études diététiques, nottament en reliant les informations caloriques
        - **Créativité culinaire** : Découvrir des combinaisons innovantes en explorant les
          frontières entre clusters
        - **Détection d'anomalies** : Identifier des recettes avec des combinaisons inhabituelles

        ### Limites et perspectives

        **Limites :**
        - La co-occurrence ne capture pas l'ordre ou les quantités des ingrédients
        - Les ingrédients très rares ne sont pas représentés et ceux trop présent
        peuvent être mal représentés

        **Perspectives d'amélioration :**
        - Clustering hiérarchique pour révéler plusieurs niveaux de granularité
        - Intégration d'informations sémantiques (catégories nutritionnelles, origines)
        - Modèles de recommandation basés sur les embeddings d'ingrédients
        """
        )

    def render_analysis_summary(self, analyzer: IngredientsAnalyzer) -> None:
        """Affiche le résumé détaillé du processus d'analyse.

        Présente des informations sur le regroupement d'ingrédients similaires,
        la normalisation effectuée et des exemples de mappings. Utile pour
        comprendre les transformations appliquées aux données brutes.

        Args:
            analyzer: Instance de l'analyseur contenant les résultats du
                traitement (groupes d'ingrédients, mappings, etc.).

        Notes:
            Affiche plusieurs sections extensibles:
            - Exemples de regroupements d'ingrédients similaires
            - Debug de la normalisation pour des ingrédients courants
            - Tests de normalisation en temps réel
            - Résumé complet du pipeline de traitement
        """
        # Afficher quelques exemples de regroupements d'ingrédients
        if hasattr(analyzer, "ingredient_groups") and analyzer.ingredient_groups:
            with st.expander("🔗 Exemples de regroupements d'ingrédients similaires"):
                # Afficher les groupes avec plus d'un élément
                multi_groups = [g for g in analyzer.ingredient_groups if len(g) > 1]

                if multi_groups:
                    # Afficher les 10 premiers groupes
                    for i, group in enumerate(multi_groups[:10]):
                        members_display = " | ".join(group[:5])
                        if len(group) > 5:
                            members_display += f" (+ {len(group) - 5} autres)"
                        st.write(f"**Groupe {i + 1}:** {members_display}")

                    st.info(f"Total: {len(multi_groups)} groupes d'ingrédients similaires détectés")

                    # Debug pour des ingrédients problématiques
                    debug_info = analyzer.debug_ingredient_mapping(["pepper", "egg", "salt", "butter", "onion"])
                    if "search_results" in debug_info:
                        st.write("**🔍 Debug - Exemples de normalisation:**")
                        for term, matches in debug_info["search_results"].items():
                            if matches:
                                st.write(f"*{term.title()}:*")
                                for match in matches[:3]:  # Limiter à 3 résultats
                                    # Montrer aussi la normalisation
                                    normalized = analyzer.normalize_ingredient(match["ingredient"])
                                    status = (
                                        "✅ Représentant"
                                        if match["is_representative"]
                                        else f"➡️ Mappé vers '{match['representative']}'"
                                    )
                                    st.write(
                                        f"  • `{match['ingredient']}` → `{normalized}` {status}"
                                    )

                    # Exemple de normalisation en temps réel
                    st.write("**🧪 Test de normalisation:**")
                    test_ingredients = [
                        "large eggs",
                        "fresh ground black pepper",
                        "unsalted butter",
                        "red onions",
                        "whole milk",
                        "extra virgin olive oil",
                    ]
                    for ing in test_ingredients:
                        normalized = analyzer.normalize_ingredient(ing)
                        st.write(f"• `{ing}` → `{normalized}`")

                    # Résumé complet du processus
                    with st.expander("📋 Résumé Complet du Data Processing"):
                        summary = analyzer.get_processing_summary()
                        if "error" not in summary:
                            col1, col2 = st.columns(2)

                            with col1:
                                st.write("**📊 Données d'entrée:**")
                                st.write(
                                    f"• Recettes: {
                                        summary['input_data']['total_recipes']:,}"
                                )
                                st.write(
                                    f"• Ingrédients bruts: {
                                        summary['input_data']['total_raw_ingredients']:,}"
                                )
                                st.write(
                                    f"• Moyenne par recette: {
                                        summary['input_data']['avg_ingredients_per_recipe']}"
                                )

                                st.write("**🔄 Normalisation:**")
                                st.write(
                                    f"• Ingrédients uniques bruts: {
                                        summary['normalization']['total_unique_raw']:,}"
                                )
                                st.write(
                                    f"• Après normalisation: {
                                        summary['normalization']['total_normalized']:,}"
                                )
                                st.write(
                                    f"• Réduction: {
                                        summary['normalization']['reduction_ratio']}%"
                                )

                            with col2:
                                st.write("**🔗 Regroupement:**")
                                st.write(
                                    f"• Groupes multiples: {
                                        summary['grouping']['groups_with_multiple_items']}"
                                )
                                st.write(
                                    f"• Plus grand groupe: {
                                        summary['grouping']['largest_group_size']} éléments"
                                )

                                st.write("**📈 Matrice Co-occurrence:**")
                                st.write(
                                    f"• Dimensions: {
                                        summary['cooccurrence_matrix']['dimensions']}"
                                )
                                st.write(f"• Co-occurrences: {summary['cooccurrence_matrix']['total_cooccurrences']:,}")
                                st.write(f"• Paires non-nulles: {summary['cooccurrence_matrix']['non_zero_pairs']:,}")
                                st.write(
                                    f"• Sparsité: {
                                        summary['cooccurrence_matrix']['sparsity']}%"
                                )
                else:
                    st.warning("Aucun regroupement détecté. Tous les ingrédients sont considérés comme uniques.")

    def run(self) -> None:
        """Point d'entrée principal de la page.

        Orchestre l'ensemble du workflow de la page de clustering:
        1. Chargement automatique des données
        2. Affichage de la sidebar avec paramètres
        3. Exécution de l'analyse (process, clustering, visualisation)
        4. Affichage des résultats interactifs

        Cette méthode gère également le cache de session Streamlit pour
        persister les résultats entre les interactions utilisateur.

        Raises:
            Exception: Affiche les erreurs via st.error mais ne les propage pas
                pour maintenir l'interface fonctionnelle.
        """
        self.logger.info("Starting ingredients clustering analysis")

        # Introduction et User Story
        with st.expander("🎯 Objectifs et méthodologie de l'analyse", expanded=True):
            st.markdown(
                """
            ### Peut-on regrouper les ingrédients selon leurs usages culinaires ?

            Cette analyse explore les patterns de co-occurrence d'ingrédients dans les recettes pour
            identifier les associations culinaires naturelles. En analysant des milliers de recettes,
            nous révélons les combinaisons d'ingrédients qui apparaissent fréquemment ensemble.

            **Questions centrales :** Quels ingrédients sont naturellement associés ? Existe-t-il des
            familles d'ingrédients distinctes ? Comment les ingrédients se regroupent-ils en fonction
            de leurs profils d'utilisation ?

            **Approche :** Analyse NLP des listes d'ingrédients, construction d'une matrice de
            co-occurrence, clustering automatique par K-means, et visualisation en 2D par t-SNE.

            **Problématique :** Dans un espace culinaire où des milliers d'ingrédients peuvent être
            combinés, comment identifier automatiquement les groupes d'ingrédients qui partagent des
            contextes d'utilisation similaires ?
            """
            )

        # Sidebar pour les paramètres
        params = self.render_sidebar()
        self.logger.debug(f"Clustering parameters: {params}")

        # Chargement automatique des données
        self.logger.debug("Loading and preparing data")
        data = self._load_and_prepare_data()

        # Traitement des données
        if data is not None:
            self.logger.info(f"Dataset loaded successfully: {len(data)} recipes")

            # Initialisation de l'analyseur
            analyzer = IngredientsAnalyzer(data)

            # Cache controls dans la sidebar
            self._render_cache_controls(analyzer)

            # Vérifier si les paramètres ont changé
            params_changed = False
            if "last_params" in st.session_state:
                last_params = st.session_state["last_params"]
                if (
                    last_params["n_ingredients"] != params["n_ingredients"]
                    or last_params["n_clusters"] != params["n_clusters"]
                    or last_params["tsne_perplexity"] != params["tsne_perplexity"]
                ):
                    params_changed = True

            # Lancer l'analyse si bouton cliqué, première fois, ou paramètres changés
            should_analyze = params["analyze_button"] or "ingredient_names" not in st.session_state or params_changed

            if should_analyze:
                self.logger.info(
                    f"Starting clustering analysis with parameters: n_ingredients={
                        params['n_ingredients']}, n_clusters={
                        params['n_clusters']}"
                )
                with st.spinner("Analyse en cours..."):
                    # Traitement des ingrédients
                    self.logger.debug(
                        f"Processing ingredients with n_ingredients={
                            params['n_ingredients']}"
                    )
                    ingredients_matrix, ingredient_names = analyzer.process_ingredients(params["n_ingredients"])
                    self.logger.info(f"Processed ingredients matrix: {ingredients_matrix.shape}")

                    # Clustering
                    self.logger.debug(f"Performing clustering with n_clusters={params['n_clusters']}")
                    clusters = analyzer.perform_clustering(ingredients_matrix, params["n_clusters"])
                    self.logger.info(f"Clustering completed: {len(set(clusters))} unique clusters found")

                    # Sauvegarde des résultats dans la session
                    st.session_state["ingredient_names"] = ingredient_names
                    st.session_state["clusters"] = clusters
                    st.session_state["ingredients_matrix"] = ingredients_matrix
                    st.session_state["analyzer"] = analyzer
                    st.session_state["last_params"] = params.copy()

            # Affichage des résultats si disponibles
            if "ingredient_names" in st.session_state:
                self.logger.debug("Displaying cached clustering results")
                ingredient_names = st.session_state["ingredient_names"]
                ingredients_matrix = st.session_state["ingredients_matrix"]
                clusters = st.session_state["clusters"]
                analyzer = st.session_state["analyzer"]

                # Statistiques générales
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📊 Recettes analysées", f"{len(data):,}")
                with col2:
                    st.metric("🥘 Ingrédients retenus", f"{len(ingredient_names)}")
                with col3:
                    st.metric("🎯 Clusters créés", f"{params['n_clusters']}")

                # ÉTAPE 1 : Prétraitement NLP
                self._render_step_1_preprocessing(analyzer)

                # ÉTAPE 2 : Matrice de co-occurrence
                self._render_step_2_cooccurrence(ingredient_names, ingredients_matrix)

                # ÉTAPE 3 : Clustering
                self._render_step_3_clustering(clusters, ingredient_names, params["n_clusters"])

                # ÉTAPE 4 : Visualisation t-SNE
                self._render_step_4_visualization(analyzer, clusters, params["tsne_perplexity"])

                # Conclusion
                self._render_conclusion(ingredient_names, clusters, params["n_clusters"])

                # Statistiques dans la sidebar
                self.render_sidebar_statistics(clusters, ingredient_names)

        else:
            st.error("Impossible de charger les données. Vérifiez la présence du fichier de données.")

        # Footer
        st.markdown("---")
        st.caption(
            "💡 **Configuration** : Ajustez les paramètres dans la sidebar pour explorer différentes configurations de clustering."
        )
