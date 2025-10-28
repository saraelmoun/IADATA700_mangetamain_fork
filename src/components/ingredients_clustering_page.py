from __future__ import annotations

"""Page de clustering des ingrédients basée sur la co-occurrence."""

from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.data_loader import DataLoader
from core.ingredients_analyzer import IngredientsAnalyzer
from core.logger import get_logger


class IngredientsClusteringPage:
    """Page pour l'analyse de clustering des ingrédients."""
    
    def __init__(self, default_recipes_path: str = "data/RAW_recipes.csv"):
        """
        Initialise la page de clustering.
        
        Args:
            default_recipes_path: Chemin par défaut vers le fichier de recettes
        """
        self.default_recipes_path = default_recipes_path
        self.logger = get_logger()
        self.logger.info("Initializing IngredientsClusteringPage")
    
    @st.cache_data
    def _load_and_prepare_data(_self):
        """Charge automatiquement le dataset au démarrage."""
        try:
            data_loader = DataLoader(_self.default_recipes_path)
            data = data_loader.load_data()
            return data
        except Exception as e:
            st.error(f"Erreur lors du chargement des données : {e}")
            return None
    
    def _render_cache_controls(self, analyzer: IngredientsAnalyzer):
        """Render cache management controls in sidebar."""
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
                if st.button("🗑️ Clear Cache", help="Supprimer tous les fichiers de cache", key="clear_ingredients_cache"):
                    from core.cache_manager import get_cache_manager
                    cache_manager = get_cache_manager()
                    deleted_files = cache_manager.clear(analyzer_name="ingredients")
                    if deleted_files > 0:
                        st.sidebar.success(f"Cache effacé! ({deleted_files} fichiers)")
                        st.rerun()
                    else:
                        st.sidebar.info("Aucun fichier de cache à supprimer")
            
            with col2:
                if st.button("ℹ️ Info Cache", help="Afficher les détails du cache", key="info_ingredients_cache"):
                    st.sidebar.json(cache_info)
                    
            # Show total cache files
            if cache_info["cache_files_count"] > 0:
                st.sidebar.caption(f"📁 {cache_info['cache_files_count']} fichier(s) de cache")
        else:
            st.sidebar.warning("Cache désactivé")
    
    def render_sidebar(self) -> dict:
        """
        Affiche la sidebar avec les paramètres de clustering.
        
        Returns:
            Dictionnaire avec les paramètres sélectionnés
        """
        st.sidebar.header("🔧 Paramètres de Clustering")
        
        # Paramètres de clustering
        n_ingredients = st.sidebar.slider(
            "Nombre d'ingrédients à analyser",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            help="Nombre d'ingrédients les plus fréquents à inclure dans l'analyse"
        )
        
        n_clusters = st.sidebar.slider(
            "Nombre de clusters",
            min_value=2,
            max_value=20,
            value=5,
            step=1,
            help="Nombre de groupes d'ingrédients à créer"
        )
        
        # Paramètres t-SNE
        st.sidebar.subheader("🎨 Paramètres Visualisation")
        tsne_perplexity = st.sidebar.slider(
            "Perplexité t-SNE",
            min_value=5,
            max_value=50,
            value=30,
            step=5,
            help="Contrôle la densité des groupes dans la visualisation"
        )
        
        # Bouton d'analyse dans la sidebar
        analyze_button = st.sidebar.button("🚀 Lancer l'analyse", type="primary")
        
        return {
            "n_ingredients": n_ingredients,
            "n_clusters": n_clusters,
            "tsne_perplexity": tsne_perplexity,
            "analyze_button": analyze_button
        }
    
    def render_cooccurrence_analysis(self, ingredient_names: list, ingredients_matrix: pd.DataFrame) -> None:
        """
        Affiche l'analyse de co-occurrence interactive.
        
        Args:
            ingredient_names: Liste des noms d'ingrédients
            ingredients_matrix: Matrice de co-occurrence
        """
        st.subheader("🔍 Analyse de Co-occurrence")
        
        # Création de trois colonnes pour les menus déroulants
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            ingredient1 = st.selectbox(
                "Premier ingrédient",
                options=ingredient_names,
                index=0,
                key="ingredient1"
            )
        
        with col2:
            ingredient2 = st.selectbox(
                "Deuxième ingrédient",
                options=ingredient_names,
                index=1 if len(ingredient_names) > 1 else 0,
                key="ingredient2"
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
                        help=f"Nombre de recettes contenant '{ingredient1}' ET '{ingredient2}'"
                    )
                
                with col_metric2:
                    if max_score > 0:
                        percentile = (cooccurrence_score / max_score) * 100
                        st.metric(
                            label="Percentile",
                            value=f"{percentile:.1f}%",
                            help="Position par rapport au score maximum"
                        )
                
                with col_metric3:
                    if avg_score > 0:
                        ratio_avg = cooccurrence_score / avg_score
                        st.metric(
                            label="Ratio vs Moyenne",
                            value=f"{ratio_avg:.1f}x",
                            help=f"Ratio par rapport à la moyenne ({avg_score:.1f})"
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
    
    def render_clusters(self, clusters: np.ndarray, ingredient_names: list, n_clusters: int) -> None:
        """
        Affiche les clusters d'ingrédients.
        
        Args:
            clusters: Labels des clusters
            ingredient_names: Liste des noms d'ingrédients
            n_clusters: Nombre de clusters
        """
        st.subheader("🎯 Clusters d'Ingrédients")
        
        # Affichage par cluster avec couleurs
        colors = ["🔴", "🟠", "🟡", "🟢", "🔵", "🟣", "⚫", "⚪", "🟤", "🔘"]
        
        for cluster_id in range(n_clusters):
            cluster_ingredients = [
                ingredient_names[i] for i, cluster in enumerate(clusters) 
                if cluster == cluster_id
            ]
            
            color_emoji = colors[cluster_id % len(colors)]
            
            with st.expander(
                f"{color_emoji} Cluster {cluster_id + 1} ({len(cluster_ingredients)} ingrédients)", 
                expanded=True
            ):
                # Affichage en colonnes pour une meilleure lisibilité
                cols = st.columns(4)
                for i, ingredient in enumerate(cluster_ingredients):
                    cols[i % 4].write(f"• **{ingredient}**")
    
    def render_tsne_visualization(self, analyzer: IngredientsAnalyzer, clusters: np.ndarray, tsne_perplexity: int) -> None:
        """
        Affiche la visualisation t-SNE.
        
        Args:
            analyzer: Instance de l'analyseur d'ingrédients
            clusters: Labels des clusters
            tsne_perplexity: Paramètre de perplexité pour t-SNE
        """
        col_title, col_button = st.columns([3, 1])
        with col_title:
            st.subheader("🎨 Visualisation t-SNE 2D des Clusters")
        with col_button:
            regenerate_tsne = st.button("🔄 Régénérer t-SNE", help="Regénérer la visualisation avec de nouveaux paramètres")
        
        # Générer t-SNE au premier lancement ou si demandé
        should_generate_tsne = 'tsne_data' not in st.session_state or regenerate_tsne
        
        if should_generate_tsne:
            with st.spinner("Génération de la visualisation t-SNE..."):
                tsne_data = analyzer.generate_tsne_visualization(clusters, perplexity=tsne_perplexity)
                st.session_state['tsne_data'] = tsne_data
        else:
            tsne_data = st.session_state['tsne_data']
        
        if "error" not in tsne_data:
            # Créer le graphique de dispersion avec Plotly
            fig_tsne = go.Figure()
            
            # Palette de couleurs hexadécimales pour t-SNE
            tsne_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", 
                          "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9"]
            
            n_clusters = tsne_data['n_clusters']
            
            # Ajouter les points par cluster pour avoir des couleurs distinctes
            for cluster_id in range(n_clusters):
                # Filtrer les données pour ce cluster
                cluster_mask = [label == cluster_id for label in tsne_data['cluster_labels']]
                cluster_x = [x for i, x in enumerate(tsne_data['x_coords']) if cluster_mask[i]]
                cluster_y = [y for i, y in enumerate(tsne_data['y_coords']) if cluster_mask[i]]
                cluster_names = [name for i, name in enumerate(tsne_data['ingredient_names']) if cluster_mask[i]]
                
                color = tsne_colors[cluster_id % len(tsne_colors)]
                
                fig_tsne.add_trace(go.Scatter(
                    x=cluster_x,
                    y=cluster_y,
                    mode='markers+text',
                    marker=dict(
                        size=12,
                        color=color,
                        line=dict(width=2, color='white'),
                        opacity=0.8
                    ),
                    text=cluster_names,
                    textposition="top center",
                    textfont=dict(size=10),
                    name=f"Cluster {cluster_id + 1}",
                    hovertemplate="<b>%{text}</b><br>Cluster: " + f"{cluster_id + 1}<br>" +
                                  "Coordonnées: (%{x:.2f}, %{y:.2f})<extra></extra>"
                ))
            
            # Mise en forme du graphique
            fig_tsne.update_layout(
                title="Visualisation t-SNE des Ingrédients par Cluster",
                xaxis_title="Dimension t-SNE 1",
                yaxis_title="Dimension t-SNE 2",
                showlegend=True,
                height=600,
                hovermode='closest',
                plot_bgcolor='rgba(245,245,245,0.8)',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Afficher le graphique
            st.plotly_chart(fig_tsne, use_container_width=True)
            
            # Informations sur t-SNE
            with st.expander("ℹ️ À propos de la visualisation t-SNE"):
                st.markdown("""
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
                """)
                
                st.markdown(f"""
                **Paramètres utilisés :**
                - Perplexité : {tsne_data['tsne_params']['perplexity']}
                - Itérations max : {tsne_data['tsne_params']['max_iter']}
                - Seed aléatoire : {tsne_data['tsne_params']['random_state']}
                """)
        else:
            st.error("Erreur lors de la génération de la visualisation t-SNE")
    
    def render_sidebar_statistics(self, clusters: Optional[np.ndarray], ingredient_names: Optional[list]) -> None:
        """
        Affiche les statistiques dans la sidebar.
        
        Args:
            clusters: Labels des clusters (optionnel)
            ingredient_names: Liste des noms d'ingrédients (optionnel)
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
            colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9"]
            
            fig = go.Figure()
            
            for i, count in enumerate(cluster_counts):
                percentage = (count / len(ingredient_names)) * 100
                color = colors[i % len(colors)]
                
                fig.add_trace(go.Bar(
                    x=[count],
                    y=[f"Cluster {i+1}"],
                    orientation='h',
                    name=f"Cluster {i+1}",
                    marker_color=color,
                    text=f"{count} ({percentage:.1f}%)",
                    textposition="outside",
                    showlegend=False
                ))
            
            fig.update_layout(
                title="",
                xaxis_title="Nombre d'ingrédients",
                yaxis_title="",
                height=min(400, len(cluster_counts) * 40 + 100),
                margin=dict(l=10, r=10, t=10, b=10),
                font=dict(size=10)
            )
            
            st.sidebar.plotly_chart(fig, use_container_width=True)
    
    def render_analysis_summary(self, analyzer: IngredientsAnalyzer) -> None:
        """
        Affiche le résumé de l'analyse.
        
        Args:
            analyzer: Instance de l'analyseur d'ingrédients
        """
        # Afficher quelques exemples de regroupements d'ingrédients
        if hasattr(analyzer, 'ingredient_groups') and analyzer.ingredient_groups:
            with st.expander("🔗 Exemples de regroupements d'ingrédients similaires"):
                # Afficher les groupes avec plus d'un élément
                multi_groups = [g for g in analyzer.ingredient_groups if len(g) > 1]
                
                if multi_groups:
                    # Afficher les 10 premiers groupes
                    for i, group in enumerate(multi_groups[:10]):
                        members_display = ' | '.join(group[:5])
                        if len(group) > 5:
                            members_display += f" (+ {len(group)-5} autres)"
                        st.write(f"**Groupe {i+1}:** {members_display}")
                    
                    st.info(f"Total: {len(multi_groups)} groupes d'ingrédients similaires détectés")
                    
                    # Debug pour des ingrédients problématiques
                    debug_info = analyzer.debug_ingredient_mapping(['pepper', 'egg', 'salt', 'butter', 'onion'])
                    if 'search_results' in debug_info:
                        st.write("**🔍 Debug - Exemples de normalisation:**")
                        for term, matches in debug_info['search_results'].items():
                            if matches:
                                st.write(f"*{term.title()}:*")
                                for match in matches[:3]:  # Limiter à 3 résultats
                                    # Montrer aussi la normalisation
                                    normalized = analyzer.normalize_ingredient(match['ingredient'])
                                    status = "✅ Représentant" if match['is_representative'] else f"➡️ Mappé vers '{match['representative']}'"
                                    st.write(f"  • `{match['ingredient']}` → `{normalized}` {status}")
                    
                    # Exemple de normalisation en temps réel
                    st.write("**🧪 Test de normalisation:**")
                    test_ingredients = [
                        "large eggs", "fresh ground black pepper", "unsalted butter", 
                        "red onions", "whole milk", "extra virgin olive oil"
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
                                st.write(f"• Recettes: {summary['input_data']['total_recipes']:,}")
                                st.write(f"• Ingrédients bruts: {summary['input_data']['total_raw_ingredients']:,}")
                                st.write(f"• Moyenne par recette: {summary['input_data']['avg_ingredients_per_recipe']}")
                                
                                st.write("**🔄 Normalisation:**")
                                st.write(f"• Ingrédients uniques bruts: {summary['normalization']['total_unique_raw']:,}")
                                st.write(f"• Après normalisation: {summary['normalization']['total_normalized']:,}")
                                st.write(f"• Réduction: {summary['normalization']['reduction_ratio']}%")
                            
                            with col2:
                                st.write("**🔗 Regroupement:**")
                                st.write(f"• Groupes multiples: {summary['grouping']['groups_with_multiple_items']}")
                                st.write(f"• Plus grand groupe: {summary['grouping']['largest_group_size']} éléments")
                                
                                st.write("**📈 Matrice Co-occurrence:**")
                                st.write(f"• Dimensions: {summary['cooccurrence_matrix']['dimensions']}")
                                st.write(f"• Co-occurrences: {summary['cooccurrence_matrix']['total_cooccurrences']:,}")
                                st.write(f"• Paires non-nulles: {summary['cooccurrence_matrix']['non_zero_pairs']:,}")
                                st.write(f"• Sparsité: {summary['cooccurrence_matrix']['sparsity']}%")
                else:
                    st.warning("Aucun regroupement détecté. Tous les ingrédients sont considérés comme uniques.")
    
    def run(self) -> None:
        """Point d'entrée principal de la page."""
        self.logger.info("Starting ingredients clustering analysis")
        st.markdown("---")
        
        # Chargement automatique des données
        self.logger.debug("Loading and preparing data")
        data = self._load_and_prepare_data()
        
        # Sidebar pour les paramètres
        params = self.render_sidebar()
        self.logger.debug(f"Clustering parameters: {params}")
        
        # Zone principale
        st.header("📈 Analyse des données")
        
        # Traitement des données
        if data is not None:
            self.logger.info(f"Dataset loaded successfully: {len(data)} recipes")
            st.success(f"Dataset chargé : {len(data)} recettes")
            
            # Initialisation de l'analyseur
            analyzer = IngredientsAnalyzer(data)
            
            # Cache controls dans la sidebar
            self._render_cache_controls(analyzer)
            
            # Lancer l'analyse automatiquement ou avec le bouton
            if params["analyze_button"] or 'ingredient_names' not in st.session_state:
                self.logger.info("Starting clustering analysis with parameters")
                with st.spinner("Analyse en cours..."):
                    # Traitement des ingrédients
                    self.logger.debug(f"Processing ingredients with n_ingredients={params['n_ingredients']}")
                    ingredients_matrix, ingredient_names = analyzer.process_ingredients(params["n_ingredients"])
                    self.logger.info(f"Processed ingredients matrix: {ingredients_matrix.shape}")
                    
                    # Clustering
                    self.logger.debug(f"Performing clustering with n_clusters={params['n_clusters']}")
                    clusters = analyzer.perform_clustering(ingredients_matrix, params["n_clusters"])
                    self.logger.info(f"Clustering completed: {len(set(clusters))} unique clusters found")
                    
                    # Sauvegarde des résultats dans la session
                    st.session_state['ingredient_names'] = ingredient_names
                    st.session_state['clusters'] = clusters
                    st.session_state['ingredients_matrix'] = ingredients_matrix
                    st.session_state['analyzer'] = analyzer
                    
                self.logger.info("Analysis completed successfully")
                st.success("Analyse terminée!")
                
                # Afficher le résumé de l'analyse
                self.render_analysis_summary(analyzer)
            
            # Affichage des résultats si disponibles
            if 'ingredient_names' in st.session_state:
                self.logger.debug("Displaying cached clustering results")
                ingredient_names = st.session_state['ingredient_names']
                ingredients_matrix = st.session_state['ingredients_matrix']
                clusters = st.session_state['clusters']
                analyzer = st.session_state['analyzer']
                
                # Analyse de co-occurrence
                self.render_cooccurrence_analysis(ingredient_names, ingredients_matrix)
                
                # Affichage des clusters
                self.render_clusters(clusters, ingredient_names, params["n_clusters"])
                
                # Visualisation t-SNE
                self.render_tsne_visualization(analyzer, clusters, params["tsne_perplexity"])
                
                # Statistiques dans la sidebar
                self.render_sidebar_statistics(clusters, ingredient_names)
        
        else:
            st.error("Impossible de charger les données. Vérifiez la présence du fichier de données.")
        
        # Informations dans la sidebar
        with st.sidebar:
            st.markdown("---")
            
            with st.expander("ℹ️ À propos de l'analyse"):
                st.markdown("""
                **Analyseur de Recettes Food.com**
                
                🎯 **Objectif** : Identifier des groupes d'ingrédients qui apparaissent fréquemment ensemble
                
                📊 **Méthodes** :
                - Preprocessing des ingrédients
                - Matrice de co-occurrence
                - Clustering K-means
                - Visualisation interactive
                
                💡 **Utilisation** :
                1. Ajustez les paramètres
                2. Lancez l'analyse
                3. Explorez les clusters
                4. Analysez les co-occurrences
                """)
        
        # Footer
        st.markdown("---")
        st.markdown("🍳 **Clustering d'Ingrédients** - Développé avec ❤️ et Streamlit pour l'analyse de recettes Food.com")