from __future__ import annotations

"""Streamlit page: Analyse de popularité des recettes.

Analyse des relations entre popularité, notes et caractéristiques structurelles.
"""
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
        st.sidebar.markdown("### Visualisation")
        plot_type = st.sidebar.selectbox(
            "Type de graphique", 
            ["Scatter", "Histogram"],
            help="Scatter: points individuels, Histogram: nombre d'observations par bins"
        )
        
        if plot_type == "Histogram":
            n_bins = st.sidebar.slider("Nombre de bins", 10, 50, 20)
            bin_agg = "count"  # Fixé à count seulement
        else:
            n_bins = 20
            bin_agg = "count"
            
        alpha = st.sidebar.slider("Transparence", 0.1, 1.0, 0.6, 0.1)
        
        # Preprocessing section
        st.sidebar.markdown("### Preprocessing")
        outlier_threshold = st.sidebar.slider(
            "Seuil outliers", 
            min_value=1.0, 
            max_value=20.0, 
            value=10.0, 
            step=1.0,
            help="Multiplicateur IQR pour filtrer les outliers techniques (minutes, n_steps, n_ingredients). Plus élevé = moins de filtrage."
        )
        
        return {
            "plot_type": plot_type,
            "n_bins": n_bins,
            "bin_agg": bin_agg,
            "alpha": alpha,
            "outlier_threshold": outlier_threshold
        }
    
    def _render_cache_controls(self, analyzer: InteractionsAnalyzer):
        """Render cache management controls in sidebar."""
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
        st.subheader("Segmentation par popularité")
        
        # Create popularity segments
        segmented_data = analyzer.create_popularity_segments(pop_rating)
        
        # Get threshold information from analyzer
        thresholds = analyzer._popularity_segments_info['thresholds']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution of segments with precise intervals
            segment_counts = segmented_data['popularity_segment'].value_counts()
            st.write("**Distribution des segments avec intervalles précis:**")
            
            # Display segments with their exact interaction count ranges
            for segment in ['Low', 'Medium', 'High', 'Viral']:
                if segment in segment_counts.index:
                    count = segment_counts[segment]
                    percentage = (count / len(segmented_data)) * 100
                    
                    # Define interval text based on segment
                    if segment == 'Low':
                        interval = f"1 à {int(thresholds['low_max'])} fois"
                    elif segment == 'Medium':
                        interval = f"{int(thresholds['low_max']) + 1} à {int(thresholds['medium_max'])} fois"
                    elif segment == 'High':
                        interval = f"{int(thresholds['medium_max']) + 1} à {int(thresholds['high_max'])} fois"
                    else:  # Viral
                        interval = f"Plus de {int(thresholds['high_max'])} fois"
                    
                    st.write(f"- **{segment}** ({interval}): {count:,} recettes ({percentage:.1f}%)")
        
        with col2:
            # Average rating by segment
            segment_ratings = segmented_data.groupby('popularity_segment')['avg_rating'].agg(['mean', 'std', 'count'])
            st.write("**Note moyenne par segment:**")
            for segment in ['Low', 'Medium', 'High', 'Viral']:
                if segment in segment_ratings.index:
                    mean_rating = segment_ratings.loc[segment, 'mean']
                    std_rating = segment_ratings.loc[segment, 'std']
                    count = segment_ratings.loc[segment, 'count']
                    st.write(f"- {segment}: {mean_rating:.2f} ± {std_rating:.2f} ({count:,} recettes)")
        
        # Visualization of segments
        self._plot_popularity_segments(segmented_data)
    def _plot_popularity_segments(self, segmented_data: pd.DataFrame):
        """Create visualization for popularity segments."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Boxplot of ratings by segment
        segment_order = ['Low', 'Medium', 'High', 'Viral']
        segments_present = [s for s in segment_order if s in segmented_data['popularity_segment'].values]
        
        if segments_present:
            sns.boxplot(data=segmented_data, x='popularity_segment', y='avg_rating', 
                       order=segments_present, ax=ax1)
            ax1.set_title('Distribution des notes par segment de popularité')
            ax1.set_xlabel('Segment de popularité')
            ax1.set_ylabel('Note moyenne')
            ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Scatter plot colored by segment
        colors = {'Low': 'blue', 'Medium': 'green', 'High': 'orange', 'Viral': 'red'}
        for segment in segments_present:
            segment_data = segmented_data[segmented_data['popularity_segment'] == segment]
            ax2.scatter(segment_data['avg_rating'], segment_data['interaction_count'], 
                       c=colors.get(segment, 'gray'), label=segment, alpha=0.6, s=30)
        
        ax2.set_xlabel('Note moyenne')
        ax2.set_ylabel('Nombre d\'interactions')
        ax2.set_title('Popularité vs Note par segment')
        ax2.legend()
        ax2.set_yscale('log')  # Log scale for better visualization
        
        plt.tight_layout()
        st.pyplot(fig)
    
    def _render_recipe_categorization(self, analyzer: InteractionsAnalyzer, agg: pd.DataFrame):
        """Render sophisticated recipe categorization analysis."""
        st.markdown("---")
        st.header("🔬 SECTION : Analyse des caractéristiques")
        st.subheader("Catégorisation des recettes")
        
        st.markdown("""
        Cette section révèle les archétypes de recettes qui réussissent selon leurs caractéristiques : 
        temps de préparation, complexité, et nombre d'ingrédients.
        
        **Questions :** Existe-t-il des profils gagnants ? Les recettes rapides surpassent-elles 
        les élaborées ? Les créations simples rivalisent-elles avec les complexes ?
        """)
        
        # Introduction de la méthode avec storytelling
        with st.expander("🔬 Notre laboratoire de classification", expanded=True):
            st.markdown("""
            Imaginez que nous sommes des botanistes découvrant de nouvelles espèces de recettes. 
            Pour les classifier, nous avons développé un système à quatre dimensions qui capture 
            l'essence de chaque création culinaire :
            
            **🧩 La dimension Complexité** révèle l'ambition créative : des recettes zen et minimalistes 
            aux symphonies culinaires élaborées. Nous mesurons cette richesse en combinant étapes et ingrédients 
            pour créer un "indice de sophistication".
            
            **⏱️ La dimension Temporelle** capture le rythme de vie : des créations express pour les pressés 
            aux rituels culinaires pour les contemplatifs. Quatre univers temporels émergent naturellement.
            
            **⚡ La dimension Efficacité** révèle la performance pure : le rapport magique entre satisfaction 
            obtenue et temps investi. C'est notre "indice de génie culinaire" - certaines recettes 
            accomplissent des miracles en quelques minutes !
            
            **📏 La dimension Richesse** explore la diversité des saveurs : des créations épurées aux 
            festins d'ingrédients. Chaque recette révèle sa philosophie culinaire à travers ce spectre.
            """)
        
        # Create categorized data
        categorized_data = analyzer.create_recipe_categories(agg)
        
        # Get insights
        insights = analyzer.get_category_insights(categorized_data)
        
        # Display category distributions with improved formatting
        st.markdown("### Distribution des recettes par catégorie")
        
        # Create columns for different category types
        categories = ['complexity_category', 'duration_category', 'efficiency_category', 'recipe_size_category']
        category_labels = {
            'complexity_category': '🧩 Complexité',
            'duration_category': '⏱️ Durée',
            'efficiency_category': '⚡ Efficacité', 
            'recipe_size_category': '📏 Taille'
        }
        available_categories = [cat for cat in categories if cat in categorized_data.columns]
        
        if available_categories:
            # Display in a more structured way
            cols = st.columns(min(len(available_categories), 2))
            
            for i, category in enumerate(available_categories):
                with cols[i % 2]:
                    # Use emoji and better formatting
                    label = category_labels.get(category, category.replace('_', ' ').title())
                    st.markdown(f"**{label}**")
                    
                    if category in insights:
                        # Create a nice formatted display
                        for cat_name, count in insights[category]['distribution'].items():
                            percentage = (count / len(categorized_data)) * 100
                            # Add progress bar for visual representation
                            st.write(f"• **{cat_name}**: {count:,} recettes ({percentage:.1f}%)")
                            st.progress(percentage / 100)
                    st.write("")  # Add spacing
        
        # Visualization of categories vs ratings
        st.markdown("""
        ### 🎨 Le théâtre des performances
        
        Maintenant que nous connaissons nos "acteurs" (les différentes catégories), observons leur 
        performance sur scène ! Ces visualisations révèlent quels archétydes de recettes brillent 
        le plus aux yeux du public. Préparez-vous à des surprises...
        """)
        
        self._plot_category_analysis(categorized_data, available_categories)
        
        # Advanced insights
        st.markdown("""
        ### 🏆 Le palmarès des champions
        
        Après cette exploration visuelle, il est temps de révéler les véritables champions de chaque catégorie. 
        Quels sont les archétypes qui dominent ? Quelles stratégies culinaires triomphent ? 
        Les insights qui suivent pourraient bien changer votre vision de la cuisine !
        """)
        
        self._render_category_insights(categorized_data, insights)
    
    def _plot_category_analysis(self, categorized_data: pd.DataFrame, categories: list):
        """Create enhanced visualizations for category analysis."""
        if not categories:
            return
        
        st.markdown("### Analyse visuelle des catégories")
        
        # Category labels with emojis
        category_labels = {
            'complexity_category': '🧩 Complexité',
            'duration_category': '⏱️ Durée', 
            'efficiency_category': '⚡ Efficacité',
            'recipe_size_category': '📏 Taille (ingrédients)'
        }
        
        # Create enhanced visualizations
        for category in categories:
            if category in categorized_data.columns:
                st.markdown(f"#### {category_labels.get(category, category.replace('_', ' ').title())}")
                
                # Get unique categories and create consistent color mapping
                unique_categories = sorted(categorized_data[category].unique())
                color_palette = sns.color_palette("viridis", len(unique_categories))
                color_mapping = dict(zip(unique_categories, color_palette))
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Boxplot with consistent colors
                    fig1, ax1 = plt.subplots(figsize=(8, 5))
                    try:
                        # Use the consistent color mapping with explicit order (fix seaborn warning)
                        box_plot = sns.boxplot(data=categorized_data, x=category, y='avg_rating', 
                                             ax=ax1, palette=color_mapping, order=unique_categories, 
                                             hue=category, legend=False)
                        
                        # Enhance the plot
                        ax1.set_title(f'Distribution des notes par {category.replace("_", " ")}', 
                                     fontsize=12, fontweight='bold')
                        ax1.set_xlabel('')  # Remove default xlabel
                        ax1.set_ylabel('Note moyenne', fontsize=10)
                        ax1.tick_params(axis='x', rotation=45, labelsize=9)
                        ax1.grid(True, alpha=0.3)
                        
                        # Add mean values as text
                        means = categorized_data.groupby(category)['avg_rating'].mean()
                        for i, (cat_name, mean_val) in enumerate(means.items()):
                            ax1.text(i, mean_val + 0.05, f'{mean_val:.2f}', 
                                   ha='center', va='bottom', fontweight='bold', fontsize=9)
                        
                        plt.tight_layout()
                        st.pyplot(fig1)
                        
                    except Exception as e:
                        st.error(f"Erreur dans la visualisation: {str(e)}")
                
                with col2:
                    # Bar plot with counts and percentages using consistent colors
                    fig2, ax2 = plt.subplots(figsize=(8, 5))
                    try:
                        # Get category counts in the SAME ORDER as boxplot
                        category_counts_ordered = []
                        colors_ordered = []
                        
                        for cat in unique_categories:
                            count = len(categorized_data[categorized_data[category] == cat])
                            category_counts_ordered.append(count)
                            colors_ordered.append(color_mapping[cat])
                        
                        total_count = len(categorized_data)
                        
                        # Create bar plot with consistent colors and black borders
                        bars = ax2.bar(range(len(unique_categories)), category_counts_ordered, 
                                     color=colors_ordered, alpha=0.8, edgecolor='black', linewidth=1.2)
                        
                        # Customize the plot
                        ax2.set_title(f'Répartition des recettes par {category.replace("_", " ")}', 
                                     fontsize=12, fontweight='bold')
                        ax2.set_xlabel('')
                        ax2.set_ylabel('Nombre de recettes', fontsize=10)
                        ax2.set_xticks(range(len(unique_categories)))
                        ax2.set_xticklabels(unique_categories, rotation=45, ha='right', fontsize=9)
                        ax2.grid(True, alpha=0.3, axis='y')
                        
                        # Add count and percentage labels on bars
                        for i, (bar, count) in enumerate(zip(bars, category_counts_ordered)):
                            height = bar.get_height()
                            percentage = (count / total_count) * 100
                            ax2.text(bar.get_x() + bar.get_width()/2., height + max(category_counts_ordered)*0.01,
                                   f'{count:,}\n({percentage:.1f}%)', ha='center', va='bottom', 
                                   fontsize=8, fontweight='bold')
                        
                        plt.tight_layout()
                        st.pyplot(fig2)
                        
                    except Exception as e:
                        st.error(f"Erreur dans la visualisation: {str(e)}")
                
                # Add separator
                st.markdown("---")
    
    def _render_category_insights(self, categorized_data: pd.DataFrame, insights: dict):
        """Render advanced insights about categories with enhanced formatting."""
        
        # Category labels with emojis
        category_labels = {
            'complexity_category': '🧩 Complexité',
            'duration_category': '⏱️ Durée',
            'efficiency_category': '⚡ Efficacité', 
            'recipe_size_category': '📏 Taille'
        }
        
        # Find interesting correlations with enhanced analysis
        key_findings = []
        recommendations = []
        
        # Check for high-performing categories
        for category, data in insights.items():
            if category != 'popularity_segments' and 'avg_rating_by_category' in data:
                ratings_by_cat = data['avg_rating_by_category']
                if len(ratings_by_cat) < 2:
                    continue
                    
                max_rating_cat = max(ratings_by_cat, key=ratings_by_cat.get)
                min_rating_cat = min(ratings_by_cat, key=ratings_by_cat.get)
                rating_diff = ratings_by_cat[max_rating_cat] - ratings_by_cat[min_rating_cat]
                
                category_name = category_labels.get(category, category.replace('_', ' ').title())
                
                if rating_diff > 0.15:  # Significant difference
                    key_findings.append({
                        'category': category_name,
                        'best': max_rating_cat,
                        'worst': min_rating_cat,
                        'best_rating': ratings_by_cat[max_rating_cat],
                        'worst_rating': ratings_by_cat[min_rating_cat],
                        'difference': rating_diff
                    })
                    
                    # Generate recommendations
                    if 'complexity' in category.lower():
                        if 'Simple' in max_rating_cat:
                            recommendations.append("💡 **Simplicité gagnante**: Les recettes simples obtiennent de meilleures notes")
                        elif 'Complex' in max_rating_cat:
                            recommendations.append("🎯 **Sophistication appréciée**: Les recettes complexes séduisent davantage")
                    
                    elif 'duration' in category.lower():
                        if 'Express' in max_rating_cat:
                            recommendations.append("⚡ **Rapidité valorisée**: Les recettes express sont mieux notées")
                        elif 'Long' in max_rating_cat or 'Marathon' in max_rating_cat:
                            recommendations.append("🍳 **Patience récompensée**: Les recettes longues obtiennent de meilleures notes")
                    
                    elif 'efficiency' in category.lower():
                        if 'High' in max_rating_cat:
                            recommendations.append("🚀 **Efficacité optimale**: Le rapport qualité/temps est crucial")
        
        # Révélation narrative des découvertes
        st.markdown("""
        **🎭 Le moment de vérité est arrivé !** Nos données viennent de révéler les secrets 
        les mieux gardés de la réussite culinaire. Chaque catégorie raconte une histoire fascinante 
        sur ce que les gens apprécient vraiment en cuisine.
        """)
        
        # Display key findings in an attractive format
        col1, col2 = st.columns(2)
        
        with col1:
            if key_findings:
                st.markdown("#### **Champions par catégorie**")
                for finding in key_findings:
                    with st.container():
                        st.markdown(f"""
                        **{finding['category']} :**
                        
                        **{finding['best']}** domine avec {finding['best_rating']:.2f} points, 
                        devançant **{finding['worst']}** ({finding['worst_rating']:.2f}) 
                        de {finding['difference']:.2f} points.
                        """)
                        st.markdown("---")
            else:
                st.markdown("""
                #### 🤝 **L'équilibre parfait**
                
                Fascinant ! Nos données révèlent un phénomène rare : toutes les catégories 
                performent de manière équilibrée. C'est la preuve que la diversité culinaire 
                est la clé du succès - il n'y a pas une seule voie vers l'excellence !
                """)
        
        with col2:
            if recommendations:
                st.markdown("#### **Insights clés**")
                for i, rec in enumerate(recommendations):
                    st.markdown(f"""
                    **Insight #{i+1} :**
                    
                    {rec.replace('💡 **', '').replace('🎯 **', '').replace('⚡ **', '').replace('🍳 **', '').replace('🚀 **', '').replace('**:', '')}
                    """)
            else:
                st.markdown("""
                #### 🌈 **La beauté de la diversité**
                
                Nos données révèlent quelque chose de merveilleux : il n'existe pas de formule unique 
                du succès culinaire ! Cette diversité est une richesse qui prouve que chaque style 
                de cuisine a sa place et son public.
                """)
        
        # Enhanced summary statistics
        with st.expander("📊 **Les coulisses de l'analyse**", expanded=False):
            st.markdown("""
            #### � **Pour les curieux de données**
            
            Vous voulez plonger dans les détails ? Cette section révèle tous les chiffres 
            qui soutiennent nos découvertes. Parfait pour vérifier nos conclusions ou 
            explorer d'autres patterns cachés !
            """)
            
            categorical_cols = [col for col in categorized_data.columns if 'category' in col]
            
            if categorical_cols and 'avg_rating' in categorized_data.columns:
                for cat_col in categorical_cols:
                    if cat_col in categorized_data.columns:
                        category_name = category_labels.get(cat_col, cat_col.replace('_', ' ').title())
                        st.markdown(f"**{category_name}**")
                        correlation_data = categorized_data.groupby(cat_col)['avg_rating'].agg([
                            'mean', 'std', 'count', 'min', 'max'
                        ]).round(3)
                        
                        # Add ranking
                        correlation_data['rank'] = correlation_data['mean'].rank(ascending=False).astype(int)
                        correlation_data = correlation_data.sort_values('mean', ascending=False)
                        
                        # Format column names
                        correlation_data.columns = ['Moyenne', 'Écart-type', 'Nombre', 'Min', 'Max', 'Rang']
                        
                        st.dataframe(correlation_data, width='stretch')
                        st.write("")

    def _render_viral_recipe_analysis(self, analyzer: InteractionsAnalyzer, agg: pd.DataFrame, 
                                    interactions_df: pd.DataFrame, recipes_df: pd.DataFrame):
        """Render temporal analysis of viral recipes with 3D visualization."""
        st.markdown("---")
        st.markdown("---")
        st.header("🔬 ÉTAPE 4 : Analyse temporelle")
        
        st.markdown("""
        **Question :** Comment évoluent les recettes à fort engagement dans le temps ?
        
        Cette analyse examine l'évolution temporelle de la qualité et du nombre d'interactions 
        pour les recettes du segment viral (>95e percentile).
        
        **Approche :** Visualisation 3D des trajectoires temporelles pour identifier les phases 
        d'accélération de l'engagement.
        """)
        
        # Identify viral recipes
        pop_rating = analyzer.popularity_vs_rating()
        segmented_data = analyzer.create_popularity_segments(pop_rating)
        viral_recipes = segmented_data[segmented_data['popularity_segment'] == 'Viral']
        
        if len(viral_recipes) == 0:
            st.warning("Aucune recette virale détectée dans les données actuelles.")
            return
        
        # TOP 10 des recettes les plus virales
        st.markdown("### Top 10 des recettes les plus virales")
        
        # Merge with recipe names and sort by interaction count
        top_viral = viral_recipes.copy()
        if 'name' in recipes_df.columns:
            top_viral = top_viral.merge(
                recipes_df[['id', 'name']], 
                left_on='recipe_id', 
                right_on='id', 
                how='left'
            )
        
        # Sort by interaction count (most viral first) and take top 10
        top_viral = top_viral.sort_values('interaction_count', ascending=False).head(10)
        
        # Recalculer les stats avec les mêmes filtres que le graphique 3D
        
        # Recalculate stats using only complete data (same as 3D graph)
        corrected_stats = []
        for _, row in top_viral.iterrows():
            recipe_id = row['recipe_id']
            recipe_interactions = interactions_df[interactions_df['recipe_id'] == recipe_id].copy()
            
            if len(recipe_interactions) > 0:
                # Apply same filtering as 3D graph
                date_columns = [col for col in interactions_df.columns if 'date' in col.lower()]
                if date_columns:
                    date_col = date_columns[0]
                    recipe_interactions[date_col] = pd.to_datetime(recipe_interactions[date_col], errors='coerce')
                    complete_data = recipe_interactions.dropna(subset=[date_col, 'rating']).copy()
                    
                    if len(complete_data) > 0:
                        corrected_stats.append({
                            'recipe_id': recipe_id,
                            'interaction_count_filtered': len(complete_data),
                            'avg_rating_filtered': complete_data['rating'].mean(),
                            'name': row.get('name', f'Recipe {recipe_id}')
                        })
        
        # Create corrected display dataframe
        if corrected_stats:
            corrected_df = pd.DataFrame(corrected_stats)
            corrected_df = corrected_df.sort_values('interaction_count_filtered', ascending=False)
            
            display_cols = ['name', 'recipe_id', 'interaction_count_filtered', 'avg_rating_filtered']
            top_viral_display = corrected_df[display_cols].copy()
            top_viral_display.columns = ['Nom de la recette', 'ID', 'Nb interactions (dates valides)', 'Note moyenne (dates valides)']
        else:
            # Fallback to original data if no corrected data available
            display_cols = ['recipe_id', 'interaction_count', 'avg_rating']
            if 'name' in top_viral.columns:
                display_cols = ['name', 'recipe_id', 'interaction_count', 'avg_rating']
            
            top_viral_display = top_viral[display_cols].copy()
            top_viral_display.columns = ['Nom de la recette', 'ID', 'Nb interactions', 'Note moyenne'] if 'name' in top_viral.columns else ['ID', 'Nb interactions', 'Note moyenne']
        
        # Add rank column
        top_viral_display.insert(0, 'Rang', range(1, len(top_viral_display) + 1))
        
        # Format the display
        if 'Nom de la recette' in top_viral_display.columns:
            top_viral_display['Nom de la recette'] = top_viral_display['Nom de la recette'].apply(
                lambda x: x[:60] + "..." if len(str(x)) > 60 else str(x)
            )
        
        # Format interaction count column (handle both old and new column names)
        interaction_col = next((col for col in top_viral_display.columns if 'interactions' in col.lower()), None)
        if interaction_col:
            top_viral_display[interaction_col] = top_viral_display[interaction_col].apply(
                lambda x: f"{x:,.0f}"
            )
        
        # Format rating column (handle both old and new column names)
        rating_col = next((col for col in top_viral_display.columns if 'note moyenne' in col.lower()), None)
        if rating_col:
            top_viral_display[rating_col] = top_viral_display[rating_col].apply(
                lambda x: f"{x:.2f} ⭐"
            )
        
        # Display the table
        st.dataframe(
            top_viral_display, 
            use_container_width=True,
            hide_index=True
        )
        
        # Quick stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🥇 Recette #1", f"{top_viral.iloc[0]['interaction_count']:,.0f} interactions")
        with col2:
            total_interactions = top_viral['interaction_count'].sum()
            st.metric("📊 Total Top 10", f"{total_interactions:,.0f} interactions")
        with col3:
            avg_rating_top10 = top_viral['avg_rating'].mean()
            st.metric("⭐ Note moyenne", f"{avg_rating_top10:.2f}")
        
        # Recipe selection interface for 3D analysis
        st.markdown("### Sélection des recettes")
        
        st.markdown("""
        **Choisissez les recettes du Top 10 à analyser en 3D :**
        """)
        
        # Create selection options from top 10
        if 'name' in top_viral.columns:
            recipe_display = top_viral.apply(
                lambda row: f"#{row.name+1} - {row['name'][:40]}... (🔥 {row['interaction_count']:.0f} interactions, ⭐ {row['avg_rating']:.1f})", 
                axis=1
            ).tolist()
        else:
            recipe_display = top_viral.apply(
                lambda row: f"#{row.name+1} - Recipe ID: {row['recipe_id']} (� {row['interaction_count']:.0f} interactions, ⭐ {row['avg_rating']:.1f})", 
                axis=1
            ).tolist()
        
        selected_indices = st.multiselect(
            "Sélectionnez une ou plusieurs recettes du Top 10 pour l'analyse 3D (maximum 5)",
            options=range(len(recipe_display)),
            format_func=lambda x: recipe_display[x],
            default=[0] if len(recipe_display) > 0 else [],
            max_selections=5
        )
        
        if not selected_indices:
            st.info("Veuillez sélectionner au moins une recette virale pour continuer l'analyse.")
            return
            
        selected_recipe_ids = top_viral.iloc[selected_indices]['recipe_id'].tolist()
        
        # Temporal analysis
        st.markdown("### Visualisation 3D")
        
        st.markdown("""
        **Lecture du graphique :**
        - **X** : Date de l'interaction
        - **Y** : Note moyenne à cette date  
        - **Z** : Nombre cumulé d'interactions
        - **Trajectoire** : Plus elle s'accélère, plus la recette devient virale
        
        📊 **Note :** La visualisation 3D utilise les données brutes pour préserver toutes les recettes.
        Les statistiques d'analyse peuvent différer car elles utilisent des données nettoyées.
        """)
        
        # Create 3D visualization using RAW data to preserve all recipes
        # Note: 3D visualization shows actual recipe trajectories without preprocessing
        # to ensure no recipes are excluded from visualization
        self.logger.info("Using raw data for 3D visualization to preserve all recipe trajectories")
        self._create_3d_visualization_real(selected_recipe_ids, interactions_df, recipe_display, selected_indices)

    def _create_3d_visualization_real(self, recipe_ids: list, interactions_df: pd.DataFrame,
                                    recipe_display: list, selected_indices: list):
        """Create 3D visualization with real temporal data from the dataset."""
        
        # Check for available date columns
        date_columns = [col for col in interactions_df.columns if 'date' in col.lower()]
        
        if not date_columns:
            st.error("Aucune colonne de date trouvée. Colonnes disponibles : " + 
                    ", ".join(interactions_df.columns.tolist()))
            return
        
        # Use the first date column found, or let user choose if multiple
        if len(date_columns) == 1:
            date_col = date_columns[0]
            st.info(f"Utilisation de la colonne de date : **{date_col}**")
        else:
            st.info(f"Colonnes de date disponibles : {', '.join(date_columns)}")
            date_col = st.selectbox("Choisissez la colonne de date :", date_columns)
        
        # Ensure date column is in datetime format
        try:
            interactions_df[date_col] = pd.to_datetime(interactions_df[date_col])
        except Exception as e:
            st.error(f"Erreur lors de la conversion de la colonne '{date_col}' en date : {str(e)}")
            return
        
        # Add interval selection for temporal sampling
        st.subheader("Paramètres temporels")
        interval_days = st.selectbox(
            "Afficher un point tous les :",
            [1, 7, 14, 30, 90],
            index=1,  # Default: tous les 7 jours
            format_func=lambda x: f"{x} jour{'s' if x > 1 else ''}" if x < 30 else f"{x//30} mois"
        )
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
        
        for i, recipe_id in enumerate(recipe_ids):
            # Filter interactions for this recipe
            recipe_interactions = interactions_df[interactions_df['recipe_id'] == recipe_id].copy()
            
            if len(recipe_interactions) == 0:
                continue
            
            # Ensure date column is properly converted to datetime
            recipe_interactions[date_col] = pd.to_datetime(recipe_interactions[date_col], errors='coerce')
            
            # Filter out rows with missing essential data (date, rating, recipe_id)
            complete_data = recipe_interactions.dropna(subset=[date_col, 'rating']).copy()
            
            if len(complete_data) == 0:
                st.warning(f"Aucune donnée complète trouvée pour la recette {recipe_id}")
                continue
            
            # Sort by date to ensure strict chronological order
            complete_data = complete_data.sort_values(date_col).reset_index(drop=True)
            
            # Convert dates to numeric for plotting (days since first interaction)
            first_date = complete_data[date_col].min()
            complete_data['days_since_start'] = (complete_data[date_col] - first_date).dt.days
            
            # Filter based on temporal interval (every X days) - NO ARTIFICIAL SAMPLING
            if interval_days > 1:
                # Group by interval periods and take the last interaction of each period
                complete_data['period'] = complete_data['days_since_start'] // interval_days
                # Take the last interaction of each period (most recent within that interval)
                display_data = complete_data.groupby('period').last().reset_index()
            else:
                # Show ALL points (every single interaction)
                display_data = complete_data.copy()
            
            # Calculate cumulative statistics correctly
            display_data = display_data.sort_values('days_since_start').reset_index(drop=True)
            
            # For each point, calculate the cumulative average rating from ALL previous interactions
            cumulative_avg_ratings = []
            cumulative_interactions = []
            
            for idx in range(len(display_data)):
                current_day = display_data.iloc[idx]['days_since_start']
                # Get all interactions up to and including current day from original complete_data
                interactions_up_to_day = complete_data[complete_data['days_since_start'] <= current_day]
                
                # Calculate cumulative average rating (from day 1 to current day)
                cumulative_avg_rating = interactions_up_to_day['rating'].mean()
                cumulative_avg_ratings.append(cumulative_avg_rating)
                
                # Cumulative interaction count
                cumulative_interactions.append(len(interactions_up_to_day))
            
            display_data['cumulative_avg_rating'] = cumulative_avg_ratings
            display_data['cumulative_interactions'] = cumulative_interactions
            
            # Extract coordinates for 3D plot - only real data points
            x_dates = display_data['days_since_start'].values
            y_ratings = display_data['cumulative_avg_rating'].values  # Cumulative average ratings
            z_cumulative = display_data['cumulative_interactions'].values
            
            # Ensure we have valid data to plot
            if len(x_dates) == 0 or np.any(np.isnan(x_dates)) or np.any(np.isnan(y_ratings)):
                st.warning(f"Données invalides pour la recette {recipe_id}")
                continue
            
            # Plot the 3D trajectory
            ax.plot(x_dates, y_ratings, z_cumulative, 
                   color=colors[i % len(colors)], 
                   marker='o', 
                   markersize=5,
                   label=f"{recipe_display[selected_indices[i]][:30]}..." if len(recipe_display[selected_indices[i]]) > 30 else recipe_display[selected_indices[i]],
                   linewidth=2.5,
                   alpha=0.8)
            
            # Add projections/shadows on the time-rating plane (XY plane, Z=0)
            ax.plot(x_dates, y_ratings, zs=0, zdir='z', 
                   color=colors[i % len(colors)], 
                   alpha=0.3, 
                   linewidth=1.5,
                   linestyle='--')
            
            # Add trajectory start and end markers
            if len(x_dates) > 1:
                # Start point (green)
                ax.scatter(x_dates[0], y_ratings[0], z_cumulative[0], 
                          color='green', s=100, alpha=0.8, marker='^')
                # End point (red)
                ax.scatter(x_dates[-1], y_ratings[-1], z_cumulative[-1], 
                          color='red', s=100, alpha=0.8, marker='v')
                
                # Start and end projections on XY plane
                ax.scatter(x_dates[0], y_ratings[0], 0, 
                          color='green', s=50, alpha=0.5, marker='^')
                ax.scatter(x_dates[-1], y_ratings[-1], 0, 
                          color='red', s=50, alpha=0.5, marker='v')
            
            # Calculate viral acceleration (changes in trajectory curvature)
            if len(x_dates) >= 3:
                # Calculate distances between consecutive points
                distances = []
                for j in range(1, len(x_dates)):
                    dist = np.sqrt((x_dates[j] - x_dates[j-1])**2 + 
                                  (y_ratings[j] - y_ratings[j-1])**2 + 
                                  (z_cumulative[j] - z_cumulative[j-1])**2)
                    distances.append(dist)
                
                # Find point of maximum acceleration (viral takeoff)
                if len(distances) > 1:
                    accelerations = np.diff(distances)
                    max_accel_idx = np.argmax(accelerations) + 1  # +1 because diff reduces length
                    
                    if max_accel_idx < len(x_dates):
                        ax.scatter(x_dates[max_accel_idx], y_ratings[max_accel_idx], z_cumulative[max_accel_idx], 
                                  color='gold', s=150, alpha=0.9, marker='*', 
                                  edgecolors='black', linewidth=1)
        
        # Formatting and labels
        ax.set_xlabel('Jours depuis la première interaction', fontsize=12)
        ax.set_ylabel('Note moyenne cumulative', fontsize=12)
        ax.set_zlabel('Interactions cumulées', fontsize=12)
        ax.set_title('Évolution Temporelle des Recettes Virales', fontsize=14, fontweight='bold')
        
        # Disable default shadow projections on XZ and YZ planes
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        # Make panes transparent
        ax.xaxis.pane.set_edgecolor('gray')
        ax.yaxis.pane.set_edgecolor('gray') 
        ax.zaxis.pane.set_edgecolor('gray')
        ax.xaxis.pane.set_alpha(0.1)
        ax.yaxis.pane.set_alpha(0.1)
        ax.zaxis.pane.set_alpha(0.1)
        
        # Legend with custom positioning
        legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        legend.set_title("Recettes", prop={'weight': 'bold'})
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3)
        
        # Custom viewing angle for better perspective
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Statistical insights
        st.markdown("""
        **Légende :**
        - 🔺 **Vert** : Première interaction
        - 🔻 **Rouge** : État actuel  
        - ⭐ **Or** : Accélération maximale
        
        **Interprétation :**
        - **X** : Jours depuis la première interaction
        - **Y** : Note moyenne cumulative (évolution de la qualité perçue)
        - **Z** : Interactions cumulées
        - **Trajectoire ascendante** : Recette qui gagne en popularité et qualité
        """)
        
        # Display detailed statistics
        st.markdown("### Statistiques par recette")
        
        stats_data = []
        for i, recipe_id in enumerate(recipe_ids):
            recipe_interactions = interactions_df[interactions_df['recipe_id'] == recipe_id].copy()
            if len(recipe_interactions) > 0:
                # Filter only complete data (same logic as 3D plot)
                recipe_interactions[date_col] = pd.to_datetime(recipe_interactions[date_col], errors='coerce')
                complete_data = recipe_interactions.dropna(subset=[date_col, 'rating']).copy()
                
                if len(complete_data) > 0:
                    first_interaction = complete_data[date_col].min()
                    last_interaction = complete_data[date_col].max()
                    duration = (last_interaction - first_interaction).days
                    total_interactions = len(complete_data)
                    avg_rating = complete_data['rating'].mean()
                    
                    stats_data.append({
                        'Recette': recipe_display[selected_indices[i]][:40] + "..." if len(recipe_display[selected_indices[i]]) > 40 else recipe_display[selected_indices[i]],
                        'Première interaction': first_interaction.strftime('%Y-%m-%d'),
                        'Dernière interaction': last_interaction.strftime('%Y-%m-%d'),
                        'Durée (jours)': duration,
                        'Total interactions (complètes)': total_interactions,
                        'Note moyenne': f"{avg_rating:.2f}",
                        'Interactions/jour': f"{total_interactions/max(duration, 1):.1f}"
                    })
        
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)

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
            ("avg_rating", "interaction_count", "Scatter"): "Note moyenne selon le nombre d'interactions",
            ("interaction_count", "avg_rating", "Scatter"): "Nombre d'interactions selon la note moyenne",
            ("minutes", "avg_rating", "Scatter"): "Note moyenne selon la durée de préparation",
            ("n_steps", "avg_rating", "Scatter"): "Note moyenne selon le nombre d'étapes",
            ("n_ingredients", "avg_rating", "Scatter"): "Note moyenne selon le nombre d'ingrédients",
            ("minutes", "interaction_count", "Scatter"): "Nombre d'interactions selon la durée de préparation",
            ("n_steps", "interaction_count", "Scatter"): "Nombre d'interactions selon le nombre d'étapes",
            ("n_ingredients", "interaction_count", "Scatter"): "Nombre d'interactions selon le nombre d'ingrédients",
            
            # Histograms avec count
            ("avg_rating", "", "Histogram"): "Distribution des notes moyennes",
            ("interaction_count", "", "Histogram"): "Distribution du nombre d'interactions",
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
            "rating": "Note"
        }
        
        x_label = var_labels.get(x, x.replace('_', ' ').title())
        y_label = var_labels.get(y, y.replace('_', ' ').title())
        
        if plot_type == "Histogram":
            return f"Distribution de {x_label}"
        else:  # Scatter
            return f"{y_label} selon {x_label}"
    
    def _create_plot(self, data: pd.DataFrame, x: str, y: str, size: str | None = None, 
                    title: str = "", plot_type: str = "Scatter", n_bins: int = 20, 
                    bin_agg: str = "count", alpha: float = 0.6):
        """Create plot based on selected type with improved visualization."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if plot_type == "Scatter":
            self._scatter_plot(data, x, y, size, ax, alpha)
        elif plot_type == "Histogram":
            self._histogram_plot(data, x, y, size, ax, n_bins, bin_agg)
        
        # Utiliser un titre prédéfini si aucun titre n'est fourni
        if not title:
            title = self._get_plot_title(x, y, plot_type, bin_agg)
            
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Labels des axes en français
        var_labels = {
            "avg_rating": "Note moyenne",
            "interaction_count": "Nombre d'interactions", 
            "minutes": "Durée (minutes)",
            "n_steps": "Nombre d'étapes",
            "n_ingredients": "Nombre d'ingrédients",
            "rating": "Note"
        }
        
        x_label = var_labels.get(x, x.replace('_', ' ').title())
        y_label = var_labels.get(y, y.replace('_', ' ').title())
        
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
            scatter = ax.scatter(data[x], data[y], 
                               s=data[size] * 10, # Scale size
                               c=data[size], 
                               cmap='viridis', alpha=alpha, edgecolors='none')
            cbar = plt.colorbar(scatter, ax=ax)
            
            # Label plus explicite pour la colorbar
            if size == "avg_rating":
                cbar.set_label("Moyenne des notes")
            else:
                size_label = size.replace('_', ' ').title()
                cbar.set_label(size_label)
        else:
            # Plot all data points without sampling
            ax.scatter(data[x], data[y], alpha=alpha, s=30, c='steelblue', edgecolors='none')
    
    def _histogram_plot(self, data: pd.DataFrame, x: str, y: str, size: str | None, ax, n_bins: int, bin_agg: str):
        """Create histogram counting observations per bin."""
        # Create bins for x-axis
        data_clean = data.dropna(subset=[x, y])
        if len(data_clean) == 0:
            ax.text(0.5, 0.5, "Pas de données valides", ha='center', va='center', transform=ax.transAxes)
            return
            
        # Create proper bins with explicit edges
        x_min, x_max = data_clean[x].min(), data_clean[x].max()
        bin_edges = np.linspace(x_min, x_max, n_bins + 1)
        
        # Assign each point to a bin
        data_clean = data_clean.copy()
        data_clean['bin_idx'] = pd.cut(data_clean[x], bins=bin_edges, include_lowest=True, labels=False)
        
        # Calculate bin centers for plotting
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = (x_max - x_min) / n_bins
        
        # Count observations per bin
        agg_data = data_clean.groupby('bin_idx').size().reset_index(name='count')
        y_values = agg_data['count']
        y_label = "Nombre d'observations"
        
        # Handle size aggregation if provided (count observations with that size)
        size_values = None
        if size and size in data_clean.columns:
            size_agg = data_clean.groupby('bin_idx')[size].mean()  # Moyenne de la variable size par bin
            
            # Align with y_values
            size_values = []
            for idx in agg_data['bin_idx']:
                if idx in size_agg.index:
                    size_values.append(size_agg[idx])
                else:
                    size_values.append(0)
        
        # Create the bar plot
        valid_indices = agg_data['bin_idx'].dropna()
        plot_x = [bin_centers[int(idx)] for idx in valid_indices if int(idx) < len(bin_centers)]
        plot_y = y_values[:len(plot_x)]
        
        if size_values and len(size_values) >= len(plot_x):
            # Color bars by size value using the same colormap as scatter plots
            size_plot = size_values[:len(plot_x)]
            
            # Normalize size values for coloring
            if len(size_plot) > 0 and np.max(size_plot) > np.min(size_plot):
                norm = plt.Normalize(vmin=np.min(size_plot), vmax=np.max(size_plot))
                colors = [plt.cm.viridis(norm(s)) for s in size_plot]
            else:
                colors = ['steelblue'] * len(plot_x)
            
            bars = ax.bar(plot_x, plot_y, 
                         width=bin_width * 0.8,
                         color=colors,
                         alpha=0.7, 
                         edgecolor='black', 
                         linewidth=0.5)
            
            # Add colorbar consistent with scatter plots
            if len(size_plot) > 0 and np.max(size_plot) > np.min(size_plot):
                sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax)
                
                # Label plus explicite pour la colorbar
                size_label = size.replace('_', ' ').title()
                if size == "avg_rating":
                    cbar.set_label("Moyenne des notes (du bin)")
                else:
                    cbar.set_label(f"Moyenne {size_label} (du bin)")
        else:
            # Simple histogram without size coloring
            bars = ax.bar(plot_x, plot_y, 
                         width=bin_width * 0.8,
                         alpha=0.7, 
                         color='steelblue', 
                         edgecolor='black', 
                         linewidth=0.5)
        
        # Add value labels on top of bars (more readable)
        for i, (x_pos, height) in enumerate(zip(plot_x, plot_y)):
            if not pd.isna(height) and height > 0:
                ax.text(x_pos, height + height * 0.02, f'{int(height)}', 
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Improve axis labels
        ax.set_ylabel(y_label)
        
        # Set proper x-axis limits and ticks
        ax.set_xlim(x_min - bin_width/2, x_max + bin_width/2)
        
        # Add subtle grid for better readability
        ax.grid(True, alpha=0.3, axis='y')
    # ---------------- Main Render ---------------- #
    def run(self):
        st.header("🔥 Analyse de la popularité des recettes")
        
        # Introduction analytique
        with st.expander("🎯 Objectifs et méthodologie de l'analyse", expanded=True):
            st.markdown("""
            ### Qu'est-ce qui rend une recette populaire ?
            
            Cette analyse explore la relation entre la qualité des recettes (notes des utilisateurs) et leur 
            succès (nombre d'interactions). Nous examinons comment les caractéristiques des recettes influencent 
            leur adoption par la communauté.
            
            **Questions centrales :** La qualité garantit-elle la popularité ? Quels sont les facteurs 
            déterminants du succès d'une recette ? Existe-t-il des profils de recettes particulièrement 
            attractifs ?
            
            **Approche :** Analyse comparative entre qualité et engagement, segmentation des recettes par 
            popularité, identification des caractéristiques discriminantes.
            
            **Données :** Preprocessing sélectif qui conserve toutes les interactions authentiques tout en 
            filtrant les anomalies techniques (seuil configurable dans la barre latérale).
            """)

        # Section explicative du preprocessing (cachée par défaut)
        with st.expander("⚙️ Choix et justification du preprocessing", expanded=False):
            st.markdown("""
            ### Pourquoi un preprocessing sélectif ?
            
            Le preprocessing appliqué vise à préserver l'authenticité des données tout en éliminant 
            les artefacts techniques qui fausseraient l'analyse.
            
            **Principe directeur :** Conserver toutes les interactions légitimes, filtrer uniquement 
            les anomalies techniques évidentes.
            """)
            
            # Vérification en temps réel des valeurs manquantes
            st.markdown("#### 🔍 Vérification des valeurs manquantes")
            
            # Chargement temporaire pour vérification
            temp_interactions_df, temp_recipes_df = self._load_data()
            
            # Analyse des valeurs manquantes dans les colonnes critiques pour l'analyse
            critical_interactions_cols = ['rating', 'recipe_id', 'user_id']
            critical_recipes_cols = ['minutes', 'n_steps', 'n_ingredients', 'id']
            
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
                st.success("""
                **🎯 Conclusion :** Aucune valeur manquante dans les colonnes critiques pour l'analyse numérique.
                
                ✅ **KNN imputation non nécessaire** - toutes les données numériques (rating, minutes, n_steps, n_ingredients) sont complètes.
                
                ℹ️ Les valeurs manquantes détectées sont dans des colonnes textuelles (review, description) non utilisées dans les calculs.
                """)
            else:
                st.warning("""
                **⚠️ Attention :** Des valeurs manquantes existent dans les colonnes critiques.
                
                🔧 **KNN imputation recommandée** pour maintenir l'intégrité de l'analyse.
                """)
            
            # Statistiques globales
            total_cells_interactions = temp_interactions_df.shape[0] * temp_interactions_df.shape[1]
            total_cells_recipes = temp_recipes_df.shape[0] * temp_recipes_df.shape[1]
            total_missing = interactions_total_missing + recipes_total_missing
            total_cells = total_cells_interactions + total_cells_recipes
            
            completeness_rate = ((total_cells - total_missing) / total_cells) * 100
            
            st.info(f"**Taux de complétude global :** {completeness_rate:.2f}% ({total_cells - total_missing:,} cellules complètes sur {total_cells:,})")
            
            st.markdown("""
            **🔧 Méthode IQR (Interquartile Range) :**
            - **Seuil configurable** : 1.0 à 20.0 (par défaut : 10.0)
            - **Calcul** : Q1 - seuil×IQR ≤ valeurs ≤ Q3 + seuil×IQR
            - **Cibles** : Variables numériques continues (temps, étapes, ingrédients)
            
            **Ce qui est préservé :**
            - **Toutes les notes utilisateurs** (aucun filtrage sur les ratings)
            - **Toutes les interactions datées** (comportements authentiques)
            - **Recettes avec caractéristiques extrêmes mais plausibles**
            
            **Ce qui est filtré :**
            - Temps de préparation = MAX_INT (erreurs système)
            - Nombres d'étapes aberrants (ex: 999+ étapes)
            - Incohérences techniques dans les métadonnées
            
            **Impact typique :** 80-99% des données conservées selon le seuil choisi.
            Le seuil est ajustable dans la barre latérale pour explorer différents niveaux de filtrage.
            """)

        params = self._sidebar()
        plot_type = params["plot_type"]
        n_bins = params["n_bins"]
        bin_agg = params["bin_agg"]
        alpha = params["alpha"]
        outlier_threshold = params["outlier_threshold"]

        with st.spinner("Chargement des données..."):
            self.logger.info("Loading data for popularity analysis")
            interactions_df, recipes_df = self._load_data()
            self.logger.debug(f"Loaded interactions: {interactions_df.shape}, recipes: {recipes_df.shape}")

        # Configuration preprocessing sélective : filtrer les valeurs techniques aberrantes mais garder toutes les notes
        config_selective = PreprocessingConfig(
            enable_preprocessing=True,
            outlier_method="iqr",     # Méthode IQR avec seuil configurable
            outlier_threshold=outlier_threshold   # Seuil configuré via sidebar
        )
        analyzer = InteractionsAnalyzer(
            interactions=interactions_df, 
            recipes=recipes_df,
            preprocessing=config_selective,
            cache_enabled=True  # Cache activé pour de meilleures performances
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
                    delta=f"{filtered_percentage:.1f}% du total"
                )
            with col2:
                if preprocessing_stats and 'outliers_removed' in preprocessing_stats:
                    outliers_removed = preprocessing_stats['outliers_removed']
                    st.metric(
                        "🚫 Outliers supprimés", 
                        f"{outliers_removed:,}",
                        delta=f"-{(outliers_removed/original_count)*100:.1f}%" if outliers_removed > 0 else "0%"
                    )
                else:
                    st.metric("🚫 Outliers supprimés", "0")
            with col3:
                st.metric(
                    "⚙️ Seuil IQR actuel", 
                    f"{outlier_threshold:.1f}x",
                    help="Multiplicateur appliqué à l'écart interquartile"
                )
            
            # Détails des features traitées
            if preprocessing_stats and 'features_processed' in preprocessing_stats:
                features_processed = preprocessing_stats['features_processed']
                if features_processed:
                    st.info(f"**Features analysées :** {', '.join(features_processed)}")
                    
                    # Bouton pour afficher les détails complets du preprocessing
                    if st.button("🔍 Voir les détails du preprocessing"):
                        st.json(preprocessing_stats)
                        
            st.success(f"✅ **Preprocessing terminé** - {filtered_percentage:.1f}% des données conservées avec le seuil {outlier_threshold:.1f}x")
            
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

        # Phase 1 : Analyse de la relation qualité-popularité
        st.markdown("---")
        st.header("📈 ÉTAPE 1 : Relation qualité-popularité")
        
        st.markdown("""
        **Question :** Les recettes bien notées génèrent-elles plus d'interactions ?
        
        Cette première analyse croise la note moyenne des recettes avec leur nombre d'interactions 
        pour évaluer la corrélation entre qualité perçue et engagement utilisateur.
        
        **Métrique :** Corrélation entre note moyenne et nombre d'interactions par recette.
        """)
        
        try:
            pop_rating = analyzer.popularity_vs_rating()
            fig1 = self._create_plot(
                pop_rating, x="avg_rating", y="interaction_count", 
                plot_type=plot_type, n_bins=n_bins, bin_agg=bin_agg, alpha=alpha
            )
            st.pyplot(fig1)
            
            # Analyse des résultats
            st.markdown("""
            **� Observations :** La distribution révèle plusieurs clusters de recettes avec des niveaux 
            d'engagement distincts. Les recettes à haute popularité ne présentent pas systématiquement 
            les meilleures notes, suggérant l'existence de facteurs additionnels.
            
            **🔍 Implication :** Cette distribution non-linéaire indique que la popularité s'organise 
            en segments distincts plutôt qu'en progression continue.
            """)
            
            # Phase 2 : Segmentation par popularité
            st.markdown("---")
            st.header("📊 ÉTAPE 2 : Segmentation par engagement")
            
            st.markdown("""
            **Objectif :** Identifier et caractériser les différents segments de popularité.
            
            La distribution observée suggère l'existence de groupes distincts de recettes. Nous appliquons 
            une segmentation basée sur les percentiles pour révéler la structure naturelle de la popularité.
            
            **Méthode :** Segmentation par percentiles (25e, 75e, 95e) du nombre d'interactions.
            """)
            
            # Segmentation par popularité avec contexte narratif
            st.markdown("---")
            self._render_popularity_segmentation(analyzer, pop_rating)
            
            # Créer la segmentation pour obtenir les seuils et ajouter l'explication
            segmented_data = analyzer.create_popularity_segments(pop_rating)
            thresholds = analyzer._popularity_segments_info['thresholds']
            
            st.markdown(f"""
            **� Caractérisation des segments identifiés :**
            
            L'analyse révèle quatre segments distincts basés sur le niveau d'engagement :
            
            - **Engagement Faible** : 1 à {int(thresholds['low_max'])} interactions
              (25% des recettes - souvent de qualité mais visibilité limitée)
              
            - **Engagement Modéré** : {int(thresholds['low_max']) + 1} à {int(thresholds['medium_max'])} interactions  
              (50% des recettes - performance stable et audience fidèle)
              
            - **Engagement Élevé** : {int(thresholds['medium_max']) + 1} à {int(thresholds['high_max'])} interactions
              (20% des recettes - forte popularité établie)
              
            - **Engagement Viral** : Plus de {int(thresholds['high_max'])} interactions
              (5% des recettes - phénomènes d'adoption exceptionnelle)
            
            **🔍 Constat :** Cette segmentation confirme que la popularité suit une distribution 
            de type Pareto plutôt qu'une progression linéaire.
            """)
            
        except ValueError as e:
            st.info(f"Impossible de tracer Note vs Popularité: {e}")

        # Phase 3 : Analyse des caractéristiques déterminantes
        st.markdown("---")
        st.markdown("---")
        st.header("🔬 ÉTAPE 3 : Facteurs d'influence")
        
        st.markdown("""
        **Objectif :** Identifier les caractéristiques intrinsèques des recettes qui corrèlent 
        avec une popularité élevée.
        
        Au-delà de la qualité, trois dimensions techniques peuvent influencer l'adoption d'une recette :
        le temps de préparation, la complexité (nombre d'étapes) et les ingrédients requis.
        
        **Méthode :** Analyse de corrélation entre caractéristiques techniques et niveau d'engagement.
        """)
        
        # Caractéristiques (feature) vs Popularité avec la note comme taille
        feature_order = ["minutes", "n_steps", "n_ingredients"]
        feature_labels = {
            "minutes": "Temps (minutes)",
            "n_steps": "Nombre d'étapes",
            "n_ingredients": "Nombre d'ingrédients",
        }
        features = [f for f in feature_order if f in agg.columns]
        if features:
            for feat in features:
                # Contexte analytique pour chaque caractéristique
                if feat == "minutes":
                    st.markdown("""
                    #### ⏱️ Impact du temps de préparation
                    
                    **Hypothèse :** Les recettes rapides sont plus populaires dans une société pressée.
                    **Variable :** Temps de préparation en minutes vs nombre d'interactions.
                    **Indicateur qualité :** Taille des points = note moyenne.
                    """)
                elif feat == "n_steps":
                    st.markdown("""
                    #### 📋 Influence de la complexité procédurale
                    
                    **Hypothèse :** La complexité (nombre d'étapes) peut freiner l'adoption mais améliorer la satisfaction.
                    **Variable :** Nombre d'étapes vs nombre d'interactions.
                    **Observation :** Équilibre entre accessibilité et sophistication.
                    """)
                elif feat == "n_ingredients":
                    st.markdown("""
                    #### 🥘 Effet de la diversité des ingrédients
                    
                    **Hypothèse :** Plus d'ingrédients = recette plus complexe et potentiellement dissuasive.
                    **Variable :** Nombre d'ingrédients vs nombre d'interactions.
                    **Analyse :** Impact de la richesse compositionnelle sur l'engagement.
                    """)
                
                try:
                    df_pop_feat = analyzer.popularity_vs_feature(feat)
                    # Merge pour récupérer la note moyenne si disponible
                    if 'pop_rating' in locals():
                        # Limiter les colonnes pour éviter suffixes _x / _y sur interaction_count
                        pr_min = pop_rating[["recipe_id", "avg_rating"]].copy()
                        merged = df_pop_feat.merge(pr_min, on="recipe_id", how="left")
                    else:
                        merged = df_pop_feat
                    # Normalisation de nom si interaction_count a été suffixé accidentellement
                    if 'interaction_count_x' in merged.columns and 'interaction_count' not in merged.columns:
                        merged.rename(columns={'interaction_count_x': 'interaction_count'}, inplace=True)
                    if 'interaction_count_y' in merged.columns and 'interaction_count' not in merged.columns:
                        merged.rename(columns={'interaction_count_y': 'interaction_count'}, inplace=True)
                    y_col = 'interaction_count' if 'interaction_count' in merged.columns else merged.columns[1]
                    if 'avg_rating' in merged.columns:
                        fig = self._create_plot(
                            merged, x=feat, y=y_col, size="avg_rating",
                            plot_type=plot_type, n_bins=n_bins, bin_agg=bin_agg, alpha=alpha
                        )
                    else:
                        fig = self._create_plot(
                            merged, x=feat, y=y_col,
                            plot_type=plot_type, n_bins=n_bins, bin_agg=bin_agg, alpha=alpha
                        )
                    st.pyplot(fig)
                    
                    # Analyse narrative spécifique pour chaque caractéristique
                    if feat == "minutes":
                        st.markdown("""
                        **🔍 Ce que révèle le graphique du temps :**
                        
                        Regardez attentivement la distribution des points ! Si vous observez une concentration 
                        de gros points (bonnes notes) dans certaines zones de temps, cela révèle les "sweet spots" 
                        temporels. Les recettes ultra-rapides (moins de 15 minutes) peuvent manquer de sophistication, 
                        tandis que les marathons culinaires (plus de 2 heures) peuvent décourager même les plus motivés.
                        
                        **Le secret** semble résider dans un équilibre : suffisamment de temps pour créer quelque chose 
                        de satisfaisant, mais pas au point de transformer la cuisine en corvée.
                        """)
                    elif feat == "n_steps":
                        st.markdown("""
                        **🔍 Le verdict sur la complexité :**
                        
                        Ce graphique révèle l'un des paradoxes les plus fascinants de la cuisine ! 
                        Si vous voyez de gros points (bonnes notes) concentrés autour de 5-8 étapes, 
                        cela confirme qu'il existe un "niveau de défi optimal".
                        
                        **L'insight psychologique** : les gens aiment se sentir accomplir quelque chose 
                        (d'où l'attrait pour un certain nombre d'étapes), mais sans se sentir overwhelmed. 
                        C'est le sweet spot entre "trop simple = ennuyeux" et "trop complexe = décourageant".
                        """)
                    elif feat == "n_ingredients":
                        st.markdown("""
                        **🔍 La révélation des ingrédients :**
                        
                        Ce dernier graphique pourrait bien vous surprendre ! La relation entre nombre d'ingrédients 
                        et succès révèle comment nous percevons la "richesse" d'une recette.
                        
                        **Le phénomène psychologique** : trop peu d'ingrédients peut sembler "basique", 
                        mais trop d'ingrédients peut paraître "intimidant" ou "coûteux". 
                        Observez où se concentrent les meilleures notes pour découvrir le nombre magique 
                        qui équilibre richesse et accessibilité.
                        """)
                
                except ValueError as e:
                    st.caption(f"{feat}: {e}")
        else:
            st.info("Aucune des colonnes minutes / n_steps / n_ingredients n'est présente dans l'agrégat.")
        
        # Quatrième acte : L'analyse temporelle des recettes virales
        self._render_viral_recipe_analysis(analyzer, agg, interactions_df, recipes_df)

        # Synthèse et conclusions
        st.markdown("---")
        st.subheader("Synthèse des résultats")
        
        st.markdown("""
        ### 🔍 Conclusions principales
        
        **1. Relation qualité-popularité :** Non-linéaire avec formation de clusters distincts
        selon le niveau d'engagement.
        
        **2. Segmentation :** Distribution de type Pareto avec 4 segments identifiés 
        (faible/modéré/élevé/viral) représentant des dynamiques d'adoption distinctes.
        
        **3. Facteurs d'influence :** Les caractéristiques techniques (temps, complexité, ingrédients) 
        montrent des corrélations variables avec la popularité, suggérant des optimums locaux.
        
        **4. Phénomènes viraux :** Les recettes à engagement exceptionnel (>95e percentile) 
        présentent des patterns temporels d'accélération spécifiques.
        
        ### 🎯 Implications stratégiques
        
        Ces résultats fournissent un cadre analytique pour optimiser la visibilité et l'engagement 
        des contenus culinaires, en identifiant les facteurs critiques de succès selon le segment visé.
        """)
        
        st.markdown("---")
        st.caption("💡 **Configuration** : Ajustez les paramètres de preprocessing et visualisation pour explorer différentes perspectives analytiques.")
        