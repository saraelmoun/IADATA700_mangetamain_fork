from __future__ import annotations

"""Streamlit page: Popularité des recettes (popularité vs rating & features).

Focus: explore how interaction_co        else:
            if len(data) > 10000:
                data_sample = data.sample(n=10000, random_state=42)
                ax.text(0.02, 0.98, f"Échantillon de {len(data_sample):,} points sur {len(data):,}", 
                        transform=ax.transAxes, fontsize=10, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            else:
                data_sample = dataxy for popularity) relates to
average rating and structural recipe features (minutes, n_steps, n_ingredients).
"""
from pathlib import Path
from dataclasses import dataclass
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from core.data_loader import DataLoader
from core.interactions_analyzer import InteractionsAnalyzer


@dataclass
class PopularityAnalysisConfig:
    interactions_path: Path
    recipes_path: Path
    min_interactions_default: int = 5


class PopularityAnalysisPage:
    def __init__(self, interactions_path: str | Path, recipes_path: str | Path):
        self.config = PopularityAnalysisConfig(
            interactions_path=Path(interactions_path),
            recipes_path=Path(recipes_path),
        )

    # ---------------- Sidebar ---------------- #
    def _sidebar(self):
        st.sidebar.markdown("### Visualisation")
        plot_type = st.sidebar.selectbox(
            "Type de graphique", 
            ["Scatter", "Histogram"],
            help="Scatter: points individuels, Histogram: agrégation par bins"
        )
        
        if plot_type == "Histogram":
            n_bins = st.sidebar.slider("Nombre de bins", 10, 50, 20)
            bin_agg = st.sidebar.selectbox("Agrégation", ["mean", "median", "count"])
        else:
            n_bins = 20
            bin_agg = "mean"
            
        alpha = st.sidebar.slider("Transparence", 0.1, 1.0, 0.6, 0.1)
        
        # Cache management section
        st.sidebar.markdown("### Cache Management")
        
        return {
            "plot_type": plot_type,
            "n_bins": n_bins,
            "bin_agg": bin_agg,
            "alpha": alpha
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
                st.sidebar.success("✅ Cache disponible")
                # Show cache details
                if "cache_age_minutes" in cache_info:
                    age_str = f"{cache_info['cache_age_minutes']:.1f} min"
                    size_str = f"{cache_info['cache_size_mb']:.1f} MB"
                    st.sidebar.info(f"Age: {age_str}, Taille: {size_str}")
            else:
                st.sidebar.info("🔄 Cache sera créé après preprocessing")
                
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
            st.sidebar.warning("⚠️ Cache désactivé")
    
    def _render_popularity_segmentation(self, analyzer: InteractionsAnalyzer, pop_rating: pd.DataFrame):
        """Render popularity segmentation analysis."""
        st.subheader("🎯 Segmentation par popularité")
        
        # Create popularity segments
        segmented_data = analyzer.create_popularity_segments(pop_rating)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution of segments
            segment_counts = segmented_data['popularity_segment'].value_counts()
            st.write("**Distribution des segments:**")
            for segment, count in segment_counts.items():
                percentage = (count / len(segmented_data)) * 100
                st.write(f"- {segment}: {count:,} recettes ({percentage:.1f}%)")
        
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
            import seaborn as sns
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
        st.subheader("🏷️ Catégorisation sophistiquée des recettes")
        
        # Introduction et explication de la démarche
        with st.expander("📖 À propos de cette analyse", expanded=True):
            st.markdown("""
            ### 🎯 Objectif de l'analyse
            Cette section analyse les recettes selon plusieurs dimensions sophistiquées pour comprendre 
            quels types de recettes performent le mieux en termes de notation.
            
            ### 🔍 Méthode de catégorisation
            Chaque recette est automatiquement classée selon 4 critères :
            
            **🧩 Complexité** : Basée sur le nombre d'étapes + ingrédients
            - *Simple* : Recettes rapides avec peu d'ingrédients (≤25e percentile)
            - *Modérée* : Recettes standards (25e-67e percentile) 
            - *Complexe* : Recettes élaborées (>67e percentile)
            
            **⏱️ Durée** : Temps de préparation
            - *Express* : ≤15 minutes
            - *Normal* : 15-45 minutes
            - *Long* : 45-120 minutes
            - *Marathon* : >120 minutes
            
            **⚡ Efficacité** : Rapport qualité/temps (note/minute)
            - *Haute efficacité* : Excellente note pour le temps investi
            - *Efficacité moyenne* : Performance standard
            - *Faible efficacité* : Beaucoup de temps pour la note obtenue
            
            **📏 Taille** : Nombre d'ingrédients
            - *Minimale* : ≤5 ingrédients
            - *Standard* : 6-10 ingrédients
            - *Riche* : 11-15 ingrédients
            - *Élaborée* : >15 ingrédients
            """)
        
        # Create categorized data
        categorized_data = analyzer.create_recipe_categories(agg)
        
        # Get insights
        insights = analyzer.get_category_insights(categorized_data)
        
        # Display category distributions with improved formatting
        st.markdown("### 📊 Distribution des recettes par catégorie")
        
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
        self._plot_category_analysis(categorized_data, available_categories)
        
        # Advanced insights
        self._render_category_insights(categorized_data, insights)
    
    def _plot_category_analysis(self, categorized_data: pd.DataFrame, categories: list):
        """Create enhanced visualizations for category analysis."""
        if not categories:
            return
        
        st.markdown("### 📈 Analyse visuelle des catégories")
        
        # Import here to avoid issues
        import seaborn as sns
        
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
        st.markdown("### � Insights et conclusions")
        
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
        
        # Display key findings in an attractive format
        col1, col2 = st.columns(2)
        
        with col1:
            if key_findings:
                st.markdown("#### 🎯 **Principales découvertes**")
                for finding in key_findings:
                    with st.container():
                        st.markdown(f"**{finding['category']}**")
                        st.success(f"🏆 **Meilleur**: {finding['best']} ({finding['best_rating']:.2f})")
                        st.error(f"📉 **Moins bon**: {finding['worst']} ({finding['worst_rating']:.2f})")
                        st.info(f"📊 **Écart**: {finding['difference']:.2f} points")
                        st.markdown("---")
            else:
                st.info("📊 Les catégories montrent des performances relativement homogènes")
        
        with col2:
            if recommendations:
                st.markdown("#### 💡 **Recommandations stratégiques**")
                for rec in recommendations:
                    st.markdown(rec)
                    st.write("")
            else:
                st.markdown("#### 📈 **Analyse générale**")
                st.write("Les données suggèrent une performance équilibrée entre les différentes catégories.")
        
        # Enhanced summary statistics
        with st.expander("� **Tableau de bord détaillé**", expanded=False):
            st.markdown("#### 📈 Statistiques complètes par catégorie")
            
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

    # ---------------- Data Loading ---------------- #
    def _load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        inter_loader = DataLoader(self.config.interactions_path)
        rec_loader = DataLoader(self.config.recipes_path)
        interactions_df = inter_loader.load_data()
        recipes_df = rec_loader.load_data()
        return interactions_df, recipes_df

    # ---------------- Visualization helpers ---------------- #
    def _create_plot(self, data: pd.DataFrame, x: str, y: str, size: str | None = None, 
                    title: str = "", plot_type: str = "Scatter", n_bins: int = 20, 
                    bin_agg: str = "mean", alpha: float = 0.6):
        """Create plot based on selected type with improved visualization."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if plot_type == "Scatter":
            self._scatter_plot(data, x, y, size, ax, alpha)
        elif plot_type == "Histogram":
            self._histogram_plot(data, x, y, size, ax, n_bins, bin_agg)
            
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(x.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(y.replace('_', ' ').title(), fontsize=12)
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
            cbar.set_label(size.replace('_', ' ').title())
        else:
            # Plot all data points without sampling
            ax.scatter(data[x], data[y], alpha=alpha, s=30, c='steelblue', edgecolors='none')
    
    def _histogram_plot(self, data: pd.DataFrame, x: str, y: str, size: str | None, ax, n_bins: int, bin_agg: str):
        """Create histogram with proper bar visualization."""
        # Create bins for x-axis
        data_clean = data.dropna(subset=[x, y])
        if len(data_clean) == 0:
            ax.text(0.5, 0.5, "Pas de données valides", ha='center', va='center', transform=ax.transAxes)
            return
            
        # Create bins and aggregate
        x_bins = pd.cut(data_clean[x], bins=n_bins, include_lowest=True)
        
        # Prepare aggregation dictionary
        agg_dict = {y: bin_agg if bin_agg in ['mean', 'median'] else 'mean'}
        
        if size and size in data_clean.columns:
            if bin_agg == "mean":
                agg_dict[size] = 'mean'
            elif bin_agg == "median":
                agg_dict[size] = 'median'
            else:  # count
                agg_dict[size] = 'count'
        else:
            # Create a count column for sizing when no size parameter
            data_clean = data_clean.copy()
            data_clean['_count'] = 1
            agg_dict['_count'] = 'count'
        
        # Aggregate data by bins
        agg_data = data_clean.groupby(x_bins).agg(agg_dict).reset_index()
        
        # Get bin edges and centers - ensure numeric conversion
        try:
            agg_data['x_left'] = pd.to_numeric(agg_data[x].apply(lambda x: x.left))
            agg_data['x_right'] = pd.to_numeric(agg_data[x].apply(lambda x: x.right))
            agg_data['x_center'] = pd.to_numeric(agg_data[x].apply(lambda x: x.mid))
            agg_data['bin_width'] = agg_data['x_right'] - agg_data['x_left']
        except (AttributeError, TypeError):
            # Fallback for non-interval data
            agg_data['x_center'] = range(len(agg_data))
            agg_data['bin_width'] = 0.8
        
        # Create histogram bars
        if size and size in agg_data.columns:
            # Color bars by size value
            bars = ax.bar(agg_data['x_center'], agg_data[y], 
                         width=agg_data['bin_width'] * 0.8,
                         alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Color bars based on size values
            norm = plt.Normalize(vmin=agg_data[size].min(), vmax=agg_data[size].max())
            for bar, size_val in zip(bars, agg_data[size]):
                bar.set_color(plt.cm.viridis(norm(size_val)))
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label(f"{bin_agg.title()} {size.replace('_', ' ').title()}")
        else:
            # Simple histogram without size coloring
            size_col = '_count'
            ax.bar(agg_data['x_center'], agg_data[y], 
                  width=agg_data['bin_width'] * 0.8,
                  alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
        
        # Add value labels on top of bars (for readability)
        for i, (center, height) in enumerate(zip(agg_data['x_center'], agg_data[y])):
            if not pd.isna(height):
                ax.text(center, height + height * 0.01, f'{height:.1f}', 
                       ha='center', va='bottom', fontsize=8, alpha=0.8)

    def _scatter(self, data: pd.DataFrame, x: str, y: str, size: str | None = None, title: str = ""):
        """Legacy method - maintained for backward compatibility."""
        return self._create_plot(data, x, y, size, title, "Scatter")

    # ---------------- Main Render ---------------- #
    def run(self):
        st.header("🔥 Analyse de la popularité des recettes")

        params = self._sidebar()
        plot_type = params["plot_type"]
        n_bins = params["n_bins"]
        bin_agg = params["bin_agg"]
        alpha = params["alpha"]

        with st.spinner("Chargement des données..."):
            interactions_df, recipes_df = self._load_data()

        analyzer = InteractionsAnalyzer(interactions=interactions_df, recipes=recipes_df)
        
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

        st.subheader("Table d'agrégation (Top 200)")
        st.dataframe(agg.head(200))

        # Debug collapsible section to help comprendre pourquoi certains graphiques n'apparaissent pas
        with st.expander("🔍 Debug colonnes agrégées"):
            st.write("Colonnes disponibles dans l'agrégat:")
            st.code(", ".join(agg.columns))
            missing = [c for c in ["minutes", "n_steps", "n_ingredients"] if c not in agg.columns]
            if missing:
                st.warning(
                    "Colonnes manquantes pour les features: " + ", ".join(missing) +
                    "\nCauses possibles: \n- Fichier recettes sans ces colonnes\n- Nom de colonne différent\n- n_ingredients non dérivable car colonne 'ingredients' absente"
                )

        # Note moyenne vs Popularité (interaction_count toujours en ordonnée)
        st.subheader("Note moyenne vs Popularité")
        try:
            pop_rating = analyzer.popularity_vs_rating()
            fig1 = self._create_plot(
                pop_rating, x="avg_rating", y="interaction_count", 
                title="Note moyenne vs Popularité", plot_type=plot_type,
                n_bins=n_bins, bin_agg=bin_agg, alpha=alpha
            )
            st.pyplot(fig1)
            
            # Segmentation par popularité
            self._render_popularity_segmentation(analyzer, pop_rating)
            
        except ValueError as e:
            st.info(f"Impossible de tracer Note vs Popularité: {e}")

        # Caractéristiques (feature) vs Popularité avec la note comme taille
        feature_order = ["minutes", "n_steps", "n_ingredients"]
        feature_labels = {
            "minutes": "Temps (minutes)",
            "n_steps": "Nombre d'étapes",
            "n_ingredients": "Nombre d'ingrédients",
        }
        features = [f for f in feature_order if f in agg.columns]
        if features:
            st.subheader("Caractéristiques vs Popularité (taille = note moyenne)")
            for feat in features:
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
                            title=f"{feature_labels.get(feat, feat)} vs Popularité",
                            plot_type=plot_type, n_bins=n_bins, bin_agg=bin_agg, alpha=alpha
                        )
                    else:
                        fig = self._create_plot(
                            merged, x=feat, y=y_col,
                            title=f"{feature_labels.get(feat, feat)} vs Popularité",
                            plot_type=plot_type, n_bins=n_bins, bin_agg=bin_agg, alpha=alpha
                        )
                    st.pyplot(fig)
                except ValueError as e:
                    st.caption(f"{feat}: {e}")
        else:
            st.info("Aucune des colonnes minutes / n_steps / n_ingredients n'est présente dans l'agrégat.")
        
        # Nouvelle section: Catégorisation sophistiquée
        self._render_recipe_categorization(analyzer, agg)

        st.markdown("---")
        st.caption("Astuce: ajustez le seuil d'interactions pour réduire le bruit des recettes peu vues.")
