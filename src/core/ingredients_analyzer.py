from __future__ import annotations

"""Analyseur d'ingrédients avec clustering basé sur la co-occurrence."""

import ast
import re
from collections import Counter
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from .data_loader import DataLoader


class IngredientsAnalyzer:
    """Classe pour analyser les ingrédients et effectuer le clustering basé sur la co-occurrence."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialise l'analyseur avec les données de recettes.
        
        Args:
            data: DataFrame contenant les recettes
        """
        self.data = data
        self.ingredient_names = []
        self.ingredients_matrix = None
        self.ingredient_groups = []
        self.ingredient_mapping = {}
    
    def normalize_ingredient(self, txt: str) -> str:
        """
        Normalise un ingrédient pour garder seulement l'essentiel.
        
        Args:
            txt: Nom de l'ingrédient brut
            
        Returns:
            Ingrédient normalisé à sa forme la plus générale
        """
        if pd.isna(txt):
            return ""
        
        t = str(txt).lower()
        t = re.sub(r'[^a-z0-9\s]', ' ', t)
        t = re.sub(r'\s+', ' ', t).strip()
        
        # Liste exhaustive de tous les mots non importants à supprimer
        stop_words = {
            # Adjectifs de taille/quantité
            'large', 'small', 'medium', 'big', 'little', 'tiny', 'huge', 'extra',
            'half', 'quarter', 'whole', 'sliced', 'diced', 'chopped', 'minced',
            'thick', 'thin', 'fine', 'coarse', 'rough',
            
            # Couleurs
            'black', 'white', 'red', 'green', 'yellow', 'brown', 'dark', 'light',
            'golden', 'pale', 'deep', 'bright',
            
            # États/préparations
            'fresh', 'dried', 'frozen', 'canned', 'raw', 'cooked', 'baked',
            'grilled', 'fried', 'boiled', 'steamed', 'roasted', 'smoked',
            'ground', 'crushed', 'powdered', 'granulated', 'shredded', 'grated',
            
            # Qualités
            'organic', 'natural', 'pure', 'virgin', 'extra', 'premium', 'quality',
            'grade', 'top', 'best', 'good', 'bad', 'old', 'new', 'aged',
            
            # Goûts/textures  
            'sweet', 'sour', 'bitter', 'salty', 'hot', 'mild', 'spicy', 'bland',
            'soft', 'hard', 'crispy', 'crunchy', 'smooth', 'rough', 'tender',
            'sharp', 'mild', 'strong', 'weak',
            
            # Parties/cuts
            'boneless', 'skinless', 'bone', 'skin', 'lean', 'fat', 'trimmed',
            'untrimmed', 'breast', 'thigh', 'wing', 'leg', 'fillet', 'steak',
            
            # Emballage/contenant
            'canned', 'jarred', 'bottled', 'bagged', 'boxed', 'packaged',
            'container', 'package', 'can', 'jar', 'bottle', 'bag', 'box',
            
            # Marques/types génériques
            'brand', 'style', 'type', 'kind', 'sort', 'variety',
            
            # Sel/graisse spécifiques
            'unsalted', 'salted', 'low', 'reduced', 'free', 'zero', 'diet',
            'light', 'heavy', 'full', 'skim', 'non',
            
            # Prépositions et articles
            'with', 'without', 'and', 'or', 'in', 'on', 'of', 'for', 'to',
            'the', 'a', 'an'
        }
        
        # Garder seulement les mots importants (substantifs principaux)
        tokens = []
        for w in t.split():
            if w not in stop_words and len(w) > 1:
                tokens.append(w)
        
        # Si on n'a plus rien, prendre le mot le plus long de l'original
        if not tokens:
            original_words = [w for w in t.split() if len(w) > 1]
            if original_words:
                tokens = [max(original_words, key=len)]
        
        # Singularisation agressive
        norm_tokens = []
        for w in tokens:
            # Règles de singularisation
            if w.endswith('ies') and len(w) > 4:
                norm_tokens.append(w[:-3] + 'y')
            elif w.endswith('es') and len(w) > 3 and not w.endswith(('ses', 'ches', 'shes', 'xes')):
                norm_tokens.append(w[:-2])
            elif w.endswith('s') and len(w) > 3 and not w.endswith(('ss', 'us', 'is', 'as', 'os')):
                norm_tokens.append(w[:-1])
            else:
                norm_tokens.append(w)
        
        # Prendre seulement le mot principal (le plus long généralement)
        if len(norm_tokens) > 1:
            main_ingredient = max(norm_tokens, key=len)
            return main_ingredient
        elif norm_tokens:
            return norm_tokens[0]
        else:
            return ""
    
    def build_similarity_groups(self, ingredients: list, threshold: float = 0.55) -> list:
        """
        Regroupe les ingrédients similaires basé sur la normalisation.
        
        Args:
            ingredients: Liste des ingrédients à regrouper
            threshold: Seuil de similarité (non utilisé dans cette implémentation)
            
        Returns:
            Liste de groupes d'ingrédients similaires
        """
        if not ingredients:
            return []
        
        # Regroupement basé sur la normalisation stricte
        norm_to_ingredients = {}
        
        for ingredient in ingredients:
            normalized = self.normalize_ingredient(ingredient)
            if normalized and len(normalized) > 1:
                if normalized not in norm_to_ingredients:
                    norm_to_ingredients[normalized] = []
                norm_to_ingredients[normalized].append(ingredient)
        
        # Créer les groupes basés sur la normalisation
        rule_groups = {}
        for i, (normalized, ingredient_list) in enumerate(norm_to_ingredients.items()):
            if len(ingredient_list) >= 1:
                rule_groups[i] = ingredient_list
        
        # Construire la liste finale des groupes
        final_groups = list(rule_groups.values())
        
        # Trier chaque groupe par longueur (plus court = plus général = représentant)
        for group in final_groups:
            group.sort(key=lambda x: (len(x), x))
        
        # Trier les groupes par taille décroissante
        final_groups.sort(key=len, reverse=True)
        
        return final_groups
    
    def clean_ingredient(self, ingredient: str) -> str:
        """
        Nettoie et standardise un ingrédient.
        
        Args:
            ingredient: Nom de l'ingrédient brut
            
        Returns:
            Ingrédient nettoyé
        """
        if pd.isna(ingredient):
            return ""
        
        # Conversion en minuscules
        ingredient = str(ingredient).lower().strip()
        
        # Suppression des caractères spéciaux et des quantités
        ingredient = re.sub(r'[0-9]+\.?[0-9]*\s*(cups?|tablespoons?|teaspoons?|ounces?|pounds?|grams?|ml|l)', '', ingredient)
        ingredient = re.sub(r'[^\w\s-]', '', ingredient)
        ingredient = re.sub(r'\s+', ' ', ingredient).strip()
        
        # Suppression des mots courants non significatifs
        stop_words = ['fresh', 'dried', 'chopped', 'minced', 'sliced', 'diced', 'ground', 'crushed', 'whole', 'large', 'small', 'medium']
        words = ingredient.split()
        words = [word for word in words if word not in stop_words]
        
        return ' '.join(words)
    
    def parse_ingredients(self, ingredients_str: str) -> list:
        """
        Parse une chaîne d'ingrédients et retourne une liste d'ingrédients nettoyés.
        
        Args:
            ingredients_str: Chaîne contenant les ingrédients (format liste Python)
            
        Returns:
            Liste des ingrédients nettoyés
        """
        if pd.isna(ingredients_str):
            return []
        
        try:
            # Utiliser ast.literal_eval pour parser les listes Python
            if isinstance(ingredients_str, str):
                ingredients_list = ast.literal_eval(ingredients_str)
            elif isinstance(ingredients_str, list):
                ingredients_list = ingredients_str
            else:
                # Fallback: séparation par virgules
                ingredients_list = str(ingredients_str).split(',')
            
            # Nettoyage de chaque ingrédient
            cleaned_ingredients = []
            for ingredient in ingredients_list:
                cleaned = self.clean_ingredient(ingredient)
                if cleaned and len(cleaned) > 2:
                    cleaned_ingredients.append(cleaned)
            
            return cleaned_ingredients
            
        except Exception:
            # En cas d'erreur, traiter comme une chaîne simple
            return [self.clean_ingredient(str(ingredients_str))]
    
    def get_most_common_ingredients(self, n_ingredients: int = 50) -> list:
        """
        Obtient les ingrédients les plus fréquents avec regroupement.
        
        Args:
            n_ingredients: Nombre d'ingrédients à retourner
            
        Returns:
            Liste des ingrédients représentatifs les plus fréquents
        """
        all_ingredients = []
        
        # Détection automatique de la colonne des ingrédients
        ingredient_columns = [col for col in self.data.columns if 'ingredient' in col.lower()]
        if not ingredient_columns:
            possible_columns = ['ingredients', 'recipe_ingredient_parts', 'ingredients_list']
            ingredient_columns = [col for col in possible_columns if col in self.data.columns]
        
        if not ingredient_columns:
            raise ValueError("Aucune colonne d'ingrédients trouvée dans les données")
        
        ingredient_column = ingredient_columns[0]
        
        # Extraction de tous les ingrédients
        for ingredients_str in self.data[ingredient_column].dropna():
            ingredients = self.parse_ingredients(ingredients_str)
            all_ingredients.extend(ingredients)
        
        # Comptage initial des ingrédients
        ingredient_counts = Counter(all_ingredients)
        
        # Prendre plus d'ingrédients pour le regroupement
        n_for_grouping = max(200, n_ingredients * 4)
        top_ingredients_raw = [ingredient for ingredient, count in ingredient_counts.most_common(n_for_grouping)]
        
        # Regroupement par similarité
        self.ingredient_groups = self.build_similarity_groups(top_ingredients_raw)
        
        # Créer un mapping ingredient -> représentant
        self.ingredient_mapping = {}
        representative_counts = {}
        
        for group in self.ingredient_groups:
            if len(group) == 1:
                ingredient = group[0]
                self.ingredient_mapping[ingredient] = ingredient
                representative_counts[ingredient] = ingredient_counts[ingredient]
            else:
                # Choisir le représentant (le plus fréquent)
                group_with_counts = [(ing, ingredient_counts[ing]) for ing in group]
                group_with_counts.sort(key=lambda x: (-x[1], len(x[0])))
                representative = group_with_counts[0][0]
                
                # Calculer le score total du groupe
                total_count = sum(ingredient_counts[ing] for ing in group)
                representative_counts[representative] = total_count
                
                # Mapper tous les membres du groupe vers le représentant
                for ingredient in group:
                    self.ingredient_mapping[ingredient] = representative
        
        # Trier les représentants par fréquence totale
        sorted_representatives = sorted(representative_counts.items(), key=lambda x: x[1], reverse=True)
        final_ingredients = [rep for rep, count in sorted_representatives[:n_ingredients]]
        
        return final_ingredients
    
    def create_cooccurrence_matrix(self, top_ingredients: list) -> pd.DataFrame:
        """
        Crée une matrice de co-occurrence.
        
        Args:
            top_ingredients: Liste des ingrédients représentatifs à analyser
            
        Returns:
            Matrice de co-occurrence sous forme de DataFrame
        """
        # Créer une matrice DataFrame 
        co_occurrence_matrix = pd.DataFrame(
            0, index=top_ingredients, columns=top_ingredients, dtype=int
        )
        
        # Détection de la colonne des ingrédients
        ingredient_columns = [col for col in self.data.columns if 'ingredient' in col.lower()]
        if not ingredient_columns:
            possible_columns = ['ingredients', 'recipe_ingredient_parts', 'ingredients_list']
            ingredient_columns = [col for col in possible_columns if col in self.data.columns]
        
        ingredient_column = ingredient_columns[0]
        
        # Parcourir chaque recette
        for ingredients_str in self.data[ingredient_column].dropna():
            ingredients = self.parse_ingredients(ingredients_str)
            
            # Mapper les ingrédients vers leurs représentants
            mapped_ingredients = []
            for ing in ingredients:
                if ing in self.ingredient_mapping:
                    representative = self.ingredient_mapping[ing]
                    if representative in top_ingredients:
                        mapped_ingredients.append(representative)
            
            # Éliminer les doublons dans cette recette
            present_ingredients = list(set(mapped_ingredients))
            
            # Mettre à jour la matrice avec toutes les combinaisons
            for ingr1, ingr2 in combinations(present_ingredients, 2):
                co_occurrence_matrix.at[ingr1, ingr2] += 1
                co_occurrence_matrix.at[ingr2, ingr1] += 1  # matrice symétrique
        
        return co_occurrence_matrix
    
    def process_ingredients(self, n_ingredients: int = 50) -> tuple[pd.DataFrame, list]:
        """
        Pipeline principal : traite les ingrédients et crée la matrice de co-occurrence.
        
        Args:
            n_ingredients: Nombre d'ingrédients représentatifs finaux
            
        Returns:
            Tuple (matrice de co-occurrence DataFrame, noms des ingrédients représentatifs)
        """
        # Obtenir les ingrédients représentatifs avec regroupement
        self.ingredient_names = self.get_most_common_ingredients(n_ingredients)
        
        # Créer la matrice de co-occurrence
        self.ingredients_matrix = self.create_cooccurrence_matrix(self.ingredient_names)
        
        return self.ingredients_matrix, self.ingredient_names
    
    def perform_clustering(self, co_occurrence_df: pd.DataFrame, n_clusters: int = 5) -> np.ndarray:
        """
        Effectue le clustering sur la matrice de co-occurrence.
        
        Args:
            co_occurrence_df: Matrice de co-occurrence
            n_clusters: Nombre de clusters
            
        Returns:
            Labels des clusters pour chaque ingrédient
        """
        # Appliquer KMeans sur la matrice de co-occurrence
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        
        # Utiliser seulement les données numériques
        matrix_data = co_occurrence_df.select_dtypes(include=[np.number])
        
        cluster_labels = kmeans.fit_predict(matrix_data)
        
        return cluster_labels
    
    def generate_tsne_visualization(self, cluster_labels: np.ndarray, perplexity: int = 30, random_state: int = 42) -> dict:
        """
        Génère une visualisation t-SNE 2D des ingrédients colorés par cluster.
        
        Args:
            cluster_labels: Labels des clusters pour chaque ingrédient
            perplexity: Paramètre de perplexité pour t-SNE
            random_state: Seed pour reproductibilité
            
        Returns:
            Données pour la visualisation (coordonnées, labels, couleurs)
        """
        if not hasattr(self, 'ingredients_matrix') or self.ingredients_matrix is None:
            return {"error": "Matrice de co-occurrence non disponible"}
        
        # Utiliser la matrice de co-occurrence comme données d'entrée pour t-SNE
        matrix_data = self.ingredients_matrix.values
        
        # Ajuster la perplexité si nécessaire
        n_samples = len(self.ingredient_names)
        perplexity = min(perplexity, n_samples - 1, 30)
        
        # Appliquer t-SNE
        tsne = TSNE(
            n_components=2, 
            perplexity=perplexity, 
            random_state=random_state,
            max_iter=1000,
            learning_rate='auto',
            init='random'
        )
        
        # Transformer les données
        tsne_coords = tsne.fit_transform(matrix_data)
        
        # Palette de couleurs pour les clusters
        colors = [
            "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", 
            "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
            "#F8C471", "#82E0AA", "#AED6F1", "#E8DAEF", "#FADBD8"
        ]
        
        # Préparer les données pour la visualisation
        visualization_data = {
            "x_coords": tsne_coords[:, 0].tolist(),
            "y_coords": tsne_coords[:, 1].tolist(),
            "ingredient_names": self.ingredient_names,
            "cluster_labels": cluster_labels.tolist(),
            "colors": [colors[label % len(colors)] for label in cluster_labels],
            "n_clusters": len(set(cluster_labels)),
            "tsne_params": {
                "perplexity": perplexity,
                "max_iter": 1000,
                "random_state": random_state
            }
        }
        
        return visualization_data
    
    def debug_ingredient_mapping(self, search_terms: Optional[list] = None) -> dict:
        """
        Méthode de debug pour voir comment les ingrédients sont mappés.
        
        Args:
            search_terms: Liste de termes à rechercher dans les mappings
            
        Returns:
            Informations de debug sur les mappings
        """
        if not hasattr(self, 'ingredient_mapping'):
            return {"error": "Aucun mapping d'ingrédients disponible. Lancez d'abord l'analyse."}
        
        debug_info = {
            "total_mappings": len(self.ingredient_mapping),
            "unique_representatives": len(set(self.ingredient_mapping.values())),
            "groups_with_multiple_items": []
        }
        
        # Grouper par représentant
        reverse_mapping = {}
        for ingredient, representative in self.ingredient_mapping.items():
            if representative not in reverse_mapping:
                reverse_mapping[representative] = []
            reverse_mapping[representative].append(ingredient)
        
        # Identifier les groupes avec plusieurs éléments
        for representative, ingredients in reverse_mapping.items():
            if len(ingredients) > 1:
                debug_info["groups_with_multiple_items"].append({
                    "representative": representative,
                    "members": ingredients,
                    "count": len(ingredients)
                })
        
        # Recherche spécifique si demandée
        if search_terms:
            debug_info["search_results"] = {}
            for term in search_terms:
                matches = []
                term_lower = term.lower()
                for ingredient, representative in self.ingredient_mapping.items():
                    if term_lower in ingredient.lower() or term_lower in representative.lower():
                        matches.append({
                            "ingredient": ingredient,
                            "representative": representative,
                            "is_representative": ingredient == representative
                        })
                debug_info["search_results"][term] = matches
        
        return debug_info
    
    def get_processing_summary(self) -> dict:
        """
        Génère un résumé complet du processus de traitement des données.
        
        Returns:
            Statistiques complètes du pipeline de traitement
        """
        if not hasattr(self, 'ingredient_mapping'):
            return {"error": "Pipeline non exécuté. Lancez d'abord process_ingredients()."}
        
        # Statistiques de base
        total_recipes = len(self.data)
        total_raw_ingredients = 0
        
        # Compter les ingrédients bruts
        ingredient_column = [col for col in self.data.columns if 'ingredient' in col.lower()][0]
        for ingredients_str in self.data[ingredient_column].dropna():
            ingredients = self.parse_ingredients(ingredients_str)
            total_raw_ingredients += len(ingredients)
        
        # Statistiques de regroupement
        unique_normalized = len(set(self.ingredient_mapping.values()))
        total_mapped = len(self.ingredient_mapping)
        
        # Groupes multiples
        reverse_mapping = {}
        for ingredient, representative in self.ingredient_mapping.items():
            if representative not in reverse_mapping:
                reverse_mapping[representative] = []
            reverse_mapping[representative].append(ingredient)
        
        multi_groups = {rep: ingredients for rep, ingredients in reverse_mapping.items() if len(ingredients) > 1}
        
        # Statistiques de la matrice
        matrix_size = len(self.ingredient_names)
        total_cooccurrences = int(self.ingredients_matrix.values.sum() / 2)
        non_zero_pairs = int((self.ingredients_matrix.values > 0).sum() / 2)
        
        return {
            "input_data": {
                "total_recipes": total_recipes,
                "total_raw_ingredients": total_raw_ingredients,
                "avg_ingredients_per_recipe": round(total_raw_ingredients / total_recipes, 2)
            },
            "normalization": {
                "total_unique_raw": total_mapped,
                "total_normalized": unique_normalized,
                "reduction_ratio": round((1 - unique_normalized/total_mapped) * 100, 1)
            },
            "grouping": {
                "groups_with_multiple_items": len(multi_groups),
                "largest_group_size": max([len(ingredients) for ingredients in multi_groups.values()]) if multi_groups else 0,
                "example_large_groups": dict(list(multi_groups.items())[:3])
            },
            "cooccurrence_matrix": {
                "dimensions": f"{matrix_size}x{matrix_size}",
                "total_cooccurrences": total_cooccurrences,
                "non_zero_pairs": non_zero_pairs,
                "sparsity": round((1 - non_zero_pairs/(matrix_size*(matrix_size-1)/2)) * 100, 1)
            },
            "final_ingredients": self.ingredient_names[:10]
        }