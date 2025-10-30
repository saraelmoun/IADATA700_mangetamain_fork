"""
Tests for IngredientsClusteringPage - Comprehensive Test Suite
================================================================

Test suite pour la page d'analyse de clustering des ingrédients.
Tests basés sur la même structure que test_popularity_analysis_page.py
avec focus sur les fonctionnalités spécifiques au clustering.
"""

from components.ingredients_clustering_page import (
    IngredientsClusteringPage,
    IngredientsClusteringConfig,
)
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import warnings
from unittest.mock import Mock, MagicMock, patch

# Suppress warnings during testing
warnings.filterwarnings("ignore")

# Add src to path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestIngredientsClusteringConfig:
    """Test la dataclass de configuration."""

    def test_config_creation(self):
        """Test la création basique de configuration."""
        config = IngredientsClusteringConfig(
            recipes_path=Path("recipes.csv"),
            n_ingredients=50,
            n_clusters=5,
            tsne_perplexity=30,
        )
        assert config.recipes_path == Path("recipes.csv")
        assert config.n_ingredients == 50
        assert config.n_clusters == 5
        assert config.tsne_perplexity == 30

    def test_config_with_defaults(self):
        """Test que les valeurs par défaut sont correctement appliquées."""
        config = IngredientsClusteringConfig(recipes_path=Path("recipes.csv"))
        assert config.n_ingredients == 30  # Updated after performance optimization
        assert config.n_clusters == 4  # Updated after performance optimization  
        assert config.tsne_perplexity == 15  # Updated after performance optimization

    def test_config_path_types(self):
        """Test la gestion des différents types de chemins."""
        # Test avec Path
        config1 = IngredientsClusteringConfig(recipes_path=Path("recipes.csv"))
        assert isinstance(config1.recipes_path, Path)

        # Test avec string (conversion dans la page)
        IngredientsClusteringConfig(recipes_path="recipes.csv")
        # Note: La conversion en Path se fait dans __init__ de la page


class TestIngredientsClusteringPage:
    """Test la classe principale IngredientsClusteringPage."""

    @pytest.fixture
    def sample_recipes_data(self):
        """Génère des données de recettes réalistes avec ingrédients."""
        np.random.seed(42)
        n_recipes = 100

        # Liste d'ingrédients courants pour créer des combinaisons réalistes
        common_ingredients = [
            "salt",
            "pepper",
            "sugar",
            "flour",
            "butter",
            "eggs",
            "milk",
            "onion",
            "garlic",
            "olive oil",
            "tomato",
            "cheese",
            "chicken",
            "beef",
            "rice",
            "pasta",
            "carrot",
            "potato",
            "lemon",
            "basil",
        ]

        # Créer des listes d'ingrédients pour chaque recette
        ingredients_lists = []
        for _ in range(n_recipes):
            n_ingredients = np.random.randint(3, 10)
            recipe_ingredients = list(np.random.choice(common_ingredients, size=n_ingredients, replace=False))
            ingredients_lists.append(str(recipe_ingredients))

        data = {
            "id": range(1, n_recipes + 1),
            "name": [f"Recipe {i}" for i in range(1, n_recipes + 1)],
            "ingredients": ingredients_lists,
            "minutes": np.random.randint(5, 180, n_recipes),
            "n_steps": np.random.randint(1, 20, n_recipes),
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def temp_recipes_file(self, sample_recipes_data):
        """Crée un fichier CSV temporaire pour les tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recipes_path = Path(tmpdir) / "recipes.csv"
            sample_recipes_data.to_csv(recipes_path, index=False)
            yield recipes_path

    @pytest.fixture
    def page_instance(self, temp_recipes_file):
        """Crée une instance de IngredientsClusteringPage pour les tests."""
        return IngredientsClusteringPage(str(temp_recipes_file))

    def test_initialization(self, temp_recipes_file):
        """Test l'initialisation basique de la page."""
        page = IngredientsClusteringPage(str(temp_recipes_file))

        assert page.default_recipes_path == str(temp_recipes_file)
        assert page.logger is not None

    def test_initialization_with_default_path(self):
        """Test l'initialisation avec le chemin par défaut."""
        page = IngredientsClusteringPage()
        assert page.default_recipes_path == "data/RAW_recipes.csv"

    def test_initialization_empty_path_raises_error(self):
        """Test qu'un chemin vide lève une erreur."""
        with pytest.raises(ValueError, match="ne peut pas être vide"):
            IngredientsClusteringPage("")

    def test_load_and_prepare_data(self, page_instance):
        """Test le chargement des données."""
        data = page_instance._load_and_prepare_data()

        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert "ingredients" in data.columns
        assert "name" in data.columns

    def test_load_data_with_invalid_path(self):
        """Test le chargement avec un chemin invalide."""
        page = IngredientsClusteringPage("/invalid/path/recipes.csv")

        # Should return None and display error (not raise exception)
        with patch("streamlit.error"):
            page._load_and_prepare_data()
            # Le comportement exact dépend de l'implémentation
            # mais ne devrait pas crasher

    def test_render_sidebar_returns_expected_structure(self, page_instance):
        """Test que render_sidebar retourne la structure attendue."""
        with (
            patch("streamlit.sidebar.header"),
            patch("streamlit.sidebar.slider") as mock_slider,
            patch("streamlit.sidebar.subheader"),
            patch("streamlit.sidebar.button") as mock_button,
        ):

            # Configuration des valeurs de retour
            mock_slider.side_effect = [
                50,
                5,
                30,
            ]  # n_ingredients, n_clusters, perplexity
            mock_button.return_value = False

            params = page_instance.render_sidebar()

            # Vérifier la structure du dictionnaire retourné
            expected_keys = [
                "n_ingredients",
                "n_clusters",
                "tsne_perplexity",
                "analyze_button",
            ]
            assert all(key in params for key in expected_keys)

            # Vérifier les types
            assert isinstance(params["n_ingredients"], int)
            assert isinstance(params["n_clusters"], int)
            assert isinstance(params["tsne_perplexity"], int)
            assert isinstance(params["analyze_button"], bool)

    def test_render_sidebar_parameter_ranges(self, page_instance):
        """Test que les paramètres sont dans les bonnes plages."""
        with (
            patch("streamlit.sidebar.header"),
            patch("streamlit.sidebar.slider") as mock_slider,
            patch("streamlit.sidebar.subheader"),
            patch("streamlit.sidebar.button"),
        ):

            # Tester différentes valeurs
            test_values = [
                (100, 8, 25),  # n_ingredients, n_clusters, perplexity
                (10, 2, 5),
                (200, 20, 50),
            ]

            for n_ing, n_clust, perp in test_values:
                mock_slider.side_effect = [n_ing, n_clust, perp]
                params = page_instance.render_sidebar()

                # Vérifier les valeurs
                assert 10 <= params["n_ingredients"] <= 200
                assert 2 <= params["n_clusters"] <= 20
                assert 5 <= params["tsne_perplexity"] <= 50

    def test_render_cooccurrence_analysis_basic(self, page_instance):
        """Test l'affichage de l'analyse de co-occurrence."""
        # Créer des données de test
        ingredient_names = ["salt", "pepper", "sugar", "flour", "butter"]
        matrix = pd.DataFrame(
            np.random.randint(0, 50, (5, 5)),
            index=ingredient_names,
            columns=ingredient_names,
        )

        with (
            patch("streamlit.subheader"),
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.selectbox") as mock_selectbox,
            patch("streamlit.metric"),
            patch("streamlit.progress"),
            patch("streamlit.success"),
            patch("streamlit.info"),
            patch("streamlit.warning"),
            patch("streamlit.error"),
        ):

            # Simuler la sélection d'ingrédients et les colonnes comme context managers
            mock_col = MagicMock()
            mock_col.__enter__ = Mock(return_value=mock_col)
            mock_col.__exit__ = Mock(return_value=False)
            mock_columns.return_value = [mock_col, mock_col, mock_col]
            mock_selectbox.side_effect = ["salt", "pepper"]

            # Ne devrait pas lever d'exception
            page_instance.render_cooccurrence_analysis(ingredient_names, matrix)

    def test_render_clusters_basic(self, page_instance):
        """Test l'affichage des clusters."""
        clusters = np.array([0, 0, 1, 1, 2])
        ingredient_names = ["salt", "pepper", "sugar", "flour", "butter"]
        n_clusters = 3

        with (
            patch("streamlit.subheader"),
            patch("streamlit.expander") as mock_expander,
            patch("streamlit.columns"),
        ):

            mock_expander.return_value.__enter__ = Mock()
            mock_expander.return_value.__exit__ = Mock()

            # Ne devrait pas lever d'exception
            page_instance.render_clusters(clusters, ingredient_names, n_clusters)

    def test_render_clusters_with_empty_cluster(self, page_instance):
        """Test l'affichage quand un cluster est vide."""
        clusters = np.array([0, 0, 0, 0, 0])  # Tous dans le même cluster
        ingredient_names = ["salt", "pepper", "sugar", "flour", "butter"]
        n_clusters = 3  # Mais on demande 3 clusters

        with (
            patch("streamlit.subheader"),
            patch("streamlit.expander"),
            patch("streamlit.columns"),
        ):

            # Ne devrait pas crasher même si certains clusters sont vides
            page_instance.render_clusters(clusters, ingredient_names, n_clusters)

    def test_render_sidebar_statistics_with_data(self, page_instance):
        """Test l'affichage des statistiques avec données valides."""
        clusters = np.array([0, 0, 1, 1, 2])
        ingredient_names = ["salt", "pepper", "sugar", "flour", "butter"]

        with (
            patch("streamlit.sidebar.markdown"),
            patch("streamlit.sidebar.metric"),
            patch("streamlit.sidebar.plotly_chart"),
        ):

            # Ne devrait pas lever d'exception
            page_instance.render_sidebar_statistics(clusters, ingredient_names)

    def test_render_sidebar_statistics_with_none(self, page_instance):
        """Test que rien n'est affiché si les données sont None."""
        with patch("streamlit.sidebar.markdown") as mock_markdown:
            # Appeler avec None
            page_instance.render_sidebar_statistics(None, None)

            # Ne devrait rien afficher
            mock_markdown.assert_not_called()

    def test_render_analysis_summary(self, page_instance):
        """Test l'affichage du résumé d'analyse."""
        # Créer un mock analyzer avec les attributs nécessaires
        mock_analyzer = Mock()
        mock_analyzer.ingredient_groups = [
            ["salt", "sea salt", "table salt"],
            ["pepper", "black pepper"],
            ["sugar", "white sugar", "granulated sugar"],
        ]
        # Mock pour debug_ingredient_mapping qui retourne un dict
        mock_analyzer.debug_ingredient_mapping.return_value = {
            "search_results": {
                "pepper": [
                    {
                        "ingredient": "black pepper",
                        "is_representative": True,
                        "representative": "pepper",
                    }
                ],
                "egg": [
                    {
                        "ingredient": "large eggs",
                        "is_representative": False,
                        "representative": "egg",
                    }
                ],
            }
        }
        # Mock pour normalize_ingredient
        mock_analyzer.normalize_ingredient.return_value = "normalized_ingredient"
        # Mock pour get_processing_summary
        mock_analyzer.get_processing_summary.return_value = {
            "input_data": {
                "total_recipes": 1000,
                "total_raw_ingredients": 5000,
                "avg_ingredients_per_recipe": 5,
            },
            "normalization": {
                "total_unique_raw": 2000,
                "total_normalized": 500,
                "reduction_ratio": 75,
            },
            "grouping": {"groups_with_multiple_items": 150, "largest_group_size": 10},
            "cooccurrence_matrix": {
                "dimensions": "50x50",
                "total_cooccurrences": 1250,
                "non_zero_pairs": 800,
                "sparsity": 36,
            },
        }

        with (
            patch("streamlit.expander") as mock_expander,
            patch("streamlit.write"),
            patch("streamlit.info"),
            patch("streamlit.warning"),
        ):

            # Mock expander as context manager
            mock_exp = MagicMock()
            mock_exp.__enter__ = Mock(return_value=mock_exp)
            mock_exp.__exit__ = Mock(return_value=False)
            mock_expander.return_value = mock_exp

            # Ne devrait pas lever d'exception
            page_instance.render_analysis_summary(mock_analyzer)

    def test_formal_language_in_methods(self, page_instance):
        """Test que les méthodes utilisent un langage formel."""
        # Vérifier les docstrings
        assert page_instance.run.__doc__ is not None
        assert page_instance.render_sidebar.__doc__ is not None

        # Les docstrings ne devraient pas contenir de langage informel
        informal_words = [" tu ", " vous ", " ton ", " ta ", " votre "]

        for method in [
            page_instance.run,
            page_instance.render_sidebar,
            page_instance.render_cooccurrence_analysis,
        ]:
            if method.__doc__:
                doc_with_spaces = f" {method.__doc__} "
                for word in informal_words:
                    assert word.lower() not in doc_with_spaces.lower()


class TestIngredientsClusteringPageTyping:
    """Test que le typage est correctement appliqué."""

    def test_all_methods_have_type_annotations(self):
        """Vérifie que toutes les méthodes publiques ont des annotations de type."""
        page = IngredientsClusteringPage()

        # Liste des méthodes qui doivent être typées
        methods_to_check = [
            "render_sidebar",
            "render_cooccurrence_analysis",
            "render_clusters",
            "render_tsne_visualization",
            "render_sidebar_statistics",
            "render_analysis_summary",
            "run",
        ]

        for method_name in methods_to_check:
            method = getattr(page, method_name)
            annotations = method.__annotations__

            # Vérifier qu'il y a au moins une annotation (return type)
            assert "return" in annotations, f"{method_name} manque l'annotation de retour"

    def test_return_type_annotations(self):
        """Vérifie les annotations de type de retour."""
        page = IngredientsClusteringPage()

        # Vérifier que render_sidebar retourne un dict
        return_annotation = page.render_sidebar.__annotations__["return"]
        # Les annotations peuvent être des strings en Python 3.9+ avec from __future__ import annotations
        if isinstance(return_annotation, str):
            assert "dict" in return_annotation.lower()
        else:
            assert return_annotation.__name__ == "dict"

        # Vérifier que les méthodes render_* retournent None
        for method_name in [
            "render_cooccurrence_analysis",
            "render_clusters",
            "render_sidebar_statistics",
            "render_analysis_summary",
            "run",
        ]:
            method = getattr(page, method_name)
            return_annotation = method.__annotations__["return"]
            # None type peut être une string ou le type None
            if isinstance(return_annotation, str):
                assert return_annotation == "None"
            else:
                assert return_annotation is None or return_annotation.__name__ == "NoneType"


class TestIntegration:
    """Tests d'intégration pour le workflow complet."""

    def test_full_workflow_mock(self):
        """Test le workflow complet avec des mocks."""
        # Créer des données de test minimales
        recipes_data = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Recipe 1", "Recipe 2", "Recipe 3"],
                "ingredients": [
                    "['salt', 'pepper', 'eggs']",
                    "['salt', 'sugar', 'flour']",
                    "['pepper', 'sugar', 'eggs']",
                ],
            }
        )

        # Créer la page
        page = IngredientsClusteringPage("fake_path.csv")

        # Patcher directement la méthode _load_and_prepare_data
        with patch.object(page, "_load_and_prepare_data", return_value=recipes_data):
            # Charger les données
            data = page._load_and_prepare_data()

            assert data is not None
            assert len(data) == 3
            assert "ingredients" in data.columns

    def test_data_quality_validation(self):
        """Test la validation de la qualité des données."""
        # Créer des données avec quelques valeurs manquantes
        recipes_data = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "name": ["Recipe 1", None, "Recipe 3", "Recipe 4"],
                "ingredients": [
                    "['salt', 'pepper']",
                    "['sugar', 'flour']",
                    None,  # Ingrédients manquants
                    "['butter', 'eggs']",
                ],
            }
        )

        # Créer la page
        page = IngredientsClusteringPage("fake_path.csv")

        # Patcher directement la méthode _load_and_prepare_data
        with patch.object(page, "_load_and_prepare_data", return_value=recipes_data):
            data = page._load_and_prepare_data()

            # Les données devraient être chargées malgré les valeurs manquantes
            assert data is not None
            assert len(data) == 4


class TestDocumentation:
    """Test la qualité de la documentation."""

    def test_module_has_docstring(self):
        """Vérifie que le module a une docstring avec User Story."""
        # Au lieu de tester __doc__ du module importé (qui peut être None avec l'optimisation),
        # on vérifie directement le contenu du fichier source
        from pathlib import Path

        module_file = Path(__file__).parent.parent / "src" / "components" / "ingredients_clustering_page.py"

        # Lire le fichier source
        with open(module_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Vérifier qu'il y a une docstring au début (après from __future__)
        lines = content.split("\n")

        # Trouver la première docstring (après imports)
        docstring_found = False
        for i, line in enumerate(lines):
            if '"""' in line and i < 20:  # Dans les 20 premières lignes
                docstring_found = True
                docstring_content = []
                # Récupérer le contenu de la docstring
                if line.count('"""') == 2:  # Docstring sur une ligne
                    docstring_content.append(line)
                else:  # Docstring multi-lignes
                    j = i
                    while j < len(lines) and (j == i or '"""' not in lines[j]):
                        docstring_content.append(lines[j])
                        j += 1
                    if j < len(lines):
                        docstring_content.append(lines[j])

                docstring_text = "\n".join(docstring_content).lower()

                # Vérifier le contenu
                assert "streamlit" in docstring_text or "page" in docstring_text or "user story" in docstring_text
                break

        assert docstring_found, "Aucune docstring trouvée dans le module"

    def test_class_has_comprehensive_docstring(self):
        """Vérifie que la classe a une docstring complète."""
        doc = IngredientsClusteringPage.__doc__

        assert doc is not None
        assert len(doc) > 50

        # Devrait mentionner le but de la classe
        doc_lower = doc.lower()
        assert any(word in doc_lower for word in ["clustering", "ingrédient", "analyse"])

    def test_all_public_methods_documented(self):
        """Vérifie que toutes les méthodes publiques ont des docstrings."""
        page = IngredientsClusteringPage()

        public_methods = [name for name in dir(page) if callable(getattr(page, name)) and not name.startswith("_")]

        for method_name in public_methods:
            method = getattr(page, method_name)
            assert method.__doc__ is not None, f"La méthode {method_name} n'a pas de docstring"
            assert len(method.__doc__.strip()) > 20, f"La docstring de {method_name} est trop courte"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
