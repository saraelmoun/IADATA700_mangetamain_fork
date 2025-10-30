"""
Tests pour le CacheableMixin.
"""

from core.cache_manager import CacheManager
from core.cacheable_mixin import CacheableMixin
import tempfile
import sys
import os

import pytest

# Ajouter le répertoire src au path pour les imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestCacheableMixin:
    """Tests pour le mixin de cache."""

    def setup_method(self):
        """Setup avant chaque test."""
        # Créer un répertoire temporaire pour les tests
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup après chaque test."""
        # Nettoyer le cache de test
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cacheable_mixin_basic_usage(self):
        """Test d'utilisation basique du CacheableMixin."""

        class TestAnalyzer(CacheableMixin):
            def __init__(self):
                self.enable_cache(True)
                self.call_count = 0

            def _get_default_cache_params(self):
                return {"test": "params"}

            def expensive_operation(self, param1, param2=None):
                return self.cached_operation(
                    operation_name="expensive_op",
                    operation_func=lambda: self._compute_expensive(param1, param2),
                    cache_params={
                        "param1": param1,
                        "param2": param2,
                        **self._get_default_cache_params(),
                    },
                )

            def _compute_expensive(self, param1, param2=None):
                self.call_count += 1
                return f"result_{param1}_{param2}"

        # Créer une instance de test avec cache personnalisé
        analyzer = TestAnalyzer()
        analyzer._cache_manager = CacheManager(base_cache_dir=self.temp_dir)

        # Premier appel - calcul depuis scratch
        result1 = analyzer.expensive_operation("test", "value")
        assert result1 == "result_test_value"
        assert analyzer.call_count == 1

        # Deuxième appel - depuis le cache
        result2 = analyzer.expensive_operation("test", "value")
        assert result2 == "result_test_value"
        assert analyzer.call_count == 1  # Pas de nouveau calcul

        # Appel avec paramètres différents - nouveau calcul
        result3 = analyzer.expensive_operation("test2", "value")
        assert result3 == "result_test2_value"
        assert analyzer.call_count == 2

    def test_enable_cache_initialization(self):
        """Test de l'initialisation du cache."""

        class TestAnalyzer(CacheableMixin):
            def _get_default_cache_params(self):
                return {}

        analyzer = TestAnalyzer()

        # Par défaut, le cache est activé dans __init__
        assert analyzer._cache_enabled is True

        # Désactiver le cache
        analyzer.enable_cache(False)
        assert analyzer._cache_enabled is False
        assert hasattr(analyzer, "_cache_manager")
        assert hasattr(analyzer, "_analyzer_name")

    def test_cache_disabled(self):
        """Test avec cache désactivé."""

        class TestAnalyzer(CacheableMixin):
            def __init__(self):
                self.enable_cache(False)
                self.call_count = 0

            def _get_default_cache_params(self):
                return {}

            def operation(self):
                return self.cached_operation(
                    operation_name="test_op",
                    operation_func=self._compute,
                    cache_params={},
                )

            def _compute(self):
                self.call_count += 1
                return "computed_result"

        analyzer = TestAnalyzer()

        # Chaque appel doit recalculer
        result1 = analyzer.operation()
        assert result1 == "computed_result"
        assert analyzer.call_count == 1

        result2 = analyzer.operation()
        assert result2 == "computed_result"
        assert analyzer.call_count == 2  # Nouveau calcul car cache désactivé

    def test_cache_with_complex_data(self):
        """Test avec des données complexes."""

        class TestAnalyzer(CacheableMixin):
            def __init__(self):
                self.enable_cache(True)

            def _get_default_cache_params(self):
                return {"analyzer_id": "test"}

            def complex_operation(self, data_dict, data_list):
                return self.cached_operation(
                    operation_name="complex_op",
                    operation_func=lambda: self._compute_complex(data_dict, data_list),
                    cache_params={
                        "data_dict": data_dict,
                        "data_list": data_list,
                        **self._get_default_cache_params(),
                    },
                )

            def _compute_complex(self, data_dict, data_list):
                return {
                    "dict_keys": list(data_dict.keys()),
                    "list_sum": sum(data_list),
                    "combined": f"{len(data_dict)}_{len(data_list)}",
                }

        analyzer = TestAnalyzer()
        analyzer._cache_manager = CacheManager(base_cache_dir=self.temp_dir)

        test_dict = {"a": 1, "b": 2}
        test_list = [1, 2, 3, 4, 5]

        # Premier appel
        result1 = analyzer.complex_operation(test_dict, test_list)
        expected = {"dict_keys": ["a", "b"], "list_sum": 15, "combined": "2_5"}
        assert result1 == expected

        # Deuxième appel - depuis le cache
        result2 = analyzer.complex_operation(test_dict, test_list)
        assert result2 == expected

    def test_cache_error_handling(self):
        """Test de gestion d'erreur dans les opérations cachées."""

        class TestAnalyzer(CacheableMixin):
            def __init__(self):
                self.enable_cache(True)

            def _get_default_cache_params(self):
                return {}

            def failing_operation(self, should_fail=True):
                return self.cached_operation(
                    operation_name="failing_op",
                    operation_func=lambda: self._compute_failing(should_fail),
                    cache_params={"should_fail": should_fail},
                )

            def _compute_failing(self, should_fail):
                if should_fail:
                    raise ValueError("Computation failed")
                return "success"

        analyzer = TestAnalyzer()
        analyzer._cache_manager = CacheManager(base_cache_dir=self.temp_dir)

        # Test avec erreur
        with pytest.raises(ValueError, match="Computation failed"):
            analyzer.failing_operation(should_fail=True)

        # Test sans erreur
        result = analyzer.failing_operation(should_fail=False)
        assert result == "success"

    def test_cache_with_different_analyzer_names(self):
        """Test que différents noms d'analyseur ont des caches séparés."""

        class TestAnalyzer1(CacheableMixin):
            def __init__(self):
                self.enable_cache(True)
                self._analyzer_name = "analyzer1"

            def _get_default_cache_params(self):
                return {}

            def operation(self):
                return self.cached_operation(
                    operation_name="test_op",
                    operation_func=lambda: "result1",
                    cache_params={},
                )

        class TestAnalyzer2(CacheableMixin):
            def __init__(self):
                self.enable_cache(True)
                self._analyzer_name = "analyzer2"

            def _get_default_cache_params(self):
                return {}

            def operation(self):
                return self.cached_operation(
                    operation_name="test_op",
                    operation_func=lambda: "result2",
                    cache_params={},
                )

        # Partager le même cache manager
        cache_manager = CacheManager(base_cache_dir=self.temp_dir)

        analyzer1 = TestAnalyzer1()
        analyzer1._cache_manager = cache_manager

        analyzer2 = TestAnalyzer2()
        analyzer2._cache_manager = cache_manager

        # Les résultats doivent être différents et indépendants
        result1 = analyzer1.operation()
        result2 = analyzer2.operation()

        assert result1 == "result1"
        assert result2 == "result2"

        # Vérifier que les deux ont des entrées de cache séparées
        info = cache_manager.get_info()
        assert "analyzer1" in info["analyzers"]
        assert "analyzer2" in info["analyzers"]

    def test_cache_params_hashing(self):
        """Test que différents paramètres de cache génèrent différentes clés."""

        class TestAnalyzer(CacheableMixin):
            def __init__(self):
                self.enable_cache(True)
                self.call_count = 0

            def _get_default_cache_params(self):
                return {"base": "param"}

            def operation(self, value, multiplier=1):
                return self.cached_operation(
                    operation_name="math_op",
                    operation_func=lambda: self._compute(value, multiplier),
                    cache_params={
                        "value": value,
                        "multiplier": multiplier,
                        **self._get_default_cache_params(),
                    },
                )

            def _compute(self, value, multiplier):
                self.call_count += 1
                return value * multiplier

        analyzer = TestAnalyzer()
        analyzer._cache_manager = CacheManager(base_cache_dir=self.temp_dir)

        # Différentes combinaisons de paramètres
        assert analyzer.operation(5, 2) == 10
        assert analyzer.call_count == 1

        assert analyzer.operation(5, 3) == 15  # Nouveau calcul (multiplier différent)
        assert analyzer.call_count == 2

        assert analyzer.operation(5, 2) == 10  # Cache hit (mêmes paramètres)
        assert analyzer.call_count == 2

        assert analyzer.operation(10, 2) == 20  # Nouveau calcul (value différent)
        assert analyzer.call_count == 3

    def test_abstract_method_requirement(self):
        """Test que _get_default_cache_params fonctionne par défaut."""

        class IncompleteAnalyzer(CacheableMixin):
            pass  # N'override pas _get_default_cache_params

        analyzer = IncompleteAnalyzer()

        # Devrait fonctionner avec l'implémentation par défaut
        result = analyzer.cached_operation("test", lambda: "result")
        assert result == "result"

        # Et doit être en cache maintenant
        result2 = analyzer.cached_operation("test", lambda: "should not execute")
        assert result2 == "result"  # Cache hit

    def test_cache_manager_integration(self):
        """Test d'intégration avec le CacheManager."""

        class TestAnalyzer(CacheableMixin):
            def __init__(self):
                super().__init__()  # Utiliser l'initialisation normale
                self._analyzer_name = "test"  # Forcer le nom

            def _get_default_cache_params(self):
                return {"integration": "test"}

            def operation(self):
                return self.cached_operation(
                    operation_name="integration_op",
                    operation_func=lambda: "integration_result",
                    cache_params=self._get_default_cache_params(),
                )

        analyzer = TestAnalyzer()
        analyzer._cache_manager = CacheManager(base_cache_dir=self.temp_dir)

        # Opération
        result = analyzer.operation()
        assert result == "integration_result"

        # Vérifier dans le cache manager
        info = analyzer._cache_manager.get_info()
        assert info["total_files"] == 1
        assert "test" in info["analyzers"]

        # Tester le clear cache
        deleted = analyzer._cache_manager.clear()
        assert deleted == 1

        info_after = analyzer._cache_manager.get_info()
        assert info_after["total_files"] == 0

    def test_cache_with_mutable_params(self):
        """Test avec des paramètres mutables dans le cache."""

        class TestAnalyzer(CacheableMixin):
            def __init__(self):
                self.enable_cache(True)
                self.call_count = 0

            def _get_default_cache_params(self):
                return {}

            def operation_with_list(self, items):
                return self.cached_operation(
                    operation_name="list_op",
                    operation_func=lambda: self._compute_list(items),
                    cache_params={"items": items},
                )

            def _compute_list(self, items):
                self.call_count += 1
                return sum(items)

        analyzer = TestAnalyzer()
        analyzer._cache_manager = CacheManager(base_cache_dir=self.temp_dir)

        # Test avec liste
        result1 = analyzer.operation_with_list([1, 2, 3])
        assert result1 == 6
        assert analyzer.call_count == 1

        # Même liste - cache hit
        result2 = analyzer.operation_with_list([1, 2, 3])
        assert result2 == 6
        assert analyzer.call_count == 1

        # Liste différente - nouveau calcul
        result3 = analyzer.operation_with_list([1, 2, 4])
        assert result3 == 7
        assert analyzer.call_count == 2
