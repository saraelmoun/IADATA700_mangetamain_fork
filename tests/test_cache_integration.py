"""
Tests d'intégration pour le système de cache complet.
"""

import tempfile
from unittest.mock import patch
import sys
import os

import pandas as pd
import pytest

# Ajouter le répertoire src au path pour les imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.cache_manager import CacheManager, get_cache_manager
from core.cacheable_mixin import CacheableMixin
from core.logger import get_logger


class TestCacheSystemIntegration:
    """Tests d'intégration du système de cache complet."""

    def setup_method(self):
        """Setup avant chaque test."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup après chaque test."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_cache_workflow(self):
        """Test complet du workflow de cache."""
        
        class MockAnalyzer(CacheableMixin):
            def __init__(self, data):
                self.data = data
                self.logger = get_logger("test_analyzer")
                self.computation_count = 0
                self.enable_cache(True)

            def _get_default_cache_params(self):
                return {
                    "data_shape": self.data.shape,
                    "data_hash": hash(str(self.data.columns.tolist()))
                }

            def expensive_computation(self, operation_type="sum"):
                """Simule une opération coûteuse."""
                return self.cached_operation(
                    operation_name="expensive_computation",
                    operation_func=lambda: self._compute_expensive(operation_type),
                    cache_params={"operation_type": operation_type, **self._get_default_cache_params()}
                )

            def _compute_expensive(self, operation_type):
                """Calcul réel de l'opération."""
                self.computation_count += 1
                self.logger.info(f"Computing {operation_type} (call #{self.computation_count})")
                
                if operation_type == "sum":
                    return self.data.sum().sum()
                elif operation_type == "mean":
                    return self.data.mean().mean()
                else:
                    raise ValueError(f"Unknown operation: {operation_type}")

        # Créer des données de test
        test_data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50],
            'C': [100, 200, 300, 400, 500]
        })

        # Créer l'analyseur avec cache personnalisé
        analyzer = MockAnalyzer(test_data)
        analyzer._cache_manager = CacheManager(base_cache_dir=self.temp_dir)

        # Premier calcul - depuis scratch
        result1 = analyzer.expensive_computation("sum")
        expected_sum = test_data.sum().sum()  # 15 + 150 + 1500 = 1665
        assert result1 == expected_sum
        assert analyzer.computation_count == 1

        # Deuxième calcul identique - depuis le cache
        result2 = analyzer.expensive_computation("sum")
        assert result2 == expected_sum
        assert analyzer.computation_count == 1  # Pas de nouveau calcul

        # Calcul différent - nouveau calcul
        result3 = analyzer.expensive_computation("mean")
        expected_mean = test_data.mean().mean()  # (3 + 30 + 300) / 3 = 111
        assert result3 == expected_mean
        assert analyzer.computation_count == 2

        # Vérifier les informations de cache
        cache_info = analyzer._cache_manager.get_info()
        assert cache_info["total_files"] == 2
        assert "mock" in cache_info["analyzers"]  # "mockanalyzer" devient "mock"
        
        analyzer_info = cache_info["analyzers"]["mock"]
        assert analyzer_info["files"] == 2
        assert "expensive_computation" in analyzer_info["operations"]

    def test_cache_persistence_across_instances(self):
        """Test de persistance du cache entre différentes instances."""
        
        class PersistentAnalyzer(CacheableMixin):
            def __init__(self, identifier):
                self.identifier = identifier
                self.logger = get_logger(f"persistent_{identifier}")
                self.enable_cache(True)

            def _get_default_cache_params(self):
                return {"identifier": self.identifier}

            def compute_value(self, input_value):
                return self.cached_operation(
                    operation_name="compute_value",
                    operation_func=lambda: self._expensive_compute(input_value),
                    cache_params={"input": input_value, **self._get_default_cache_params()}
                )

            def _expensive_compute(self, input_value):
                return input_value * 2 + self.identifier

        # Partager le même cache manager
        shared_cache = CacheManager(base_cache_dir=self.temp_dir)

        # Première instance
        analyzer1 = PersistentAnalyzer(100)
        analyzer1._cache_manager = shared_cache
        result1 = analyzer1.compute_value(5)  # 5 * 2 + 100 = 110
        assert result1 == 110

        # Deuxième instance avec même identifier
        analyzer2 = PersistentAnalyzer(100)
        analyzer2._cache_manager = shared_cache
        result2 = analyzer2.compute_value(5)  # Devrait venir du cache
        assert result2 == 110

        # Troisième instance avec identifier différent
        analyzer3 = PersistentAnalyzer(200)
        analyzer3._cache_manager = shared_cache
        result3 = analyzer3.compute_value(5)  # 5 * 2 + 200 = 210
        assert result3 == 210

        # Vérifier que le cache contient les deux analyses
        info = shared_cache.get_info()
        assert info["total_files"] == 2

    def test_error_handling_integration(self):
        """Test de gestion d'erreur intégrée."""
        
        class ErrorProneAnalyzer(CacheableMixin):
            def __init__(self):
                self.logger = get_logger("error_analyzer")
                self.enable_cache(True)

            def _get_default_cache_params(self):
                return {"version": "1.0"}

            def risky_operation(self, should_fail=False):
                return self.cached_operation(
                    operation_name="risky_op",
                    operation_func=lambda: self._risky_compute(should_fail),
                    cache_params={"should_fail": should_fail, **self._get_default_cache_params()}
                )

            def _risky_compute(self, should_fail):
                if should_fail:
                    self.logger.error("Computation failed as requested")
                    raise RuntimeError("Intentional failure")
                else:
                    self.logger.info("Computation succeeded")
                    return "success_result"

        analyzer = ErrorProneAnalyzer()
        analyzer._cache_manager = CacheManager(base_cache_dir=self.temp_dir)

        # Test avec succès
        result_success = analyzer.risky_operation(should_fail=False)
        assert result_success == "success_result"

        # Test avec échec
        with pytest.raises(RuntimeError, match="Intentional failure"):
            analyzer.risky_operation(should_fail=True)

        # Vérifier que seul le succès est en cache
        info = analyzer._cache_manager.get_info()
        assert info["total_files"] == 1

    def test_logging_integration(self):
        """Test d'intégration avec le système de logging."""
        
        class LoggingAnalyzer(CacheableMixin):
            def __init__(self):
                self.logger = get_logger("logging_test")
                self.enable_cache(True)

            def _get_default_cache_params(self):
                return {"log_test": True}

            def logged_operation(self, value):
                self.logger.info(f"Starting operation with value: {value}")
                
                result = self.cached_operation(
                    operation_name="logged_op",
                    operation_func=lambda: self._logged_compute(value),
                    cache_params={"value": value, **self._get_default_cache_params()}
                )
                
                self.logger.info(f"Operation completed with result: {result}")
                return result

            def _logged_compute(self, value):
                self.logger.debug("Performing actual computation")
                return value ** 2

        analyzer = LoggingAnalyzer()
        analyzer._cache_manager = CacheManager(base_cache_dir=self.temp_dir)

        # Test avec capture de logs
        with patch.object(analyzer.logger, 'info') as mock_info, \
             patch.object(analyzer.logger, 'debug') as mock_debug:
            
            # Premier appel
            result = analyzer.logged_operation(5)
            assert result == 25
            
            # Vérifier les appels de logging
            assert mock_info.call_count >= 2  # Au moins start et complete
            assert mock_debug.call_count >= 1  # Au moins un debug lors du calcul

            # Reset mocks
            mock_info.reset_mock()
            mock_debug.reset_mock()

            # Deuxième appel (cache hit)
            result2 = analyzer.logged_operation(5)
            assert result2 == 25
            
            # Vérifier les logs du cache hit
            assert mock_debug.call_count >= 1  # Au moins un debug pour cache hit
            assert mock_info.call_count >= 2  # Toujours start et complete

    def test_global_cache_manager_integration(self):
        """Test d'intégration avec l'instance globale du cache manager."""
        
        class GlobalCacheAnalyzer(CacheableMixin):
            def __init__(self):
                self.logger = get_logger("global_cache_test")
                self.enable_cache(True)
                # Utiliser l'instance globale
                self._cache_manager = get_cache_manager()

            def _get_default_cache_params(self):
                return {"uses_global": True}

            def global_operation(self, data):
                return self.cached_operation(
                    operation_name="global_op",
                    operation_func=lambda: self._compute_global(data),
                    cache_params={"data": data, **self._get_default_cache_params()}
                )

            def _compute_global(self, data):
                return f"processed_{data}"

        # Créer deux instances utilisant le cache global
        analyzer1 = GlobalCacheAnalyzer()
        analyzer2 = GlobalCacheAnalyzer()

        # Premier calcul avec analyzer1
        result1 = analyzer1.global_operation("test_data")
        assert result1 == "processed_test_data"

        # Deuxième calcul avec analyzer2 - devrait utiliser le cache
        result2 = analyzer2.global_operation("test_data")
        assert result2 == "processed_test_data"

        # Vérifier que les deux utilisent la même instance de cache
        assert analyzer1._cache_manager is analyzer2._cache_manager

