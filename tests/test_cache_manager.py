"""
Tests pour le CacheManager.
"""

import pickle
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
import os

import pandas as pd
import pytest

# Ajouter le répertoire src au path pour les imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.cache_manager import CacheManager, get_cache_manager


class TestCacheManager:
    """Tests pour le gestionnaire de cache centralisé."""

    def setup_method(self):
        """Setup avant chaque test."""
        # Créer un répertoire temporaire pour les tests
        self.temp_dir = tempfile.mkdtemp()
        self.cache_manager = CacheManager(base_cache_dir=self.temp_dir)

    def teardown_method(self):
        """Cleanup après chaque test."""
        # Nettoyer le cache de test
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cache_manager_initialization(self):
        """Test de l'initialisation du CacheManager."""
        assert self.cache_manager.base_cache_dir == Path(self.temp_dir)
        assert self.cache_manager.base_cache_dir.exists()
        assert self.cache_manager.logger is not None

    def test_generate_key(self):
        """Test de génération de clé de cache."""
        params = {"param1": "value1", "param2": 42}
        key1 = self.cache_manager._generate_key("test_analyzer", "test_operation", params)
        key2 = self.cache_manager._generate_key("test_analyzer", "test_operation", params)
        
        # Même paramètres = même clé
        assert key1 == key2
        assert len(key1) == 32  # MD5 hash length
        
        # Paramètres différents = clés différentes
        different_params = {"param1": "value2", "param2": 42}
        key3 = self.cache_manager._generate_key("test_analyzer", "test_operation", different_params)
        assert key1 != key3

    def test_get_cache_path(self):
        """Test de génération de chemin de cache."""
        path = self.cache_manager._get_cache_path("test_analyzer", "test_operation", "test_key")
        
        expected_path = Path(self.temp_dir) / "test_analyzer" / "test_operation" / "test_key.pkl"
        assert path == expected_path
        assert path.parent.exists()  # Répertoire créé automatiquement

    def test_set_and_get_simple_data(self):
        """Test de sauvegarde et récupération de données simples."""
        analyzer_name = "test_analyzer"
        operation = "test_operation"
        params = {"test_param": "test_value"}
        test_data = {"result": "test_result", "count": 42}
        
        # Sauvegarder
        success = self.cache_manager.set(analyzer_name, operation, params, test_data)
        assert success is True
        
        # Récupérer
        cached_data = self.cache_manager.get(analyzer_name, operation, params)
        assert cached_data == test_data

    def test_set_and_get_dataframe(self):
        """Test avec DataFrame pandas."""
        analyzer_name = "test_analyzer"
        operation = "dataframe_operation"
        params = {"rows": 3, "cols": 2}
        
        # Créer un DataFrame de test
        test_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        # Sauvegarder
        success = self.cache_manager.set(analyzer_name, operation, params, test_df)
        assert success is True
        
        # Récupérer
        cached_df = self.cache_manager.get(analyzer_name, operation, params)
        pd.testing.assert_frame_equal(cached_df, test_df)

    def test_get_cache_miss(self):
        """Test de cache miss."""
        result = self.cache_manager.get("nonexistent", "operation", {"param": "value"})
        assert result is None

    def test_get_with_different_params(self):
        """Test que des paramètres différents donnent un cache miss."""
        analyzer_name = "test_analyzer"
        operation = "test_operation"
        params1 = {"param": "value1"}
        params2 = {"param": "value2"}
        test_data = "test_data"
        
        # Sauvegarder avec params1
        self.cache_manager.set(analyzer_name, operation, params1, test_data)
        
        # Essayer de récupérer avec params2 (différents)
        result = self.cache_manager.get(analyzer_name, operation, params2)
        assert result is None

    def test_clear_all_cache(self):
        """Test de suppression complète du cache."""
        # Ajouter plusieurs entrées
        self.cache_manager.set("analyzer1", "op1", {"p": 1}, "data1")
        self.cache_manager.set("analyzer1", "op2", {"p": 2}, "data2")
        self.cache_manager.set("analyzer2", "op1", {"p": 3}, "data3")
        
        # Supprimer tout le cache
        deleted_count = self.cache_manager.clear()
        assert deleted_count == 3
        
        # Vérifier que le cache est vide
        assert self.cache_manager.get("analyzer1", "op1", {"p": 1}) is None
        assert self.cache_manager.get("analyzer1", "op2", {"p": 2}) is None
        assert self.cache_manager.get("analyzer2", "op1", {"p": 3}) is None

    def test_clear_analyzer_cache(self):
        """Test de suppression du cache d'un analyseur spécifique."""
        # Ajouter plusieurs entrées
        self.cache_manager.set("analyzer1", "op1", {"p": 1}, "data1")
        self.cache_manager.set("analyzer1", "op2", {"p": 2}, "data2")
        self.cache_manager.set("analyzer2", "op1", {"p": 3}, "data3")
        
        # Supprimer seulement analyzer1
        deleted_count = self.cache_manager.clear(analyzer_name="analyzer1")
        assert deleted_count == 2
        
        # Vérifier que analyzer1 est supprimé mais pas analyzer2
        assert self.cache_manager.get("analyzer1", "op1", {"p": 1}) is None
        assert self.cache_manager.get("analyzer1", "op2", {"p": 2}) is None
        assert self.cache_manager.get("analyzer2", "op1", {"p": 3}) == "data3"

    def test_clear_operation_cache(self):
        """Test de suppression du cache d'une opération spécifique."""
        # Ajouter plusieurs entrées
        self.cache_manager.set("analyzer1", "op1", {"p": 1}, "data1")
        self.cache_manager.set("analyzer1", "op2", {"p": 2}, "data2")
        self.cache_manager.set("analyzer1", "op1", {"p": 3}, "data3")  # même op, params différents
        
        # Supprimer seulement op1 de analyzer1
        deleted_count = self.cache_manager.clear(analyzer_name="analyzer1", operation="op1")
        assert deleted_count == 2  # 2 fichiers op1 avec des paramètres différents
        
        # Vérifier que op1 est supprimé mais pas op2
        assert self.cache_manager.get("analyzer1", "op1", {"p": 1}) is None
        assert self.cache_manager.get("analyzer1", "op1", {"p": 3}) is None
        assert self.cache_manager.get("analyzer1", "op2", {"p": 2}) == "data2"

    def test_get_info_empty_cache(self):
        """Test d'info sur un cache vide."""
        info = self.cache_manager.get_info()
        
        assert info['base_directory'] == str(self.cache_manager.base_cache_dir)
        assert info['analyzers'] == {}
        assert info['total_files'] == 0
        assert info['total_size_mb'] == 0.0

    def test_get_info_with_data(self):
        """Test de récupération d'informations sur un cache avec données."""
        # Ajouter des données de test
        self.cache_manager.set("test_analyzer", "operation1", {"param": "value"}, "data1")
        self.cache_manager.set("test_analyzer", "operation2", {"param": "value"}, "data2")
        
        info = self.cache_manager.get_info()
        
        assert info['total_files'] == 2
        assert 'base_directory' in info
        assert 'analyzers' in info
        assert len(info['analyzers']) == 1
        assert 'test_analyzer' in info['analyzers']
        assert info['analyzers']['test_analyzer']['files'] == 2
        # La taille peut être 0 si les données sont petites, on vérifie qu'elle est >= 0
        assert info['total_size_mb'] >= 0

    def test_cache_metadata(self):
        """Test des métadonnées stockées avec les données."""
        analyzer_name = "test_analyzer"
        operation = "test_operation"
        params = {"test": "value"}
        data = "test_data"
        
        # Sauvegarder
        self.cache_manager.set(analyzer_name, operation, params, data)
        
        # Lire directement le fichier pour vérifier les métadonnées
        cache_key = self.cache_manager._generate_key(analyzer_name, operation, params)
        cache_path = self.cache_manager._get_cache_path(analyzer_name, operation, cache_key)
        
        with open(cache_path, 'rb') as f:
            cache_content = pickle.load(f)
        
        assert 'data' in cache_content
        assert 'timestamp' in cache_content
        assert 'analyzer' in cache_content
        assert 'operation' in cache_content
        assert 'params' in cache_content
        
        assert cache_content['data'] == data
        assert cache_content['analyzer'] == analyzer_name
        assert cache_content['operation'] == operation
        assert cache_content['params'] == params

    def test_error_handling_corrupted_cache(self):
        """Test de gestion d'erreur avec cache corrompu."""
        analyzer_name = "test_analyzer"
        operation = "test_operation"
        params = {"test": "value"}
        
        # Créer un fichier de cache corrompu
        cache_key = self.cache_manager._generate_key(analyzer_name, operation, params)
        cache_path = self.cache_manager._get_cache_path(analyzer_name, operation, cache_key)
        
        # Écrire des données invalides
        with open(cache_path, 'w') as f:
            f.write("invalid pickle data")
        
        # Essayer de lire le cache corrompu
        result = self.cache_manager.get(analyzer_name, operation, params)
        assert result is None  # Devrait retourner None en cas d'erreur

    def test_error_handling_set_failure(self):
        """Test de gestion d'erreur lors de la sauvegarde."""
        # Mock pour simuler une erreur d'écriture
        with patch('builtins.open', side_effect=PermissionError("Access denied")):
            success = self.cache_manager.set("test", "test", {}, "data")
            assert success is False

    def test_large_data_caching(self):
        """Test avec de gros volumes de données."""
        # Créer un gros DataFrame
        large_df = pd.DataFrame({
            f'col_{i}': list(range(1000)) for i in range(10)
        })
        
        analyzer_name = "test_analyzer"
        operation = "large_operation"
        params = {"size": "large"}
        
        # Sauvegarder
        success = self.cache_manager.set(analyzer_name, operation, params, large_df)
        assert success is True
        
        # Récupérer
        cached_df = self.cache_manager.get(analyzer_name, operation, params)
        pd.testing.assert_frame_equal(cached_df, large_df)
        
        # Vérifier la taille dans les infos
        info = self.cache_manager.get_info()
        assert info['total_size_mb'] > 0


class TestCacheManagerSingleton:
    """Tests pour l'instance globale du cache manager."""

    def test_get_cache_manager_singleton(self):
        """Test que get_cache_manager retourne toujours la même instance."""
        manager1 = get_cache_manager()
        manager2 = get_cache_manager()
        
        assert manager1 is manager2

    def test_get_cache_manager_initialization(self):
        """Test de l'initialisation de l'instance globale."""
        manager = get_cache_manager()
        
        assert isinstance(manager, CacheManager)
        assert manager.base_cache_dir.name == "cache"
        assert manager.base_cache_dir.exists()

    def test_global_cache_manager_persistence(self):
        """Test de persistance de l'instance globale."""
        manager = get_cache_manager()
        
        # Ajouter des données
        manager.set("global_test", "operation", {"param": "value"}, "test_data")
        
        # Récupérer l'instance à nouveau
        manager2 = get_cache_manager()
        
        # Les données doivent être présentes
        result = manager2.get("global_test", "operation", {"param": "value"})
        assert result == "test_data"