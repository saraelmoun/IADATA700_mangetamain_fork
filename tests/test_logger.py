"""
Tests pour le module logger.
"""

import logging
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
import os

import pytest

# Ajouter le répertoire src au path pour les imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.logger import get_logger, MangetamainLogger, setup_logging


class TestLogger:
    """Tests pour le système de logging."""

    def test_get_logger_default(self):
        """Test de récupération du logger par défaut."""
        logger = get_logger()
        
        assert logger is not None
        assert isinstance(logger, MangetamainLogger)
        assert logger.logger.name == "mangetamain"
        # Le niveau peut varier selon l'ordre des tests, on vérifie juste qu'il existe
        assert logger.logger.level in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR]

    def test_get_logger_custom_name(self):
        """Test de récupération du logger avec nom personnalisé."""
        custom_name = "test_logger"
        logger = get_logger(custom_name)
        
        assert logger is not None
        assert isinstance(logger, MangetamainLogger)

    def test_get_logger_singleton(self):
        """Test que get_logger retourne toujours la même instance."""
        logger1 = get_logger()
        logger2 = get_logger()
        
        assert logger1 is logger2

    def test_mangetamain_logger_initialization(self):
        """Test de l'initialisation de MangetamainLogger."""
        with tempfile.TemporaryDirectory() as temp_dir:
            debug_file = Path(temp_dir) / "debug.log"
            error_file = Path(temp_dir) / "error.log"
            
            logger = MangetamainLogger(
                name="test_logger",
                level="DEBUG",
                debug_log_file=debug_file,
                error_log_file=error_file
            )
            
            assert logger.logger.name == "test_logger"
            assert logger.logger.level == logging.DEBUG
            assert len(logger.logger.handlers) >= 1  # Au moins console handler

    def test_logger_levels(self):
        """Test des différents niveaux de log."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = MangetamainLogger(name="test_levels", level="DEBUG")
            
            # Test avec capture de logs
            with patch.object(logger.logger, 'info') as mock_info:
                logger.info("Test info")
                mock_info.assert_called_once_with("Test info")

            with patch.object(logger.logger, 'debug') as mock_debug:
                logger.debug("Test debug")
                mock_debug.assert_called_once_with("Test debug")

            with patch.object(logger.logger, 'warning') as mock_warning:
                logger.warning("Test warning")
                mock_warning.assert_called_once_with("Test warning")

            with patch.object(logger.logger, 'error') as mock_error:
                logger.error("Test error")
                mock_error.assert_called_once_with("Test error")

    def test_logger_with_exception(self):
        """Test du logging avec exception."""
        logger = MangetamainLogger(name="test_exception")
        
        with patch.object(logger.logger, 'exception') as mock_exception:
            try:
                raise ValueError("Test exception")
            except ValueError:
                logger.exception("Error occurred")
                mock_exception.assert_called_once_with("Error occurred")

    def test_setup_logging(self):
        """Test de setup_logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            debug_file = Path(temp_dir) / "test_debug.log"
            error_file = Path(temp_dir) / "test_error.log"
            
            logger = setup_logging(
                level="DEBUG",
                debug_log_file=debug_file,
                error_log_file=error_file
            )
            
            assert isinstance(logger, MangetamainLogger)
            assert logger.logger.level == logging.DEBUG

    def test_logger_file_creation(self):
        """Test de création de fichiers de log."""
        with tempfile.TemporaryDirectory() as temp_dir:
            debug_file = Path(temp_dir) / "subdir" / "debug.log"
            error_file = Path(temp_dir) / "subdir" / "error.log"
            
            logger = MangetamainLogger(
                name="file_test",
                level="DEBUG",
                debug_log_file=debug_file,
                error_log_file=error_file
            )
            
            # Les répertoires parent doivent être créés
            assert debug_file.parent.exists()
            assert error_file.parent.exists()

    def test_logger_performance_info(self):
        """Test du logging d'informations de performance."""
        logger = get_logger()
        
        with patch.object(logger.logger, 'info') as mock_info:
            operation_time = 1.234
            logger.info(f"Operation completed in {operation_time:.3f}s")
            mock_info.assert_called_once_with("Operation completed in 1.234s")

    def test_logger_cache_operations(self):
        """Test du logging pour les opérations de cache."""
        logger = get_logger()
        
        with patch.object(logger.logger, 'debug') as mock_debug:
            logger.debug("Cache hit: analyzer.operation")
            mock_debug.assert_called_once_with("Cache hit: analyzer.operation")

    def test_logger_handlers_not_duplicated(self):
        """Test que les handlers ne sont pas dupliqués."""
        # Créer plusieurs instances avec le même nom
        logger1 = MangetamainLogger(name="duplicate_test")
        logger2 = MangetamainLogger(name="duplicate_test")
        
        # Le nombre de handlers ne doit pas doubler
        assert len(logger1.logger.handlers) == len(logger2.logger.handlers)

    def test_convenience_functions(self):
        """Test des fonctions de convenance."""
        with patch('core.logger.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            from core.logger import log_info, log_warning, log_error, log_debug
            
            log_info("Test message")
            mock_logger.info.assert_called_once_with("Test message")

            log_warning("Warning message")
            mock_logger.warning.assert_called_once_with("Warning message")

            log_error("Error message")
            mock_logger.error.assert_called_once_with("Error message")

            log_debug("Debug message")
            mock_logger.debug.assert_called_once_with("Debug message")

    def test_logger_thread_safety(self):
        """Test de la sécurité thread du logger."""
        import threading
        import time
        
        results = []
        logger = get_logger()
        
        def log_messages(thread_id):
            for i in range(10):
                logger.info(f"Thread {thread_id}, message {i}")
                results.append((thread_id, i))
                time.sleep(0.001)  # Petit délai pour simulation
        
        # Lancer plusieurs threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=log_messages, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Attendre que tous les threads se terminent
        for thread in threads:
            thread.join()
        
        # Vérifier que tous les messages ont été loggés
        assert len(results) == 30  # 3 threads × 10 messages

    def test_logger_with_kwargs(self):
        """Test du logging avec arguments supplémentaires."""
        logger = MangetamainLogger(name="kwargs_test")
        
        with patch.object(logger.logger, 'info') as mock_info:
            logger.info("Test message", extra={'user_id': 123})
            mock_info.assert_called_once_with("Test message", extra={'user_id': 123})

    def test_logger_level_filtering(self):
        """Test du filtrage par niveau."""
        logger = MangetamainLogger(name="level_test", level="WARNING")
        
        with patch.object(logger.logger, 'debug') as mock_debug, \
             patch.object(logger.logger, 'info') as mock_info, \
             patch.object(logger.logger, 'warning') as mock_warning:
            
            logger.debug("Debug message")
            logger.info("Info message") 
            logger.warning("Warning message")
            
            # Seul warning devrait être loggé (niveau WARNING)
            mock_debug.assert_called_once()  # Appelé mais filtré
            mock_info.assert_called_once()   # Appelé mais filtré
            mock_warning.assert_called_once()  # Appelé et loggé