"""
Gestionnaire de cache centralisé pour tous les analyseurs.
Permet une gestion uniforme du cache across l'application.
"""

import hashlib
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar

from .logger import get_logger

T = TypeVar("T")


class CacheManager:
    """Gestionnaire de cache centralisé avec support multi-analyseurs."""

    def __init__(self, base_cache_dir: str = "cache"):
        """
        Initialise le gestionnaire de cache.

        Args:
            base_cache_dir: Répertoire de base pour le cache
        """
        self.base_cache_dir = Path(base_cache_dir)
        self.base_cache_dir.mkdir(exist_ok=True)
        self.logger = get_logger()

    def _generate_key(
        self, analyzer_name: str, operation: str, params: Dict[str, Any]
    ) -> str:
        """
        Génère une clé de cache unique basée sur l'analyseur, l'opération et les paramètres.

        Args:
            analyzer_name: Nom de l'analyseur (ex: "interactions", "ingredients")
            operation: Nom de l'opération (ex: "aggregate", "cluster")
            params: Paramètres de l'opération

        Returns:
            Clé de cache unique
        """
        # Créer un dictionnaire ordonné pour une sérialisation cohérente
        cache_data = {
            "analyzer": analyzer_name,
            "operation": operation,
            "params": params,
        }

        # Sérialiser et hasher
        serialized = str(sorted(cache_data.items()))
        return hashlib.md5(serialized.encode()).hexdigest()

    def _get_cache_path(
        self, analyzer_name: str, operation: str, cache_key: str
    ) -> Path:
        """
        Génère le chemin du fichier de cache.

        Args:
            analyzer_name: Nom de l'analyseur
            operation: Nom de l'opération
            cache_key: Clé de cache

        Returns:
            Chemin vers le fichier de cache
        """
        # Structure: cache/analyzer_name/operation/cache_key.pkl
        cache_dir = self.base_cache_dir / analyzer_name / operation
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{cache_key}.pkl"

    def get(
        self, analyzer_name: str, operation: str, params: Dict[str, Any]
    ) -> Optional[T]:
        """
        Récupère un objet du cache.

        Args:
            analyzer_name: Nom de l'analyseur
            operation: Nom de l'opération
            params: Paramètres de l'opération

        Returns:
            Objet mis en cache ou None si pas trouvé
        """
        try:
            cache_key = self._generate_key(analyzer_name, operation, params)
            cache_path = self._get_cache_path(analyzer_name, operation, cache_key)

            if not cache_path.exists():
                self.logger.debug(f"Cache miss: {analyzer_name}.{operation}")
                return None

            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)

            # Vérifier la validité (optionnel: ajouter TTL plus tard)
            if "timestamp" in cached_data and "data" in cached_data:
                self.logger.debug(f"Cache hit: {analyzer_name}.{operation}")
                return cached_data["data"]
            else:
                self.logger.warning(
                    f"Invalid cache format for {analyzer_name}.{operation}"
                )
                return None

        except Exception as e:
            self.logger.warning(
                f"Error loading cache for {analyzer_name}.{operation}: {e}"
            )
            return None

    def set(
        self, analyzer_name: str, operation: str, params: Dict[str, Any], data: T
    ) -> bool:
        """
        Sauvegarde un objet dans le cache.

        Args:
            analyzer_name: Nom de l'analyseur
            operation: Nom de l'opération
            params: Paramètres de l'opération
            data: Données à mettre en cache

        Returns:
            True si la sauvegarde a réussi
        """
        try:
            cache_key = self._generate_key(analyzer_name, operation, params)
            cache_path = self._get_cache_path(analyzer_name, operation, cache_key)

            # Préparer les données avec métadonnées
            cache_data = {
                "data": data,
                "timestamp": datetime.now().isoformat(),
                "analyzer": analyzer_name,
                "operation": operation,
                "params": params,
            }

            with open(cache_path, "wb") as f:
                pickle.dump(cache_data, f)

            self.logger.debug(f"Cache saved: {analyzer_name}.{operation}")
            return True

        except Exception as e:
            self.logger.warning(
                f"Error saving cache for {analyzer_name}.{operation}: {e}"
            )
            return False

    def clear(
        self, analyzer_name: Optional[str] = None, operation: Optional[str] = None
    ) -> int:
        """
        Nettoie le cache.

        Args:
            analyzer_name: Si spécifié, nettoie seulement cet analyseur
            operation: Si spécifié (avec analyzer_name), nettoie seulement cette opération

        Returns:
            Nombre de fichiers supprimés
        """
        deleted_count = 0

        try:
            if analyzer_name is None:
                # Nettoyer tout le cache
                target_path = self.base_cache_dir
            elif operation is None:
                # Nettoyer un analyseur
                target_path = self.base_cache_dir / analyzer_name
            else:
                # Nettoyer une opération spécifique
                target_path = self.base_cache_dir / analyzer_name / operation

            if target_path.exists():
                for cache_file in target_path.rglob("*.pkl"):
                    cache_file.unlink()
                    deleted_count += 1

                # Supprimer les dossiers vides
                if analyzer_name is None:
                    # Ne pas supprimer le dossier racine du cache
                    pass
                else:
                    try:
                        if operation is not None:
                            target_path.rmdir()  # Supprimer le dossier d'opération s'il est vide
                        # Essayer de supprimer le dossier d'analyseur s'il est vide
                        analyzer_path = self.base_cache_dir / analyzer_name
                        if analyzer_path.exists() and not any(analyzer_path.iterdir()):
                            analyzer_path.rmdir()
                    except OSError:
                        pass  # Dossier pas vide, c'est normal

            self.logger.info(f"Cache cleared: {deleted_count} files deleted")
            return deleted_count

        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
            return 0

    def get_info(self) -> Dict[str, Any]:
        """
        Retourne des informations sur le cache.

        Returns:
            Dictionnaire avec les statistiques du cache
        """
        info = {
            "base_directory": str(self.base_cache_dir),
            "analyzers": {},
            "total_files": 0,
            "total_size_mb": 0.0,
        }

        try:
            for analyzer_dir in self.base_cache_dir.iterdir():
                if analyzer_dir.is_dir():
                    analyzer_name = analyzer_dir.name
                    analyzer_info = {"operations": {}, "files": 0, "size_mb": 0.0}

                    for operation_dir in analyzer_dir.iterdir():
                        if operation_dir.is_dir():
                            operation_name = operation_dir.name
                            operation_files = list(operation_dir.glob("*.pkl"))
                            operation_size = sum(
                                f.stat().st_size for f in operation_files
                            )

                            analyzer_info["operations"][operation_name] = {
                                "files": len(operation_files),
                                "size_mb": round(operation_size / (1024 * 1024), 2),
                            }

                            analyzer_info["files"] += len(operation_files)
                            analyzer_info["size_mb"] += operation_size / (1024 * 1024)

                    analyzer_info["size_mb"] = round(analyzer_info["size_mb"], 2)
                    info["analyzers"][analyzer_name] = analyzer_info
                    info["total_files"] += analyzer_info["files"]
                    info["total_size_mb"] += analyzer_info["size_mb"]

            info["total_size_mb"] = round(info["total_size_mb"], 2)

        except Exception as e:
            self.logger.error(f"Error getting cache info: {e}")

        return info


# Instance globale pour faciliter l'utilisation
_cache_manager = None


def get_cache_manager() -> CacheManager:
    """Retourne l'instance globale du gestionnaire de cache."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager
