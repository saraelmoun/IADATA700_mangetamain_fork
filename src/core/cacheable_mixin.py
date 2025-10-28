"""
Mixin pour ajouter des capacités de cache aux analyseurs.
"""

from typing import Any, Callable, Dict, Optional, TypeVar

from .cache_manager import get_cache_manager

T = TypeVar('T')


class CacheableMixin:
    """Mixin qui ajoute des capacités de cache à une classe."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache_manager = get_cache_manager()
        self._cache_enabled = True
        self._analyzer_name = self.__class__.__name__.lower().replace('analyzer', '')
    
    def enable_cache(self, enabled: bool = True) -> None:
        """Active ou désactive le cache pour cet analyseur."""
        self._cache_enabled = enabled
        
        # Initialiser les attributs s'ils n'existent pas encore
        if not hasattr(self, '_cache_manager'):
            self._cache_manager = get_cache_manager()
        if not hasattr(self, '_analyzer_name'):
            self._analyzer_name = self.__class__.__name__.lower().replace('analyzer', '')
    
    def cached_operation(self, 
                        operation_name: str, 
                        operation_func: Callable[[], T], 
                        cache_params: Optional[Dict[str, Any]] = None) -> T:
        """
        Exécute une opération avec mise en cache automatique.
        
        Args:
            operation_name: Nom de l'opération (pour la clé de cache)
            operation_func: Fonction à exécuter si pas en cache
            cache_params: Paramètres pour générer la clé de cache
            
        Returns:
            Résultat de l'opération (depuis le cache ou calculé)
        """
        if not self._cache_enabled:
            return operation_func()
        
        # Utiliser les paramètres fournis ou ceux par défaut
        if cache_params is None:
            cache_params = self._get_default_cache_params()
        
        # Essayer de charger depuis le cache
        cached_result = self._cache_manager.get(
            analyzer_name=self._analyzer_name,
            operation=operation_name,
            params=cache_params
        )
        
        if cached_result is not None:
            return cached_result
        
        # Exécuter l'opération et sauvegarder en cache
        result = operation_func()
        self._cache_manager.set(
            analyzer_name=self._analyzer_name,
            operation=operation_name,
            params=cache_params,
            data=result
        )
        
        return result
    
    def _get_default_cache_params(self) -> Dict[str, Any]:
        """
        Retourne les paramètres par défaut pour le cache.
        À override dans les classes dérivées.
        """
        return {}
    
    def clear_cache(self, operation: Optional[str] = None) -> int:
        """
        Nettoie le cache pour cet analyseur.
        
        Args:
            operation: Si spécifié, nettoie seulement cette opération
            
        Returns:
            Nombre de fichiers supprimés
        """
        return self._cache_manager.clear(
            analyzer_name=self._analyzer_name,
            operation=operation
        )
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Retourne les informations de cache pour cet analyseur."""
        full_info = self._cache_manager.get_info()
        return full_info.get('analyzers', {}).get(self._analyzer_name, {
            'operations': {},
            'files': 0,
            'size_mb': 0.0
        })