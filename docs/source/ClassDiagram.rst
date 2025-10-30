Architecture du Projet
=======================

Diagramme de Classes
--------------------

Le diagramme suivant illustre l'architecture modulaire de l'application Mangetamain :

.. image:: ClassDiagram.png
   :width: 100%
   :alt: Diagramme de classes de l'application Mangetamain

Composants Principaux
---------------------

**Application Core**
   - `App` : Point d'entrée principal de l'application Streamlit
   - `AppConfig` : Configuration centralisée de l'application

**Analyseurs de Données**
   - `DataLoader` : Chargement et préprocessing des données CSV
   - `DataExplorer` : Exploration et analyse des datasets
   - `IngredientsAnalyzer` : Analyse spécialisée des ingrédients
   - `InteractionsAnalyzer` : Analyse des interactions utilisateur-recettes

**Interface Utilisateur**
   - `IngredientsClusteringPage` : Page d'analyse de clustering des ingrédients
   - `PopularityAnalysisPage` : Page d'analyse de popularité des recettes

**Système de Cache**
   - `CacheManager` : Gestionnaire centralisé du cache
   - `CacheableMixin` : Mixin pour l'intégration du cache dans les analyseurs

**Utilitaires**
   - `Logger` : Système de logging configuré
   - `download_data` : Module de téléchargement automatique des données S3

Cette architecture modulaire permet une séparation claire des responsabilités et facilite la maintenance et l'évolution du code.