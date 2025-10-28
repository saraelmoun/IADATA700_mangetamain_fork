# Tests pour IADATA700_mangetamain

## Structure des tests

# Tests pour IADATA700_mangetamain

## Vue d'ensemble

Suite de tests complète avec **115 tests** couvrant tous les modules principaux du projet avec **34% de couverture globale**.

## Structure des tests

```
tests/
├── __init__.py                      # Package initialization
├── test_app_extended.py             # Tests application Streamlit (9 tests)
├── test_cache_integration.py        # Tests intégration système cache (5 tests)
├── test_cache_manager.py            # Tests gestionnaire de cache (19 tests)
├── test_cacheable_mixin.py          # Tests mixin cache pour analyseurs (10 tests)
├── test_data_explorer.py            # Tests exploration de données (15 tests)
├── test_data_loader.py              # Tests chargement de données (18 tests)
├── test_interactions_analyzer.py    # Tests analyseur d'interactions (18 tests)
├── test_logger.py                   # Tests système de logging (11 tests)
├── test_popularity_analysis_page.py # Tests page analyse popularité (10 tests)
└── README.md                        # Cette documentation
```

## Lancer les tests

### Tous les tests
```bash
# Exécution complète
uv run python -m pytest tests/ -v

# Avec couverture de code
uv run python -m pytest --cov=src --cov-report=html --cov-report=term-missing tests/
```

### Tests spécifiques
```bash
# Un fichier particulier
uv run pytest tests/test_cache_manager.py -v

# Un test spécifique
uv run pytest tests/test_cache_manager.py::TestCacheManager::test_set_and_get_simple_data -v
```

---

## Détail des tests par fichier

### test_app_extended.py - Application Streamlit (9 tests)

**Fixtures :** `temp_csv_file`, `mock_streamlit` (mock complet Streamlit)

| Test | Description |
|------|-------------|
| `test_app_initialization_with_custom_config` | Vérifie l'initialisation de l'app avec une configuration personnalisée |
| `test_app_initialization_default_config` | Vérifie l'initialisation avec la configuration par défaut |
| `test_sidebar_home_page_configuration` | Teste la configuration de la sidebar pour la page d'accueil |
| `test_sidebar_interactions_dataset` | Teste la sélection du dataset interactions dans la sidebar |
| `test_run_clustering_page` | Vérifie l'exécution complète de la page de clustering d'ingrédients |
| `test_run_popularity_page` | Vérifie l'exécution complète de la page d'analyse de popularité |
| `test_render_home_page_success_basic` | Teste le rendu réussi de la page d'accueil avec DataLoader et DataExplorer |
| `test_render_home_page_unexpected_error` | Vérifie la gestion des erreurs inattendues lors du rendu |
| `test_main_function_basic` | Teste que la fonction main() créé et exécute l'application correctement |

### test_cache_manager.py - Gestionnaire de cache (19 tests)

**Setup :** `setup_method` (répertoire temporaire), `teardown_method` (nettoyage)

| Test | Description |
|------|-------------|
| `test_cache_manager_initialization` | Vérifie l'initialisation du CacheManager avec répertoire et logger |
| `test_generate_key` | Teste la génération de clés MD5 déterministes pour les paramètres |
| `test_get_cache_path` | Vérifie la création de chemins hiérarchiques pour le stockage |
| `test_set_and_get_simple_data` | Teste le cycle complet sauvegarde/récupération de données simples |
| `test_set_and_get_dataframe` | Vérifie la sérialisation et désérialisation de DataFrames pandas |
| `test_get_cache_miss` | Teste le comportement lors d'un cache miss (clé inexistante) |
| `test_get_with_different_params` | Vérifie l'isolation du cache entre différents paramètres |
| `test_clear_all_cache` | Teste la suppression complète de tous les fichiers de cache |
| `test_clear_analyzer_cache` | Vérifie la suppression sélective du cache d'un analyseur spécifique |
| `test_clear_operation_cache` | Teste la suppression du cache d'une opération particulière |
| `test_get_info_empty_cache` | Vérifie les informations retournées pour un cache vide |
| `test_get_info_with_data` | Teste les statistiques du cache avec des données présentes |
| `test_cache_metadata` | Vérifie la structure des métadonnées stockées avec les données |
| `test_error_handling_corrupted_cache` | Teste la gestion d'un fichier de cache corrompu |
| `test_error_handling_set_failure` | Vérifie la gestion des erreurs lors de l'écriture dans le cache |
| `test_large_data_caching` | Teste les performances avec de gros volumes de données |
| `test_get_cache_manager_singleton` | Vérifie que get_cache_manager retourne toujours la même instance |
| `test_get_cache_manager_initialization` | Teste l'initialisation de l'instance globale du cache manager |
| `test_global_cache_manager_persistence` | Vérifie la persistance des données entre différents accès à l'instance globale |

### test_cacheable_mixin.py - Mixin de cache (10 tests)

**Setup :** `setup_method` (répertoire temporaire), `teardown_method` (nettoyage)

| Test | Description |
|------|-------------|
| `test_cacheable_mixin_basic_usage` | Vérifie l'utilisation basique du mixin avec cache hit et miss |
| `test_enable_cache_initialization` | Teste l'activation et désactivation du cache via enable_cache() |
| `test_cache_disabled` | Vérifie que les opérations recalculent toujours quand le cache est désactivé |
| `test_cache_with_complex_data` | Teste la sérialisation de structures de données complexes (dict, listes) |
| `test_cache_error_handling` | Vérifie la propagation correcte des exceptions dans les opérations cachées |
| `test_cache_with_different_analyzer_names` | Teste l'isolation du cache entre différents analyseurs |
| `test_cache_params_hashing` | Vérifie la génération de clés uniques selon les paramètres |
| `test_abstract_method_requirement` | Teste le fonctionnement avec l'implémentation par défaut |
| `test_cache_manager_integration` | Vérifie l'intégration complète avec le CacheManager |
| `test_cache_with_mutable_params` | Teste le cache avec des paramètres mutables (listes) |

### test_cache_integration.py - Intégration cache (5 tests)

**Setup :** `setup_method` (répertoire temporaire), `teardown_method` (nettoyage)

| Test | Description |
|------|-------------|
| `test_full_cache_workflow` | Teste le workflow complet CacheManager + CacheableMixin avec analyseur mock |
| `test_cache_persistence_across_instances` | Vérifie la persistance du cache entre différentes instances d'analyseurs |
| `test_error_handling_integration` | Teste la gestion d'erreur intégrée dans le système de cache |
| `test_logging_integration` | Vérifie l'intégration entre le système de cache et de logging |
| `test_global_cache_manager_integration` | Teste l'utilisation de l'instance globale du cache manager |

### test_logger.py - Système de logging (11 tests)

**Setup :** Tests directs sans setup particulier

| Test | Description |
|------|-------------|
| `test_get_logger_default` | Vérifie la récupération du logger par défaut avec le nom "mangetamain" |
| `test_get_logger_custom_name` | Teste la création d'un logger avec un nom personnalisé |
| `test_get_logger_singleton` | Vérifie que get_logger() retourne toujours la même instance |
| `test_mangetamain_logger_initialization` | Teste l'initialisation complète de MangetamainLogger avec fichiers |
| `test_logger_levels` | Vérifie le fonctionnement de tous les niveaux (debug, info, warning, error) |
| `test_logger_with_exception` | Teste le logging des exceptions avec stack trace |
| `test_setup_logging` | Vérifie la fonction de configuration globale du logging |
| `test_logger_file_creation` | Teste la création automatique des répertoires pour les fichiers de log |
| `test_logger_handlers_not_duplicated` | Vérifie qu'il n'y a pas de duplication des handlers lors de créations multiples |
| `test_logger_with_kwargs` | Teste le passage d'arguments supplémentaires aux méthodes de logging |
| `test_logger_level_filtering` | Vérifie le filtrage des messages selon le niveau configuré |

### test_data_loader.py - Chargement de données (18 tests)

**Fixtures :** `sample_csv_data`, `temp_csv_file`, `temp_parquet_file`

| Test | Description |
|------|-------------|
| `test_initialization_with_string_path` | Vérifie l'initialisation du DataLoader avec un chemin en string |
| `test_initialization_with_path_object` | Teste l'initialisation avec un objet Path |
| `test_initialization_with_cache_disabled` | Vérifie l'initialisation avec le cache désactivé |
| `test_load_csv_file` | Teste le chargement d'un fichier CSV avec preprocessing des colonnes |
| `test_load_parquet_file` | Vérifie le chargement d'un fichier Parquet |
| `test_file_not_found_error` | Teste la gestion d'erreur pour un fichier inexistant |
| `test_unsupported_file_format_error` | Vérifie l'erreur pour un format de fichier non supporté |
| `test_cache_behavior_default` | Teste que les données sont mises en cache par défaut |
| `test_force_reload` | Vérifie le rechargement forcé qui bypasse le cache |
| `test_get_data_loads_if_needed` | Teste le chargement lazy via get_data() |
| `test_get_data_returns_cached` | Vérifie que get_data() retourne les données mises en cache |
| `test_column_preprocessing` | Teste la normalisation des noms de colonnes (lowercase, spaces) |
| `test_preprocess_method_direct` | Vérifie la méthode preprocess() appelée directement |
| `test_preprocess_preserves_data` | Teste que le preprocessing préserve les valeurs des données |
| `test_empty_csv_file` | Vérifie la gestion d'un fichier CSV vide |
| `test_csv_with_special_characters` | Teste le chargement de CSV avec caractères spéciaux et encodage UTF-8 |
| `test_path_conversion` | Vérifie la conversion automatique des strings en objets Path |
| `test_full_workflow` | Teste le workflow complet d'initialisation à l'accès aux données |

### test_data_explorer.py - Exploration de données (15 tests)

**Fixtures :** `sample_data`, `temp_csv_file`, `sample_loader`

| Test | Description |
|------|-------------|
| `test_initialization_with_dataframe` | Vérifie l'initialisation du DataExplorer avec un DataFrame existant |
| `test_initialization_with_loader` | Teste l'initialisation avec un DataLoader pour chargement lazy |
| `test_initialization_with_both_df_and_loader` | Vérifie l'initialisation avec DataFrame ET DataLoader |
| `test_initialization_without_arguments` | Teste que l'initialisation échoue sans DataFrame ni DataLoader |
| `test_initialization_with_none_arguments` | Vérifie l'échec avec des arguments explicitement None |
| `test_df_property_with_existing_dataframe` | Teste l'accès à la propriété df avec DataFrame déjà chargé |
| `test_df_property_with_loader_lazy_loading` | Vérifie le chargement lazy via la propriété df |
| `test_df_property_without_loader_raises_error` | Teste l'erreur quand pas de données ni de loader |
| `test_reload_with_loader` | Vérifie la fonction reload() avec un DataLoader configuré |
| `test_reload_without_force` | Teste reload() sans le paramètre force |
| `test_reload_without_loader_raises_error` | Vérifie l'erreur de reload() sans DataLoader |
| `test_integration_with_data_loader` | Teste l'intégration complète DataExplorer + DataLoader |
| `test_data_consistency_across_operations` | Vérifie la cohérence des données entre différents accès |
| `test_empty_dataframe` | Teste le comportement avec un DataFrame vide |
| `test_dataframe_with_missing_values` | Vérifie la gestion des valeurs manquantes (NaN) |

### test_interactions_analyzer.py - Analyseur d'interactions (18 tests)

**Fixtures :** `sample_interactions_data`, `sample_recipes_data`, `sample_recipes_with_missing`, `analyzer_basic`, `analyzer_with_preprocessing`

| Test | Description |
|------|-------------|
| `test_initialization_basic` | Vérifie l'initialisation de base de l'InteractionsAnalyzer |
| `test_data_merging` | Teste la fusion correcte des données interactions et recettes |
| `test_aggregation_basic` | Vérifie les calculs d'agrégation (count, moyenne des ratings) |
| `test_aggregation_sorting` | Teste le tri des résultats par nombre d'interactions |
| `test_preprocessing_disabled` | Vérifie que le preprocessing peut être désactivé |
| `test_preprocessing_enabled` | Teste l'application du preprocessing quand activé |
| `test_outlier_removal_iqr` | Vérifie la suppression d'outliers par méthode IQR |
| `test_missing_values_preservation` | Teste la préservation des valeurs manquantes (pas d'imputation) |
| `test_popularity_segmentation` | Vérifie la segmentation des recettes par popularité |
| `test_recipe_categorization` | Teste la catégorisation des recettes (complexité, durée) |
| `test_efficiency_score_calculation` | Vérifie le calcul du score d'efficacité (rating/time) |
| `test_category_insights` | Teste la génération d'insights par catégorie |
| `test_cache_disabled` | Vérifie que le cache peut être désactivé |
| `test_missing_recipe_id_column` | Teste l'erreur quand la colonne recipe_id est manquante |
| `test_empty_dataframes` | Vérifie la gestion de DataFrames vides |
| `test_invalid_preprocessing_config` | Teste la gestion d'une configuration de preprocessing invalide |
| `test_full_pipeline_integration` | Vérifie le pipeline complet de données brutes aux insights |
| `test_data_consistency_across_operations` | Teste la cohérence des données entre opérations répétées |

### test_popularity_analysis_page.py - Page analyse popularité (10 tests)

**Fixtures :** `sample_interactions_data`, `sample_recipes_data`, `temp_csv_files`, `page_with_temp_files`

| Test | Description |
|------|-------------|
| `test_config_initialization` | Vérifie l'initialisation de PopularityAnalysisConfig avec paths |
| `test_config_with_string_paths` | Teste la configuration avec des chemins en string |
| `test_config_equality` | Vérifie la comparaison d'égalité entre configurations |
| `test_page_initialization_with_string_paths` | Teste l'initialisation de la page avec chemins string |
| `test_page_initialization_with_path_objects` | Vérifie l'initialisation avec objets Path |
| `test_config_access` | Teste l'accès aux propriétés de configuration |
| `test_data_structure_validation` | Vérifie la validation de la structure attendue des données |
| `test_data_content_validation` | Teste la validation du contenu des données (types, valeurs) |
| `test_configuration_consistency` | Vérifie la cohérence de configuration entre instances |
| `test_invalid_path_handling` | Teste la gestion des chemins invalides (None) |

---

## Couverture de code

### Résumé par module
| Module | Couverture | Lignes couvertes | Lignes manquées |
|--------|------------|------------------|-----------------|
| `data_loader.py` | **100%** | 44/44 | 0 |
| `data_explorer.py` | **95%** | 41/43 | 2 |
| `cacheable_mixin.py` | **91%** | 30/33 | 3 |
| `cache_manager.py` | **90%** | 95/105 | 10 |
| `logger.py` | **89%** | 55/62 | 7 |
| `app.py` | **77%** | 101/132 | 31 |
| `interactions_analyzer.py` | **70%** | 186/267 | 81 |
| `ingredients_analyzer.py` | **13%** | 32/244 | 212 |
| `popularity_analysis_page.py` | **5%** | 36/714 | 678 |
| `ingredients_clustering_page.py` | **9%** | 23/253 | 230 |

### Statistiques globales
- **Total des tests :** 115 tests
- **Temps d'exécution :** ~6.7 secondes
- **Taux de réussite :** 100%
- **Couverture globale :** 34% (643/1899 lignes)

### Commandes utiles
```bash
# Couverture avec rapport HTML
uv run python -m pytest --cov=src --cov-report=html tests/

# Tests les plus lents
uv run python -m pytest --durations=10 tests/

# Tests en parallèle (si pytest-xdist installé)
uv run python -m pytest -n auto tests/
```

## Lancer les tests

### Méthode recommandée
```bash
python run_tests.py
```

### Méthode directe avec pytest
```bash
uv run pytest tests/ -v
```

### Tests spécifiques
```bash
# Tester uniquement InteractionsAnalyzer
uv run pytest tests/test_interactions_analyzer.py -v

# Tester une méthode spécifique
uv run pytest tests/test_interactions_analyzer.py::TestInteractionsAnalyzer::test_aggregation_basic -v
```

## Couverture de tests pour InteractionsAnalyzer

### ✅ Fonctionnalités de base (5 tests)
- `test_initialization_basic` - Initialisation de base
- `test_data_merging` - Fusion des données interactions/recettes  
- `test_aggregation_basic` - Calculs d'agrégation
- `test_aggregation_sorting` - Tri des résultats
- `test_preprocessing_disabled` - Preprocessing désactivé

### ✅ Pipeline de preprocessing (4 tests)
- `test_preprocessing_enabled` - Preprocessing activé
- `test_outlier_removal_iqr` - Suppression d'outliers IQR
- `test_missing_values_preservation` - Préservation des valeurs manquantes (plus d'imputation KNN)

### ✅ Feature engineering (4 tests)
- `test_popularity_segmentation` - Segmentation par popularité
- `test_recipe_categorization` - Catégorisation des recettes
- `test_efficiency_score_calculation` - Calcul du score d'efficacité
- `test_category_insights` - Génération d'insights

### ✅ Système de cache (3 tests)
- `test_cache_disabled` - Cache désactivé
- `test_cache_info_structure` - Structure des infos cache
- `test_cache_key_generation` - Génération des clés de cache

### ✅ Gestion d'erreurs (3 tests)
- `test_missing_recipe_id_column` - Colonne manquante
- `test_empty_dataframes` - DataFrames vides
- `test_invalid_preprocessing_config` - Configuration invalide

### ✅ Tests d'intégration (2 tests)
- `test_full_pipeline_integration` - Pipeline complet
- `test_data_consistency_across_operations` - Cohérence des données

## Résultats des tests

```
==================================================== test session starts ====================================================
collected 20 items                                                                                                          

TestInteractionsAnalyzer::test_initialization_basic PASSED                       [  5%]
TestInteractionsAnalyzer::test_data_merging PASSED                               [ 10%]
TestInteractionsAnalyzer::test_aggregation_basic PASSED                          [ 15%]
TestInteractionsAnalyzer::test_aggregation_sorting PASSED                        [ 20%]
TestInteractionsAnalyzer::test_preprocessing_disabled PASSED                     [ 25%]
TestInteractionsAnalyzer::test_preprocessing_enabled PASSED                      [ 30%]
TestInteractionsAnalyzer::test_outlier_removal_iqr PASSED                        [ 35%]
TestInteractionsAnalyzer::test_missing_values_preservation PASSED                [ 40%]
TestInteractionsAnalyzer::test_popularity_segmentation PASSED                    [ 45%]
TestInteractionsAnalyzer::test_recipe_categorization PASSED                      [ 50%]
TestInteractionsAnalyzer::test_efficiency_score_calculation PASSED               [ 55%]
TestInteractionsAnalyzer::test_category_insights PASSED                          [ 60%]
TestInteractionsAnalyzer::test_cache_disabled PASSED                             [ 65%]
TestInteractionsAnalyzer::test_cache_info_structure PASSED                       [ 70%]
TestInteractionsAnalyzer::test_cache_key_generation PASSED                       [ 75%]
TestInteractionsAnalyzer::test_missing_recipe_id_column PASSED                   [ 80%]
TestInteractionsAnalyzer::test_empty_dataframes PASSED                           [ 85%]
TestInteractionsAnalyzer::test_invalid_preprocessing_config PASSED               [ 90%]
TestInteractionsAnalyzer::test_full_pipeline_integration PASSED                  [ 95%]
TestInteractionsAnalyzer::test_data_consistency_across_operations PASSED         [100%]

==================================================== 20 passed in 1.84s =====================================================
```

## Prochaines étapes

- [ ] Tests pour `DataLoader`
- [ ] Tests pour les composants Streamlit
- [ ] Tests d'intégration pour l'application complète
- [ ] Tests de performance
- [ ] Tests avec couverture de code