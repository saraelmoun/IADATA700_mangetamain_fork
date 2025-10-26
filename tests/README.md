# Tests pour IADATA700_mangetamain

## Structure des tests

```
tests/
├── __init__.py                      # Package initialization
├── test_interactions_analyzer.py   # Tests pour InteractionsAnalyzer
└── README.md                       # Cette documentation
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