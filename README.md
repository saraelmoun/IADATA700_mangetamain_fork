# IADATA700_mangetamain

Dans le cadre d'un enseignement à Telecom Paris, ce projet consiste en une application web interactive d'analyse de données pour une entreprise fictive : **Mangetamain** ; leader dans la recommandation B2C de recettes de cuisine à l'ancienne bio.

## 🚀 Application Streamlit

### 📋 Pages disponibles
1. **🏠 Home** - Exploration générale des données de recettes
2. **🍳 Analyse de clustering des ingrédients** - Clustering basé sur la co-occurrence des ingrédients

### 🛠️ Lancement
```bash
uv sync
uv run streamlit run src/app.py
```

### 📂 Structure du projet
```
src/
├── app.py                          # Application principale Streamlit
├── core/                          # Modules de base
│   ├── data_loader.py            # Chargement des données
│   ├── data_explorer.py          # Exploration de base
│   └── ingredients_analyzer.py   # Analyse des ingrédients
├── components/                   # Composants de l'application
│   └── ingredients_clustering_page.py  # Page de clustering des ingrédients
└── utils/                        # Utilitaires (vide actuellement)
```

### 📊 Données requises
Les chemins par défaut :
- **Recettes** : `data/RAW_recipes.csv`

> 💡 **Prérequis** : Le fichier de données doit être présent localement dans le dossier `data/` à la racine du projet.

### ✨ Fonctionnalités
- **Page Home** : Exploration générale des recettes avec métriques de base
- **Page Clustering** : 
  - Sélection du nombre d'ingrédients à analyser
  - Configuration du nombre de clusters K-means
  - Matrice de co-occurrence interactive
  - Visualisation t-SNE des clusters
  - Liste détaillée des ingrédients par cluster
