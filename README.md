# IADATA700_mangetamain

Dans le cadre d'un enseignement à Telecom Paris, ce projet consiste en une application web interactive d'analyse de données pour une entreprise fictive : **Mangetamain** ; leader dans la recommandation B2C de recettes de cuisine à l'ancienne bio.

## 🚀 Application Streamlit

### 📋 Pages disponibles
1. **🏠 Home** - Exploration générale des données (recettes ou interactions)
2. **🍳 Analyse de clustering des ingrédients** - Clustering basé sur la co-occurrence
3. **🔥 Analyse popularité des recettes** - Popularité (nombre d'interactions) vs note moyenne & caractéristiques (minutes, n_steps, n_ingredients)

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
│   ├── data_explorer.py          # Exploration de base (accès aux données)
│   ├── interactions_analyzer.py  # Agrégations popularité / notes / features
│   └── ingredients_analyzer.py   # Analyse des ingrédients
├── components/                   # Composants de l'application
│   ├── ingredients_clustering_page.py     # Page clustering des ingrédients
│   └── popularity_analysis_page.py         # Page analyse popularité
└── utils/                        # Utilitaires (vide actuellement)
```

### 📊 Données requises
Chemins par défaut :
- **Recettes** : `data/RAW_recipes.csv`
- **Interactions** : `data/RAW_interactions.csv`

> 💡 **Prérequis** : Le fichier de données doit être présent localement dans le dossier `data/` à la racine du projet.

### ✨ Fonctionnalités
- **Page Home** : Exploration générale des données + métriques
- **Clustering Ingrédients** :
  - Sélection du nombre d'ingrédients à analyser
  - Regroupement normalisé + co-occurrences
  - Clustering K-means + t-SNE
  - Analyse de groupes & debug mappings
- **Popularité Recettes** :
  - Agrégat par recette : interaction_count, avg_rating, minutes, n_steps, n_ingredients
  - Scatter Note moyenne vs Popularité
  - Scatter Caractéristiques vs Popularité (taille = note)
  - Aperçu DataFrame fusionné (diagnostic)
  - Filtre sur interactions minimales

### 🧩 Diagramme UML
Un diagramme de classes PlantUML est disponible : `docs/class-diagram.puml`.

Pour le générer en PNG (nécessite PlantUML + Java) :
```bash
plantuml docs/class-diagram.puml
```
Ou via l'extension VS Code PlantUML.

