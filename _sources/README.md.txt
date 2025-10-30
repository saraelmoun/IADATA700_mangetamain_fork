# IADATA700_mangetamain
Dans le cadre d'un enseignement à Telecom Paris, ce projet consiste en une application web interactive d'analyse de données pour une entreprise fictive : Mangetamain ; leader dans la recommandation B2C de recettes de cuisine à l'ancienne bio.

## Application Streamlit

Version simplifiée et modulaire avec pages : Display, Analysis1 (recettes), Analysis2 (recettes + interactions).

### Lancer
```
uv sync
uv run streamlit run src/app.py
```

Les chemins par défaut :
- Recettes : `data/RAW_recipes.csv`
- Interactions : `data/RAW_interactions.csv`

> Prérequis données : pour le moment l'application suppose que ces deux fichiers existent localement dans un dossier `data/` à la racine. Aucun téléchargement automatique n'est encore implémenté.


### Fonctionnalités actuelles
- Page Display : aperçu (10 premières lignes) du dataset sélectionné (recettes ou interactions)
- Page Analysis1 : aperçu + métriques ingrédients (diversité, moyenne par recette)
- Page Analysis2 : aperçu des deux jeux + scatter "popularité vs note moyenne" (interactions)
- Chargement paresseux des données via `DataLoader`
- Architecture extensible via explorateurs spécialisés

### Diagramme de classes (PlantUML)
Le diagramme suivant décrit l'architecture principale (base + explorateurs + application) :



![Schéma](ClassDiagram.png)


Dans VS Code (extension PlantUML) vous pouvez simplement ouvrir le fichier et utiliser "Preview Current Diagram".

#### Évolution possible
- Ajouter d'autres pages d'analyse (ex: qualité nutritionnelle, temporalité)
- Extraire un registre d'analyses plug-and-play
- Remplacer `seaborn` par matplotlib pur pour alléger les dépendances

---
_Ce README reflète l'état simplifié actuel après nettoyage des fonctionnalités inutilisées._

