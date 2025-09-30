"""Application Streamlit minimale autonome.

Lancer avec:
    streamlit run src/streamlit_app.py

Fonctionnalités:
- Chargement d'un CSV (chemin paramétrable)
- Détection/derivation du nombre d'ingrédients
- Histogramme + KDE
"""
from __future__ import annotations

import pathlib
from typing import Optional

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st


def load_csv(path: str | pathlib.Path, sep: str = ",", sample_size: Optional[int] = None) -> pd.DataFrame:
    path = pathlib.Path(path)
    if not path.exists():
        st.error(f"Fichier introuvable: {path}")
        st.stop()
    df = pd.read_csv(path, sep=sep)
    if sample_size and 0 < sample_size < len(df):
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)
    # Derive ingredient counts if needed
    if not any(c in df.columns for c in ["nb_ingredients", "n_ingredients", "ingredients_count"]):
        if "ingredients" in df.columns:
            def _count_ingr(x):
                if isinstance(x, (list, tuple, set)):
                    return len(x)
                if isinstance(x, str):
                    x = x.strip()
                    if not x:
                        return 0
                    return len([p for p in x.split(',') if p.strip()])
                return 0
            df["n_ingredients"] = df["ingredients"].apply(_count_ingr)
    return df


def main() -> None:
    st.title("POC Recettes - Distribution nombre d'ingrédients")
    st.sidebar.header("Paramètres")
    default_path = "data/raw/RAW_recipes.csv"
    csv_path = st.sidebar.text_input("Chemin CSV", value=default_path)
    sep = st.sidebar.text_input("Séparateur", value=",")
    sample = st.sidebar.number_input("Échantillon (0 = complet)", min_value=0, value=0, step=500)

    df = load_csv(csv_path, sep=sep, sample_size=sample or None)
    st.write(f"Données chargées: {len(df):,} lignes")

    candidate_cols = [c for c in ["nb_ingredients", "n_ingredients", "ingredients_count"] if c in df.columns]
    if not candidate_cols:
        st.warning("Aucune colonne explicite trouvée, vérifie 'ingredients'.")
        if "n_ingredients" in df.columns:
            candidate_cols = ["n_ingredients"]
        else:
            st.stop()

    chosen = st.selectbox("Colonne du nombre d'ingrédients", candidate_cols)
    serie = pd.to_numeric(df[chosen], errors="coerce").dropna().astype(int)

    fig, ax = plt.subplots(figsize=(7,4))
    ax.hist(serie, bins=20, color='skyblue', edgecolor='black')
    ax.set_xlabel("Nombre d'ingrédients")
    ax.set_ylabel("Nombre de recettes")
    ax.set_title(f"Distribution ({chosen})")
    st.pyplot(fig)

    st.subheader("Statistiques")
    stats = serie.describe(percentiles=[0.1,0.25,0.5,0.75,0.9]).to_frame(name="valeur")
    st.dataframe(stats)

    st.caption("POC minimal Streamlit autonome.")


if __name__ == "__main__":  # pragma: no cover
    main()
