#!/usr/bin/env python3
"""
Script de lancement pour l'application IADATA700_mangetamain.

Ce script s'assure que les données sont disponibles avant de lancer Streamlit.
"""

import os
import sys
import time
import subprocess
from pathlib import Path


def check_data_files(verbose=True):
    """Vérifie si tous les fichiers de données requis sont présents."""
    data_dir = Path("data")
    required_files = ["RAW_recipes.csv", "RAW_interactions.csv"]

    if not data_dir.exists():
        if verbose:
            print("❌ Dossier 'data' inexistant")
        return False

    missing_files = []
    for file_name in required_files:
        file_path = data_dir / file_name
        if not file_path.exists():
            missing_files.append(file_name)
            if verbose:
                print(f"❌ Fichier manquant: {file_name}")
        else:
            file_size = file_path.stat().st_size
            if file_size < 1000:  # Fichier trop petit (probablement corrompu)
                missing_files.append(file_name)
                if verbose:
                    print(f"⚠️  {file_name} trop petit ({file_size} bytes) - probablement corrompu")
            elif verbose:
                print(f"✅ {file_name} présent ({file_size:,} bytes)")

    return len(missing_files) == 0


def download_data():
    """Télécharge les données manquantes depuis S3."""
    print("🔄 Téléchargement des données en cours...")
    print("   ⏳ Cela peut prendre quelques minutes pour les gros fichiers...")

    try:
        from download_data import ensure_data_files

        ensure_data_files()

        # Attente pour la synchronisation du système de fichiers
        print("⏳ Attente de la synchronisation des fichiers (5s)...")
        time.sleep(5)

        print("✅ Téléchargement terminé!")
        return True
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement: {e}")
        return False


def wait_for_files_ready(max_attempts=3):
    """Attend que les fichiers soient complètement disponibles."""
    for attempt in range(max_attempts):
        if attempt > 0:
            print(f"🔄 Nouvelle vérification ({attempt + 1}/{max_attempts})...")
            time.sleep(3)

        if check_data_files(verbose=(attempt == max_attempts - 1)):
            return True

    return False


def launch_streamlit():
    """Lance l'application Streamlit."""
    print("🚀 Lancement de l'application Streamlit...")

    try:
        # Import des modules nécessaires
        import webbrowser
        import signal

        # Commande Streamlit
        cmd = ["uv", "run", "streamlit", "run", "src/app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]

        print("📋 Commande:", " ".join(cmd))
        print("🌐 L'application sera accessible sur: http://localhost:8501")
        print("⚠️  Pour arrêter l'application: Ctrl+C")
        print("=" * 50)

        # Gestion propre des signaux
        def signal_handler(sig, frame):
            print("\n🛑 Arrêt de l'application...")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        # Lancement direct avec subprocess.run() mais interruptible
        try:
            subprocess.run(cmd, check=True)
        except KeyboardInterrupt:
            print("\n� Application arrêtée par l'utilisateur")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Erreur lors du lancement de Streamlit: {e}")
            return False

        return True

    except FileNotFoundError:
        print("❌ Erreur: 'uv' non trouvé. Assurez-vous qu'uv est installé.")
        return False
    except Exception as e:
        print(f"❌ Erreur lors du lancement: {e}")
        return False


def main():
    """Point d'entrée principal."""
    print("🍳 Mangetamain - Démarrage de l'application")
    print("=" * 50)

    # Vérification initiale des données
    print("📂 Vérification des fichiers de données...")
    if check_data_files():
        print("✅ Tous les fichiers de données sont présents")
    else:
        print("⬇️  Téléchargement des données manquantes...")
        if not download_data():
            print("❌ Impossible de télécharger les données. Arrêt.")
            return 1

        # Vérification avec retry et attente
        print("🔍 Vérification finale des fichiers...")
        if not wait_for_files_ready():
            print("❌ Les fichiers ne sont toujours pas disponibles après téléchargement.")
            print("   Vérifiez votre connexion internet et l'espace disque disponible.")
            return 1

        print("✅ Tous les fichiers sont maintenant disponibles!")

    print("=" * 50)

    # Lancement de Streamlit
    if not launch_streamlit():
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
