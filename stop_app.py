#!/usr/bin/env python3
"""
Script pour arrêter tous les processus Streamlit en cours.
"""

import subprocess
import sys


def stop_streamlit():
    """Arrête tous les processus Streamlit."""
    print("🛑 Recherche des processus Streamlit en cours...")

    try:
        # Recherche des processus streamlit
        result = subprocess.run(["pgrep", "-f", "streamlit"], capture_output=True, text=True)

        if result.returncode == 0:
            pids = result.stdout.strip().split("\n")
            pids = [pid for pid in pids if pid]  # Filtrer les lignes vides

            if pids:
                print(f"📋 Processus trouvés: {', '.join(pids)}")

                # Arrêt des processus
                for pid in pids:
                    try:
                        subprocess.run(["kill", pid], check=True)
                        print(f"✅ Processus {pid} arrêté")
                    except subprocess.CalledProcessError:
                        print(f"❌ Impossible d'arrêter le processus {pid}")

                print("🎉 Tous les processus Streamlit ont été arrêtés!")
            else:
                print("✅ Aucun processus Streamlit en cours")
        else:
            print("✅ Aucun processus Streamlit en cours")

    except FileNotFoundError:
        print("❌ Commande 'pgrep' non trouvée, utilisation alternative...")

        # Méthode alternative avec ps
        try:
            result = subprocess.run(["ps", "aux"], capture_output=True, text=True)

            lines = result.stdout.split("\n")
            streamlit_processes = []

            for line in lines:
                if "streamlit" in line and "grep" not in line:
                    parts = line.split()
                    if len(parts) > 1:
                        pid = parts[1]
                        streamlit_processes.append(pid)

            if streamlit_processes:
                print(f"📋 Processus trouvés: {', '.join(streamlit_processes)}")
                for pid in streamlit_processes:
                    try:
                        subprocess.run(["kill", pid], check=True)
                        print(f"✅ Processus {pid} arrêté")
                    except subprocess.CalledProcessError:
                        print(f"❌ Impossible d'arrêter le processus {pid}")
                print("🎉 Tous les processus Streamlit ont été arrêtés!")
            else:
                print("✅ Aucun processus Streamlit en cours")

        except Exception as e:
            print(f"❌ Erreur: {e}")
            return False

    return True


if __name__ == "__main__":
    stop_streamlit()
