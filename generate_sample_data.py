"""
Génère un dataset synthétique réaliste pour tester le pipeline
sans avoir le dataset réel FFE/IFCE.

Usage :
    python generate_sample_data.py --rows 5000 --output data/dataset.csv
"""

import argparse
import os
import numpy as np
import pandas as pd

RACES  = ["Selle Français", "KWPN", "Anglo-Arabe", "Lusitanien", "Hanovrien", "Autre"]
SEXES  = ["Hongre", "Jument", "Étalon"]
DISCS  = ["Saut d'obstacles", "Dressage", "Cross", "Équifun", "Hunter"]
NIVEAUX= ["Club", "Amateur", "Régional", "National", "International"]
CATS   = ["Amateur", "Pro", "Jeune Cavalier", "Poney"]
CLUBS  = [f"Club_{i}" for i in range(1, 60)]

np.random.seed(42)


def generate_dataset(n: int = 5000) -> pd.DataFrame:
    rows = []
    for i in range(n):
        cheval_id  = np.random.randint(1, int(n * 0.4))
        cavalier_id = np.random.randint(1, int(n * 0.3))
        age_cheval  = np.random.randint(3, 22)
        sexe_cheval = np.random.choice(SEXES, p=[0.45, 0.40, 0.15])
        race_cheval = np.random.choice(RACES, p=[0.30, 0.20, 0.15, 0.10, 0.15, 0.10])
        discipline  = np.random.choice(DISCS,  p=[0.40, 0.25, 0.15, 0.10, 0.10])
        niveau      = np.random.choice(NIVEAUX, p=[0.35, 0.30, 0.20, 0.10, 0.05])
        age_cavalier = np.random.randint(10, 65)
        categorie_cavalier = np.random.choice(CATS, p=[0.50, 0.20, 0.20, 0.10])
        club        = np.random.choice(CLUBS)
        nb_participants = np.random.randint(5, 100)

        # Historique simulé
        horse_participations = np.random.randint(0, 200)
        rider_participations  = np.random.randint(0, 300)

        # Taux de réussite de base
        horse_win_rate = np.clip(np.random.beta(2, 3), 0.05, 0.95)
        rider_win_rate = np.clip(np.random.beta(2, 3), 0.05, 0.95)

        # Score composite (feature latente pour simuler la réalité)
        age_effect  = 1.0 - abs(age_cheval - 10) / 15       # pic ~10 ans
        exp_effect  = min(horse_participations / 100, 1.0)
        niveau_map  = {"Club": 1, "Amateur": 2, "Régional": 3, "National": 4, "International": 5}
        niv_factor  = 1 / niveau_map[niveau]
        score = (
            0.30 * horse_win_rate
            + 0.25 * rider_win_rate
            + 0.15 * age_effect
            + 0.15 * exp_effect
            + 0.15 * niv_factor
            + np.random.normal(0, 0.08)
        )
        resultat_binaire = int(score > np.random.uniform(0.38, 0.55))
        classement = max(1, int(nb_participants * (1 - score) + np.random.randint(-3, 4)))
        classement = min(classement, nb_participants)

        rows.append({
            "cheval_id"          : cheval_id,
            "cavalier_id"        : cavalier_id,
            "cheval_nom"         : f"Cheval_{cheval_id:04d}",
            "cavalier_nom"       : f"Cavalier_{cavalier_id:04d}",
            "age_cheval"         : age_cheval,
            "sexe_cheval"        : sexe_cheval,
            "race_cheval"        : race_cheval,
            "age_cavalier"       : age_cavalier,
            "categorie_cavalier" : categorie_cavalier,
            "club"               : club,
            "discipline"         : discipline,
            "niveau"             : niveau,
            "nombre_participants": nb_participants,
            "horse_participations": horse_participations,
            "rider_participations": rider_participations,
            "horse_win_rate"     : round(horse_win_rate, 3),
            "rider_win_rate"     : round(rider_win_rate, 3),
            "classement"         : classement,
            "resultat_binaire"   : resultat_binaire,
        })

    df = pd.DataFrame(rows)

    # Introduire ~5% de valeurs manquantes de façon aléatoire
    for col in ["age_cavalier", "horse_win_rate", "race_cheval"]:
        mask = np.random.rand(len(df)) < 0.05
        df.loc[mask, col] = np.nan

    return df


def main():
    parser = argparse.ArgumentParser(description="Générateur de dataset synthétique équestre")
    parser.add_argument("--rows",   type=int, default=5000, help="Nombre de lignes")
    parser.add_argument("--output", default="data/dataset.csv", help="Chemin de sortie")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df = generate_dataset(args.rows)
    df.to_csv(args.output, index=False)

    print(f"✅ Dataset synthétique généré : {args.output}")
    print(f"   Lignes : {len(df)} | Colonnes : {len(df.columns)}")
    print(f"   Taux réussite : {df['resultat_binaire'].mean()*100:.1f}%")
    print(f"\n   Premières lignes :")
    print(df.head(3).to_string())


if __name__ == "__main__":
    main()
