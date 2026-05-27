"""
Script d'adaptation du nouveau dataset FFE (avec cavaliers, disciplines variées).
À lancer une seule fois :
    python adapt_dataset_v2.py --input data/dataset_brut_v2.csv --output data/dataset.csv
"""

import argparse
import os
import re
import pandas as pd
import numpy as np


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def extract_hauteur(discipline: str) -> int:
    """Extrait la hauteur en cm depuis le nom de la discipline (ex: 'CSO 80 cm' → 80)."""
    m = re.search(r"(\d+)\s*cm", str(discipline), re.IGNORECASE)
    return int(m.group(1)) if m else 0


def extract_discipline_famille(discipline: str) -> str:
    """Extrait la famille de discipline (ex: 'CSO 80 cm' → 'CSO')."""
    d = str(discipline).strip()
    # Retire les mesures en cm et les chiffres isolés
    famille = re.sub(r"\d+\s*cm.*", "", d).strip()
    famille = re.sub(r"\s*\(\+\d+\).*", "", famille).strip()
    if not famille:
        return "Autre"
    return famille


def extract_niveau_epreuve(epreuve: str) -> str:
    """Extrait le niveau compétitif depuis le nom de l'épreuve."""
    e = str(epreuve).lower()
    if "elite"      in e: return "Elite"
    if "pro"        in e: return "Pro"
    if "national"   in e or "chp de france" in e or "championnat" in e: return "National"
    if "regional"   in e or "circuit reg"   in e: return "Regional"
    if "coupe"      in e: return "Coupe"
    if "poney 3"    in e or "club 3" in e: return "Niveau3"
    if "poney 2"    in e or "club 2" in e: return "Niveau2"
    if "poney 1"    in e or "club 1" in e: return "Niveau1"
    if "as poney"   in e: return "As"
    if "benjamin"   in e: return "Benjamin"
    if "moustique"  in e: return "Moustique"
    if "poussin"    in e: return "Poussin"
    if "minime"     in e: return "Minime"
    if "cadet"      in e: return "Cadet"
    return "Club"


def extract_type_epreuve(epreuve: str) -> str:
    """Grand Prix / Vitesse / Mania / Carrousel / Equipe / etc."""
    e = str(epreuve).lower()
    if "grand prix"  in e: return "GrandPrix"
    if "vitesse"     in e: return "Vitesse"
    if "mania"       in e: return "Mania"
    if "carrousel"   in e: return "Carrousel"
    if "tda"         in e: return "TDA"
    if "equipe"      in e or "mixte" in e or "feminine" in e: return "Equipe"
    if "trec"        in e: return "TREC"
    return "Autre"


def is_equipe(equipe: str) -> int:
    return 0 if str(equipe).strip().upper() == "INDIVIDUEL" else 1


# ──────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ──────────────────────────────────────────────

def adapt(input_path: str, output_path: str) -> None:
    print(f"📂 Chargement : {input_path}")
    df = pd.read_csv(input_path, sep=None, engine="python")
    print(f"   Shape brut : {df.shape}")

    out = pd.DataFrame()

    # ── Identifiants ──────────────────────────────────────────
    out["concours_id"]  = df["s2_Numéro"]
    out["cheval_nom"]   = df["s4_Équidé"].str.strip().str.upper()
    out["cavalier_nom"] = df["s4_Cavalier"].str.strip().str.upper()
    out["club"]         = df["s4_Club engageur"].str.strip()

    # IDs numériques stables
    out["cheval_id"]   = pd.factorize(out["cheval_nom"])[0]  + 1
    out["cavalier_id"] = pd.factorize(out["cavalier_nom"])[0] + 1
    out["club_id"]     = pd.factorize(out["club"])[0]         + 1

    # ── Caractéristiques du cheval ────────────────────────────
    out["sexe_cheval"]   = df["s5_Sexe"].str.strip()      # Femelle / Hongre / Mâle
    out["robe_cheval"]   = df["s5_Robe"].str.strip()       # Bai / Alezan / Noir…
    out["taille_cheval"] = df["s5_Taille"].str.strip()     # Catégorie A à F
    out["pere_cheval"]   = df["s5_Père"].str.strip()
    out["mere_cheval"]   = df["s5_Mère"].str.strip()

    # ── Épreuve ───────────────────────────────────────────────
    out["discipline_brute"] = df["s3_Discipline"].str.strip()
    out["epreuve_brute"]    = df["s3_Épreuve"].str.strip()

    out["discipline_famille"] = df["s3_Discipline"].apply(extract_discipline_famille)
    out["hauteur_cm"]         = df["s3_Discipline"].apply(extract_hauteur)
    out["niveau_epreuve"]     = df["s3_Épreuve"].apply(extract_niveau_epreuve)
    out["type_epreuve"]       = df["s3_Épreuve"].apply(extract_type_epreuve)
    out["is_equipe"]          = df["s4_Équipe"].apply(is_equipe)

    # Nombre de participants par épreuve (dans le concours)
    out["nombre_participants"] = df.groupby(
        ["s2_Numéro", "s3_Épreuve"]
    )["s4_Clt."].transform("count")

    # ── Résultats ─────────────────────────────────────────────
    out["classement"] = df["s4_Clt."]

    pts_raw = df["s4_Pts qualif. Chpt"].replace("NONE", "0")
    out["pts_qualification"] = pd.to_numeric(pts_raw, errors="coerce").fillna(0)

    # ── Cible binaire ─────────────────────────────────────────
    # Réussite = top 3 OU points de qualification >= 8
    # → ~30% de réussites = dataset raisonnablement équilibré
    out["resultat_binaire"] = (
        (out["classement"] <= 3) | (out["pts_qualification"] >= 8)
    ).astype(int)

    print(f"\n   Taux de réussite (cible) : {out['resultat_binaire'].mean()*100:.1f}%")

    # ── Features historiques (calculées par cheval) ───────────
    # Tri arbitraire par concours_id pour avoir un ordre reproductible
    out_sorted = out.sort_values("concours_id").copy()

    def rolling_winrate(series):
        """Taux de réussite glissant sans fuite de données (shift avant expanding)."""
        return series.shift(1).expanding().mean().fillna(0.3)

    # Historique cheval
    out_sorted["horse_win_rate"] = (
        out_sorted.groupby("cheval_id")["resultat_binaire"]
        .transform(rolling_winrate)
    )
    out_sorted["horse_participations"] = (
        out_sorted.groupby("cheval_id").cumcount()
    )

    # Historique cavalier
    out_sorted["rider_win_rate"] = (
        out_sorted.groupby("cavalier_id")["resultat_binaire"]
        .transform(rolling_winrate)
    )
    out_sorted["rider_participations"] = (
        out_sorted.groupby("cavalier_id").cumcount()
    )

    # Historique club
    out_sorted["club_win_rate"] = (
        out_sorted.groupby("club_id")["resultat_binaire"]
        .transform(rolling_winrate)
    )

    # Features combinées
    out_sorted["couple_synergy"]   = out_sorted["horse_win_rate"] * out_sorted["rider_win_rate"]
    out_sorted["experience_ratio"] = out_sorted["horse_participations"] / (out_sorted["hauteur_cm"] / 10 + 1)
    out_sorted["niveau_num"]       = out_sorted["niveau_epreuve"].map({
        "Moustique": 1, "Poussin": 1, "Benjamin": 2, "Minime": 2,
        "Cadet": 3, "Club": 3, "Niveau1": 3, "Niveau2": 4,
        "Niveau3": 4, "As": 4, "Regional": 5, "Coupe": 5,
        "National": 6, "Pro": 7, "Elite": 8,
    }).fillna(3)

    # ── Features d'interaction ────────────────────────────────
    # Permettent au modèle d'apprendre des nuances type "ce top
    # cheval performe-t-il MIEUX en Elite qu'en Club ?". Sans ces
    # croisements, win_rate domine et écrase toute différenciation
    # entre niveaux/hauteurs pour un même duo elite (saturation).
    out_sorted["horse_wr_x_niveau"]  = out_sorted["horse_win_rate"] * out_sorted["niveau_num"]
    out_sorted["horse_wr_x_hauteur"] = out_sorted["horse_win_rate"] * (out_sorted["hauteur_cm"] / 100)
    out_sorted["rider_wr_x_niveau"]  = out_sorted["rider_win_rate"] * out_sorted["niveau_num"]
    out_sorted["rider_wr_x_hauteur"] = out_sorted["rider_win_rate"] * (out_sorted["hauteur_cm"] / 100)
    out_sorted["synergy_x_niveau"]   = out_sorted["couple_synergy"] * out_sorted["niveau_num"]

    # ── Nettoyage final ───────────────────────────────────────
    out_sorted = out_sorted[out_sorted["classement"] >= 0]

    # Colonnes finales à conserver (dans l'ordre logique)
    cols_finales = [
        # Identifiants (exclus des features, utiles pour debug)
        "concours_id", "cheval_id", "cavalier_id", "club_id",
        "cheval_nom", "cavalier_nom", "club",
        # Features cheval
        "sexe_cheval", "robe_cheval", "taille_cheval",
        # Features épreuve
        "discipline_famille", "hauteur_cm", "niveau_epreuve",
        "type_epreuve", "is_equipe", "nombre_participants", "niveau_num",
        # Features historiques (calculées)
        "horse_win_rate", "horse_participations",
        "rider_win_rate", "rider_participations",
        "club_win_rate", "couple_synergy", "experience_ratio",
        # Features d'interaction (croisements win_rate × contexte)
        "horse_wr_x_niveau", "horse_wr_x_hauteur",
        "rider_wr_x_niveau", "rider_wr_x_hauteur",
        "synergy_x_niveau",
        # Points de qualification
        "pts_qualification",
        # Cibles
        "classement", "resultat_binaire",
    ]
    out_sorted = out_sorted[cols_finales]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out_sorted.to_csv(output_path, index=False)

    print(f"\n✅ Dataset adapté sauvegardé : {output_path}")
    print(f"   Lignes     : {len(out_sorted)}")
    print(f"   Colonnes   : {len(out_sorted.columns)}")
    print(f"   Cavaliers  : {out_sorted['cavalier_id'].nunique()}")
    print(f"   Chevaux    : {out_sorted['cheval_id'].nunique()}")
    print(f"   Disciplines: {out_sorted['discipline_famille'].nunique()}")
    print(f"   Réussite   : {out_sorted['resultat_binaire'].mean()*100:.1f}%")
    print(f"\n   Colonnes finales :")
    for c in out_sorted.columns:
        print(f"     - {c}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adapte le nouveau dataset FFE v2")
    parser.add_argument("--input",  default="data/dataset_brut_v2.csv")
    parser.add_argument("--output", default="data/dataset.csv")
    args = parser.parse_args()
    adapt(args.input, args.output)