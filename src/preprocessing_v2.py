"""
Module de préparation et nettoyage des données équestres.
Compatible avec le dataset v2 (cavaliers nommés, disciplines variées).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 1. CHARGEMENT
# ──────────────────────────────────────────────

def load_dataset(path: str) -> pd.DataFrame:
    logger.info(f"Chargement du dataset : {path}")
    df = pd.read_csv(path, sep=None, engine="python")
    logger.info(f"Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


# ──────────────────────────────────────────────
# 2. NETTOYAGE
# ──────────────────────────────────────────────

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^a-z0-9_]", "", regex=True)
    )
    initial = len(df)
    df.drop_duplicates(inplace=True)
    logger.info(f"Doublons supprimés : {initial - len(df)}")

    if "classement" in df.columns:
        df = df[df["classement"] >= 0]
    if "hauteur_cm" in df.columns:
        df = df[df["hauteur_cm"] >= 0]

    logger.info(f"Dataset après nettoyage : {df.shape}")
    return df


# ──────────────────────────────────────────────
# 3. FEATURE ENGINEERING (si pas déjà fait par adapt_dataset_v2.py)
# ──────────────────────────────────────────────

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Crée les features combinées si absentes (dataset non encore adapté)."""
    df = df.copy()

    # Synergies déjà calculées par adapt_dataset_v2 → on ne recalcule que si absent
    if "couple_synergy" not in df.columns:
        if "horse_win_rate" in df.columns and "rider_win_rate" in df.columns:
            df["couple_synergy"] = df["horse_win_rate"] * df["rider_win_rate"]

    if "competition_density" not in df.columns and "nombre_participants" in df.columns:
        df["competition_density"] = pd.cut(
            df["nombre_participants"],
            bins=[0, 10, 25, 50, 999],
            labels=["small", "medium", "large", "xlarge"],
        ).astype(str)

    if "niveau_num" not in df.columns and "niveau_epreuve" in df.columns:
        niveau_map = {
            "Moustique": 1, "Poussin": 1, "Benjamin": 2, "Minime": 2,
            "Cadet": 3, "Club": 3, "Niveau1": 3, "Niveau2": 4,
            "Niveau3": 4, "As": 4, "Regional": 5, "Coupe": 5,
            "National": 6, "Pro": 7, "Elite": 8,
        }
        df["niveau_num"] = df["niveau_epreuve"].map(niveau_map).fillna(3)

    if "experience_ratio" not in df.columns:
        if "horse_participations" in df.columns and "hauteur_cm" in df.columns:
            df["experience_ratio"] = df["horse_participations"] / (df["hauteur_cm"] / 10 + 1)

    logger.info(f"Feature engineering OK — shape : {df.shape}")
    return df


# ──────────────────────────────────────────────
# 4. COLONNES FEATURES & CIBLES
# ──────────────────────────────────────────────

CATEGORICAL_COLS = [
    # Cheval
    "sexe_cheval", "robe_cheval", "taille_cheval",
    # Épreuve
    "discipline_famille", "niveau_epreuve", "type_epreuve",
    # Ancien dataset
    "race_cheval", "discipline", "niveau", "categorie_cavalier",
    "club", "competition_density",
]

NUMERIC_COLS = [
    # Épreuve
    "hauteur_cm", "nombre_participants", "niveau_num", "is_equipe",
    # Historiques
    "horse_win_rate", "horse_participations",
    "rider_win_rate", "rider_participations",
    "club_win_rate",
    # Features combinées
    "couple_synergy", "experience_ratio",
    # Points
    "pts_qualification",
    # Ancien dataset
    "age_cheval", "age_cavalier",
]

TARGET_BINARY = "resultat_binaire"
TARGET_RANK   = "classement"


# ──────────────────────────────────────────────
# 5. ENCODAGE & IMPUTATION
# ──────────────────────────────────────────────

def encode_and_impute(df: pd.DataFrame, fit: bool = True, encoders: dict = None):
    df = df.copy()
    if encoders is None:
        encoders = {}

    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str).str.strip().str.lower()
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].fillna("inconnu"))
            encoders[col] = le
        else:
            le = encoders.get(col)
            if le:
                known   = set(le.classes_)
                fallback = le.classes_[0]   # ← jamais "inconnu" : toujours une classe connue
                df[col]  = df[col].apply(lambda x: x if x in known else fallback)
                df[col]  = le.transform(df[col])

    for col in NUMERIC_COLS:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if fit:
            imp = SimpleImputer(strategy="median")
            df[[col]] = imp.fit_transform(df[[col]])
            encoders[f"imp_{col}"] = imp
        else:
            imp = encoders.get(f"imp_{col}")
            if imp:
                df[[col]] = imp.transform(df[[col]])

    return df, encoders


# ──────────────────────────────────────────────
# 6. CONSTRUCTION MATRICE FEATURES / CIBLE
# ──────────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame, target: str = TARGET_BINARY):
    exclude = [
        "cheval_id", "cavalier_id", "club_id",
        "cheval_nom", "cavalier_nom", "club",
        "concours_id", "date",
        "discipline_brute", "epreuve_brute",
        "pere_cheval", "mere_cheval",
        TARGET_BINARY, TARGET_RANK,
    ]
    feature_cols = [c for c in df.columns if c not in exclude]
    target_col   = target if target in df.columns else TARGET_BINARY

    X = df[feature_cols].copy()
    y = df[target_col].copy() if target_col in df.columns else None

    logger.info(f"Features retenues ({len(feature_cols)}) : {feature_cols}")
    return X, y, feature_cols


# ──────────────────────────────────────────────
# 7. PIPELINE COMPLET
# ──────────────────────────────────────────────

def full_preprocessing_pipeline(path: str, target: str = TARGET_BINARY):
    df = load_dataset(path)
    df = clean_dataset(df)
    df = feature_engineering(df)
    df, encoders = encode_and_impute(df, fit=True)
    X, y, feature_cols = build_feature_matrix(df, target=target)
    return X, y, feature_cols, encoders


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/dataset.csv"
    if os.path.exists(path):
        X, y, cols, enc = full_preprocessing_pipeline(path)
        print(f"\n✅ Preprocessing OK")
        print(f"   X : {X.shape}")
        print(f"   Features : {cols}")
        print(f"   Distribution cible : {y.value_counts().to_dict()}")
    else:
        logger.warning(f"Fichier introuvable : {path}")