"""
Script principal d'entraînement.
Usage :
    python train.py --data data/dataset.csv --target resultat_binaire --model RandomForest
"""

import argparse
import os
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

# Ajoute le dossier src au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocessing_v2 import full_preprocessing_pipeline, encode_and_impute, build_feature_matrix, load_dataset, clean_dataset, feature_engineering
from eda           import run_full_eda
from model_training import full_training_pipeline

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Entraînement du modèle équestre")
    parser.add_argument("--data",   default="data/dataset.csv", help="Chemin vers le CSV")
    parser.add_argument("--target", default="resultat_binaire",  help="Colonne cible")
    parser.add_argument("--model",  default="RandomForest",
                        choices=["LogisticRegression", "DecisionTree", "RandomForest", "XGBoost", "LightGBM"],
                        help="Modèle à entraîner")
    parser.add_argument("--test-size",   type=float, default=0.2, help="Fraction test (défaut : 0.2)")
    parser.add_argument("--no-eda",      action="store_true",     help="Désactiver l'EDA")
    parser.add_argument("--no-cv",       action="store_true",     help="Désactiver la validation croisée")
    parser.add_argument("--random-state",type=int,   default=42,  help="Graine aléatoire")
    return parser.parse_args()


def main():
    args = parse_args()

    print("╔══════════════════════════════════════════════╗")
    print("║  PROJET IA — PRÉDICTION CONCOURS ÉQUESTRES  ║")
    print("╚══════════════════════════════════════════════╝\n")

    # ── 1. Chargement & EDA ─────────────────────────────
    df_raw = load_dataset(args.data)

    if not args.no_eda:
        print("\n[1/4] Analyse exploratoire …")
        run_full_eda(df_raw, target_col=args.target)

    # ── 2. Preprocessing ────────────────────────────────
    print("\n[2/4] Preprocessing …")
    df_clean  = clean_dataset(df_raw)
    df_feat   = feature_engineering(df_clean)
    df_enc, encoders = encode_and_impute(df_feat, fit=True)
    X, y, feature_cols = build_feature_matrix(df_enc, target=args.target)

    if y is None:
        print("❌ Colonne cible introuvable dans le dataset.")
        sys.exit(1)

    print(f"  Dataset final : {X.shape[0]} lignes × {X.shape[1]} features")
    print(f"  Distribution cible : {y.value_counts().to_dict()}")

    # ── 3. Split ────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )
    print(f"  Train : {len(X_train)} | Test : {len(X_test)}")

    # ── 4. Entraînement & Évaluation ────────────────────
    print(f"\n[3/4] Entraînement — {args.model} …")
    model = full_training_pipeline(
        X_train, y_train, X_test, y_test,
        feature_names=feature_cols,
        best_model_name=args.model,
        run_cv=not args.no_cv,
    )

    # ── 5. Sauvegarde des artefacts ──────────────────────
    print("\n[4/4] Sauvegarde des artefacts …")
    joblib.dump(encoders,     os.path.join(MODELS_DIR, "encoders.joblib"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_cols.joblib"))
    print(f"  ✅ Encoders       → models/encoders.joblib")
    print(f"  ✅ Feature cols   → models/feature_cols.joblib")
    print(f"  ✅ Modèle         → models/{args.model}.joblib")

    print("\n🏁 Entraînement terminé avec succès !")
    print("   Lancer l'interface : streamlit run app.py")


if __name__ == "__main__":
    main()
