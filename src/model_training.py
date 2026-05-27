"""
Module d'entraînement, évaluation et interprétabilité des modèles.
Supporte : Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM
"""

import os
import joblib
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model   import LogisticRegression
from sklearn.tree           import DecisionTreeClassifier
from sklearn.ensemble       import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics        import (
    classification_report, f1_score, roc_auc_score,
    precision_recall_curve, roc_curve, ConfusionMatrixDisplay,
    accuracy_score
)
from sklearn.pipeline       import Pipeline
from sklearn.preprocessing  import StandardScaler
from sklearn.calibration    import CalibratedClassifierCV

try:
    from xgboost  import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

warnings.filterwarnings("ignore")

MODELS_DIR  = os.path.join(os.path.dirname(__file__), "../models")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "../outputs/models")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# 1. DÉFINITION DES MODÈLES
# ──────────────────────────────────────────────

def get_models(random_state: int = 42) -> dict:
    """Retourne le catalogue des modèles à comparer."""
    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=random_state, class_weight="balanced")),
        ]),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=10, random_state=random_state, class_weight="balanced"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=5,
            random_state=random_state, n_jobs=-1, class_weight="balanced"
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=random_state, n_jobs=-1,
        )
    if HAS_LGB:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            random_state=random_state, n_jobs=-1, verbose=-1,
        )
    return models


# ──────────────────────────────────────────────
# 2. VALIDATION CROISÉE
# ──────────────────────────────────────────────

def cross_validate_models(
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compare tous les modèles en validation croisée stratifiée."""
    models = get_models(random_state)
    skf    = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scoring = ["f1", "precision", "recall", "roc_auc", "accuracy"]

    results = []
    for name, model in models.items():
        print(f"  ▸ {name} …", end=" ", flush=True)
        cv_res = cross_validate(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
        row = {"model": name}
        for metric in scoring:
            vals = cv_res[f"test_{metric}"]
            row[f"{metric}_mean"] = vals.mean().round(4)
            row[f"{metric}_std"]  = vals.std().round(4)
        results.append(row)
        print(f"F1={row['f1_mean']:.3f} ± {row['f1_std']:.3f}")

    df_results = pd.DataFrame(results).sort_values("f1_mean", ascending=False)
    df_results.to_csv(os.path.join(OUTPUTS_DIR, "cv_results.csv"), index=False)
    print(f"\n  Résultats CV sauvegardés → cv_results.csv")
    return df_results


# ──────────────────────────────────────────────
# 3. ENTRAÎNEMENT & SAUVEGARDE
# ──────────────────────────────────────────────

def train_best_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str = "RandomForest",
    random_state: int = 42,
    calibrate:   bool = False,
):
    """
    Entraîne le modèle choisi et le sauvegarde.

    Si `calibrate=True` (par défaut), on enveloppe l'estimateur dans un
    CalibratedClassifierCV (Platt scaling, 5-fold). Pourquoi :
      - LightGBM et autres boosters sortent des probas non-calibrées qui
        saturent à 100 % / 0 % pour les zones bien apprises (cf. un duo
        à horse_win_rate=0.9 ⇒ proba=100 % alors que rien n'est jamais
        certain en concours).
      - Platt scaling apprend une sigmoïde sur les sorties OOF, lisse
        les probas extrêmes vers des valeurs réalistes (95-98 %), et
        améliore la calibration sans dégrader la décision binaire.
    """
    models = get_models(random_state)
    if model_name not in models:
        raise ValueError(f"Modèle inconnu : {model_name}. Disponibles : {list(models)}")

    base = models[model_name]
    if calibrate:
        # method='sigmoid' (Platt scaling). Plus stable qu'isotonic qui
        # saturait à exactement 1.0 sur les top duos (bins purs). Sigmoid
        # plafonne plus doucement vers ~0.985.
        # cv=5 garantit que la sigmoïde apprend sur des sorties OOF.
        model = CalibratedClassifierCV(base, method = 'sigmoid', cv = 5)
        print(f"\n  ⚙ Entraînement de {model_name} (+ calibration Platt) sur {X_train.shape[0]} exemples …")
    else:
        model = base
        print(f"\n  ⚙ Entraînement de {model_name} (sans calibration) sur {X_train.shape[0]} exemples …")
    model.fit(X_train, y_train)

    path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
    joblib.dump(model, path)
    print(f"  ✅ Modèle sauvegardé → {path}")
    return model


def load_model(model_name: str):
    """Charge un modèle depuis le disque."""
    path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modèle introuvable : {path}")
    return joblib.load(path)


# ──────────────────────────────────────────────
# 4. ÉVALUATION COMPLÈTE
# ──────────────────────────────────────────────

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str = "model") -> dict:
    """Génère un rapport complet : métriques, courbes ROC/PR, matrice de confusion."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy" : accuracy_score(y_test, y_pred),
        "f1"       : f1_score(y_test, y_pred, average="binary"),
        "roc_auc"  : roc_auc_score(y_test, y_prob) if y_prob is not None else None,
    }
    print(f"\n══════ Évaluation — {model_name} ══════")
    print(classification_report(y_test, y_pred, target_names=["Échec", "Réussite"]))
    print(f"  AUC-ROC : {metrics['roc_auc']:.4f}" if metrics["roc_auc"] else "")

    # Matrice de confusion
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["Échec", "Réussite"],
        ax=axes[0], colorbar=False, cmap="Blues"
    )
    axes[0].set_title("Matrice de confusion")

    # Courbe ROC
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        axes[1].plot(fpr, tpr, color="#2ecc71", lw=2, label=f"AUC={metrics['roc_auc']:.3f}")
        axes[1].plot([0, 1], [0, 1], "k--", lw=0.8)
        axes[1].set_xlabel("Taux faux positifs")
        axes[1].set_ylabel("Taux vrais positifs")
        axes[1].set_title("Courbe ROC")
        axes[1].legend()

        # Courbe Précision-Rappel
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        axes[2].plot(rec, prec, color="#e67e22", lw=2)
        axes[2].set_xlabel("Rappel")
        axes[2].set_ylabel("Précision")
        axes[2].set_title("Courbe Précision-Rappel")

    plt.suptitle(f"Évaluation — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fname = os.path.join(OUTPUTS_DIR, f"eval_{model_name}.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Graphiques sauvegardés → eval_{model_name}.png")

    # Sauvegarde métriques JSON
    with open(os.path.join(OUTPUTS_DIR, f"metrics_{model_name}.json"), "w") as f:
        json.dump({k: float(v) if v is not None else None for k, v in metrics.items()}, f, indent=2)

    return metrics


# ──────────────────────────────────────────────
# 5. FEATURE IMPORTANCE & SHAP
# ──────────────────────────────────────────────

def plot_feature_importance(model, feature_names: list, model_name: str = "model", top_n: int = 20) -> None:
    """Extrait et visualise les importances de features."""
    estimator = model.named_steps["clf"] if hasattr(model, "named_steps") else model

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importances = np.abs(estimator.coef_[0])
    else:
        print(f"  ⚠ Impossible d'extraire les importances pour {model_name}")
        return

    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    fi.plot.barh(ax=ax, color=sns.color_palette("viridis", len(fi)))
    ax.invert_yaxis()
    ax.set_title(f"Feature Importance — {model_name} (Top {top_n})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fname = os.path.join(OUTPUTS_DIR, f"feature_importance_{model_name}.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  → feature_importance_{model_name}.png sauvegardé")


def plot_shap_values(model, X_sample: pd.DataFrame, model_name: str = "model", max_display: int = 20) -> None:
    """Génère les graphiques SHAP (summary + beeswarm)."""
    if not HAS_SHAP:
        print("  ⚠ SHAP non installé — pip install shap")
        return

    estimator = model.named_steps["clf"] if hasattr(model, "named_steps") else model

    try:
        if hasattr(estimator, "predict_proba") and hasattr(estimator, "feature_importances_"):
            explainer = shap.TreeExplainer(estimator)
        else:
            explainer = shap.LinearExplainer(estimator, X_sample)

        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Summary bar
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(shap_values, X_sample, plot_type="bar", max_display=max_display, show=False)
        plt.title(f"SHAP Summary — {model_name}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUTS_DIR, f"shap_bar_{model_name}.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # Beeswarm
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(shap_values, X_sample, max_display=max_display, show=False)
        plt.title(f"SHAP Beeswarm — {model_name}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUTS_DIR, f"shap_beeswarm_{model_name}.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  → SHAP sauvegardé pour {model_name}")

    except Exception as e:
        print(f"  ⚠ Erreur SHAP : {e}")


# ──────────────────────────────────────────────
# 6. COMPARAISON VISUELLE DES MODÈLES
# ──────────────────────────────────────────────

def plot_cv_comparison(cv_df: pd.DataFrame) -> None:
    """Graphique comparatif des modèles (F1, AUC, Accuracy)."""
    metrics = ["f1_mean", "roc_auc_mean", "accuracy_mean", "precision_mean", "recall_mean"]
    metrics = [m for m in metrics if m in cv_df.columns]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(cv_df))
    width = 0.15
    palette = sns.color_palette("viridis", len(metrics))

    for i, metric in enumerate(metrics):
        label = metric.replace("_mean", "").upper()
        ax.bar(x + i * width, cv_df[metric], width, label=label, color=palette[i])

    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(cv_df["model"], rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Comparaison des modèles (validation croisée)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUTS_DIR, "models_comparison.png"), dpi=150)
    plt.close(fig)
    print("  → models_comparison.png sauvegardé")


# ──────────────────────────────────────────────
# PIPELINE COMPLET
# ──────────────────────────────────────────────

def full_training_pipeline(
    X_train, y_train, X_test, y_test,
    feature_names: list,
    best_model_name: str = "RandomForest",
    run_cv: bool = True,
):
    """Pipeline complet : CV → entraînement → évaluation → interprétabilité."""
    if run_cv:
        print("\n── Validation croisée ──")
        cv_df = cross_validate_models(X_train, y_train)
        plot_cv_comparison(cv_df)
        print(cv_df[["model", "f1_mean", "roc_auc_mean"]].to_string(index=False))

    model = train_best_model(X_train, y_train, model_name=best_model_name)
    evaluate_model(model, X_test, y_test, model_name=best_model_name)
    plot_feature_importance(model, feature_names, model_name=best_model_name)

    sample_size = min(500, len(X_test))
    plot_shap_values(model, X_test.iloc[:sample_size], model_name=best_model_name)

    return model


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    print("Test rapide avec données synthétiques …")
    X, y = make_classification(n_samples=2000, n_features=12, n_informative=8, random_state=42)
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(12)])
    y = pd.Series(y)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    full_training_pipeline(X_tr, y_tr, X_te, y_te, list(X.columns))
