"""
Module d'Analyse Exploratoire des Données (EDA).
Génère automatiquement un rapport visuel complet dans /outputs/eda/
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../outputs/eda")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Palette cohérente au projet
PALETTE = sns.color_palette("viridis", 10)
sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.1)


# ──────────────────────────────────────────────────────────
# 1. STATISTIQUES DESCRIPTIVES
# ──────────────────────────────────────────────────────────

def descriptive_stats(df: pd.DataFrame) -> None:
    """Affiche et sauvegarde les statistiques descriptives."""
    print("\n══════════════════ STATISTIQUES DESCRIPTIVES ══════════════════")
    print(df.describe(include="all").T.to_string())

    # Valeurs manquantes
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"missing": missing, "pct": missing_pct})
    missing_df = missing_df[missing_df["missing"] > 0].sort_values("pct", ascending=False)
    if not missing_df.empty:
        print("\n── Valeurs manquantes ──")
        print(missing_df.to_string())

        fig, ax = plt.subplots(figsize=(10, max(4, len(missing_df) * 0.4)))
        missing_df["pct"].plot.barh(ax=ax, color=PALETTE[3])
        ax.set_xlabel("% manquant")
        ax.set_title("Valeurs manquantes par colonne")
        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "missing_values.png"), dpi=150)
        plt.close(fig)
        print("  → missing_values.png sauvegardé")


# ──────────────────────────────────────────────────────────
# 2. DISTRIBUTION DE LA CIBLE
# ──────────────────────────────────────────────────────────

def plot_target_distribution(df: pd.DataFrame, target_col: str = "resultat_binaire") -> None:
    if target_col not in df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Distribution de la variable cible", fontsize=14, fontweight="bold")

    # Binaire
    counts = df[target_col].value_counts()
    axes[0].pie(
        counts,
        labels=["Échec (0)", "Réussite (1)"],
        autopct="%1.1f%%",
        colors=[PALETTE[0], PALETTE[6]],
        startangle=90,
    )
    axes[0].set_title("Répartition succès / échec")

    # Classement si disponible
    if "classement" in df.columns:
        df["classement"].dropna().clip(upper=50).hist(
            ax=axes[1], bins=30, color=PALETTE[4], edgecolor="white"
        )
        axes[1].set_xlabel("Classement")
        axes[1].set_title("Distribution des classements (≤ 50)")
    else:
        axes[1].set_visible(False)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "target_distribution.png"), dpi=150)
    plt.close(fig)
    print("  → target_distribution.png sauvegardé")


# ──────────────────────────────────────────────────────────
# 3. DISTRIBUTIONS NUMÉRIQUES
# ──────────────────────────────────────────────────────────

NUMERIC_COLS_EDA = ["age_cheval", "age_cavalier", "nombre_participants",
                    "horse_win_rate", "rider_win_rate", "couple_synergy"]


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    cols = [c for c in NUMERIC_COLS_EDA if c in df.columns]
    if not cols:
        return

    n = len(cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        axes[i].hist(df[col].dropna(), bins=40, color=PALETTE[i % 10], edgecolor="white")
        axes[i].set_title(col.replace("_", " ").title())
        axes[i].set_xlabel(col)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Distributions des variables numériques", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "numeric_distributions.png"), dpi=150)
    plt.close(fig)
    print("  → numeric_distributions.png sauvegardé")


# ──────────────────────────────────────────────────────────
# 4. VARIABLES CATÉGORIELLES
# ──────────────────────────────────────────────────────────

CAT_COLS_EDA = ["discipline", "niveau", "sexe_cheval", "race_cheval",
                "categorie_cavalier", "competition_density"]


def plot_categorical_counts(df: pd.DataFrame) -> None:
    cols = [c for c in CAT_COLS_EDA if c in df.columns]
    if not cols:
        return

    for col in cols:
        fig, ax = plt.subplots(figsize=(10, 4))
        top = df[col].astype(str).value_counts().head(20)
        top.plot.barh(ax=ax, color=PALETTE[5])
        ax.invert_yaxis()
        ax.set_title(f"Répartition — {col.replace('_', ' ').title()}")
        ax.set_xlabel("Nombre d'occurrences")
        plt.tight_layout()
        fname = f"cat_{col}.png"
        fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150)
        plt.close(fig)
        print(f"  → {fname} sauvegardé")


# ──────────────────────────────────────────────────────────
# 5. CORRÉLATIONS
# ──────────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame, target_col: str = "resultat_binaire") -> None:
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty or len(num_df.columns) < 2:
        return

    corr = num_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(max(10, len(corr) * 0.8), max(8, len(corr) * 0.7)))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdYlGn", center=0, ax=ax,
        linewidths=0.5, linecolor="white",
    )
    ax.set_title("Matrice de corrélation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close(fig)
    print("  → correlation_heatmap.png sauvegardé")

    # Corrélations avec la cible
    if target_col in num_df.columns:
        target_corr = (
            corr[target_col]
            .drop(target_col)
            .sort_values(key=abs, ascending=False)
            .head(15)
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        target_corr.plot.barh(ax=ax, color=[PALETTE[6] if v > 0 else PALETTE[0] for v in target_corr])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"Corrélation avec '{target_col}'", fontsize=13, fontweight="bold")
        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "target_correlations.png"), dpi=150)
        plt.close(fig)
        print("  → target_correlations.png sauvegardé")


# ──────────────────────────────────────────────────────────
# 6. PERFORMANCE PAR DISCIPLINE & NIVEAU
# ──────────────────────────────────────────────────────────

def plot_performance_by_category(df: pd.DataFrame, target_col: str = "resultat_binaire") -> None:
    if target_col not in df.columns:
        return

    for cat in ["discipline", "niveau", "race_cheval"]:
        if cat not in df.columns:
            continue
        perf = (
            df.groupby(cat)[target_col]
            .agg(taux_reussite="mean", nb_participations="count")
            .sort_values("taux_reussite", ascending=False)
            .head(15)
        )
        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax2 = ax1.twinx()
        perf["taux_reussite"].plot.bar(ax=ax1, color=PALETTE[6], alpha=0.8, label="Taux réussite")
        ax2.plot(range(len(perf)), perf["nb_participations"], "o-", color=PALETTE[1], label="Nb participations")
        ax1.set_ylabel("Taux de réussite (%)")
        ax2.set_ylabel("Nombre de participations")
        ax1.set_title(f"Performance par {cat.replace('_', ' ').title()}", fontsize=13, fontweight="bold")
        ax1.tick_params(axis="x", rotation=45)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        plt.tight_layout()
        fname = f"perf_by_{cat}.png"
        fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150)
        plt.close(fig)
        print(f"  → {fname} sauvegardé")


# ──────────────────────────────────────────────────────────
# 7. ÂGE VS PERFORMANCE
# ──────────────────────────────────────────────────────────

def plot_age_vs_performance(df: pd.DataFrame, target_col: str = "resultat_binaire") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Âge vs Performance", fontsize=14, fontweight="bold")

    for ax, age_col, label in zip(
        axes,
        ["age_cheval", "age_cavalier"],
        ["Âge du cheval", "Âge du cavalier"],
    ):
        if age_col not in df.columns or target_col not in df.columns:
            ax.set_visible(False)
            continue
        perf_age = df.groupby(age_col)[target_col].mean()
        counts   = df.groupby(age_col)[target_col].count()
        ax2 = ax.twinx()
        ax.plot(perf_age.index, perf_age.values, "o-", color=PALETTE[6], linewidth=2)
        ax2.bar(counts.index, counts.values, alpha=0.25, color=PALETTE[3])
        ax.set_xlabel(label)
        ax.set_ylabel("Taux de réussite")
        ax2.set_ylabel("Nb participations")
        ax.set_title(label)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "age_vs_performance.png"), dpi=150)
    plt.close(fig)
    print("  → age_vs_performance.png sauvegardé")


# ──────────────────────────────────────────────────────────
# PIPELINE EDA COMPLET
# ──────────────────────────────────────────────────────────

def run_full_eda(df: pd.DataFrame, target_col: str = "resultat_binaire") -> None:
    print(f"\n{'═'*60}")
    print("  ANALYSE EXPLORATOIRE DES DONNÉES — PROJET ÉQUESTRE")
    print(f"{'═'*60}\n")
    print(f"Fichiers de sortie : {OUTPUT_DIR}\n")

    descriptive_stats(df)
    plot_target_distribution(df, target_col)
    plot_numeric_distributions(df)
    plot_categorical_counts(df)
    plot_correlation_heatmap(df, target_col)
    plot_performance_by_category(df, target_col)
    plot_age_vs_performance(df, target_col)

    print(f"\n✅ EDA terminé — tous les graphiques sont dans {OUTPUT_DIR}")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "../data/dataset.csv"
    if os.path.exists(path):
        df = pd.read_csv(path, sep=None, engine="python")
        run_full_eda(df)
    else:
        print(f"Fichier introuvable : {path}")
