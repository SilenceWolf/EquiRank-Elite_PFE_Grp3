# ============================================================
# © 2026 PFE EquiRank Elite — Tous droits réservés
# Équipe : Louis Guillory · Karlotta Martin · Mathéo Isidoro
#         · Mattéo D'Andrea · Melvin Bandeira · Yohan Cana
# Licence : PROPRIETARY — voir fichier LICENSE
# ============================================================

"""
ffe_crawler — Outil dédié au PFE pour crawler la Fédération Française d'Équitation
via le portail Telemat/SIF, et générer un dataset au format dataset_brut_v2.csv.

Inspiré de la session "FFE final" (id ab904d27014e) du crawler générique mais
fonctionnellement autonome : la chaîne de 5 étapes (calendrier → concours →
épreuves → engagements → fiche équidé) est ici figée. Seules les interactions
indispensables sont exposées à l'utilisateur sur une page Streamlit unique :
  1. Sélection du calendrier (dates de début / fin)
  2. Discipline (radio "Toutes disciplines" / "Equifun" / ...)
  3. Identifiants de connexion FFE (pour la fiche équidé authentifiée s5)

Les data_selectors et le chaînage entre étapes (colonne URL utilisée pour la
répétition) sont volontairement immuables : ils encodent ce qu'on a validé
ensemble durant la mise au point de la session.
"""

from .core.runner import runFfeChain
from .core.dataset_builder import buildDatasetBrutV2
from .core.ffe_chain import buildFfeChain, DEFAULT_ENTRY_URL

__all__ = [
    'runFfeChain',
    'buildDatasetBrutV2',
    'buildFfeChain',
    'DEFAULT_ENTRY_URL',
]
