# ============================================================
# © 2026 PFE EquiRank Elite — Tous droits réservés
# Équipe : Louis Guillory · Karlotta Martin · Mathéo Isidoro
#         · Mattéo D'Andrea · Melvin Bandeira · Yohan Cana
# Licence : PROPRIETARY — voir fichier LICENSE
# ============================================================

"""
equirank — interface de prédiction du PFE.

Empile la page Equirank (HTML + style hippique dark/gold) au-dessus du
modèle PFE entraîné dans models/. Trois entrées :

  - server.py  : FastAPI qui sert la page et expose /api/predict
  - predictor.py: charge model + dataset, reconstruit les features
                  pour un duo cheval/cavalier/discipline et renvoie
                  une proba + des stats agrégées
  - liaison.py : version Streamlit minimale (test interactif du backend)
"""

from .predictor import EquirankPredictor, getDefaultPredictor

__all__ = ['EquirankPredictor', 'getDefaultPredictor']
