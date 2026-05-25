# ============================================================
# © 2026 PFE EquiRank Elite — Tous droits réservés
# Équipe : Louis Guillory · Karlotta Martin · Mathéo Isidoro
#         · Mattéo D'Andrea · Melvin Bandeira · Yohan Cana
# Licence : PROPRIETARY — voir fichier LICENSE
# ============================================================

"""
Version Streamlit minimaliste de la prédiction Equirank — utile pour
tester rapidement la logique de `predictor.py` sans avoir à lancer le
serveur FastAPI ni ouvrir le navigateur.

C'est l'évolution PFE de l'ancien script `equi/liaison.py` :
  - on n'utilise plus `dataset_equirank.csv` mais le `dataset.csv`
    adapté du PFE (sortie de adapt_dataset_v2.py) ;
  - on n'utilise plus `model.pkl` mais le `.joblib` produit par
    `train.py` (modèle LightGBM / XGBoost / ...) ;
  - le tout passe par `EquirankPredictor` pour rester strictement
    aligné sur ce que sert `/api/predict`.

Lancement :
    streamlit run equirank/liaison.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Permet de lancer depuis n'importe où — on ajoute la racine PFE au path
_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from equirank.predictor import getDefaultPredictor


st.set_page_config(
    page_title = 'Equirank Liaison — Test backend',
    page_icon  = '🐎',
    layout     = 'centered',
)

st.title('🐎 Equirank — Test direct du backend')
st.caption(
    'Banc d\'essai du module `equirank.predictor`. Pour la page complète, '
    'lance plutôt `uvicorn equirank.server:app --reload`.'
)

predictor = getDefaultPredictor()

try:
    disciplines = predictor.listDisciplines()
except FileNotFoundError as exc:
    st.error(f'Configuration incomplète : {exc}')
    st.stop()

with st.form('predict_form'):
    cheval     = st.text_input('Nom du cheval',     value = '')
    cavalier   = st.text_input('Nom du cavalier',   value = '')
    discipline = st.selectbox('Discipline', options = disciplines)
    submitted  = st.form_submit_button('Prédire')

if submitted:
    result = predictor.predict(
        cheval     = cheval,
        cavalier   = cavalier,
        discipline = discipline,
    )

    if isinstance(result, dict):
        st.error(result.get('error', 'Erreur inconnue'))
    else:
        d = result.to_dict()
        proba = d['proba']
        if proba >= 60:
            st.success(f"★ Favori solide — probabilité de réussite : {proba} %")
        elif proba >= 35:
            st.warning(f"◆ Outsider crédible — probabilité : {proba} %")
        else:
            st.error(f"▲ Outsider — probabilité : {proba} %")

        col1, col2 = st.columns(2)
        with col1:
            st.metric('Cheval — réussites',  f"{d['cheval_victoires']} / {d['cheval_courses']}")
            st.metric('Cavalier — réussites', f"{d['jockey_victoires']} / {d['jockey_courses']}")
        with col2:
            st.metric('Duo — réussites',     f"{d['duo_victoires']} / {d['duo_courses']}")
            st.metric('Modèle',              d['model_name'])

        st.info(d['forme'])

        with st.expander('Payload JSON renvoyé par /api/predict'):
            st.json(d)
