# ============================================================
# © 2026 PFE EquiRank Elite — Tous droits réservés
# Équipe : Louis Guillory · Karlotta Martin · Mathéo Isidoro
#         · Mattéo D'Andrea · Melvin Bandeira · Yohan Cana
# Licence : PROPRIETARY — voir fichier LICENSE
# ============================================================

"""
Description figée de la chaîne FFE.

On reprend ici tels quels les data_selectors, field_values et required_columns
qui avaient été ajustés sur la session "FFE final" (id ab904d27014e). Ces
choix sont volontairement non-paramétrables : c'est la signature même de la
chaîne FFE. Seuls trois leviers restent ouverts à l'utilisateur :

  • dates `deb` / `fin` du calendrier de l'étape s1
  • discipline (radio) — propage la même valeur à toutes les étapes
  • identifiants FFE — uniquement utilisés par s5 qui requiert l'auth

L'URL d'entrée s1 est l'écran calendrier Telemat/SIF FFE. Le chaînage entre
étapes (s2 → s3 → s4 → s5) est résolu par le ChainRunner du moteur via
parent_step_id ; on n'a donc pas à figer les URLs intermédiaires.
"""

from __future__ import annotations

import sys
from pathlib import Path

# On injecte le moteur copié sous engine/ dans le sys.path pour que les
# `from crawler import …` / `from log import …` continuent de fonctionner
# tels quels, sans avoir à toucher les modules copiés.
_ENGINE_DIR = Path(__file__).resolve().parent.parent / 'engine'
if str(_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(_ENGINE_DIR))

from session.chain import CrawlChain
from session.step  import CrawlStep


# URL "calendrier Telemat/SIF FFE — toutes disciplines". C'est le point
# d'entrée historique de la session ab904d27014e. Le crawler suit ensuite
# automatiquement les liens vers les concours.
DEFAULT_ENTRY_URL = (
    'https://www.telemat.org/FFE/sif/'
    '?cs=4.57906319cd19ab638818b32201477a972c88'
)


# Liste blanche des disciplines reconnues côté Telemat (label exact tel
# que servi par le formulaire). On expose la même liste dans l'app
# Streamlit, et c'est cette chaîne qui est injectée dans field_values.
SUPPORTED_DISCIPLINES: tuple[str, ...] = (
    'Toutes disciplines',
    'CSO',
    'Dressage',
    'CCE',
    'Hunter',
    'Equifun',
    'Pony-Games',
    'Endurance',
    'Attelage',
    'Voltige',
    'Horse-Ball',
    'TREC',
    'Western',
)


def buildFfeChain(
    deb:        str,
    fin:        str,
    discipline: str = 'Toutes disciplines',
    entry_url:  str = DEFAULT_ENTRY_URL,
    chain_name: str = 'FFE final',
    include_s5: bool = True,
) -> CrawlChain:
    """
    Construit la CrawlChain figée FFE pour les paramètres fournis.

    Args:
        deb        : date de début au format YYYY-MM-DD
        fin        : date de fin au format YYYY-MM-DD
        discipline : valeur du radio "discipline" — voir SUPPORTED_DISCIPLINES
        entry_url  : URL du calendrier — par défaut DEFAULT_ENTRY_URL
        chain_name : nom logique de la chaîne (pour les logs)
        include_s5 : True (défaut) inclut la fiche équidé authentifiée ;
                     False arrête après s4. Pratique pour les concours
                     futurs où l'auth n'apporte rien (et où on ne veut
                     pas demander ses credentials à l'utilisateur).
    """

    chain = CrawlChain.new(chain_name)

    # ── s1 : calendrier — sélecteur ".on" capture les jours actifs ─────
    chain.steps.append(CrawlStep(
        step_id        = 's1',
        entry_urls     = [entry_url],
        parent_step_id = None,
        data_selector  = '.on',
        field_values   = {
            'toutes_disciplines': discipline,
            'deb':                deb,
            'fin':                fin,
        },
    ))

    # ── s2 : liste des concours dans la période ──────────────────────
    # entry_urls vide → ChainRunner les déduit des url_rows de s1
    chain.steps.append(CrawlStep(
        step_id        = 's2',
        parent_step_id = 's1',
        data_selector  = 'td[data-label="Numéro"]',
        field_values   = {
            'toutes_disciplines': discipline,
        },
    ))

    # ── s3 : liste des épreuves d'un concours ─────────────────────────
    chain.steps.append(CrawlStep(
        step_id        = 's3',
        parent_step_id = 's2',
        data_selector  = (
            'td[data-label="Date"], '
            'td[data-label="Discipline"], '
            'td[data-label="Épreuve"]'
        ),
        field_values   = {
            'toutes_disciplines': discipline,
        },
    ))

    # ── s4 : engagements (chevaux engagés sur une épreuve) ────────────
    chain.steps.append(CrawlStep(
        step_id        = 's4',
        parent_step_id = 's3',
        data_selector  = '#t_engts tr',
        field_values   = {
            'toutes_disciplines': discipline,
        },
    ))

    # ── s5 : fiche détail équidé — AUTH requise ───────────────────────
    if include_s5:
        chain.steps.append(CrawlStep(
            step_id        = 's5',
            parent_step_id = 's4',
            data_selector  = '.fieldset .label',
            field_values   = {
                'toutes_disciplines': discipline,
            },
            requires_auth  = True,
        ))

    # Colonnes "URL de répétition" choisies pendant la mise au point de
    # la session — on les ré-impose pour que le filtrage avant chaînage
    # reste cohérent : s3 ne suit que href, s4 ne suit que Équidé_url.
    chain.required_columns_by_step = {
        's3': ['href'],
        's4': ['Équidé_url'],
    }

    return chain
