# ============================================================
# © 2026 Louis Guillory (Silence Wolf) — Tous droits réservés
# Licence : PROPRIETARY — voir fichier LICENSE
# ============================================================

# Une CrawlChain est une liste ordonnée de CrawlStep. Les URLs exportées
# par l'étape N alimentent l'étape N+1 via parent_step_id — c'est le
# ChainRunner qui résout ça à l'exécution, pas la dataclass elle-même.

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime    import datetime
from typing      import Any

from .step import CrawlStep


@dataclass
class CrawlChain:
    """
    Chaîne de crawls sérialisable. Sert à la fois de modèle d'exécution
    (passée au ChainRunner) et d'objet persisté dans les sessions JSON.
    """

    chain_id:   str
    name:       str
    created_at: str                                # ISO-8601
    steps:      list[CrawlStep] = field(default_factory=list)
    # Mapping step_id → liste de colonnes "obligatoires non vides"
    # appliqué par FilterColumnsPanel. Persisté avec la session pour
    # qu'au replay le filtre soit ré-appliqué automatiquement
    # (table affichée + ré-écriture du `<step>_filtered.csv`).
    required_columns_by_step: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def new(cls, name: str) -> 'CrawlChain':
        """Factory : crée une nouvelle chaîne vide avec un id fraîchement généré."""
        return cls(
            chain_id   = uuid.uuid4().hex[:12],
            name       = name,
            created_at = datetime.now().isoformat(timespec='seconds'),
            steps      = [],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            'chain_id':   self.chain_id,
            'name':       self.name,
            'created_at': self.created_at,
            'steps':      [s.to_dict() for s in self.steps],
            'required_columns_by_step': dict(self.required_columns_by_step),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> 'CrawlChain':
        return cls(
            chain_id   = d['chain_id'],
            name       = d['name'],
            created_at = d['created_at'],
            steps      = [CrawlStep.from_dict(s) for s in d.get('steps', [])],
            required_columns_by_step = dict(d.get('required_columns_by_step') or {}),
        )
