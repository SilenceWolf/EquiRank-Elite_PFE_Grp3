# ============================================================
# © 2026 Louis Guillory (Silence Wolf) — Tous droits réservés
# Licence : PROPRIETARY — voir fichier LICENSE
# ============================================================

# Calcule une empreinte stable du squelette DOM d'une page.
# Deux pages avec la même structure (même tags, mêmes classes principales,
# mêmes chemins) auront la même empreinte, peu importe leur contenu textuel.
# Utilisé par StructureComparator pour détecter si N URLs partagent la
# même structure — auquel cas on peut les crawler avec la même config.

from __future__ import annotations

import hashlib
from bs4 import BeautifulSoup, Tag


_MAX_DEPTH   = 4       # profondeur max d'exploration — au-delà c'est du bruit
_MAX_PATHS   = 2000    # cap dur pour éviter les pages monstrueuses


def fingerprint(soup: BeautifulSoup | Tag, max_depth: int = _MAX_DEPTH) -> str:
    """
    Retourne un hash SHA1 stable du squelette DOM de la page.

    La normalisation :
      - on ignore le texte
      - on ignore les id (souvent générés dynamiquement)
      - on garde le tag + la PREMIÈRE classe uniquement (évite les classes d'état)
      - on limite la profondeur pour rester rapide
    """
    root = soup.find('body') if hasattr(soup, 'find') else soup
    if root is None:
        root = soup

    paths: set[str] = set()
    _walk(root, '', paths, depth=0, max_depth=max_depth)

    serialized = '\n'.join(sorted(paths))
    return hashlib.sha1(serialized.encode('utf-8')).hexdigest()


def _walk(
    node:      Tag,
    prefix:    str,
    paths:     set[str],
    depth:     int,
    max_depth: int,
) -> None:
    """Walk récursif qui ajoute chaque chemin tag>tag>tag au set."""
    if depth > max_depth or len(paths) >= _MAX_PATHS:
        return

    if not isinstance(node, Tag) or node.name is None:
        return

    # Normalisation : tag + première classe CSS uniquement
    first_class = ''
    classes = node.get('class') or []
    if classes:
        first_class = f'.{classes[0]}'

    key      = f'{node.name}{first_class}'
    new_path = f'{prefix}>{key}' if prefix else key
    paths.add(new_path)

    for child in node.children:
        if isinstance(child, Tag):
            _walk(child, new_path, paths, depth + 1, max_depth)
