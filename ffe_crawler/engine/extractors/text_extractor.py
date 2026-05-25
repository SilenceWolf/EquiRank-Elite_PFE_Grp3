# © 2026 Louis Guillory (Silence Wolf) — Tous droits réservés
# Licence : PROPRIETARY — voir fichier LICENSE

from bs4 import Tag
from .base_extractor import BaseCrawlerExtractor


# Balises dont on veut capturer le texte directement
_TEXT_TAGS = {
    'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'span', 'li', 'dt', 'dd', 'blockquote',
    'label', 'caption', 'figcaption', 'time',
    'article', 'section', 'aside', 'header', 'footer',
    'div',  # en dernier recours — souvent un conteneur générique
}


class TextExtractor(BaseCrawlerExtractor):
    """
    Capture le texte visible de tous les éléments.
    C'est le filet de sécurité : si aucun extracteur spécialisé n'a pris en charge
    un élément, celui-ci récupère au moins son contenu textuel.
    """

    def can_handle(self, element: Tag) -> bool:
        # On peut presque toujours extraire du texte — sauf les éléments purement binaires
        return bool(element.get_text(strip=True))

    def extract(self, element: Tag, url: str) -> list[dict]:
        rows: list[dict] = []

        # On parcourt les descendants directs qui ont du texte propre
        # plutôt que de tout aplatir — sinon on duplique le texte des parents
        for el in element.find_all(_TEXT_TAGS):
            # get_text(separator=' ') préserve les espaces entre les balises inline
            text = el.get_text(separator=' ', strip=True)
            if not text:
                continue

            rows.append({
                'source_url': url,
                'extractor':  self.name,
                'type':       'text',
                'tag':        el.name,
                'content':    text,
                'extra':      _attrs_summary(el),
            })

        # Si aucun descendant ciblé, on prend le texte de la racine elle-même
        if not rows:
            root_text = element.get_text(separator=' ', strip=True)
            if root_text:
                rows.append({
                    'source_url': url,
                    'extractor':  self.name,
                    'type':       'text',
                    'tag':        element.name,
                    'content':    root_text,
                    'extra':      _attrs_summary(element),
                })

        return rows


def _attrs_summary(el: Tag) -> str:
    """Résumé compact des attributs utiles pour la colonne 'extra'."""
    useful = {k: v for k, v in el.attrs.items()
              if k in ('id', 'class', 'title', 'aria-label', 'datetime', 'data-value')}
    if not useful:
        return ''
    parts = []
    for k, v in useful.items():
        val = ' '.join(v) if isinstance(v, list) else str(v)
        parts.append(f'{k}={val}')
    return ' | '.join(parts)
