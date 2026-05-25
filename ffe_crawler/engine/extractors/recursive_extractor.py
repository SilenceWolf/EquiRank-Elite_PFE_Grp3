# ============================================================
# © 2026 Louis Guillory (Silence Wolf) — Tous droits réservés
# Licence : PROPRIETARY — voir fichier LICENSE
# ============================================================

# RecursiveExtractor : descend dans les enfants d'un élément jusqu'à
# trouver les "feuilles" (texte, <a>, <img>, <td>, <input value>) et
# capture le nom de colonne depuis le <th>, <label>, data-label ou
# aria-label le plus proche dans la hiérarchie.
#
# Cet extracteur est OPT-IN : il n'est PAS dans DEFAULT_EXTRACTORS parce
# que son can_handle retourne toujours True et court-circuiterait les
# extracteurs spécialisés. On l'instancie explicitement quand
# CrawlStep.recursive_depth > 0.
#
# Les noms de colonnes découverts deviennent automatiquement des colonnes
# du CSV grâce à la gestion dynamique de CsvWriter (output/csv_writer.py).

from __future__ import annotations

from bs4 import NavigableString, Tag

from .base_extractor import BaseCrawlerExtractor
from detection.language_detector import LanguageDetector


# Balises considérées comme "terminales" : on arrête la descente ici
# et on capture leur valeur textuelle (ou leur attribut pour les médias/liens).
_LEAF_TAGS = {'a', 'img', 'td', 'th', 'span', 'li', 'input', 'p', 'h1',
              'h2', 'h3', 'h4', 'h5', 'h6', 'strong', 'em', 'code', 'time'}

# Préfixes de classes CSS utilitaires qu'on IGNORE pour le nommage —
# ce sont du layout, pas de la sémantique (Bootstrap, Tailwind, etc.).
_UTILITY_CLASS_PREFIXES = (
    'col-', 'row-', 'container', 'btn', 'mb-', 'mt-', 'ml-', 'mr-',
    'mx-', 'my-', 'p-', 'pt-', 'pb-', 'pl-', 'pr-', 'px-', 'py-',
    'd-', 'flex-', 'justify-', 'align-', 'items-', 'self-', 'order-',
    'text-', 'font-', 'fs-', 'fw-', 'bg-', 'border', 'rounded',
    'shadow', 'w-', 'h-', 'min-', 'max-', 'gap-', 'space-',
    'absolute', 'relative', 'fixed', 'sticky', 'static', 'z-',
    'overflow', 'cursor-', 'opacity-', 'hidden', 'visible',
    'block', 'inline', 'grid', 'table-', 'list-', 'truncate',
)
_UTILITY_CLASS_EXACT = {
    'row', 'col', 'container', 'small', 'lead',
    'show', 'hide',
    # Note : 'on', 'off', 'active', 'disabled', 'badge', 'card', 'btn'
    # sont VOLONTAIREMENT exclus — ces classes sont souvent sémantiques
    # (ex : FFE SIF utilise .on pour marquer les jours cliquables du
    # calendrier, .badge pour des numéros d'événements…).
}


def _is_meaningful_class(cls: str) -> bool:
    """Heuristique : la classe peut servir de nom de colonne."""
    if not cls:
        return False
    if cls in _UTILITY_CLASS_EXACT:
        return False
    for prefix in _UTILITY_CLASS_PREFIXES:
        if cls.startswith(prefix):
            return False
    return len(cls) >= 3


class RecursiveExtractor(BaseCrawlerExtractor):
    """
    Extracteur universel qui walk les enfants d'un élément jusqu'à la
    profondeur max ou jusqu'à trouver une feuille pertinente.

    Chaque feuille produit une row avec une colonne dynamique nommée
    d'après le header le plus proche (th / label / data-label) et un
    champ 'language' détecté depuis un drapeau proche → attribut lang →
    langue de la page.
    """

    def __init__(
        self,
        max_depth:            int = 5,
        capture_headers_from: str = 'auto',   # 'th' | 'label' | 'auto'
    ) -> None:
        self._max_depth    = max_depth
        self._header_mode  = capture_headers_from
        self._lang         = LanguageDetector()
        # Cache la langue de la page par id(document) — évite de re-walker
        # jusqu'au <html> pour chaque leaf.
        self._page_lang_cache: dict[int, str] = {}

    # ── API BaseCrawlerExtractor ──────────────────────────────────────

    def can_handle(self, element: Tag) -> bool:
        # Universel : on laisse le Crawler nous passer n'importe quel élément.
        # Ne JAMAIS mettre cet extracteur dans DEFAULT_EXTRACTORS.
        return True

    def extract(self, element: Tag, url: str) -> list[dict]:
        rows: list[dict] = []
        self._walk(element, url, depth=0, rows=rows)
        return rows

    # ── Internals ──────────────────────────────────────────────────────

    def _walk(self, node: Tag, url: str, depth: int, rows: list[dict]) -> None:
        """Descend récursivement dans les enfants jusqu'à max_depth ou une feuille."""
        if depth > self._max_depth:
            return

        # Si on est sur une feuille ou à profondeur max → on capture
        if self._is_leaf(node) or depth == self._max_depth:
            row = self._capture(node, url)
            if row is not None:
                rows.append(row)
            return

        # Sinon on descend dans les enfants Tag (on ignore les NavigableString)
        for child in node.children:
            if isinstance(child, Tag):
                self._walk(child, url, depth + 1, rows)

    def _is_leaf(self, node: Tag) -> bool:
        """Une feuille = un tag terminal OU un élément sans enfant Tag."""
        if node.name in _LEAF_TAGS:
            return True
        # Si aucun enfant n'est un Tag, c'est qu'on n'a que du texte → feuille
        has_tag_child = any(isinstance(c, Tag) for c in node.children)
        return not has_tag_child

    def _capture(self, node: Tag, url: str) -> dict | None:
        """Produit une row à partir d'une feuille, avec header dynamique
        et détection de la langue (drapeau proche → ancêtre lang → page)."""
        # Valeur principale selon le type de balise. Pour les <a> on
        # enrichit la row avec href/text séparés (consommés par urls.csv).
        href = ''
        text = ''
        if node.name == 'a':
            href    = node.get('href', '') or ''
            text    = node.get_text(strip=True)
            content = href or text
        elif node.name == 'img':
            content = node.get('src', '') or node.get('alt', '')
        elif node.name == 'input':
            content = node.get('value', '') or node.get('placeholder', '')
        elif node.name == 'time':
            content = node.get('datetime', '') or node.get_text(strip=True)
        else:
            content = node.get_text(strip=True)

        # On skip les feuilles totalement vides pour ne pas polluer le CSV
        if not content:
            return None

        column_name = self._resolve_header(node)
        language    = self._detect_language(node)

        row: dict = {
            'source_url': url,
            'extractor':  self.name,
            'type':       'recursive',
            'tag':        node.name or '',
            'content':    content,
            'extra':      '',
            'language':   language,
        }
        # Pour les <a> : expose href et text pour que urls.csv les route
        if href:
            row['href'] = href
        if text:
            row['text'] = text
        # Ajoute la colonne dynamique — récupérée par le CsvWriter
        if column_name:
            row[column_name] = content
        return row

    def _detect_language(self, leaf: Tag) -> str:
        """
        Résout la langue d'une feuille :
          1. drapeau/hreflang proche de la feuille → langue annoncée
          2. attribut lang sur un ancêtre → langue héritée
          3. langue de la page (html[lang], meta) en dernier recours
        """
        # 1. proche
        code = self._lang.detect_near(leaf)
        if code:
            return code

        # 2. ancêtre lang
        code = self._lang.detect_from_ancestors(leaf)
        if code:
            return code

        # 3. page (cachée par document)
        doc_key = id(leaf.parent) if leaf.parent else id(leaf)
        if doc_key not in self._page_lang_cache:
            self._page_lang_cache[doc_key] = self._lang.detect_page(leaf)
        return self._page_lang_cache[doc_key]

    def _resolve_header(self, leaf: Tag) -> str:
        """
        Remonte les ancêtres pour trouver le nom de colonne le plus pertinent.
        Ordre selon capture_headers_from : th → label → data-label → aria-label → tag parent.
        """
        mode = self._header_mode

        # Essai 1 : <th> correspondant pour une cellule de table
        if mode in ('th', 'auto'):
            th_name = self._find_table_header(leaf)
            if th_name:
                return self._normalize(th_name)

        # Essai 2 : <label for="id"> ou <label> parent
        if mode in ('label', 'auto'):
            label_name = self._find_label(leaf)
            if label_name:
                return self._normalize(label_name)

        # Essai 3 : data-label / aria-label sur l'élément ou un ancêtre proche
        if mode == 'auto':
            for node in [leaf] + list(leaf.parents):
                if not isinstance(node, Tag):
                    continue
                for attr in ('data-label', 'aria-label', 'data-column'):
                    val = node.get(attr)
                    if val:
                        return self._normalize(val)
                # On limite la remontée à 4 niveaux — au-delà c'est du bruit
                if len(list(node.parents)) > 4:
                    break

        # Essai 4 : classe CSS sémantique (discipline, club, prix, etc.)
        # — TOUJOURS tenté en dernier recours, quel que soit le mode :
        # utile quand le HTML n'a ni th, ni label, ni data-label mais
        # nomme ses éléments par classe. On filtre les classes utilitaires
        # de layout (Bootstrap/Tailwind) pour ne garder que la sémantique.
        count = 0
        for node in [leaf] + list(leaf.parents):
            if not isinstance(node, Tag):
                continue
            classes = node.get('class') or []
            for cls in classes:
                if _is_meaningful_class(cls):
                    return self._normalize(cls)
            count += 1
            if count > 4:
                break

        # Fallback : on retourne une chaîne vide → pas de colonne dynamique ajoutée
        return ''

    def _find_table_header(self, leaf: Tag) -> str:
        """
        Si la feuille est une <td> (ou descendante d'une <td>), trouve le
        <th> correspondant en comptant l'index de la cellule dans sa <tr>.
        """
        # Trouve la <td> ancêtre (ou la feuille elle-même si c'en est une)
        td = leaf if leaf.name == 'td' else leaf.find_parent('td')
        if td is None:
            return ''

        tr = td.find_parent('tr')
        if tr is None:
            return ''

        # Index de la td dans sa ligne
        tds = [c for c in tr.find_all(['td', 'th'], recursive=False)]
        try:
            col_idx = tds.index(td)
        except ValueError:
            return ''

        # Remonte à la table et prend la première <tr> avec des <th>
        table = tr.find_parent('table')
        if table is None:
            return ''

        header_tr = None
        for candidate in table.find_all('tr'):
            ths = candidate.find_all('th', recursive=False)
            if ths:
                header_tr = candidate
                break

        if header_tr is None:
            return ''

        headers = header_tr.find_all(['th', 'td'], recursive=False)
        if col_idx < len(headers):
            return headers[col_idx].get_text(strip=True)
        return ''

    def _find_label(self, leaf: Tag) -> str:
        """
        Trouve le <label> associé à cette feuille.
        Deux cas : <label for="id"> avec id correspondant, ou <label> parent.
        """
        # Cas 1 : leaf a un id, on cherche un label[for=id] frère
        leaf_id = leaf.get('id') if isinstance(leaf, Tag) else None
        if leaf_id:
            root = leaf.find_parent() or leaf
            label = root.find('label', attrs={'for': leaf_id})
            if label:
                return label.get_text(strip=True)

        # Cas 2 : label parent direct (jusqu'à 3 niveaux)
        parent = leaf.parent
        depth  = 0
        while parent is not None and depth < 3:
            if isinstance(parent, Tag) and parent.name == 'label':
                return parent.get_text(strip=True)
            parent = parent.parent
            depth += 1
        return ''

    def _normalize(self, raw: str) -> str:
        """
        Nettoie un nom de colonne pour en faire une clé CSV propre.
        On vire les retours ligne, espaces multiples et caractères de contrôle.
        """
        cleaned = ' '.join(raw.split())
        return cleaned[:80]    # cap pour éviter les en-têtes aberrants
