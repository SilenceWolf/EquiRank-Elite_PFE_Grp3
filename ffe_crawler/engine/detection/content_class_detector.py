# ============================================================
# © 2026 Louis Guillory (Silence Wolf) — Tous droits réservés
# Licence : PROPRIETARY — voir fichier LICENSE
# ============================================================

# ContentClassDetector : analyse le DOM d'une page et détecte les classes
# CSS qui contiennent du contenu intéressant (titres, descriptions, prix,
# dates, images, liens, etc.). Retourne une liste de suggestions avec
# un sample du texte pour preview.

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from bs4 import BeautifulSoup, Tag

from log import CrawlerLogger
from crawler import Crawler


# ── Patterns de détection ──────────────────────────────────────────────────

# Chaque catégorie a une liste de patterns (regex) qui matchent sur le nom
# de la classe CSS OU l'attribut class combiné.

_CATEGORY_PATTERNS: dict[str, list[re.Pattern]] = {
    'title': [
        re.compile(r'title|titre|heading|headline|name|nom', re.I),
        re.compile(r'\bh[1-3]\b', re.I),
        re.compile(r'product.?name|item.?name|card.?title', re.I),
    ],
    'description': [
        re.compile(r'desc|description|summary|abstract|excerpt|synopsis|overview|content|text|body', re.I),
        re.compile(r'detail|info|bio|about', re.I),
    ],
    'price': [
        re.compile(r'price|prix|cost|amount|tarif|rate|fee', re.I),
        re.compile(r'sale|promo|discount|reduction', re.I),
    ],
    'date': [
        re.compile(r'date|time|posted|publish|created|updated|when|timestamp', re.I),
        re.compile(r'ago|since|schedule', re.I),
    ],
    'image': [
        re.compile(r'image|img|photo|picture|thumb|avatar|cover|poster|banner', re.I),
        re.compile(r'media|visual|illustration', re.I),
    ],
    'link': [
        re.compile(r'link|url|href|anchor|nav|permalink', re.I),
        re.compile(r'btn|button|action|cta', re.I),
    ],
    'tag': [
        re.compile(r'tag|label|badge|category|categ|genre|type|status|chip', re.I),
    ],
    'rating': [
        re.compile(r'rating|rate|score|star|review|note|vote', re.I),
    ],
    'author': [
        re.compile(r'author|auteur|writer|creator|user|member|artist', re.I),
    ],
}


@dataclass
class DetectedContentClass:
    """Une classe CSS détectée comme porteuse de contenu intéressant."""
    selector: str                      # sélecteur CSS exact (.classe ou tag.classe)
    category: str                      # title, description, price, date, image, link, tag, rating, author
    sample_text: str                   # premier sample (rétrocompat)
    count: int                         # nombre d'éléments matchés sur la page
    class_name: str                    # nom brut de la classe
    samples: list[str] = field(default_factory=list)  # tous les samples (max 10)

    def to_dict(self) -> dict[str, Any]:
        return {
            'selector': self.selector,
            'category': self.category,
            'sample_text': self.sample_text,
            'count': self.count,
            'class_name': self.class_name,
            'samples': self.samples,
        }


@dataclass
class DetectedGroup:
    """
    Un "groupe" d'éléments qui vont ensemble dans le HTML : un parent qui
    se répète N fois sur la page, chaque instance contenant le même jeu
    de classes enfants. Typiquement une carte-produit, une ligne de
    tableau résultat, un item de liste complexe.

    Exemple FFE en mode Liste :
      <div class="result-row">
        <span class="concours-name">...</span>
        <span class="concours-date">...</span>
        <a   class="concours-link">...</a>
      </div>
    → parent=.result-row (× N), enfants=[.concours-name, .concours-date,
      .concours-link]

    Le user peut cocher TOUT le groupe en un clic pour extraire les N
    enfants en colonnes distinctes dans le CSV.
    """
    parent_selector: str         # sélecteur CSS du container (.result-row)
    children:        list[str]   # sélecteurs des enfants communs (.concours-name, …)
    count:           int         # nombre d'instances du parent
    sample_preview:  str         # aperçu du premier parent (truncated)

    def to_dict(self) -> dict[str, Any]:
        return {
            'parent_selector': self.parent_selector,
            'children':        self.children,
            'count':           self.count,
            'sample_preview':  self.sample_preview,
        }


@dataclass
class DetectedContentPayload:
    """Payload retourné par ContentClassDetector.detect()."""
    classes: list[DetectedContentClass] = field(default_factory=list)
    groups:  list[DetectedGroup]        = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            'classes': [c.to_dict() for c in self.classes],
            'groups':  [g.to_dict() for g in self.groups],
        }


class ContentClassDetector:
    """
    Analyse une page et détecte les classes CSS contenant du contenu
    intéressant (titres, descriptions, prix, etc.).
    """

    def __init__(self, crawler: Crawler | None = None) -> None:
        self._crawler = crawler or Crawler()
        self._log = CrawlerLogger.get_instance()

    def detect(self, url: str, force_dynamic: bool = False) -> DetectedContentPayload:
        """Fetch la page et détecte les classes de contenu."""
        self._log.info(f'Détection des classes de contenu → {url}')

        original_force = self._crawler._force_dynamic
        if force_dynamic:
            self._crawler._force_dynamic = True
        try:
            html = self._crawler._fetch(url)
        finally:
            self._crawler._force_dynamic = original_force

        soup = BeautifulSoup(html, 'html.parser')
        detected = self._scan_classes(soup)

        # Ajoute les classes de cellules de calendrier navigables (patterns
        # trop courts ou trop génériques pour les regex de catégorie) —
        # typiquement <td class="on"><a href="..."> sur FFE SIF.
        existing_selectors = {c.selector for c in detected}
        for calendar_cls in self._scan_calendar_cells(soup):
            if calendar_cls.selector not in existing_selectors:
                detected.append(calendar_cls)
                existing_selectors.add(calendar_cls.selector)

        # Ajoute les tables ayant un id ou une classe utile, avec leur
        # sélecteur ` ... tr` ready-to-use. Couvre les cas où la table
        # cible n'a pas de classe sémantique sur ses cellules (FFE
        # Telemat : `<table id="t_engts">` rempli de <td> sans classe).
        # Sans ce scan, le user devait deviner et taper `#t_engts tr`
        # à la main — maintenant le picker le propose en clic direct.
        for table_cls in self._scan_tables(soup):
            if table_cls.selector not in existing_selectors:
                detected.append(table_cls)
                existing_selectors.add(table_cls.selector)

        # Tri final : table et calendrier en tête, puis par nombre d'éléments
        _CATEGORY_ORDER = {
            'table': 0, 'calendar': 0, 'title': 1, 'description': 2, 'price': 3, 'date': 4,
            'image': 5, 'link': 6, 'tag': 7, 'rating': 8, 'author': 9,
        }
        detected.sort(key=lambda r: (_CATEGORY_ORDER.get(r.category, 99), -r.count))

        # Détecte les groupes (parent répétitif avec enfants communs).
        groups = self._scan_groups(soup)

        self._log.success(
            f'  → {len(detected)} classe(s), {len(groups)} groupe(s) détecté(s)'
        )
        return DetectedContentPayload(classes=detected, groups=groups)

    def _scan_groups(self, soup: BeautifulSoup) -> list['DetectedGroup']:
        """
        Cherche les "patterns de répétition" : une classe de parent qui
        apparaît ≥ 3 fois sur la page, dont chaque instance contient
        ~le même jeu de classes enfants.

        Heuristique :
          - Parent classes candidates : len ≥ 3 fois sur la page
          - Pour chaque parent, collecte les classes enfants présentes
          - Classe enfant retenue si elle apparaît dans ≥ 60% des parents
            (commune — pas juste un cas isolé)
          - Groupe retenu si ≥ 2 enfants communs (sinon pas vraiment un
            "groupe" riche)

        Retourne la liste des groupes triés par count décroissant.
        """
        by_class: dict[str, list[Tag]] = {}
        for el in soup.find_all(attrs={'class': True}):
            if not isinstance(el, Tag):
                continue
            for cls in el.get('class', []):
                if not cls or len(cls) < 2 or '[' in cls:
                    continue
                by_class.setdefault(cls, []).append(el)

        groups: list[DetectedGroup] = []

        for parent_cls, instances in by_class.items():
            if len(instances) < 3:
                continue

            # On ne garde que les instances qui sont vraiment des
            # conteneurs (ont au moins 2 enfants avec classes).
            rich = [
                inst for inst in instances
                if sum(1 for c in inst.find_all(attrs={'class': True})) >= 2
            ]
            if len(rich) < 3:
                continue

            # Compte les classes enfants dans les instances riches
            child_counts: dict[str, int] = {}
            for parent in rich:
                seen: set[str] = set()
                for child in parent.find_all(attrs={'class': True}):
                    for c in child.get('class', []):
                        if not c or len(c) < 2 or '[' in c:
                            continue
                        if c == parent_cls:
                            continue
                        seen.add(c)
                for c in seen:
                    child_counts[c] = child_counts.get(c, 0) + 1

            threshold = max(2, int(len(rich) * 0.6))
            common = sorted(
                (c for c, n in child_counts.items() if n >= threshold),
                key=lambda c: -child_counts[c],
            )
            # Limite à 8 enfants pour ne pas exploser le payload
            common = common[:8]
            if len(common) < 2:
                continue

            # Preview : texte truncé du premier parent riche
            preview = rich[0].get_text(' ', strip=True)[:120]

            groups.append(DetectedGroup(
                parent_selector = f'.{parent_cls}',
                children        = [f'.{c}' for c in common],
                count           = len(rich),
                sample_preview  = preview,
            ))

        groups.sort(key=lambda g: -g.count)
        # Cap à 10 groupes pour ne pas submerger l'UI
        return groups[:10]

    def _scan_tables(self, soup: BeautifulSoup) -> list[DetectedContentClass]:
        """
        Détecte les `<table>` qui contiennent ≥ 2 lignes de données et
        qui ont un identifiant exploitable (id ou classe sémantique).

        Pourquoi ce scan dédié :
          - `_scan_classes` ne propose que des sélecteurs `.classe` —
            jamais `#id tr` ni `table.classe tr`.
          - Les tables FFE (`#t_engts`, `#t_lst_epr`) n'ont AUCUNE
            classe sémantique sur leurs `<td>` — uniquement des
            `style="text-align:center"`. Le picker passait à côté.
          - Bonus : on tolère l'absence de `<tbody>` (cas text/xml).
            Les `<tr>` directement enfants de `<table>` sont comptés.

        Pour chaque table éligible, on émet UN sélecteur prêt à coller
        dans le data_selector : `#monid tr` ou `table.maclasse tr`.
        Les rows d'en-tête (`<thead><tr>`) sont visibles dans le sample
        mais comptées en moins via `count = total - thead_count`.
        """
        results: list[DetectedContentClass] = []

        for table in soup.find_all('table'):
            if not isinstance(table, Tag):
                continue

            # Identifiant exploitable : id en priorité, sinon 1ère classe
            # sémantique (filtre Bootstrap/Tailwind utilitaires).
            table_id = table.get('id') or ''
            table_classes = [
                c for c in (table.get('class') or [])
                if c and len(c) >= 3
                and c not in {'table', 'tbody', 'thead', 'fallback-table'}
                and not c.startswith(('mt-', 'mb-', 'p-', 'd-', 'w-', 'h-'))
            ]

            if table_id:
                anchor = f'#{table_id}'
            elif table_classes:
                anchor = f'table.{table_classes[0]}'
            else:
                continue  # rien d'exploitable

            # Compte les rows DATA : tous les <tr> sauf ceux dans <thead>.
            # On utilise find_all(recursive=True) pour couvrir le cas
            # avec ET sans <tbody> explicite.
            all_trs    = table.find_all('tr')
            thead_trs  = []
            thead      = table.find('thead')
            if thead:
                thead_trs = thead.find_all('tr')
            data_trs = [tr for tr in all_trs if tr not in thead_trs]

            if len(data_trs) < 2:
                continue  # pas assez de données pour mériter une suggestion

            # Selector final : `<anchor> tr`. Pas `tbody tr` car on veut
            # justement éviter ce piège (tbody manquant en text/xml).
            selector = f'{anchor} tr'

            # Samples : texte plat des 5 premières rows data, tronqués.
            samples: list[str] = []
            for tr in data_trs[:5]:
                txt = tr.get_text(' ', strip=True)
                if txt:
                    samples.append(txt[:120])

            if not samples:
                continue

            results.append(DetectedContentClass(
                selector    = selector,
                category    = 'table',
                sample_text = samples[0],
                count       = len(data_trs),
                class_name  = table_id or (table_classes[0] if table_classes else ''),
                samples     = samples,
            ))

        return results

    def _scan_calendar_cells(self, soup: BeautifulSoup) -> list[DetectedContentClass]:
        """
        Détecte les classes de cellules de calendrier navigables.

        Heuristique : un `<td class="X">` (ou `<li class="X">`) qui
        contient un `<a href>` et qui apparaît en plusieurs exemplaires
        (≥ 3) sur la page est probablement un jour/item navigable d'un
        calendrier, d'un planning ou d'une liste paginée.

        Cas cible : FFE SIF utilise `<td class="on"><a>16</a></td>` pour
        chaque jour du calendrier qui a au moins un concours. Le nom de
        classe "on" est trop court et trop générique pour matcher les
        regex de _CATEGORY_PATTERNS — sans ce scan dédié, le user ne
        verrait jamais cette classe proposée dans le picker.
        """
        by_class: dict[str, list[Tag]] = {}

        for cell in soup.find_all(['td', 'li', 'div', 'span']):
            if not isinstance(cell, Tag):
                continue
            classes = cell.get('class', [])
            if not classes:
                continue
            # Doit contenir un <a> avec href cliquable
            a = cell.find('a')
            if not a or not a.get('href'):
                continue
            for cls in classes:
                if not cls or len(cls) < 2 or '[' in cls or ']' in cls:
                    continue
                by_class.setdefault(cls, []).append(cell)

        results: list[DetectedContentClass] = []
        for cls, cells in by_class.items():
            # Au moins 3 cellules pour qu'on suspecte un calendrier /
            # une liste navigable (évite les faux positifs isolés).
            if len(cells) < 3:
                continue

            # Skip si cette classe a déjà été cataloguée par _scan_classes
            # (priorité aux catégories spécifiques comme link/tag)
            selector = f'.{cls}'

            samples: list[str] = []
            for cell in cells[:10]:
                text = cell.get_text(' ', strip=True)
                if text:
                    samples.append(text[:60])

            if not samples:
                continue

            results.append(DetectedContentClass(
                selector    = selector,
                category    = 'calendar',
                sample_text = samples[0],
                count       = len(cells),
                class_name  = cls,
                samples     = samples,
            ))

        return results

    def _scan_classes(self, soup: BeautifulSoup) -> list[DetectedContentClass]:
        """Scanne toutes les classes du DOM et identifie celles qui matchent."""
        results: list[DetectedContentClass] = []
        seen_selectors: set[str] = set()

        # Collecter tous les éléments avec des classes
        for el in soup.find_all(attrs={'class': True}):
            if not isinstance(el, Tag):
                continue

            classes = el.get('class', [])
            if not classes:
                continue

            for cls in classes:
                if not cls or len(cls) < 2:
                    continue
                # Ignore les classes Tailwind avec crochets (ex: text-[0.8em])
                if '[' in cls or ']' in cls:
                    continue

                # Tester chaque catégorie
                for category, patterns in _CATEGORY_PATTERNS.items():
                    if any(p.search(cls) for p in patterns):
                        selector = f'.{cls}'
                        if selector in seen_selectors:
                            break
                        seen_selectors.add(selector)

                        # Compter les éléments avec cette classe
                        elements = soup.find_all(class_=cls)
                        count = len(elements)
                        if count == 0:
                            break

                        # Extraire plusieurs samples pour preview (max 10)
                        samples: list[str] = []
                        for el_sample in elements[:10]:
                            s = self._extract_sample(el_sample, category)
                            if s and s not in samples:
                                samples.append(s[:150])

                        if not samples:
                            break

                        results.append(DetectedContentClass(
                            selector=selector,
                            category=category,
                            sample_text=samples[0],
                            count=count,
                            class_name=cls,
                            samples=samples,
                        ))
                        break  # une classe = une seule catégorie

        # On dédoublonne et trie en dehors, dans detect() après merge
        # avec les cellules de calendrier.
        return results

    def _extract_sample(self, el: Tag, category: str) -> str:
        """Extrait un sample de texte ou src d'un élément pour la preview."""
        if category == 'image':
            # Pour les images, prendre le src
            img = el if el.name == 'img' else el.find('img')
            if img:
                return img.get('src', '') or img.get('data-src', '') or ''
            return ''

        # Pour tout le reste, prendre le texte
        text = el.get_text(strip=True, separator=' ')
        if not text:
            # Peut-être un lien
            a = el.find('a')
            if a:
                text = a.get_text(strip=True)
        return text
