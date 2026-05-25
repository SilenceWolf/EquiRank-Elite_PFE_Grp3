# ============================================================
# © 2026 Louis Guillory (Silence Wolf) — Tous droits réservés
# Licence : PROPRIETARY — voir fichier LICENSE
# ============================================================

# FieldDetector : inspecte une page et renvoie la liste des champs
# remplissables (formulaires, filtres hors-form, barres de recherche,
# sélecteurs de date). Le front utilise ce payload pour afficher
# dynamiquement les options à remplir avant de lancer le crawl.

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing      import Any

from bs4 import BeautifulSoup, Tag

from log     import CrawlerLogger
from crawler import Crawler


# ── Payloads ────────────────────────────────────────────────────────────────

@dataclass
class FieldOption:
    value: str
    label: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class DetectedField:
    """Un champ unique détecté sur la page (filtre, select, input de recherche...)."""
    selector:    str                         # sélecteur CSS approximatif pour info
    label:       str                         # label lisible (depuis label/placeholder/name)
    type:        str                         # 'select' | 'date' | 'text' | 'checkbox' | 'radio' | 'search'
    name:        str = ''                    # attribut name
    options:     list[FieldOption] = field(default_factory=list)
    placeholder: str | None        = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'selector':    self.selector,
            'label':       self.label,
            'type':        self.type,
            'name':        self.name,
            'options':     [o.to_dict() for o in self.options],
            'placeholder': self.placeholder,
        }


@dataclass
class FormDescriptor:
    """Un formulaire complet avec ses champs."""
    action: str
    method: str
    fields: list[DetectedField] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            'action': self.action,
            'method': self.method,
            'fields': [f.to_dict() for f in self.fields],
        }


@dataclass
class DetectedFieldsPayload:
    """Payload complet renvoyé par FieldDetector.inspect()."""
    forms:        list[FormDescriptor] = field(default_factory=list)
    filters:      list[DetectedField]  = field(default_factory=list)
    search_boxes: list[DetectedField]  = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            'forms':        [f.to_dict() for f in self.forms],
            'filters':      [f.to_dict() for f in self.filters],
            'search_boxes': [f.to_dict() for f in self.search_boxes],
        }


# ── Détecteur ───────────────────────────────────────────────────────────────

# Patterns pour reconnaître une barre de recherche même hors formulaire
_SEARCH_NAME_RE = re.compile(r'search|recherche|query|\bq\b|keyword|filter', re.I)

# Patterns pour reconnaître un input texte qui est SÉMANTIQUEMENT une date
# (calendrier JS custom, pas <input type="date"> natif). Classique sur les
# sites anciens comme FFE : name="deb", "fin", "date_debut", etc.
_DATE_NAME_RE = re.compile(
    r'^(deb(ut)?|fin|date(_(debut|fin|de|depart|arriv[ée]e))?|du|au|start|end|from|to)$',
    re.I,
)

# Classes/ids CSS typiques d'un dropdown-menu (Bootstrap et variantes)
_DROPDOWN_MENU_CLASS_RE = re.compile(
    r'\b(dropdown-menu|menu-list|select-dropdown|options-list)\b',
    re.I,
)

# Conteneurs qu'on considère comme de la navigation site — les dropdowns
# qui y vivent ne sont PAS des filtres de recherche, on les skip.
_NAV_ANCESTOR_RE = re.compile(
    r'\b(nav|navbar|header|topbar|site-?menu|main-?menu|primary-?menu)\b',
    re.I,
)

# Un dropdown avec moins de X options est sans doute un menu de navigation
# (7 points standards) et pas un filtre (régions = 21, départements = 100).
_MIN_DROPDOWN_OPTIONS = 5
_MAX_DROPDOWN_OPTIONS = 500   # garde-fou pour ne pas exploser un payload

# Classes CSS qui désignent un bloc de libellé autour d'un champ
# (card-header Bootstrap, form-label BS5, legend, etc.). Utilisé par
# _find_contextual_label quand aucun <label> natif n'est présent.
_LABEL_CLASS_RE = re.compile(
    r'\b(card-header|card-title|form-label|field-label|input-label|'
    r'control-label|field-title|legend|col-form-label)\b',
    re.I,
)

# Placeholders typiques de dates à ignorer comme label humain (on ne
# veut pas afficher "jj/mm/aaaa" comme intitulé de champ).
_DATE_PLACEHOLDER_RE = re.compile(
    r'^(jj/mm/aaaa|dd/mm/yyyy|mm/dd/yyyy|yyyy-mm-dd|aaaa-mm-jj)$',
    re.I,
)


def _is_date_placeholder(s: str) -> bool:
    return bool(_DATE_PLACEHOLDER_RE.match(s.strip()))


class FieldDetector:
    """
    Inspecte une page et produit un DetectedFieldsPayload.

    Réutilise le fetch du Crawler pour la cohérence (auto-detect JS,
    headers, cookies). Ne modifie pas le Crawler — juste appelé en lecture.
    """

    def __init__(self, crawler: Crawler | None = None) -> None:
        # Crawler injectable pour les tests ; en prod on en crée un par défaut
        self._crawler = crawler or Crawler()
        self._log     = CrawlerLogger.get_instance()

    def inspect(self, url: str, force_dynamic: bool = False) -> DetectedFieldsPayload:
        """
        Fetche la page et extrait tous les champs remplissables.

        force_dynamic : si True, force playwright même si la page semble statique.
        Utile pour les SPAs qu'on n'arrive pas à auto-détecter.
        """
        self._log.info(f'Inspection des champs → {url}')

        # Bascule temporaire en mode dynamique si demandé
        original_force = self._crawler._force_dynamic
        if force_dynamic:
            self._crawler._force_dynamic = True
        try:
            html = self._crawler._fetch(url)
        finally:
            self._crawler._force_dynamic = original_force

        soup = BeautifulSoup(html, 'html.parser')

        forms        = self._extract_forms(soup)
        filters      = self._extract_standalone_filters(soup)
        filters     += self._extract_custom_dropdowns(soup)
        search_boxes = self._extract_search_boxes(soup)

        self._log.success(
            f'  → {len(forms)} form(s), {len(filters)} filtre(s), '
            f'{len(search_boxes)} barre(s) de recherche'
        )

        # Log détaillé avec le sélecteur CSS (inclut les classes)
        for i, form in enumerate(forms):
            self._log.info(f'  form[{i}] action={form.action} method={form.method}')
            for f in form.fields:
                self._log.info(f'    ├ {f.type:<10} name={f.name!r:<20} selector={f.selector}')
        for f in filters:
            self._log.info(f'  filtre   {f.type:<10} name={f.name!r:<20} selector={f.selector}')
        for f in search_boxes:
            self._log.info(f'  search   {f.type:<10} name={f.name!r:<20} selector={f.selector}')

        return DetectedFieldsPayload(
            forms        = forms,
            filters      = filters,
            search_boxes = search_boxes,
        )

    # ── Formulaires ────────────────────────────────────────────────────────

    def _extract_forms(self, soup: BeautifulSoup) -> list[FormDescriptor]:
        forms: list[FormDescriptor] = []
        for form in soup.find_all('form'):
            descriptor = FormDescriptor(
                action = form.get('action', '') or '',
                method = (form.get('method') or 'GET').upper(),
                fields = [],
            )

            # Inputs textuels et numériques
            for inp in form.find_all('input'):
                field_type = (inp.get('type') or 'text').lower()
                if field_type in ('submit', 'button', 'reset', 'hidden', 'image', 'file'):
                    continue

                name_attr = inp.get('name', '') or ''
                if field_type == 'checkbox':
                    kind = 'checkbox'
                elif field_type == 'radio':
                    kind = 'radio'
                elif field_type == 'date':
                    kind = 'date'
                elif _DATE_NAME_RE.match(name_attr):
                    # Input texte dont le name indique clairement une date
                    # (deb, fin, date_debut…). Classique sur les vieux sites
                    # qui utilisent un date picker JS à la place du type=date.
                    kind = 'date'
                elif field_type == 'search' or bool(_SEARCH_NAME_RE.search(name_attr)):
                    kind = 'search'
                else:
                    kind = 'text'

                descriptor.fields.append(DetectedField(
                    selector    = self._selector_hint(inp),
                    label       = self._find_label_for(inp, soup)
                                   or inp.get('placeholder', '')
                                   or inp.get('name', ''),
                    type        = kind,
                    name        = inp.get('name', ''),
                    placeholder = inp.get('placeholder'),
                ))

            # Selects
            for sel in form.find_all('select'):
                options = [
                    FieldOption(
                        value = opt.get('value') or opt.get_text(strip=True),
                        label = opt.get_text(strip=True),
                    )
                    for opt in sel.find_all('option')
                ]
                descriptor.fields.append(DetectedField(
                    selector = self._selector_hint(sel),
                    label    = self._find_label_for(sel, soup) or sel.get('name', ''),
                    type     = 'select',
                    name     = sel.get('name', ''),
                    options  = options,
                ))

            # Textareas
            for ta in form.find_all('textarea'):
                descriptor.fields.append(DetectedField(
                    selector    = self._selector_hint(ta),
                    label       = self._find_label_for(ta, soup) or ta.get('name', ''),
                    type        = 'text',
                    name        = ta.get('name', ''),
                    placeholder = ta.get('placeholder'),
                ))

            if descriptor.fields:
                forms.append(descriptor)

        return forms

    # ── Filtres hors-form ──────────────────────────────────────────────────

    def _extract_standalone_filters(self, soup: BeautifulSoup) -> list[DetectedField]:
        """
        Détecte les <select> et inputs de filtrage qui ne sont PAS dans un
        <form> : souvent des filtres JS qui déclenchent un fetch AJAX.
        """
        filters: list[DetectedField] = []

        # Selects orphelins : pareil que les dropdowns Bootstrap, on
        # utilise le label comme value — stable d'une page à l'autre
        # là où l'attribut value peut être un code volatile (ou encoder
        # un token de session).
        for sel in soup.find_all('select'):
            if sel.find_parent('form'):
                continue
            options = [
                FieldOption(
                    value = opt.get_text(strip=True) or opt.get('value', ''),
                    label = opt.get_text(strip=True),
                )
                for opt in sel.find_all('option')
            ]
            filters.append(DetectedField(
                selector = self._selector_hint(sel),
                label    = self._find_label_for(sel, soup) or sel.get('name', '') or 'filtre',
                type     = 'select',
                name     = sel.get('name', ''),
                options  = options,
            ))

        # Inputs date orphelins
        for inp in soup.find_all('input', attrs={'type': 'date'}):
            if inp.find_parent('form'):
                continue
            filters.append(DetectedField(
                selector    = self._selector_hint(inp),
                label       = self._find_label_for(inp, soup) or inp.get('name', '') or 'date',
                type        = 'date',
                name        = inp.get('name', ''),
                placeholder = inp.get('placeholder'),
            ))

        return filters

    # ── Barres de recherche ────────────────────────────────────────────────

    def _extract_search_boxes(self, soup: BeautifulSoup) -> list[DetectedField]:
        """
        Détecte les inputs qui ressemblent à une barre de recherche :
        role="searchbox", type="search", ou name/placeholder qui match
        le pattern /search|recherche|query|q/.
        """
        search_boxes: list[DetectedField] = []
        seen: set[int] = set()

        # Collecter les inputs déjà capturés dans les formulaires
        form_inputs: set[int] = set()
        for form in soup.find_all('form'):
            for inp in form.find_all('input'):
                form_inputs.add(id(inp))

        for inp in soup.find_all('input'):
            if id(inp) in seen or id(inp) in form_inputs:
                continue

            field_type = (inp.get('type') or 'text').lower()
            role       = (inp.get('role') or '').lower()
            name       = inp.get('name', '') or ''
            placeholder = inp.get('placeholder', '') or ''

            is_search = (
                field_type == 'search'
                or role == 'searchbox'
                or bool(_SEARCH_NAME_RE.search(name))
                or bool(_SEARCH_NAME_RE.search(placeholder))
            )
            if not is_search:
                continue

            seen.add(id(inp))
            search_boxes.append(DetectedField(
                selector    = self._selector_hint(inp),
                label       = self._find_label_for(inp, soup)
                               or placeholder
                               or name
                               or 'recherche',
                type        = 'search',
                name        = name,
                placeholder = placeholder or None,
            ))

        return search_boxes

    # ── Dropdowns custom (Bootstrap et variantes) ─────────────────────────

    def _extract_custom_dropdowns(self, soup: BeautifulSoup) -> list[DetectedField]:
        """
        Détecte les dropdowns "déguisés" : un bouton avec
        data-toggle="dropdown" (Bootstrap) ou role="button" qui ouvre une
        liste d'<a class="dropdown-item"> ou similaire.

        Pattern classique (FFE, Bootstrap 4+) :
            <button data-toggle="dropdown">Toutes régions</button>
            <div class="dropdown-menu">
              <a class="dropdown-item" href="./?cs=4.XXX">Bretagne</a>
              <a class="dropdown-item" href="./?cs=4.YYY">Normandie</a>
              ...
            </div>

        On filtre :
          - les toggles qui sont dans la navigation du site (nav, navbar…)
            parce que ce ne sont pas des filtres de recherche
          - les dropdowns avec moins de _MIN_DROPDOWN_OPTIONS éléments
            (probablement un menu plutôt qu'un filtre)
          - les dupliqués (même liste d'options qu'un dropdown déjà trouvé)
        """
        filters: list[DetectedField] = []
        seen_fingerprints: set[str] = set()

        # Toggle candidates : tout ce qui ressemble à un ouvreur de dropdown
        toggles = soup.find_all(
            ['button', 'a', 'div', 'span'],
            attrs={'data-toggle': re.compile(r'^dropdown$', re.I)},
        )
        # Variantes Bootstrap 5 (data-bs-toggle="dropdown")
        toggles += soup.find_all(
            ['button', 'a', 'div', 'span'],
            attrs={'data-bs-toggle': re.compile(r'^dropdown$', re.I)},
        )

        for toggle in toggles:
            if self._is_in_navigation(toggle):
                continue

            menu = self._find_dropdown_menu(toggle)
            if menu is None:
                continue

            items = menu.find_all('a')
            if len(items) < _MIN_DROPDOWN_OPTIONS:
                continue

            # Construit les options — on limite pour éviter des payloads
            # monstrueux sur les pages à listes géantes.
            #
            # VALUE = LABEL : pour les dropdowns de type navigation
            # (Bootstrap / FFE SIF), le href encode un token CSRF qui
            # CHANGE entre pages, donc on ne peut pas s'en servir comme
            # clé stable (le même choix "Equifun" a un href différent sur
            # le listing vs le détail). Le LABEL, lui, est stable —
            # c'est ce que l'utilisateur identifie et ce que fetch_
            # filtered_html peut retrouver par recherche d'ancre.
            options: list[FieldOption] = []
            for item in items[:_MAX_DROPDOWN_OPTIONS]:
                text = item.get_text(' ', strip=True)
                if not text:
                    continue
                options.append(FieldOption(value=text, label=text))

            if len(options) < _MIN_DROPDOWN_OPTIONS:
                continue

            # Empreinte pour dédupliquer : signature des N premiers labels.
            # FFE peut rendre le même dropdown à plusieurs endroits (menu
            # mobile + menu desktop).
            fingerprint = '|'.join(o.label for o in options[:10])
            if fingerprint in seen_fingerprints:
                continue
            seen_fingerprints.add(fingerprint)

            label = self._derive_dropdown_label(toggle, menu, options)
            filters.append(DetectedField(
                selector = self._selector_hint(toggle),
                label    = label,
                type     = 'select',
                # name dérivé du label — sert de clé stable dans
                # field_values côté front
                name     = self._slugify_name(label),
                options  = options,
            ))

        return filters

    def _derive_dropdown_label(
        self,
        toggle:  Tag,
        menu:    Tag,
        options: list['FieldOption'],
    ) -> str:
        """
        Trouve un libellé humain lisible pour un dropdown. Cascade de
        fallbacks — l'utilisateur doit toujours voir autre chose que
        "dropdown" dans l'UI.

        Ordre :
          1. texte direct du bouton (cas idéal : "Toutes régions")
          2. aria-label / title / data-label sur le bouton
          3. texte du premier item marqué actif / selected (la plupart
             des dropdowns "filtres" ont une option "Toutes X" en 1er,
             marquée active, qui décrit parfaitement le filtre)
          4. texte du premier item (fallback)
          5. forme générique "Filtre (N options)"
        """
        # 1. Texte du bouton (idéal)
        text = toggle.get_text(' ', strip=True)
        if text:
            return text[:60]

        # 2. Attributs accessibilité du bouton
        for attr in ('aria-label', 'title', 'data-label', 'data-original-title'):
            val = toggle.get(attr, '') or ''
            val = val.strip()
            if val:
                return val[:60]

        # 3. Premier item actif du menu (ex: "Toutes régions" avec .active)
        for item in menu.find_all('a'):
            classes = ' '.join(item.get('class') or [])
            if re.search(r'\b(active|selected|current|default|tous|all)\b', classes, re.I):
                t = item.get_text(' ', strip=True)
                if t:
                    return t[:60]

        # 4. Premier item tout court
        if options:
            return options[0].label[:60]

        # 5. Fallback générique
        return f'filtre ({len(options)} options)'

    def _find_dropdown_menu(self, toggle: Tag) -> Tag | None:
        """
        Trouve le <div class="dropdown-menu"> associé à un toggle.
        Trois emplacements courants :
          1. sibling direct qui suit
          2. descendant du parent immédiat (.btn-group, .dropdown)
          3. descendant du grand-parent si .dropdown est wrappé
        """
        # 1) Sibling direct
        nxt = toggle.find_next_sibling()
        if isinstance(nxt, Tag) and self._is_dropdown_menu(nxt):
            return nxt

        # 2) Parent immédiat
        parent = toggle.parent
        if isinstance(parent, Tag):
            for child in parent.find_all(True, recursive=False):
                if child is toggle:
                    continue
                if self._is_dropdown_menu(child):
                    return child
            # chercher aussi en profondeur dans le parent
            menu = parent.find(
                lambda t: isinstance(t, Tag) and self._is_dropdown_menu(t),
            )
            if menu is not None:
                return menu

        # 3) Grand-parent
        if isinstance(parent, Tag) and isinstance(parent.parent, Tag):
            menu = parent.parent.find(
                lambda t: isinstance(t, Tag) and self._is_dropdown_menu(t),
            )
            if menu is not None:
                return menu

        return None

    def _is_dropdown_menu(self, el: Tag) -> bool:
        """Retourne True si l'élément ressemble à un dropdown-menu."""
        if not isinstance(el, Tag):
            return False
        classes = ' '.join(el.get('class') or [])
        return bool(_DROPDOWN_MENU_CLASS_RE.search(classes))

    def _is_in_navigation(self, el: Tag) -> bool:
        """
        True si l'élément est dans un contexte de navigation site
        (ne pas confondre avec un filtre de recherche).
        """
        node: Tag | None = el
        depth = 0
        while node is not None and depth < 8:
            if isinstance(node, Tag):
                if node.name == 'nav':
                    return True
                classes = ' '.join(node.get('class') or [])
                node_id  = node.get('id', '') or ''
                if _NAV_ANCESTOR_RE.search(classes) or _NAV_ANCESTOR_RE.search(node_id):
                    return True
            node = node.parent
            depth += 1
        return False

    def _slugify_name(self, text: str) -> str:
        """Convertit un libellé en clé stable utilisable comme name de champ."""
        slug = re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')
        return slug[:40] or 'dropdown'

    # ── Helpers ────────────────────────────────────────────────────────────

    def _selector_hint(self, el: Tag) -> str:
        """Produit un sélecteur CSS approximatif lisible par un humain."""
        parts = [el.name or '']
        el_id = el.get('id')
        if el_id:
            parts.append(f'#{el_id}')
        classes = el.get('class') or []
        if classes:
            parts.append('.' + '.'.join(classes[:2]))
        name = el.get('name')
        if name:
            parts.append(f'[name="{name}"]')
        return ''.join(parts) or el.name or ''

    def _find_label_for(self, el: Tag, soup: BeautifulSoup) -> str:
        """
        Trouve un libellé humain pour un champ. Ordre :
          1. <label for="id"> où id correspond
          2. <label> ancêtre direct
          3. frère précédent <label> / <span>
          4. aria-label / title sur l'élément lui-même
          5. Titre de carte Bootstrap ou équivalent : un ancêtre proche
             contient un élément avec class ~ 'card-header|form-label|
             field-label|legend|title' OU une balise <legend> / <h1-h6>
             à proximité. Classique sur les vieux back-office comme
             FFE où "Numéro de concours" est dans <div class="card-header">
             plutôt que dans un <label>.
          6. placeholder (humanisé)
        """
        # 1. label[for=id]
        el_id = el.get('id')
        if el_id:
            label = soup.find('label', attrs={'for': el_id})
            if label:
                text = label.get_text(' ', strip=True)
                if text:
                    return text

        # 2. <label> ancêtre
        parent = el.parent
        depth  = 0
        while parent is not None and depth < 3:
            if isinstance(parent, Tag) and parent.name == 'label':
                text = parent.get_text(' ', strip=True)
                if text:
                    return text
            parent = parent.parent
            depth += 1

        # 3. frère précédent
        prev = el.find_previous_sibling(['label', 'span'])
        if prev:
            text = prev.get_text(' ', strip=True)
            if text and len(text) < 80:
                return text

        # 4. aria-label / title sur l'élément
        for attr in ('aria-label', 'title', 'data-label'):
            val = (el.get(attr, '') or '').strip()
            if val:
                return val[:80]

        # 5. Titre de carte / legend / heading dans les ancêtres
        label_from_context = self._find_contextual_label(el)
        if label_from_context:
            return label_from_context

        # 6. placeholder humanisé
        placeholder = (el.get('placeholder', '') or '').strip()
        if placeholder and not _is_date_placeholder(placeholder):
            return placeholder[:80]

        return ''

    def _find_contextual_label(self, el: Tag) -> str:
        """
        Remonte jusqu'à 4 ancêtres et cherche parmi leurs ENFANTS DIRECTS
        un bloc qui ressemble à un libellé (card-header, legend, heading,
        *-label). On n'explore PAS en profondeur pour éviter de capturer
        le libellé d'un champ voisin — classique quand on remonte trop
        haut et qu'on tombe sur le premier <h3>/<label> de la page.

        Pattern FFE type :
            <div class="card">            ← ancêtre
              <div class="card-header">    ← frère enfant = label
              <div class="card-body">
                <input>                    ← notre champ
        """
        node = el.parent
        depth = 0
        while node is not None and depth < 4:
            if isinstance(node, Tag) and node.name not in ('body', 'html', 'main'):
                # Parcourt UNIQUEMENT les enfants directs
                for child in node.children:
                    if not isinstance(child, Tag):
                        continue
                    # Skip si c'est l'ancêtre qui contient notre champ
                    # (sinon le "label" inclurait le champ lui-même)
                    if child is el or el in list(child.descendants):
                        continue

                    # Tags sémantiques : <legend>, <h1>-<h6>
                    if child.name in ('legend', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
                        text = child.get_text(' ', strip=True)
                        if 2 <= len(text) <= 80:
                            return text

                    # Classes CSS explicites de libellé
                    classes = ' '.join(child.get('class') or [])
                    if _LABEL_CLASS_RE.search(classes):
                        text = child.get_text(' ', strip=True)
                        if 2 <= len(text) <= 80:
                            return text
            node = node.parent
            depth += 1
        return ''
