# ============================================================
# © 2026 Louis Guillory (Silence Wolf) — Tous droits réservés
# Licence : PROPRIETARY — voir fichier LICENSE
# ============================================================

# LanguageDetector : identifie la langue associée à un élément DOM.
#
# Stratégie en 3 temps (dans cet ordre) :
#   1. Drapeau ou indicateur de langue proche de l'élément
#      (frère, enfant, parent direct) — ex : <img src="fr.png">,
#      <span class="flag-fr">, hreflang sur un <a>, etc.
#   2. Attributs langue sur l'élément ou un ancêtre : lang="", xml:lang="".
#   3. Langue de la page (html[lang] ou meta) comme fallback.
#
# Si rien n'est trouvé → chaîne vide (pas 'unknown') pour ne pas polluer
# le CSV avec une valeur par défaut bruyante.

from __future__ import annotations

import re
from typing import Iterable

from bs4 import Tag


# Code ISO 639-1 vers libellé humain. On garde uniquement le code court
# dans le CSV pour rester compact ; le libellé sert au logging.
_LANG_NAMES: dict[str, str] = {
    'fr': 'french',      'en': 'english',     'de': 'german',
    'es': 'spanish',     'it': 'italian',     'pt': 'portuguese',
    'nl': 'dutch',       'ru': 'russian',     'ja': 'japanese',
    'zh': 'chinese',     'ko': 'korean',      'ar': 'arabic',
    'pl': 'polish',      'tr': 'turkish',     'vi': 'vietnamese',
    'id': 'indonesian',  'th': 'thai',        'he': 'hebrew',
    'sv': 'swedish',     'no': 'norwegian',   'da': 'danish',
    'fi': 'finnish',     'cs': 'czech',       'hu': 'hungarian',
    'ro': 'romanian',    'uk': 'ukrainian',   'el': 'greek',
    'hi': 'hindi',       'bn': 'bengali',
}

# Codes pays ISO 3166-1 alpha-2 vers langue associée principale.
# Utile quand le drapeau est un code pays (fr, us, gb, jp…) plutôt
# qu'un code langue. Liste restreinte aux cas fréquents — on préfère
# l'absence à une fausse détection.
_COUNTRY_TO_LANG: dict[str, str] = {
    'fr': 'fr', 'be': 'fr', 'qc': 'fr',
    'gb': 'en', 'uk': 'en', 'us': 'en', 'ca': 'en', 'au': 'en',
    'ie': 'en', 'nz': 'en',
    'de': 'de', 'at': 'de', 'ch': 'de',
    'es': 'es', 'mx': 'es', 'ar': 'es', 'cl': 'es', 'co': 'es',
    'it': 'it',
    'pt': 'pt', 'br': 'pt',
    'nl': 'nl',
    'ru': 'ru',
    'jp': 'ja',
    'cn': 'zh', 'tw': 'zh', 'hk': 'zh',
    'kr': 'ko',
    'sa': 'ar', 'eg': 'ar', 'ae': 'ar',
    'pl': 'pl',
    'tr': 'tr',
    'vn': 'vi',
    'id': 'id',
    'th': 'th',
    'il': 'he',
    'se': 'sv',
    'dk': 'da',
    'fi': 'fi',
    'cz': 'cs',
    'hu': 'hu',
    'ro': 'ro',
    'ua': 'uk',
    'gr': 'el',
    'in': 'hi',
}

# Patterns regex pour extraire un code depuis une classe ou un src d'image
# On cible les codes ISO à 2 lettres, avec des séparateurs usuels.
# Chaque pattern tente d'attraper le DERNIER code 2-letters dans un token
# préfixé — sinon "flag-icon-fr" matcherait sur "ic" au lieu de "fr".
_CODE_PATTERNS = [
    # Classes composées : "flag-icon-fr", "fi-fr", "flag-fr", "lang-fr",
    # "locale-fr", "country-fr", "lng-fr", "flag-icon-en-us" → on prend le
    # DERNIER groupe de 2 lettres avant la fin ou l'espace.
    re.compile(
        r'\b(?:flag|fi|lang|locale|country|lng)'
        r'(?:[-_][a-z]+)*'
        r'[-_](?P<code>[a-z]{2})(?:[-_][a-z]{2})?\b',
        re.I,
    ),
    # Séparateur espace : "flag fr", "icon fr"
    re.compile(
        r'\b(?:flag|icon|country|lang|locale)\s+(?P<code>[a-z]{2})\b',
        re.I,
    ),
]

# Noms de pays (EN et FR) vers code langue principal, utilisé pour les alt
# de drapeau type "Flag of France", "Drapeau allemand".
_COUNTRY_NAME_TO_LANG: dict[str, str] = {
    'france': 'fr',     'french': 'fr',     'français': 'fr',  'francais': 'fr',
    'england': 'en',    'english': 'en',    'anglais': 'en',
    'uk': 'en',         'britain': 'en',    'american': 'en',  'america': 'en',
    'germany': 'de',    'german': 'de',     'allemand': 'de',  'deutsch': 'de',
    'spain': 'es',      'spanish': 'es',    'espagnol': 'es',
    'italy': 'it',      'italian': 'it',    'italien': 'it',
    'portugal': 'pt',   'portuguese': 'pt', 'portugais': 'pt', 'brazil': 'pt',
    'netherlands': 'nl','dutch': 'nl',      'néerlandais': 'nl','neerlandais': 'nl',
    'russia': 'ru',     'russian': 'ru',    'russe': 'ru',
    'japan': 'ja',      'japanese': 'ja',   'japonais': 'ja',
    'china': 'zh',      'chinese': 'zh',    'chinois': 'zh',
    'korea': 'ko',      'korean': 'ko',     'coréen': 'ko',    'coreen': 'ko',
    'poland': 'pl',     'polish': 'pl',     'polonais': 'pl',
    'turkey': 'tr',     'turkish': 'tr',    'turc': 'tr',
    'vietnam': 'vi',    'vietnamese': 'vi', 'vietnamien': 'vi',
    'indonesia': 'id',  'indonesian': 'id',
    'thailand': 'th',   'thai': 'th',
    'israel': 'he',     'hebrew': 'he',     'hébreu': 'he',    'hebreu': 'he',
    'sweden': 'sv',     'swedish': 'sv',    'suédois': 'sv',   'suedois': 'sv',
    'denmark': 'da',    'danish': 'da',     'danois': 'da',
    'finland': 'fi',    'finnish': 'fi',    'finlandais': 'fi',
    'arabia': 'ar',     'arabic': 'ar',     'arabe': 'ar',
}
# Pattern pour src d'image : "/flags/fr.png", "fr.svg", "flag-fr.png"
_SRC_PATTERN = re.compile(
    r'(?:flag|flags?|country|lang|locale)[\-_/]?(?P<code>[a-z]{2})\.(?:png|svg|gif|jpe?g|webp)',
    re.I,
)
# Pattern pour alt de flag : "French", "français", "Flag of France"
_ALT_HINT = re.compile(r'\b(?P<code>[a-z]{2,3})\b|flag of (?P<name>\w+)', re.I)

# Mots-clés dans l'alt qui indiquent qu'on est bien sur un drapeau
_FLAG_HINT_WORDS = ('flag', 'drapeau', 'langue', 'language', 'lang', 'country')

# Stopwords discriminants par langue (minuscules). Sert UNIQUEMENT de
# fallback heuristique quand la page n'a ni <html lang> ni meta langue.
# On privilégie des mots courts très fréquents — articles, prépositions,
# auxiliaires, pronoms. Volontairement léger (pas de dépendance externe
# type langdetect). Couvre les langues les plus probables sur les sites
# auxquels on s'attend (FR/EN/DE/ES/IT/PT/NL/RU).
_STOPWORDS_BY_LANG: dict[str, frozenset[str]] = {
    'fr': frozenset(['le', 'la', 'les', 'de', 'du', 'des', 'un', 'une', 'et', 'est',
                     'dans', 'pour', 'sur', 'avec', 'par', 'que', 'qui', 'mais', 'ou',
                     'ce', 'cette', 'ces', 'sont', 'être', 'avoir', 'plus', 'tout',
                     'tous', 'aussi', 'très', 'vous', 'nous', 'je', 'il', 'elle',
                     'ils', 'elles', 'notre', 'votre', 'leur', 'au', 'aux', 'se',
                     'pas', 'ne', 'en', 'à', 'son', 'sa', 'ses']),
    'en': frozenset(['the', 'of', 'and', 'to', 'in', 'is', 'it', 'you', 'that', 'he',
                     'was', 'for', 'on', 'are', 'as', 'with', 'his', 'they', 'at',
                     'be', 'this', 'have', 'from', 'or', 'one', 'had', 'by', 'but',
                     'not', 'what', 'all', 'were', 'we', 'when', 'your', 'can',
                     'said', 'there', 'an', 'each', 'which', 'she', 'do', 'how',
                     'their', 'if', 'will']),
    'de': frozenset(['der', 'die', 'das', 'und', 'ist', 'ein', 'eine', 'zu', 'den',
                     'mit', 'nicht', 'von', 'auf', 'für', 'ich', 'du', 'er', 'sie',
                     'wir', 'ihr', 'war', 'sind', 'hat', 'haben', 'werden', 'aber',
                     'oder', 'wenn', 'auch', 'sehr', 'alle', 'noch', 'im', 'bei',
                     'als', 'nach', 'bis', 'über', 'unter', 'dem', 'des', 'einem']),
    'es': frozenset(['el', 'la', 'los', 'las', 'de', 'del', 'un', 'una', 'y', 'es',
                     'en', 'que', 'por', 'con', 'para', 'se', 'su', 'sus', 'este',
                     'esta', 'pero', 'como', 'más', 'son', 'ser', 'estar', 'tener',
                     'todo', 'todos', 'muy', 'yo', 'tú', 'él', 'ella', 'no', 'al',
                     'lo', 'le', 'les', 'nos', 'os']),
    'it': frozenset(['il', 'la', 'lo', 'gli', 'le', 'di', 'del', 'della', 'un', 'una',
                     'uno', 'e', 'è', 'in', 'che', 'per', 'con', 'da', 'su', 'non',
                     'ma', 'ci', 'si', 'ha', 'ho', 'sono', 'essere', 'avere', 'molto',
                     'tutti', 'questo', 'quello', 'al', 'dal', 'nel', 'sul']),
    'pt': frozenset(['o', 'a', 'os', 'as', 'de', 'do', 'da', 'dos', 'das', 'um', 'uma',
                     'e', 'é', 'em', 'com', 'por', 'para', 'que', 'não', 'se', 'mas',
                     'ou', 'como', 'seu', 'sua', 'ele', 'ela', 'eles', 'elas', 'nós',
                     'no', 'na', 'nos', 'nas']),
    'nl': frozenset(['de', 'het', 'een', 'en', 'van', 'is', 'in', 'op', 'te', 'dat',
                     'die', 'ik', 'je', 'niet', 'zijn', 'aan', 'voor', 'met', 'als',
                     'er', 'maar', 'om', 'had', 'zij', 'hij', 'bij', 'nog', 'naar']),
    'ru': frozenset(['и', 'в', 'не', 'на', 'я', 'что', 'он', 'с', 'как', 'а', 'то',
                     'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же',
                     'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было']),
}

# Nombre de mots minimum dans l'échantillon pour tenter une détection.
# En dessous on retourne '' — trop peu de signal pour être fiable.
_MIN_WORDS_FOR_HEURISTIC = 15

# Taille max de l'échantillon texte analysé, en caractères. Suffisant
# pour obtenir un verdict stable sans parcourir des pages énormes.
_TEXT_SAMPLE_SIZE = 3000

# Tokenizer : attrape mots latins + cyrilliques + diacritiques courants.
_WORD_RE = re.compile(r"[a-zàâäéèêëïîôöùûüÿçœæñáíóúüß\u0400-\u04ff']+", re.I)


class LanguageDetector:
    """
    Détecteur de langue stateless : toutes ses méthodes sont pures.
    Pas de Singleton — on en instancie un par session de crawl pour
    éviter des effets de bord entre crawls.
    """

    # ── API publique ──────────────────────────────────────────────────────

    def detect_near(self, element: Tag) -> str:
        """
        Cherche un drapeau / indicateur de langue à proximité immédiate
        de l'élément. Retourne un code ISO 639-1 ou chaîne vide.

        Règle : hreflang/data-lang ne sont lus QUE sur l'élément lui-même
        (ils qualifient cet élément, pas ses voisins). Pour les voisins,
        on ne considère que ce qui ressemble visuellement à un drapeau
        (img flag, span/i avec classe flag-*).
        """
        if not isinstance(element, Tag):
            return ''

        # 1. hreflang / data-lang / lang sur l'élément LUI-MÊME
        for attr in ('hreflang', 'data-lang', 'data-locale', 'data-language', 'lang'):
            val = element.get(attr)
            if val:
                return self._normalize_code(val)

        # 2. Drapeau visuel dans les descendants directs (img ou classe flag-*)
        for child in element.find_all(['img', 'i', 'span'], limit=5):
            code = self._code_from_flag_element(child)
            if code:
                return code

        # 3. Frère immédiat (1 avant + 1 après) : drapeau visuel uniquement
        for sibling in self._immediate_siblings(element):
            code = self._code_from_flag_element(sibling)
            if code:
                return code
            # Le drapeau peut être imbriqué dans le sibling, pas directement
            if isinstance(sibling, Tag):
                for nested in sibling.find_all(['img', 'i', 'span'], limit=3):
                    code = self._code_from_flag_element(nested)
                    if code:
                        return code

        return ''

    def detect_from_ancestors(self, element: Tag) -> str:
        """
        Remonte les ancêtres à la recherche d'un attribut lang/xml:lang.
        Retourne un code ISO 639-1 ou chaîne vide.
        """
        if not isinstance(element, Tag):
            return ''

        node: Tag | None = element
        while node is not None:
            if isinstance(node, Tag):
                for attr in ('lang', 'xml:lang'):
                    val = node.get(attr)
                    if val:
                        return self._normalize_code(val)
            node = node.parent
        return ''

    def detect_page(self, any_tag: Tag) -> str:
        """
        Extrait la langue de la page depuis <html lang="..."> ou des meta
        tags courants. Utilisé comme dernier fallback.
        """
        if not isinstance(any_tag, Tag):
            return ''

        # Remonte au document root
        root = any_tag
        while root.parent is not None:
            root = root.parent

        # <html lang="...">
        html_tag = root.find('html') if hasattr(root, 'find') else None
        if html_tag is not None and isinstance(html_tag, Tag):
            lang = html_tag.get('lang') or html_tag.get('xml:lang')
            if lang:
                return self._normalize_code(lang)

        # <meta http-equiv="content-language" content="fr">
        meta = root.find('meta', attrs={'http-equiv': re.compile(r'content-language', re.I)}) \
               if hasattr(root, 'find') else None
        if meta and meta.get('content'):
            return self._normalize_code(meta['content'])

        # <meta name="language" content="...">
        meta = root.find('meta', attrs={'name': re.compile(r'^language$', re.I)}) \
               if hasattr(root, 'find') else None
        if meta and meta.get('content'):
            return self._normalize_code(meta['content'])

        # <meta property="og:locale" content="fr_FR">
        meta = root.find('meta', attrs={'property': re.compile(r'og:locale$', re.I)}) \
               if hasattr(root, 'find') else None
        if meta and meta.get('content'):
            return self._normalize_code(meta['content'])

        # Dernier fallback : analyse heuristique du texte de la page.
        # Utile pour les sites qui ne déclarent rien côté HTML
        # (ex : FFE SIF — seulement <html translate="no">, pas de lang).
        text = root.get_text(' ', strip=True) if hasattr(root, 'get_text') else ''
        return self.detect_from_text(text)

    def detect_from_text(self, text: str) -> str:
        """
        Détection de langue basée sur la fréquence des stopwords dans
        un échantillon de texte. Stateless, pas de dépendance externe.
        Retourne '' si l'échantillon est trop court ou le score pas assez
        net pour trancher (moins de 5 matchs sur la langue gagnante,
        ou écart < 2 avec la 2e langue).
        """
        if not text:
            return ''
        sample = text.lower()[:_TEXT_SAMPLE_SIZE]
        words = _WORD_RE.findall(sample)
        if len(words) < _MIN_WORDS_FOR_HEURISTIC:
            return ''

        scores: dict[str, int] = {}
        for code, stopwords in _STOPWORDS_BY_LANG.items():
            scores[code] = sum(1 for w in words if w in stopwords)

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_code, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0

        # Il faut assez de matchs ET une avance nette sur la 2e langue.
        if best_score < 5 or (best_score - second_score) < 2:
            return ''
        return best_code

    def detect(self, element: Tag, page_lang: str = '') -> str:
        """
        Pipeline complet : drapeau proche → lang ancêtre → page fournie
        (ou détectée). Retourne toujours un code court ou chaîne vide.
        """
        code = self.detect_near(element)
        if code:
            return code

        code = self.detect_from_ancestors(element)
        if code:
            return code

        if page_lang:
            return self._normalize_code(page_lang)

        return self.detect_page(element)

    def label(self, code: str) -> str:
        """Libellé humain pour un code. Utile pour les logs — pas pour le CSV."""
        return _LANG_NAMES.get(code, code)

    # ── Internals ──────────────────────────────────────────────────────────

    def _immediate_siblings(self, element: Tag) -> Iterable[Tag]:
        """Retourne UNIQUEMENT le previous et le next sibling Tag immédiats.
        Exclut les node text/whitespace. Les drapeaux sont toujours collés."""
        out: list[Tag] = []
        cur = element.previous_sibling
        while cur is not None:
            if isinstance(cur, Tag):
                out.append(cur)
                break
            cur = cur.previous_sibling
        cur = element.next_sibling
        while cur is not None:
            if isinstance(cur, Tag):
                out.append(cur)
                break
            cur = cur.next_sibling
        return out

    def _code_from_flag_element(self, el: Tag) -> str:
        """
        Extrait un code langue depuis un élément qui ressemble à un DRAPEAU :
        img avec src /flags/*, span/i avec classe flag-*, etc.
        N'examine PAS les attributs génériques comme hreflang (qui sont la
        responsabilité de detect_near sur l'élément lui-même).
        """
        if not isinstance(el, Tag):
            return ''

        # Images de drapeau
        if el.name == 'img':
            src = el.get('src', '') or ''
            m = _SRC_PATTERN.search(src)
            if m:
                return self._map_to_lang(m.group('code'))
            alt = (el.get('alt', '') or '') + ' ' + (el.get('title', '') or '')
            if alt.strip():
                code = self._from_alt(alt)
                if code:
                    return code

        # Span / i avec classe flag-*
        classes = el.get('class') or []
        class_str = ' '.join(classes)
        for pattern in _CODE_PATTERNS:
            m = pattern.search(class_str)
            if m:
                return self._map_to_lang(m.group('code'))

        return ''

    def _from_alt(self, alt: str) -> str:
        """
        Essaie de tirer un code langue d'un alt/title d'image.
        Accepte :
          - Codes directs courts : "fr", "en-US"
          - Noms de langue : "French", "Français", "English"
          - Noms de pays (dans un contexte flag) : "France", "Allemagne"
        """
        alt_lower = alt.lower().strip()
        if not alt_lower:
            return ''

        # Codes directs courts
        if len(alt_lower) <= 20:
            if len(alt_lower) in (2, 3) and alt_lower.isalpha():
                return self._map_to_lang(alt_lower)
            m = re.match(r'^([a-z]{2,3})[-_]([a-z]{2,3})$', alt_lower)
            if m:
                return self._normalize_code(m.group(0))

        # Noms de langue ("French", "Français", "English")
        for code, name in _LANG_NAMES.items():
            if re.search(rf'\b{re.escape(name)}\b', alt_lower):
                return code

        # Noms de pays : on n'accepte que si "flag/drapeau" est dans l'alt
        # (sinon "France" dans un texte quelconque provoque un faux positif)
        has_flag_hint = any(hint in alt_lower for hint in _FLAG_HINT_WORDS)
        if has_flag_hint or len(alt_lower) <= 30:
            for name, code in _COUNTRY_NAME_TO_LANG.items():
                if re.search(rf'\b{re.escape(name)}\b', alt_lower):
                    return code

        return ''

    def _map_to_lang(self, code: str) -> str:
        """
        Un code court peut être soit un code langue (fr, en) soit un code
        pays (gb, us). On normalise vers un code langue.
        """
        c = code.lower().strip()
        if len(c) < 2:
            return ''
        # Si c'est déjà un code langue connu, on le garde
        if c in _LANG_NAMES:
            return c
        # Sinon on le traite comme un code pays
        return _COUNTRY_TO_LANG.get(c, '')

    def _normalize_code(self, raw: str) -> str:
        """
        Nettoie une valeur d'attribut lang/hreflang/meta :
          'fr-FR' → 'fr'
          'en_US' → 'en'
          'FR'    → 'fr'
        """
        if not raw:
            return ''
        raw = raw.strip().lower()
        # Première partie avant séparateur de région
        primary = re.split(r'[-_]', raw, maxsplit=1)[0]
        if not primary:
            return ''
        # Seuls des codes 2-3 lettres sont valides en ISO 639-1/2
        if not re.fullmatch(r'[a-z]{2,3}', primary):
            return ''
        # Si c'est un code pays connu, on mappe vers langue
        if primary in _LANG_NAMES:
            return primary
        if primary in _COUNTRY_TO_LANG:
            return _COUNTRY_TO_LANG[primary]
        return primary    # code inconnu mais formellement valide → on garde
