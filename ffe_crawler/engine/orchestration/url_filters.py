# ============================================================
# © 2026 Louis Guillory (Silence Wolf) — Tous droits réservés
# Licence : PROPRIETARY — voir fichier LICENSE
# ============================================================

# Utilitaire pour construire l'URL effective d'une page à crawler en
# combinant son URL de base et les valeurs de filtres saisies par
# l'utilisateur à l'étape 2 du wizard.
#
# Extrait de StepRunner pour pouvoir être réutilisé par les routes de
# détection (/detect-content-classes, /fetch-preview) et garantir que
# TOUS les points d'entrée qui "voient" la page cible la voient avec
# les MÊMES filtres appliqués (dates, régions, etc.) — sinon la
# preview et la détection des classes se font sur la page non-filtrée
# et l'utilisateur voit des éléments qui ne correspondent pas au
# périmètre qu'il a demandé.

from __future__ import annotations

import json
import threading
import time
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup

from log import CrawlerLogger
from crawler import Crawler
from orchestration.form_submit import FormSubmitter

_log = CrawlerLogger.get_instance()


# ── Cache HTTP en mémoire ────────────────────────────────────────────────
#
# Les fetch filtrés (POST form + nav dropdown + éventuellement Playwright)
# coûtent 1-10s sur FFE. Le front déclenche plusieurs appels quasi-
# simultanés (preview + détection) et refait tout au moindre re-render.
# Un cache en mémoire court terme élimine 80 % de ces appels redondants.
#
# Clé : (url, hash(field_values), force_dynamic). TTL 60s.
# On stocke (html, effective_url, stored_at). Thread-safe via lock.

_CACHE_TTL_S  = 60
_cache_lock   = threading.Lock()
_cache: dict[tuple, tuple[str, str, float]] = {}


def clear_cache() -> None:
    """
    Vide entièrement le cache HTTP en mémoire. Appelé au démarrage du
    sidecar pour garantir qu'une nouvelle session ne serve pas des
    résultats stales d'un run précédent du même processus (le module
    n'est pas systématiquement rechargé sur restart soft).
    """
    with _cache_lock:
        _cache.clear()


def _cache_key(
    url:           str,
    field_values:  dict[str, str] | None,
    force_dynamic: bool,
) -> tuple:
    # JSON trié → clé stable quelle que soit l'ordre d'insertion des
    # field_values côté front.
    fv_norm = json.dumps(field_values or {}, sort_keys=True)
    return (url, fv_norm, bool(force_dynamic))


def _cache_get(key: tuple) -> tuple[str, str] | None:
    now = time.time()
    with _cache_lock:
        entry = _cache.get(key)
        if entry is None:
            return None
        html, effective_url, ts = entry
        if now - ts > _CACHE_TTL_S:
            del _cache[key]
            return None
        return (html, effective_url)


def _cache_put(key: tuple, html: str, effective_url: str) -> None:
    with _cache_lock:
        _cache[key] = (html, effective_url, time.time())
        # Ménage opportuniste : supprime les entrées expirées quand la
        # map dépasse une taille raisonnable (évite croissance infinie).
        if len(_cache) > 64:
            now = time.time()
            stale = [k for k, (_h, _u, ts) in _cache.items() if now - ts > _CACHE_TTL_S]
            for k in stale:
                del _cache[k]


def fetch_filtered_html(
    url:           str,
    field_values:  dict[str, str] | None,
    force_dynamic: bool = False,
    auth_cookies:  dict[str, str] | None = None,
) -> tuple[str, str]:
    """
    Récupère le HTML de la page "telle que l'utilisateur la verrait"
    après application de ses filtres (étape 2).

    Stratégie en deux temps :

    1. Essaie d'abord `FormSubmitter.submit()` — gère les cas POST avec
       token CSRF (FFE SIF, etc.). Si un <form> matche les field_values,
       le résultat est le HTML FILTRÉ.

    2. Fallback GET : construit l'URL via `build_url_with_filters` et
       fetch avec `Crawler._fetch` (gère aussi Playwright si
       force_dynamic). Convient aux sites qui encodent les filtres
       dans la query string (ou dans des tokens de menu dropdown).

    Retourne (html, final_url) — l'URL finale peut différer de la base
    si le form a redirigé, ou si un filtre "URL-like" l'a remplacée.
    """
    # Pipeline FFE SIF-compliant :
    #   (1) Sépare scalaires (dates, etc.) vs url_like (hrefs de
    #       dropdowns capturés à l'étape 2 — peuvent être périmés dès
    #       qu'on soumet les dates car le token `cs` change).
    #   (2) Mappe chaque url_like {field_name → label_texte} depuis la
    #       page d'origine (le label est stable, l'href ne l'est pas).
    #   (3) POST le formulaire avec les dates → page filtrée par dates.
    #   (4) Sur cette page, pour chaque dropdown sélectionné, retrouve
    #       le <a> par SON LABEL (pas par href, qui est obsolète),
    #       extrait son nouveau href, GET dessus. Répète pour chaque
    #       dropdown (l'URL change à chaque clic).
    # Check cache en premier — si la même combinaison url+filters a été
    # fetchée il y a moins de _CACHE_TTL_S, retourne la réponse cached.
    cache_k    = _cache_key(url, field_values, force_dynamic)
    cache_hit  = _cache_get(cache_k)
    if cache_hit is not None:
        html, effective_url = cache_hit
        _log.info(f'[filters] CACHE HIT → {effective_url} ({len(html)} octets)')
        return html, effective_url

    _log.info(f'[filters] URL base : {url}')
    _log.info(f'[filters] field_values reçus : {field_values}')

    url_like_vals: dict[str, str] = {}
    scalar_vals:   dict[str, str] = {}
    for key, value in (field_values or {}).items():
        if not value:
            continue
        v = str(value).strip()
        if not v:
            continue
        # Ajout de './' et '../' : FFE SIF émet des hrefs relatifs
        # de type "./?cs=4.XXX" pour ses dropdowns — sans ce prefix
        # ils tombaient en scalar_vals et étaient envoyés en query param.
        if v.startswith(('?', '/', './', '../', 'http://', 'https://')):
            url_like_vals[key] = v
        else:
            scalar_vals[key] = v

    _log.info(f'[filters] dates/scalaires  : {scalar_vals}')
    _log.info(f'[filters] dropdowns (href) : {url_like_vals}')

    # On utilise UNE SEULE Session (cookies partagés) pour tout le
    # pipeline : GET initial → POST form → GET dropdown. FFE stocke
    # probablement des états côté serveur indexés par cookie de
    # session ; sans ce partage on perd les dates dès qu'on clique
    # sur un dropdown (chaque fetch = session neuve = reset).
    # auth_cookies (depuis l'auth_store) sont injectés DÈS la création
    # du FormSubmitter → le GET initial part déjà avec la session
    # post-login, FFE renvoie la version connectée des pages.
    submitter = FormSubmitter(cookies=auth_cookies)
    http_session = submitter.session

    crawler = Crawler(force_dynamic=force_dynamic)

    # (2a) Fetch la page d'origine (via submitter.session) pour
    #      résoudre les labels ET normaliser les dates, tout en
    #      récupérant les cookies initiaux dans la session partagée.
    original_soup: BeautifulSoup | None = None
    try:
        r = http_session.get(url, timeout=(15, 60))
        r.raise_for_status()
        r.encoding = r.apparent_encoding or 'utf-8'
        original_html = r.text
        original_soup = BeautifulSoup(original_html, 'html.parser')
    except Exception as exc:
        _log.warn(f'fetch_filtered_html : GET initial a échoué — {exc}')

    # (2b) Labels des dropdowns — unifie deux sources :
    #      1. Valeurs url_like (anciens sessions) : on retrouve le
    #         label en cherchant l'ancre ayant cet href sur la page
    #         d'origine.
    #      2. Valeurs scalaires qui correspondent directement au texte
    #         d'un <a> sur la page d'origine (format NOUVEAU : depuis
    #         qu'on stocke le label au lieu du href, le value est déjà
    #         un label humain type "Equifun").
    labels: dict[str, str] = {}
    if original_soup is not None:
        for field_name, old_href in url_like_vals.items():
            label = _label_of_anchor(original_soup, old_href, url)
            if label:
                labels[field_name] = label
            else:
                _log.warn(f'[filters] label introuvable pour {field_name} (href stocké : {old_href[:80]}...)')
        # Scalaire dont la valeur matche un <a> texte → dropdown-label.
        # Pour les autres scalaires, on vérifie qu'un <input>/<select>/
        # <textarea> avec ce name existe bien. Sinon ce n'est NI un form
        # field NI un dropdown sur cette page (cas typique : on est
        # arrivé sur une page de détail qui n'a plus le dropdown d'origine,
        # la valeur ne peut pas être appliquée et il ne faut surtout pas
        # la balancer en query param à FormSubmitter — risque de redirection
        # vers une page parasite qui polluerait le crawl).
        consumed: list[str] = []
        dropped:  list[str] = []
        for field_name, value in list(scalar_vals.items()):
            if _href_of_label(original_soup, value):
                labels[field_name] = value
                consumed.append(field_name)
                continue
            # Ni dropdown-label, ni date : vérifie qu'un vrai form field
            # avec ce name existe. Sinon, on le drop proprement.
            has_form_field = bool(
                original_soup.find(['input', 'select', 'textarea'], attrs={'name': field_name})
            )
            if not has_form_field:
                dropped.append(field_name)
        for k in consumed:
            del scalar_vals[k]
        for k in dropped:
            del scalar_vals[k]
        if labels:
            _log.info(f'[filters] labels résolus : {labels}')
        if dropped:
            _log.info(f'[filters] filtres non applicables sur cette page (ignorés) : {dropped}')

    # (2c) Normalisation des dates : FFE attend DD/MM/YYYY dans ses
    #      inputs, le front envoie YYYY-MM-DD. On lit la valeur actuelle
    #      de l'input sur la page d'origine et on convertit au même
    #      format si nécessaire.
    if scalar_vals and original_soup is not None:
        scalar_vals = _normalize_dates_for_form(scalar_vals, original_soup)
        _log.info(f'[filters] scalaires normalisés : {scalar_vals}')

    # (3) Soumet les dates → page_url + page_html filtrés par dates
    current_url  = url
    current_html: str | None = None
    if scalar_vals:
        try:
            result = submitter.submit(current_url, scalar_vals)
            if result is not None:
                _log.info(f'[filters] form submit OK → {result.final_url}')
                current_url  = result.final_url
                current_html = result.html
            else:
                _log.warn(f'[filters] form submit : aucun <form> matchant trouvé pour {scalar_vals}')
        except Exception as exc:
            _log.warn(f'fetch_filtered_html : form-submit a échoué — {exc}')

    # (4) Applique chaque dropdown par label sur la page courante
    if labels:
        for field_name, label in labels.items():
            if current_html is None:
                try:
                    r = http_session.get(current_url, timeout=(15, 60))
                    r.raise_for_status()
                    r.encoding = r.apparent_encoding or 'utf-8'
                    current_html = r.text
                except Exception as exc:
                    _log.warn(f'fetch_filtered_html : GET {current_url} échoué — {exc}')
                    break
            soup       = BeautifulSoup(current_html, 'html.parser')
            new_href   = _href_of_label(soup, label)
            if not new_href:
                _log.warn(f'[filters] label "{label}" ({field_name}) introuvable sur la page courante — sélection dropdown ignorée')
                continue
            current_url = urljoin(current_url, new_href)
            _log.info(f'[filters] dropdown {field_name} = "{label}" → {current_url}')
            try:
                r = http_session.get(current_url, timeout=(15, 60))
                r.raise_for_status()
                r.encoding = r.apparent_encoding or 'utf-8'
                current_html = r.text
            except Exception as exc:
                _log.warn(f'fetch_filtered_html : GET {current_url} échoué après dropdown {field_name} — {exc}')
                break

    # Si on n'a encore rien chargé (aucun filtre), fetch la page brute
    if current_html is None:
        try:
            r = http_session.get(current_url, timeout=(15, 60))
            r.raise_for_status()
            r.encoding = r.apparent_encoding or 'utf-8'
            current_html = r.text
        except Exception as exc:
            _log.warn(f'fetch_filtered_html : GET final échoué — {exc}')
            current_html = crawler._fetch(current_url)

    _log.info(f'[filters] URL finale utilisée : {current_url}')

    # Fallback auto : FFE SIF (et beaucoup d'autres) rendent leur
    # contenu en AJAX — le HTML statique qu'on récupère est la coquille,
    # les événements arrivent plus tard via JS. Si la page n'a aucun
    # <tr> (liste de résultats) et qu'on a des filtres à appliquer, on
    # retente via Playwright. Le seuil précédent basé sur "card" était
    # trop permissif car le template FFE utilise `card` partout dans la
    # nav statique.
    if not force_dynamic and field_values and current_html:
        try:
            sniff_soup = BeautifulSoup(current_html, 'html.parser')
            n_tr = len(sniff_soup.select('table tbody tr'))
            if n_tr == 0:
                _log.info(f'[filters] page sans <tr> (tr={n_tr}) → retry avec Playwright (pool persistant)')
                from orchestration.playwright_pool import (
                    fetch as _pw_fetch,
                    cookies_from_requests_session as _pw_cookies,
                )
                # Injecte les cookies de la session HTTP dans Chromium
                # pour que FFE voit la même "identité" (sinon Playwright
                # ouvre une session vierge et peut renvoyer la homepage).
                pw_cookies   = _pw_cookies(http_session)
                # On déclenche ce retry précisément parce que la page
                # n'a aucun <tr> en static. La table cible est donc
                # injectée par AJAX post-onReady (cas FFE Telemat). On
                # attend explicitement qu'un <tr> apparaisse — sans ça
                # Playwright sort dès domcontentloaded et capture le
                # DOM avant injection, on récupère 53 Ko sans <tr>.
                current_html = _pw_fetch(
                    current_url,
                    cookies           = pw_cookies,
                    wait_until        = 'networkidle',
                    timeout_ms        = 25_000,
                    wait_for_selector = 'table tbody tr',
                    wait_timeout_ms   = 12_000,
                )
                _log.info(f'[filters] retry Playwright OK : {len(current_html)} octets '
                          f'(cookies injectés : {len(pw_cookies)})')
        except Exception as exc:
            _log.warn(f'[filters] retry Playwright a échoué — {exc}')

    # Dump diagnostic : taille + titre + 3 premiers tr/li/div
    # visiblement "contenu" pour confirmer si la page rendue correspond
    # bien aux filtres demandés ou si on est sur la page brute.
    try:
        final_soup   = BeautifulSoup(current_html, 'html.parser')
        title_text   = (final_soup.title.get_text(strip=True) if final_soup.title else '')
        table_rows   = final_soup.select('table tbody tr')
        list_items   = final_soup.select('ul li')
        _log.info(f'[filters][page] taille HTML : {len(current_html)} octets')
        _log.info(f'[filters][page] title : {title_text!r}')
        _log.info(f'[filters][page] <tr> dans tbody : {len(table_rows)}')
        _log.info(f'[filters][page] <li> : {len(list_items)}')
        for i, tr in enumerate(table_rows[:3]):
            txt = tr.get_text(' ', strip=True)[:200]
            _log.info(f'[filters][page] tr[{i}] = {txt!r}')
        # Dump d'échantillons de <li> ET des principaux conteneurs de
        # contenu (<div>/<article>) pour voir si on a des events, même
        # sans <tr>. Si ni tr ni div-content → contenu chargé en AJAX.
        for i, li in enumerate(list_items[:5]):
            txt = li.get_text(' ', strip=True)[:200]
            _log.info(f'[filters][page] li[{i}] = {txt!r}')
        content_divs = [d for d in final_soup.find_all(['div', 'article'])
                        if d.get_text(strip=True) and len(d.get_text(strip=True)) > 50]
        _log.info(f'[filters][page] divs/articles avec texte > 50 char : {len(content_divs)}')
        # Cherche des indices de chargement AJAX : <script> qui fetch
        # des URLs, ou placeholders typiques "chargement…".
        scripts_count = len(final_soup.find_all('script'))
        loading_markers = final_soup.find_all(
            string=lambda s: s and ('chargement' in s.lower() or 'loading' in s.lower()),
        )
        _log.info(f'[filters][page] scripts : {scripts_count}  ·  markers chargement : {len(loading_markers)}')
    except Exception as exc:
        _log.warn(f'[filters][page] diagnostic a échoué — {exc}')

    # Vérification : on relit les valeurs ACTUELLES des champs sur la
    # page rendue, pour confirmer que le filtrage a bien pris effet.
    # Si l'input a la bonne valeur → filtre appliqué côté serveur.
    # Si l'input est vide ou a la valeur par défaut → échec silencieux.
    if field_values:
        try:
            verify_soup = BeautifulSoup(current_html, 'html.parser')
            for field_name, expected in field_values.items():
                actual = _read_field_value(verify_soup, field_name)
                exp_s  = str(expected).strip()
                act_s  = str(actual).strip()
                # Les dates peuvent arriver en ISO (front) et être
                # rendues en DD/MM/YYYY (FFE) — on les compare en ISO.
                iso_exp = _to_iso_date(exp_s)
                iso_act = _to_iso_date(act_s)
                if iso_exp and iso_act:
                    ok     = iso_exp == iso_act
                    mark   = '✓' if ok else '✗'
                elif not act_s:
                    # Pas d'input/select avec ce name (cas typique d'un
                    # dropdown Bootstrap à base de <a>). On ne peut pas
                    # vérifier via l'attribut value — on fait un fallback
                    # "label visible sur la page". Si le texte attendu
                    # apparaît comme texte d'ancre, on considère que la
                    # sélection a bien été appliquée.
                    has_label = bool(exp_s) and bool(_href_of_label(verify_soup, exp_s))
                    mark = '✓' if has_label else '?'
                    act_s = '(dropdown — label présent)' if has_label else '(non trouvé)'
                else:
                    ok   = act_s == exp_s
                    mark = '✓' if ok else '✗'
                _log.info(
                    f'[filters][verify] {mark} {field_name} '
                    f'attendu={expected!r}  reçu={act_s!r}'
                )
        except Exception as exc:
            _log.warn(f'[filters][verify] lecture des champs a échoué — {exc}')

    # Store en cache UNIQUEMENT si le résultat paraît valide — deux
    # requêtes parallèles sur la même clé peuvent s'écraser entre
    # elles, et on ne veut pas qu'un résultat "vide" (tr=0 alors qu'on
    # a des filtres) supplante la version Playwright avec contenu.
    try:
        valid_soup = BeautifulSoup(current_html, 'html.parser')
        has_rows   = len(valid_soup.select('table tbody tr')) > 0
    except Exception:
        has_rows = True  # en cas d'échec de parsing, on garde le cache
    should_cache = has_rows or not field_values
    if should_cache:
        _cache_put(cache_k, current_html, current_url)
    else:
        _log.info('[filters] résultat vide avec filtres → NON mis en cache')

    return current_html, current_url


import re as _re

_ISO_DATE_RE     = _re.compile(r'^(\d{4})-(\d{2})-(\d{2})$')
_DDMMYYYY_RE     = _re.compile(r'^(\d{2})/(\d{2})/(\d{4})$')
_MMDDYYYY_HINT   = _re.compile(r'(MM/DD|mm/dd)')
_DDMMYYYY_HINT   = _re.compile(r'(DD/MM|JJ/MM|dd/mm|jj/mm)')


def _normalize_dates_for_form(
    scalars: dict[str, str],
    soup:    BeautifulSoup,
) -> dict[str, str]:
    """
    Convertit les dates ISO (YYYY-MM-DD) du front vers le format attendu
    par l'input correspondant sur la page.

    Stratégie : pour chaque scalaire qui ressemble à une date ISO,
    inspecter l'input de même `name` dans la page :
      - si sa value actuelle matche DD/MM/YYYY → convertir
      - sinon si son placeholder contient "DD/MM" ou "JJ/MM" → convertir
      - sinon laisser tel quel (le serveur accepte peut-être l'ISO)
    """
    out: dict[str, str] = {}
    for name, value in scalars.items():
        m = _ISO_DATE_RE.match(value)
        if not m:
            out[name] = value
            continue

        yyyy, mm, dd = m.group(1), m.group(2), m.group(3)

        inp = soup.find('input', attrs={'name': name})
        wants_ddmm = False
        if inp is not None:
            curr = (inp.get('value') or '').strip()
            if _DDMMYYYY_RE.match(curr):
                wants_ddmm = True
            else:
                placeholder = (inp.get('placeholder') or '')
                pattern     = (inp.get('pattern')     or '')
                title       = (inp.get('title')       or '')
                blob = f'{placeholder} {pattern} {title}'
                if _DDMMYYYY_HINT.search(blob):
                    wants_ddmm = True

        out[name] = f'{dd}/{mm}/{yyyy}' if wants_ddmm else value
    return out


def _to_iso_date(s: str) -> str:
    """Tente de convertir une date en ISO YYYY-MM-DD. Retourne '' si
    le format n'est pas reconnu."""
    if not s:
        return ''
    m = _ISO_DATE_RE.match(s)
    if m:
        return s
    m = _DDMMYYYY_RE.match(s)
    if m:
        return f'{m.group(3)}-{m.group(2)}-{m.group(1)}'
    return ''


def _read_field_value(soup: BeautifulSoup, name: str) -> str:
    """
    Relit la valeur actuelle d'un champ (input/select/textarea) par son
    attribut `name` dans la page rendue. Utile pour confirmer qu'un
    POST de formulaire a bien été pris en compte côté serveur.

    - input : attr `value`
    - checkbox/radio : 'on' si `checked`, sinon '' (ou la value de
      l'option cochée si présente)
    - select : `value` de l'<option selected>
    - textarea : texte intérieur
    - sinon : '' (non trouvé)
    """
    # input
    inp = soup.find('input', attrs={'name': name})
    if inp is not None:
        itype = (inp.get('type') or 'text').lower()
        if itype in ('checkbox', 'radio'):
            if inp.has_attr('checked'):
                return inp.get('value') or 'on'
            return ''
        return inp.get('value') or ''

    # select
    sel = soup.find('select', attrs={'name': name})
    if sel is not None:
        opt = sel.find('option', selected=True)
        if opt is not None:
            return opt.get('value') or opt.get_text(strip=True)
        return ''

    # textarea
    ta = soup.find('textarea', attrs={'name': name})
    if ta is not None:
        return ta.get_text()

    return ''


def _label_of_anchor(
    soup:     BeautifulSoup,
    href:     str,
    base_url: str,
) -> str:
    """
    Trouve le <a> dans `soup` dont le href matche `href`, retourne son
    texte (label). Plusieurs stratégies de match en cascade :
      1. Exact sur attribut href
      2. Exact sur URL absolue résolue
      3. Match sur le token `cs=...` (FFE SIF encode l'état complet du
         filtre dedans — si les deux ont le même cs, c'est la même
         sélection, peu importe les différences de prefix/slash)
    """
    target_abs = urljoin(base_url, href)
    target_cs  = _extract_cs_token(target_abs)

    for a in soup.find_all('a', href=True):
        a_href = a.get('href') or ''
        if a_href == href:
            text = a.get_text(strip=True)
            if text:
                return text
        a_abs = urljoin(base_url, a_href)
        if a_abs == target_abs:
            text = a.get_text(strip=True)
            if text:
                return text
        if target_cs and _extract_cs_token(a_abs) == target_cs:
            text = a.get_text(strip=True)
            if text:
                return text
    return ''


def _extract_cs_token(absolute_url: str) -> str:
    """Retourne la valeur du query param `cs=` (FFE SIF) ou '' si absent."""
    try:
        qs = parse_qs(urlparse(absolute_url).query, keep_blank_values=True)
        vals = qs.get('cs') or []
        return vals[0] if vals else ''
    except Exception:
        return ''


def _href_of_label(soup: BeautifulSoup, label: str) -> str:
    """
    Retrouve l'href du <a> dont le texte strippé égale `label`. Utilisé
    pour ré-résoudre une sélection de dropdown après qu'une soumission
    de form ait changé les tokens dans tous les hrefs de la page.
    """
    norm = label.strip().lower()
    if not norm:
        return ''
    for a in soup.find_all('a', href=True):
        if a.get_text(strip=True).strip().lower() == norm:
            return a.get('href') or ''
    return ''


def build_url_with_filters(url: str, field_values: dict[str, str] | None) -> str:
    """
    Construit l'URL effective à utiliser en combinant `url` et les
    valeurs saisies.

    Deux types de valeurs gérés :

    1. "URL-like" : valeur commençant par '?', '/' ou 'http' — c'est
       le résultat d'un dropdown Bootstrap dont chaque option est un
       <a href="..."> qui encode l'état du filtre dans l'URL. On
       REMPLACE l'URL par cette valeur (le filtre est déjà
       self-contained).

    2. Scalaire (texte, date, nombre) : ajouté en query param GET.

    Les URL-like sont appliquées dans l'ordre (la dernière gagne), puis
    les scalaires sont mergés dans la query string résultante.
    """
    if not field_values:
        return url

    base_url = url
    scalar_params: dict[str, str] = {}

    for key, value in field_values.items():
        if not value:
            continue
        value_str = str(value).strip()
        if not value_str:
            continue

        if value_str.startswith('?'):
            parsed   = urlparse(base_url)
            base_url = urlunparse(parsed._replace(query=value_str.lstrip('?')))
        elif value_str.startswith('http://') or value_str.startswith('https://'):
            base_url = value_str
        elif value_str.startswith('/'):
            parsed = urlparse(base_url)
            if '?' in value_str:
                p, q = value_str.split('?', 1)
            else:
                p, q = value_str, ''
            base_url = urlunparse(parsed._replace(path=p, query=q))
        else:
            scalar_params[key] = value_str

    if scalar_params:
        parsed   = urlparse(base_url)
        existing = parse_qs(parsed.query, keep_blank_values=True)
        for k, v in scalar_params.items():
            existing[k] = [v]
        new_query = urlencode(
            [(k, v[0] if isinstance(v, list) else v) for k, v in existing.items()],
            doseq=False,
        )
        base_url = urlunparse(parsed._replace(query=new_query))

    return base_url
