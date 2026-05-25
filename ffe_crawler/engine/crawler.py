#!/usr/bin/env python3
# © 2026 Louis Guillory (Silence Wolf) — Tous droits réservés
# Licence : PROPRIETARY — voir fichier LICENSE

"""
Web Crawler Toolbox — point d'entrée principal.

Usage bibliothèque :
    from crawler import Crawler
    c = Crawler()
    result  = c.crawl("https://example.com", "my-class")
    results = []
    c.crawl_batch(["https://a.com", "https://b.com"], "my-class", results)

Usage CLI :
    python crawler.py --url "https://example.com" --class "my-class"
    python crawler.py --urls urls.txt --class "my-class" --dynamic --raw
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup, Tag

from extractors import DEFAULT_EXTRACTORS, BaseCrawlerExtractor
from log        import CrawlerLogger, LogLevel


# ──────────────────────────────────────────────────────────────────────
# Shutdown coopératif : un event partagé que tous les workers vérifient
# avant chaque requête. Permet d'arrêter PROPREMENT un crawl en cours
# (Ctrl+C, redémarrage uvicorn) au lieu d'attendre que la queue de 600
# futures se vide en envoyant un burst à FFE qui te fait bannir.
# ──────────────────────────────────────────────────────────────────────
_abort_event: threading.Event = threading.Event()


def request_shutdown() -> None:
    """Signale à tous les workers en cours de stopper le plus vite possible.
    Appelé par le lifespan FastAPI au Ctrl+C."""
    _abort_event.set()


def reset_shutdown() -> None:
    """Re-arme l'event (pour un nouveau crawl)."""
    _abort_event.clear()


def is_shutdown_requested() -> bool:
    return _abort_event.is_set()


def _get_proxy_dict() -> dict[str, str] | None:
    """
    Récupère un proxy depuis l'environnement, au format dict attendu
    par requests. Permet de changer d'IP sortante (Tor, proxy rotatif,
    VPN avec SOCKS) sans toucher au code.

    Variables d'env reconnues (priorité décroissante) :
      EQUIRANK_PROXY   — variable dédiée au crawler PFE (priorité)
      HTTPS_PROXY      — standard requests/curl
      HTTP_PROXY       — fallback

    Format accepté :
      http://host:port
      http://user:pass@host:port
      socks5://127.0.0.1:9050   (Tor par défaut — requiert `requests[socks]`)
    """
    import os
    proxy = None
    for key in ('EQUIRANK_PROXY', 'HTTPS_PROXY', 'HTTP_PROXY',
                'https_proxy',  'http_proxy'):
        proxy = os.environ.get(key)
        if proxy:
            break
    if not proxy:
        return None
    return {'http': proxy, 'https': proxy}


# ──────────────────────────────────────────────────────────────────────
# Circuit breaker global anti-bannissement
# ──────────────────────────────────────────────────────────────────────
# Si on encaisse plusieurs 429 dans une courte fenêtre, on impose une
# pause GLOBALE à TOUS les workers — pas seulement à celui qui a
# rencontré le 429. Évite la cascade de bannissement IP : quand FFE
# voit qu'on continue à taper malgré ses 429, il escalade en 403/IP
# block. Avec cette pause partagée, dès qu'un worker remarque qu'on
# est rate-limité, tous les autres se mettent au pas.
_rate_lock              = threading.Lock()
_rate_limited_until:    float = 0.0          # epoch UTC ; 0 = pas de pause
_rate_429_window:       list[float] = []     # timestamps des 429 récents

_RATE_THRESHOLD_429     = 3      # 3 × 429 dans la fenêtre → pause
_RATE_WINDOW_SEC        = 30.0   # fenêtre de comptage
_RATE_PAUSE_SEC         = 90.0   # pause à imposer (en cas de pic)


def _wait_if_rate_limited() -> None:
    """À appeler AVANT chaque requête HTTP. Bloque le thread courant si
    le circuit breaker est ouvert. Tous les workers convergent ici.
    Sort tout de suite si un shutdown coopératif est demandé."""
    while True:
        if _abort_event.is_set():
            return                          # shutdown → on libère le worker
        with _rate_lock:
            now    = time.time()
            remain = _rate_limited_until - now
        if remain <= 0:
            return
        # Sleep en petits morceaux pour rester réactif au shutdown
        time.sleep(min(remain, 1.0))


def _record_429() -> bool:
    """Enregistre un 429. Renvoie True si le circuit breaker doit
    s'ouvrir (≥ _RATE_THRESHOLD_429 sur _RATE_WINDOW_SEC)."""
    global _rate_limited_until
    with _rate_lock:
        now = time.time()
        # Garde uniquement les timestamps dans la fenêtre
        _rate_429_window[:] = [t for t in _rate_429_window if now - t <= _RATE_WINDOW_SEC]
        _rate_429_window.append(now)
        if len(_rate_429_window) >= _RATE_THRESHOLD_429:
            _rate_limited_until = now + _RATE_PAUSE_SEC
            _rate_429_window.clear()
            return True
        return False
from output     import OutputManager


# ── Dataclass résultat ─────────────────────────────────────────────────────────

@dataclass
class CrawlResult:
    url:        str
    class_name: str
    timestamp:  str
    rows:       list[dict] = field(default_factory=list)
    raw_html:   Optional[str] = None


# ── Crawler ────────────────────────────────────────────────────────────────────

class Crawler:
    """
    Crawle une ou plusieurs pages web et extrait le contenu d'une classe CSS donnée.

    Deux modes principaux :
      - crawl()       → page unique, retourne un CrawlResult
      - crawl_batch() → liste de pages, remplit un tableau results (in-place)

    Détection automatique JS : si la page semble rendue côté client (React, Vue, etc.)
    le crawler bascule sur playwright pour obtenir le DOM réel.
    """

    def __init__(
        self,
        extractors:     list[BaseCrawlerExtractor] | None = None,
        force_dynamic:  bool  = False,
        save_raw:       bool  = False,
        download_media: bool  = False,
        log_level:      LogLevel = LogLevel.INFO,
        log_file:       Path | None = None,
    ) -> None:
        self._extractors     = extractors or DEFAULT_EXTRACTORS
        self._force_dynamic  = force_dynamic
        self._save_raw       = save_raw
        self._download_media = download_media
        self._output         = OutputManager()
        self._log            = CrawlerLogger.get_instance(
            context='crawler', min_level=log_level, log_file=log_file
        )

        # Headers complets imitant Chrome — certains sites (FFE Telemat
        # notamment) servent une page dégradée quand un seul UA est
        # présent mais qu'il manque les Sec-Fetch-* / Accept attendus
        # d'un vrai navigateur. Sans ces headers, on récupère ~10 Ko de
        # stub anti-bot au lieu des 130 Ko de la page réelle.
        self._headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/124.0.0.0 Safari/537.36'
            ),
            'Accept': (
                'text/html,application/xhtml+xml,application/xml;q=0.9,'
                'image/avif,image/webp,image/apng,*/*;q=0.8,'
                'application/signed-exchange;v=b3;q=0.7'
            ),
            'Accept-Language':  'fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding':  'gzip, deflate, br',
            'Cache-Control':    'max-age=0',
            'Sec-Ch-Ua':        '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
            'Sec-Ch-Ua-Mobile': '?0',
            'Sec-Ch-Ua-Platform': '"Windows"',
            'Sec-Fetch-Dest':   'document',
            'Sec-Fetch-Mode':   'navigate',
            'Sec-Fetch-Site':   'none',
            'Sec-Fetch-User':   '?1',
            'Upgrade-Insecure-Requests': '1',
        }

    # ── API publique ───────────────────────────────────────────────────────────

    def crawl(self, url: str, class_name: str) -> CrawlResult:
        """Mode individuel — crawle une seule URL et retourne le résultat."""
        self._log.separator()
        self._log.info(f'Crawl individuel → {url}  [classe: {class_name}]')

        result = self._process_url(url, class_name)
        self._save([result], f'{url}_{class_name}')
        return result

    def crawl_batch(
        self,
        urls:       list[str],
        class_name: str,
        results:    list,           # tableau fourni par l'appelant — modifié in-place
        delay:      float = 1.0,   # délai entre requêtes en secondes (politesse)
    ) -> None:
        """
        Mode batch — crawle toutes les URLs et ajoute les CrawlResult dans `results`.
        Le tableau est modifié in-place pour que l'appelant puisse surveiller la progression.
        """
        self._log.banner(f'Crawl batch — {len(urls)} URL(s)  [classe: {class_name}]')

        for idx, url in enumerate(urls, start=1):
            self._log.info(f'[{idx}/{len(urls)}] {url}')
            try:
                result = self._process_url(url, class_name)
                results.append(result)
                self._log.success(f'  → {len(result.rows)} ligne(s) extraite(s)')
            except Exception as exc:
                self._log.error(f'  → Échec : {exc}')
                results.append(CrawlResult(
                    url=url, class_name=class_name,
                    timestamp=_now(), rows=[],
                ))

            if idx < len(urls):
                time.sleep(delay)

        self._save(results, class_name)
        self._log.success(f'Batch terminé — {len(results)} résultats sauvegardés')

    # ── Internals ──────────────────────────────────────────────────────────────

    def _process_url(self, url: str, class_name: str) -> CrawlResult:
        """Fetche la page, trouve les éléments de la classe, lance les extracteurs."""
        from datetime import datetime
        timestamp = datetime.now().isoformat(timespec='seconds')

        # Détermine si on utilise Playwright (pagination intégrée)
        use_dynamic = self._force_dynamic
        if not use_dynamic:
            static_html = self._fetch_static(url)
            use_dynamic = self._is_js_heavy(static_html)

        if use_dynamic:
            rows = self._process_url_paginated(url, class_name)
        else:
            rows = self._extract_from_html(static_html, url, class_name)

        # Déduplique sur le contenu complet de la row, pas juste text|href.
        # Les rows multi-colonnes (extraction par <th>) n'ont PAS de clé
        # "text" et peuvent partager un même "href" (cas FFE : tous les
        # cavaliers anonymisés pointent vers le même `rgpd_link` d'avertis-
        # sement). Une dédup sur (text, href) écrasait alors 39 cavaliers
        # en 1 seul. La clé est maintenant un tuple ordonné de toutes les
        # paires colonne=valeur (sauf source_url qui est constant pour une
        # même page) — deux rows sont dupes uniquement si elles ont
        # exactement les mêmes données.
        seen: set[tuple] = set()
        unique_rows: list[dict] = []
        for row in rows:
            key = tuple(sorted(
                (k, v) for k, v in row.items() if k != 'source_url'
            ))
            if key not in seen:
                seen.add(key)
                unique_rows.append(row)

        return CrawlResult(
            url=url,
            class_name=class_name,
            timestamp=timestamp,
            rows=unique_rows,
            raw_html=None,
        )

    def _extract_from_html(self, html: str, url: str, class_name: str) -> list[dict]:
        """Extrait les données structurées depuis du HTML brut.
        Garde la trace du sélecteur qui a matché chaque élément pour
        pouvoir le réutiliser comme nom de colonne dans le CSV — plus
        prévisible que de deviner depuis les classes du DOM."""
        soup = BeautifulSoup(html, 'html.parser')

        # tagged_elems : liste de (matched_selector, element)
        tagged_elems: list[tuple[str, 'Tag']] = []

        if _is_css_selector(class_name):
            for sel in class_name.split(','):
                sel = sel.strip()
                if not sel:
                    continue
                try:
                    found = soup.select(sel)
                except Exception:
                    raw = sel.lstrip('.')
                    found = soup.find_all(class_=lambda c: c and raw in c)
                    self._log.debug(f'  Sélecteur "{sel}" invalide pour CSS, fallback class search → {len(found)}')
                # Fallback "tbody manquant" : certains sites (FFE Telemat
                # en text/xml notamment) renvoient des tables SANS balise
                # `<tbody>` explicite — les `<tr>` sont enfants directs
                # de `<table>`. BeautifulSoup en mode html.parser n'auto-
                # injecte PAS de `<tbody>` virtuel sur du XML, donc un
                # sélecteur `... tbody tr` retourne 0 alors que les rows
                # existent bien. On re-tente sans `tbody` et on prévient
                # via les logs.
                if not found and 'tbody' in sel.lower():
                    sel_no_tbody = re.sub(r'\s*\btbody\b\s*', ' ', sel, flags=re.IGNORECASE).strip()
                    sel_no_tbody = re.sub(r'\s+', ' ', sel_no_tbody)
                    if sel_no_tbody and sel_no_tbody != sel:
                        try:
                            found = soup.select(sel_no_tbody)
                            if found:
                                self._log.info(
                                    f'  Sélecteur "{sel}" → 0, fallback sans tbody '
                                    f'"{sel_no_tbody}" → {len(found)} (page sans <tbody> explicite)'
                                )
                        except Exception:
                            pass
                for el in found:
                    tagged_elems.append((sel, el))
        else:
            for el in soup.find_all(class_=_class_selector(class_name)):
                tagged_elems.append((class_name, el))

        self._log.debug(f'  {len(tagged_elems)} élément(s) trouvé(s) pour "{class_name}"')
        return self._extract_structured(tagged_elems, url)

    def _process_url_paginated(self, url: str, class_name: str) -> list[dict]:
        """
        Mode Playwright avec pagination : extrait les données de la page courante,
        puis clique les boutons de pagination (1, 2, 3...) pour capturer toutes les pages.
        """
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            self._log.fatal('playwright non installé')
            raise

        all_rows: list[dict] = []

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(
                extra_http_headers={'Accept-Language': 'fr-FR,fr;q=0.9'},
                user_agent=self._headers['User-Agent'],
            )
            page.goto(url, wait_until='networkidle', timeout=30_000)
            self._handle_load_more(page)
            self._handle_infinite_scroll(page)

            # Extraire page 1
            html = page.content()
            rows = self._extract_from_html(html, url, class_name)
            all_rows.extend(rows)
            self._log.info(f'  Page 1 : {len(rows)} élément(s)')

            # Chercher des boutons de pagination
            page_num = 2
            MAX_PAGES = 50
            while page_num <= MAX_PAGES:
                if not self._click_pagination(page, page_num):
                    break

                self._log.debug(f'  → clic pagination page {page_num}')
                page.wait_for_timeout(1500)
                try:
                    page.wait_for_load_state('networkidle', timeout=10_000)
                except Exception:
                    pass

                html = page.content()
                rows = self._extract_from_html(html, url, class_name)
                if not rows:
                    break
                all_rows.extend(rows)
                self._log.info(f'  Page {page_num} : {len(rows)} élément(s)')
                page_num += 1

            if page_num > 2:
                self._log.success(f'  → {page_num - 1} page(s) parcourue(s), {len(all_rows)} élément(s) total')

            browser.close()

        return all_rows

    def _extract_structured(self, tagged_elems: list, url: str) -> list[dict]:
        """
        Extraction structurée. UNE row par "groupe" :
          - si plusieurs éléments matchés partagent le même <tr>
            ancestral, ils sont fusionnés sur la même row CSV (les
            colonnes sont nommées d'après le sélecteur de chaque match)
          - sinon, un élément isolé = sa propre row

        Cas spécial : si l'élément matché EST un <tr>, on explose ses
        cellules <td> en colonnes nommées d'après les <th> du thead
        correspondant (mapping par index). Résultat : chaque <tr> =
        une row complète avec colonnes ("Clt.", "Cavalier", etc.) — au
        lieu d'une seule colonne "tr" avec tout le texte concaténé,
        qui n'est pas exploitable. Couvre le cas courant des tableaux
        de résultats avec header propre (#t_engts, etc.).

        Permet de garder l'association naturelle des données quand on
        sélectionne plusieurs colonnes d'un tableau (date + discipline
        + épreuve atterrissent sur la même ligne CSV) sans pour autant
        forcer la récupération de cellules non demandées.
        """
        from urllib.parse import urljoin

        # Groupe les matches par <tr> ancestral. Si pas de <tr>, le
        # groupe = l'élément lui-même (ungrouped).
        groups: dict[int, dict] = {}
        order: list[int] = []  # préserve l'ordre d'apparition

        # Permet de passer directement une liste d'éléments (ancien
        # signature) — on suppose alors un sélecteur vide.
        if tagged_elems and not isinstance(tagged_elems[0], tuple):
            tagged_elems = [('', e) for e in tagged_elems]

        for matched_sel, el in tagged_elems:
            # Cas spécial <tr> : explode en colonnes par cellule
            if el.name == 'tr':
                row = self._explode_tr_to_row(el, url)
                if row:
                    key = id(el)
                    groups[key] = row
                    order.append(key)
                continue

            text = el.get_text(separator=' ', strip=True)
            if not text:
                continue

            # Pattern label-value : `<X><strong>Label :</strong>Valeur</X>`
            # → on extrait Label comme nom de colonne et Valeur comme
            # valeur. Couvre les fiches FFE Telemat où chaque méta du
            # cheval (Robe, Sexe, Taille, Père, Mère…) est dans un
            # <span class="label"> séparé.
            #
            # Stratégie de grouping :
            #   - Si l'élément est DANS un <tr> (table de fiches multi-
            #     rows) → group_key = id(tr) → 1 row par tr.
            #   - Sinon (page-fiche d'une seule entité, cas FFE cheval)
            #     → group_key = ('label_page', url) → TOUS les labels
            #     de la page atterrissent dans 1 SEULE row. Permet
            #     d'agréger Robe + Sexe + Taille + Père + Mère + …
            #     en une fiche unique exploitable directement.
            label_pair = _extract_label_value(el)
            if label_pair is not None:
                lbl, val = label_pair
                tr_anc = el.find_parent('tr')
                if tr_anc is not None:
                    group_key = id(tr_anc)
                else:
                    group_key = ('label_page', url)
                if group_key not in groups:
                    groups[group_key] = {'source_url': url}
                    order.append(group_key)
                row = groups[group_key]
                col = _normalize_col(lbl)
                if col and col not in row:
                    row[col] = val
                # Si la valeur a un href (cas Père/Mère/Naisseur lien),
                # on capture aussi <Label>_url
                href = self._find_href(el)
                if href:
                    abs_h = urljoin(url, href)
                    if 'href' not in row:
                        row['href'] = abs_h
                    url_col = f'{col}_url'
                    if col and url_col not in row:
                        row[url_col] = abs_h
                continue

            # Skip les éléments qui ont la classe "label" mais n'ont
            # PAS le pattern label-value (pas de `:` dans le strong).
            # Cas typique FFE : <span class="label"><strong>Club</strong></span>
            # qui marque juste une catégorie sans donnée associée — on
            # créerait sinon une row parasite {label: "Club"} qui pollue.
            cls_list = el.get('class') or []
            if 'label' in cls_list:
                continue

            tr = el.find_parent('tr')
            group_key = id(tr) if tr is not None else id(el)

            if group_key not in groups:
                groups[group_key] = {'source_url': url}
                order.append(group_key)
            row = groups[group_key]

            # Href associé : on garde le PREMIER trouvé pour le groupe
            # (souvent les <a> du <tr> pointent vers la même page).
            if 'href' not in row:
                href = self._find_href(el)
                if href:
                    row['href'] = urljoin(url, href)

            # Nom de colonne : sélecteur utilisateur > data-label >
            # aria-label > classe > tag. C'est un fallback strict —
            # on n'inonde pas la row avec les descendants non demandés.
            sel_name = _selector_to_col_name(matched_sel)
            col = _normalize_col(
                sel_name
                or el.get('data-label')
                or el.get('aria-label')
                or _first_meaningful_class(el)
                or (el.name or '')
            )
            if col and col not in row:
                row[col] = text

        # Filtre les groupes vides puis renvoie dans l'ordre.
        rows: list[dict] = []
        for k in order:
            row = groups[k]
            has_named = any(c not in ('source_url', 'href', 'image') for c in row)
            if has_named or 'href' in row:
                rows.append(row)
        return rows

    def _explode_tr_to_row(self, tr: Tag, url: str) -> dict | None:
        """
        Transforme un <tr> matché en UNE row avec une colonne par <td>,
        nommée d'après le <th> correspondant dans le thead (ou index de
        colonne si pas de header). Le href de la row = premier lien non
        vide trouvé dans les cellules.
        """
        from urllib.parse import urljoin

        # Skip les <tr> du <thead> : ils contiennent les libellés de
        # colonnes (Clt., Cavalier, …) qui SERVENT à nommer les colonnes
        # mais ne sont PAS de la donnée. Sans ce skip, un sélecteur
        # comme `#t_engts tr` matche aussi le <tr> d'en-tête → on émet
        # une row data où chaque cellule = nom de la colonne (ex.
        # row {Clt.: "Clt.", Cavalier: "Cavalier", …}). Bug visible sur
        # 17 CSV de runs précédents.
        if tr.find_parent('thead') is not None:
            return None

        # Cas tableau d'épreuve par équipe (FFE Telemat) :
        #   <tr class="ligneequipe">      → header d'équipe, contient
        #                                    nom + classement
        #   <tr class="tablesorter-childRow"> → membre de l'équipe,
        #                                    avec son cavalier+équidé
        # On SKIP la ligneequipe — son contenu est propagé aux membres
        # ci-dessous. Sinon on aurait une row "équipe" parasite avec
        # un nom dans la colonne Cavalier.
        tr_classes = tr.get('class') or []
        if 'ligneequipe' in tr_classes:
            return None

        # Récupère la table parente et les headers une seule fois
        table = tr.find_parent('table')
        headers: list[str] = []
        if table is not None:
            # Première tr contenant des <th> = ligne d'en-têtes
            for candidate in table.find_all('tr'):
                ths = candidate.find_all('th', recursive=False)
                if ths:
                    headers = [
                        h.get_text(' ', strip=True) or f'col_{i}'
                        for i, h in enumerate(
                            candidate.find_all(['th', 'td'], recursive=False)
                        )
                    ]
                    break

        cells = tr.find_all(['td', 'th'], recursive=False)
        if not cells:
            return None

        # Cas table sans <thead> explicite (FFE Telemat) : si TOUTES les
        # cellules de ce tr sont des <th>, c'est la ligne d'en-tête
        # implicite — pareil que thead, on skip pour ne pas la dupliquer
        # comme data.
        if all(c.name == 'th' for c in cells):
            return None

        # ── Détection contexte équipe ─────────────────────────────────
        # Si ce tr est un membre (`tablesorter-childRow`), on remonte
        # les siblings vers le haut pour trouver la `ligneequipe` la
        # plus proche, et on en extrait :
        #   - le nom de l'équipe (texte de la cellule "Cavalier",
        #     préfixé "Équipe :")
        #   - les valeurs de classement (Clt., Pts qualif., Quart)
        #     qui ne sont présentes QUE sur la ligne équipe — pas
        #     sur les membres → sans cette propagation, le membre
        #     sort sans classement, info perdue.
        team_name        = 'INDIVIDUEL'
        team_row_data: dict[str, str] = {}
        if 'tablesorter-childRow' in tr_classes:
            sib = tr.find_previous_sibling('tr')
            while sib is not None:
                sib_cls = sib.get('class') or []
                if 'ligneequipe' in sib_cls:
                    sib_cells = sib.find_all(['td', 'th'], recursive=False)
                    # Récupère nom équipe : 1re cellule textuelle non-Clt.
                    # En général index 1 (la 2e cellule) car index 0 = Clt.
                    if len(sib_cells) >= 2:
                        raw = sib_cells[1].get_text(' ', strip=True)
                        team_name = re.sub(
                            r'^Équipe\s*:\s*', '', raw, flags=re.IGNORECASE,
                        ).strip() or 'INDIVIDUEL'
                    # Récupère les autres cellules (Clt., Pts, Quart) avec
                    # leur header pour qu'on puisse fusionner par nom
                    # de colonne dans la row enfant.
                    for j, sc in enumerate(sib_cells):
                        # Skip la cellule du nom équipe (sera traité à part)
                        if j == 1:
                            continue
                        col_raw = (
                            (headers[j] if j < len(headers) and headers[j] else '')
                            or sc.get('data-label')
                            or ''
                        )
                        col = _normalize_col(col_raw)
                        if not col:
                            continue
                        val = sc.get_text(' ', strip=True)
                        # Idem que la boucle principale : ne nettoyer
                        # les ordinaux que sur les colonnes ranking,
                        # pas sur les noms de cavaliers/équidés.
                        if _is_ranking_col(col):
                            val = _clean_ordinal(val)
                        if val:
                            team_row_data[col] = val
                    break
                # Si on tombe sur autre chose qu'un childRow (et qui
                # n'est pas la ligneequipe), on n'est plus dans le
                # contexte équipe : on s'arrête.
                if 'tablesorter-childRow' not in sib_cls:
                    break
                sib = sib.find_previous_sibling('tr')

        row: dict = {'source_url': url}
        first_href = ''
        # Flag levé si on rencontre un marqueur de non-participation
        # (forfait, non-partant, hors-concours) → la row entière sera
        # filtrée (return None à la fin). On ne peut pas faire un
        # return None inline ici car on veut quand même finir d'extraire
        # toutes les cellules (au cas où le marqueur n'est qu'en milieu
        # de row, on doit s'assurer qu'il n'est PAS dans une colonne
        # texte légitime — on le fait via _is_did_not_participate qui
        # est strict sur le pattern).
        skip_row = False
        for i, cell in enumerate(cells):
            col = _normalize_col(
                (headers[i] if i < len(headers) and headers[i] else '')
                or cell.get('data-label')
                or f'col_{i}'
            )
            # Valeur texte de la cellule (accepte texte concaténé des descendants)
            val = cell.get_text(' ', strip=True)
            # Si la cellule contient un <a>, on préfère son texte lisible
            # au texte brut qui peut être vide (cas des icônes seules)
            if not val:
                a = cell.find('a')
                if a is not None:
                    val = a.get_text(' ', strip=True) or a.get('title', '')
            # Détection forfait/non-partant/hors-concours AVANT clean :
            # ces statuts font qu'on filtre la row entière (vrai absence
            # de résultat, ne doit pas polluer le dataset comme un -1).
            if _is_did_not_participate(val):
                skip_row = True

            # Normalisations RANKING UNIQUEMENT — sinon on casserait
            # des noms textuels (ex: cheval "El Caballo" → "-1" parce
            # que `_clean_ordinal` matche "el" comme abréviation
            # d'éliminé). On limite donc :
            #   - `_clean_ordinal`  ("1 re" → "1", "El." → "-1")
            #   - `_force_ranking_value` (texte non-num → "-1")
            # aux seules colonnes dont le nom ressemble à un classement
            # ou un score (Clt., Pts qualif, Quart, Score, Note...).
            if col and _is_ranking_col(col):
                val = _clean_ordinal(val)
                val = _force_ranking_value(val)
            if col and val:
                # Évite d'écraser une colonne déjà remplie (cas rare de
                # doublon dans le tr, ex: deux <td> avec même header)
                if col not in row:
                    row[col] = val

            # Capture le href PROPRE à cette cellule dans une colonne
            # `<col>_url`. Indispensable pour les tables où chaque
            # cellule pointe vers une fiche distincte (FFE : Cavalier,
            # Club, Équidé, Coach ont chacun leur lien) — sans ça, on
            # ne récupère que le 1er lien et l'utilisateur ne peut
            # répéter le crawl que sur une seule des dimensions.
            cell_href = ''
            a = cell if cell.name == 'a' else cell.find('a')
            if a is not None:
                h = self._extract_anchor_href(a)
                if h:
                    cell_href = urljoin(url, h)
            if cell_href and col:
                url_col = f'{col}_url'
                if url_col not in row:
                    row[url_col] = cell_href
            # Premier href de la row (compat ascendante : `href` reste
            # le 1er lien non vide pour le dédup et les anciens consumers)
            if cell_href and not first_href:
                first_href = cell_href
        if first_href:
            row['href'] = first_href

        # Skip row entière si statut non-participation détecté.
        if skip_row:
            return None

        # Injection de la colonne `Équipe` :
        #   - Si la row vient d'un membre d'équipe (childRow avec une
        #     ligneequipe parente), on a extrait le nom de l'équipe
        #     plus haut dans `team_name`.
        #   - Sinon, c'est une participation individuelle → "INDIVIDUEL".
        # Permet de filtrer/trier facilement par type d'épreuve.
        row['Équipe'] = team_name

        # Fusion des données équipe (Clt., Pts qualif., Quart) qui ne
        # sont présentes que sur la ligneequipe parente — on les
        # propage au membre uniquement si sa propre cellule est vide.
        for col, val in team_row_data.items():
            if not row.get(col):
                row[col] = val

        # Une row n'est utile que si au moins une colonne de donnée a été remplie
        has_data = any(k not in ('source_url', 'href', 'Équipe') for k in row)
        return row if has_data else None

    def _find_href(self, el: Tag) -> str | None:
        """
        Cherche le href associé à un élément :
        1) L'élément lui-même si c'est un <a>
        2) Un <a> enfant
        3) Un <a> parent (remonte jusqu'à 5 niveaux)

        Si le href trouvé est `javascript:void(0)` ou similaire, on
        tente d'extraire l'URL réelle depuis l'attribut `onclick` (cas
        FFE SIF qui utilise $.ajax({url:'./?cs=...'}) pour la pagination).
        """
        # L'élément est un <a>
        if el.name == 'a':
            r = self._extract_anchor_href(el)
            if r:
                return r

        # Enfant <a>
        child_a = el.find('a')
        if child_a is not None:
            r = self._extract_anchor_href(child_a)
            if r:
                return r

        # Parent <a> (remonte jusqu'à 5 niveaux)
        parent = el.parent
        depth = 0
        while parent and depth < 5:
            if isinstance(parent, Tag) and parent.name == 'a':
                r = self._extract_anchor_href(parent)
                if r:
                    return r
            parent = parent.parent
            depth += 1

        return None

    _AJAX_URL_RE = re.compile(r"""url\s*:\s*['"]([^'"]+)['"]""")

    def _extract_anchor_href(self, a: Tag) -> str | None:
        """
        Retourne l'URL d'un <a>. Si href=javascript:void(0), extrait
        l'URL depuis onclick. Retourne None si rien d'utilisable.
        """
        href = (a.get('href') or '').strip()
        if href and not href.startswith('#') and not href.lower().startswith('javascript'):
            return href
        onclick = a.get('onclick') or ''
        if onclick:
            m = self._AJAX_URL_RE.search(onclick)
            if m:
                extracted = m.group(1).strip()
                if extracted:
                    return extracted
        return None

    def _fetch(self, url: str, cookies: dict[str, str] | None = None) -> str:
        """Fetche la page — bascule sur playwright si le contenu est JS-heavy.

        `cookies` (optionnel) : passe-plat pour la session authentifiée
        (auth_store côté step_runner). Sans cookies sur un site protégé
        (ex. FFE Telemat), on récupère la page de modal de connexion
        au lieu du contenu réel.
        """
        if self._force_dynamic:
            self._log.debug('  Mode dynamique forcé → playwright')
            return self._fetch_dynamic(url)

        html = self._fetch_static(url, cookies=cookies)
        if self._is_js_heavy(html):
            self._log.info('  Page JS détectée → bascule playwright')
            return self._fetch_dynamic(url)

        return html

    def _fetch_static(self, url: str, cookies: dict[str, str] | None = None) -> str:
        # ── Shutdown coopératif : ne lance JAMAIS de nouvelle requête
        # si l'app est en train de s'arrêter. Évite la rafale de 600
        # requêtes au Ctrl+C qui fait bannir l'IP.
        if _abort_event.is_set():
            raise RuntimeError('crawl aborted (shutdown in progress)')

        # ── Anti-bannissement : respect du circuit breaker global ─────
        # Si un autre worker a déjà ouvert le breaker (≥ 3 × 429 dans
        # 30s), on attend que la pause expire AVANT d'envoyer notre
        # propre requête. Évite l'escalade 429 → 403 → IP block.
        _wait_if_rate_limited()

        # Proxy optionnel (cf. _get_proxy_dict) — permet de rouler par Tor
        # ou un proxy rotatif quand l'IP locale s'est fait ban par FFE.
        proxies = _get_proxy_dict()

        # timeout=(connect, read) : certains sites de back-office (FFE SIF,
        # intranets, etc.) sont lents à générer leur HTML. 15s pour la
        # TCP handshake, 60s pour lire la réponse complète.
        try:
            resp = requests.get(url, headers=self._headers, cookies=cookies, proxies=proxies, timeout=(15, 60))
        except requests.exceptions.ReadTimeout:
            self._log.warn(f'  ReadTimeout sur {url} — retry avec timeout étendu')
            resp = requests.get(url, headers=self._headers, cookies=cookies, proxies=proxies, timeout=(15, 120))

        # ── Retry exponentiel sur 429 (Too Many Requests) ────────────
        # FFE Telemat rate-limit dès qu'on dépasse ~5-10 req/s. On
        # respecte si possible `Retry-After`, sinon backoff 2s → 4s → 8s.
        # Plafonné à 3 tentatives. EN PLUS, chaque 429 incrémente le
        # compteur du circuit breaker — si on en encaisse 3 dans 30s,
        # une pause globale de 90s est imposée à tous les workers.
        attempts = 0
        while resp.status_code == 429 and attempts < 3:
            attempts += 1
            broke = _record_429()
            try:
                retry_after = float(resp.headers.get('Retry-After') or 0)
            except ValueError:
                retry_after = 0
            wait = retry_after if retry_after > 0 else (2 ** attempts)
            wait = min(wait, 30)
            if broke:
                self._log.warn(
                    f'  ⛔ 3 × 429 en 30s — pause GLOBALE 90s pour ne pas se '
                    f'faire blacklister par FFE. Tous les workers attendent.'
                )
            else:
                self._log.info(
                    f'  429 sur {url[:60]}… → attente {wait:.0f}s puis retry {attempts}/3'
                )
            time.sleep(wait)
            _wait_if_rate_limited()      # re-vérifie en cas de breaker
            try:
                resp = requests.get(url, headers=self._headers, cookies=cookies, proxies=proxies, timeout=(15, 60))
            except requests.exceptions.ReadTimeout:
                resp = requests.get(url, headers=self._headers, cookies=cookies, proxies=proxies, timeout=(15, 120))

        resp.raise_for_status()
        # On force UTF-8 si le serveur ne le précise pas — évite les mojibake
        resp.encoding = resp.apparent_encoding or 'utf-8'
        return resp.text

    def _fetch_dynamic(self, url: str) -> str:
        """Playwright : rend la page JS côté client et retourne le HTML final."""
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            self._log.fatal('playwright non installé — pip install playwright && playwright install chromium')
            raise

        proxy_url    = None
        launch_kwargs: dict = {'headless': True}
        try:
            import os
            for key in ('EQUIRANK_PROXY', 'HTTPS_PROXY', 'HTTP_PROXY',
                        'https_proxy', 'http_proxy'):
                v = os.environ.get(key)
                if v:
                    proxy_url = v
                    break
            if proxy_url:
                launch_kwargs['proxy'] = {'server': proxy_url}
        except Exception:
            pass

        with sync_playwright() as pw:
            browser = pw.chromium.launch(**launch_kwargs)
            page    = browser.new_page(
                extra_http_headers={'Accept-Language': 'fr-FR,fr;q=0.9'},
                user_agent=self._headers['User-Agent'],
            )
            page.goto(url, wait_until='networkidle', timeout=30_000)
            self._handle_load_more(page)
            self._handle_infinite_scroll(page)
            html = page.content()
            browser.close()

        return html

    def _click_pagination(self, page, page_num: int) -> bool:
        """
        Cherche et clique un bouton de pagination pour la page N.
        Retourne True si un bouton a été cliqué, False sinon.
        """
        try:
            clicked = page.evaluate('''(targetPage) => {
                const allBtns = [...document.querySelectorAll('button, a')];
                const numBtns = allBtns.filter(b => {
                    const t = b.textContent.trim();
                    return /^\\d+$/.test(t) && parseInt(t) <= 100;
                });

                const groups = new Map();
                for (const b of numBtns) {
                    const p = b.parentElement;
                    if (!p) continue;
                    if (!groups.has(p)) groups.set(p, []);
                    groups.get(p).push(b);
                }

                let bestGroup = null;
                let bestCount = 0;
                for (const [parent, btns] of groups) {
                    if (btns.length > bestCount) {
                        bestCount = btns.length;
                        bestGroup = btns;
                    }
                }

                if (!bestGroup || bestCount < 2) return false;

                const target = bestGroup.find(b => b.textContent.trim() === String(targetPage));
                if (target) {
                    target.click();
                    return true;
                }
                return false;
            }''', page_num)
            return bool(clicked)
        except Exception as exc:
            self._log.debug(f'  Pagination error: {exc}')
            return False

    def _handle_load_more(self, page) -> None:
        """Clique les boutons 'load more' / 'show all' / pagination tant qu'il y en a."""
        _LOAD_MORE_SELECTORS = [
            'button:has-text("load more")',
            'button:has-text("charger plus")',
            'button:has-text("voir plus")',
            'button:has-text("show more")',
            'button:has-text("show all")',
            'button:has-text("tout afficher")',
            '[class*="load-more"]',
            '[class*="loadMore"]',
            '[class*="show-more"]',
            '[class*="showMore"]',
        ]
        MAX_CLICKS = 20

        for _ in range(MAX_CLICKS):
            clicked = False
            for sel in _LOAD_MORE_SELECTORS:
                try:
                    btn = page.locator(sel).first
                    if btn.is_visible(timeout=500):
                        btn.click()
                        page.wait_for_timeout(1000)
                        clicked = True
                        self._log.debug(f'  → clic "load more" ({sel})')
                        break
                except Exception:
                    continue
            if not clicked:
                break

    def _handle_infinite_scroll(self, page) -> None:
        """Scrolle vers le bas jusqu'à ce qu'il n'y ait plus de nouveau contenu."""
        MAX_SCROLLS = 30
        prev_height = 0

        for _ in range(MAX_SCROLLS):
            page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
            page.wait_for_timeout(600)
            height = page.evaluate('document.body.scrollHeight')
            if height == prev_height:
                break
            prev_height = height

    def _is_js_heavy(self, html: str) -> bool:
        """
        Heuristique rapide pour détecter les SPAs / pages JS-rendered.
        On ne veut pas lancer playwright pour une page statique — c'est lent.
        """
        indicators = [
            'window.__NEXT_DATA__',         # Next.js
            'window.__NUXT__',              # Nuxt.js
            '<div id="root"></div>',         # React vide
            '<div id="app"></div>',          # Vue vide
            'ng-version=',                  # Angular
            '__vue_app__',                  # Vue
            'data-reactroot',               # React legacy
        ]
        return any(indicator in html for indicator in indicators)

    def _run_extractors(self, element: Tag, url: str) -> list[dict]:
        """Lance tous les extracteurs pertinents sur un élément."""
        rows: list[dict] = []
        for extractor in self._extractors:
            if extractor.can_handle(element):
                try:
                    extracted = extractor.extract(element, url)
                    rows.extend(extracted)
                    if extracted:
                        self._log.debug(
                            f'    [{extractor.name}] → {len(extracted)} ligne(s)'
                        )
                except Exception as exc:
                    self._log.warn(f'    [{extractor.name}] erreur : {exc}')
        return rows

    def _save(self, results: list[CrawlResult], query_name: str) -> None:
        if not results:
            return
        session_dir = self._output.save(
            results,
            query_slug=query_name,
            save_raw=self._save_raw,
            download_media=self._download_media,
        )
        total_rows = sum(len(r.rows) for r in results)
        self._log.box('Résultats sauvegardés', [
            f'Dossier  : {session_dir}',
            f'Fichiers : {len(results)} URL(s) crawlée(s)',
            f'Lignes   : {total_rows} ligne(s) extraite(s)',
        ])


# ── Helpers ────────────────────────────────────────────────────────────────────

# Filtre inline (dupliqué depuis extractors/recursive_extractor.py pour
# éviter un import circulaire : detection/field_detector.py importe
# Crawler, extractors importe detection, donc crawler ne peut pas
# importer extractors au top-level).
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
_UTILITY_CLASS_EXACT = {'row', 'col', 'container', 'small', 'lead', 'show', 'hide'}


def _fill_columns_from_descendants(root: 'Tag', row: dict) -> None:
    """
    Parcourt les descendants d'un élément matché et ajoute une colonne
    par sous-élément nommé (data-label > th parent > classe sémantique).
    Pensé pour les tableaux FFE : un <tr> en entrée, ses <td data-label="…">
    deviennent autant de colonnes dans la même row.
    """
    if not isinstance(root, Tag):
        return

    # Index th parent : pour chaque <td> on retrouve sa <th> de colonne
    # en comptant l'index dans la <tr> + mapping headers.
    th_map: dict['Tag', str] = {}
    table = root if root.name == 'table' else root.find_parent('table')
    if table is not None:
        header_tr = None
        for candidate in table.find_all('tr'):
            if candidate.find_all('th', recursive=False):
                header_tr = candidate
                break
        if header_tr is not None:
            headers = [h.get_text(strip=True) for h in header_tr.find_all(['th', 'td'], recursive=False)]
            for tr in table.find_all('tr'):
                tds = tr.find_all(['td', 'th'], recursive=False)
                for i, td in enumerate(tds):
                    if i < len(headers) and headers[i]:
                        th_map[td] = headers[i]

    for node in root.descendants:
        if not isinstance(node, Tag):
            continue
        # On ignore les conteneurs purs : ils ont déjà leur texte
        # pris en compte par leurs enfants. On ne capture QUE les
        # feuilles "nommées".
        col_name = ''
        # 1. data-label / aria-label / data-column
        for attr in ('data-label', 'aria-label', 'data-column'):
            v = node.get(attr)
            if v:
                col_name = _normalize_col(v)
                break
        # 2. th parent (tableau)
        if not col_name and node in th_map:
            col_name = _normalize_col(th_map[node])
        # 3. classe sémantique (sauf si l'élément a des enfants Tag —
        #    alors le leaf interne fera le boulot pour sa propre classe)
        if not col_name:
            if any(isinstance(c, Tag) for c in node.children):
                continue
            cls = _first_meaningful_class(node)
            if cls:
                col_name = cls

        if not col_name or col_name in row:
            continue

        # Valeur : pour un <a> on préfère le texte ; pour <img> l'alt ou src
        if node.name == 'a':
            val = node.get_text(strip=True) or node.get('href', '')
        elif node.name == 'img':
            val = node.get('alt') or node.get('src') or ''
        elif node.name == 'input':
            val = node.get('value') or node.get('placeholder') or ''
        else:
            val = node.get_text(separator=' ', strip=True)
        if val:
            row[col_name] = val


def _selector_to_col_name(sel: str) -> str:
    """
    Convertit un sélecteur CSS en nom de colonne lisible.
    Ex : ".discipline" → "discipline", ".col-md-3 .titre" → "titre",
    "td[data-label='Date']" → "Date".
    """
    if not sel:
        return ''
    s = sel.strip()
    # Si le sélecteur cible un attribut, en extraire la valeur
    import re as _re
    m = _re.search(r'''\[([\w-]+)=['"]?([^'"\]]+)['"]?\]''', s)
    if m and m.group(1) in ('data-label', 'aria-label', 'data-column', 'name', 'id'):
        return m.group(2).strip()
    # Sinon, dernier segment du sélecteur (après dernier espace), classe sans le point
    last = s.split()[-1].split('>')[-1].split('+')[-1].split('~')[-1].strip()
    last = last.lstrip('.#')
    # Coupe sur les pseudo-sélecteurs ou attributs
    for sep in ('[', ':'):
        if sep in last:
            last = last.split(sep)[0]
    return last


def _normalize_col(raw: str) -> str:
    cleaned = ' '.join((raw or '').split())
    return cleaned[:80]


# Suffixes ordinaux français (avec ou sans accent, séparateur optionnel).
# On veut matcher "1 re", "1re", "1ère", "1 er", "2 e", "3 ème", "21e",
# "3 e ex" / "8 e ex" (ex-aequo) etc.
# La capture ne se déclenche QUE si un suffixe ordinal est présent — sinon
# une simple valeur numérique comme "20" (points qualif) reste intacte.
_ORDINAL_RE = re.compile(
    r'^\s*(\d+)\s*(?:er|ère|ere|re|nd|nde|ème|eme|e)(?:\s*(?:ex|ex\.?|ex\s*aequo|ex\s*æquo))?\s*\.?\s*$',
    re.IGNORECASE,
)


# Heuristique : nom de colonne qui ressemble à un classement / score.
# Sert à forcer les valeurs textuelles ("Cavalier inconnu", "Niveau
# trop élevé", "Nb part. trop élevé"…) à -1 dans ces colonnes.
_RANKING_COL_RE = re.compile(
    r'\b(?:clt|classement|class|rang|place|quart|pts|points?|score|note)\b',
    re.IGNORECASE,
)
_NUMERIC_VAL_RE = re.compile(r'^-?\d+(?:\.\d+)?$')


def _is_ranking_col(col_name: str) -> bool:
    """True si le nom de colonne ressemble à un classement / score."""
    return bool(_RANKING_COL_RE.search(col_name or ''))


def _force_ranking_value(val: str) -> str:
    """
    Sur une colonne classement/score, on n'accepte que des valeurs
    NUMÉRIQUES (0, -1, 25, 12.5…). Tout texte (ex: "Cavalier ou
    licence inconnu", "Niveau examen trop élevé", "Nb part. cavalier
    trop élevé", "NONE") est remplacé par -1 — signal clair d'absence
    de résultat exploitable, sans avoir à hardcoder chaque message FFE.
    """
    s = (val or '').strip()
    if not s:
        return val
    if _NUMERIC_VAL_RE.match(s):
        return s
    return '-1'


def _extract_label_value(el) -> tuple[str, str] | None:
    """
    Détecte le pattern `<X><strong>Label :</strong>Valeur</X>` et
    retourne (label_clean, value_clean). Retourne None si l'élément
    n'a pas cette structure.

    Cas couvert : fiches FFE Telemat (cheval / cavalier / club) où
    chaque méta est dans un <span class="label"> contenant un <strong>
    avec un séparateur ":" suivi du contenu. Permet de transformer
    des spans isolés en colonnes nommées (Robe, Sexe, Taille…) au lieu
    d'une colonne unique "label" répétée pour chaque méta.

    Le séparateur est tolérant : ":" / " :" / ":  " / etc. La valeur
    peut contenir n'importe quoi (texte simple, lien <a>, icônes…) —
    on prend le texte concaténé après extraction du strong.
    """
    if not isinstance(el, Tag):
        return None
    # Premier <strong> direct ou descendant du début
    strong = el.find('strong')
    if strong is None:
        return None
    label_raw = strong.get_text(' ', strip=True)
    if not label_raw:
        return None
    # Doit se terminer par ":" (avec ou sans espace) — sinon ce n'est
    # pas un label.
    if ':' not in label_raw:
        return None
    label_clean = label_raw.rstrip().rstrip(':').rstrip().strip()
    if not label_clean:
        return None
    # Valeur = texte total de l'élément moins le texte du strong.
    full_text = el.get_text(' ', strip=True)
    strong_text = strong.get_text(' ', strip=True)
    if full_text.startswith(strong_text):
        value = full_text[len(strong_text):].strip()
    else:
        # Cas rare : le strong n'est pas en tête (ex: icône avant).
        # On tombe sur un fallback : on enlève la 1ère occurrence.
        value = full_text.replace(strong_text, '', 1).strip()
    # Nettoie les séparateurs résiduels en début (": ", "- ", etc.)
    value = value.lstrip(':').lstrip('-').strip()
    return label_clean, value


def _clean_ordinal(val: str) -> str:
    """
    "1 re" / "2 e" / "1 er" / "3ème" → "1" / "2" / "1" / "3".
    Retourne `val` inchangé si la chaîne ne ressemble pas à un classement.
    Utilisé pour nettoyer les colonnes Clt./Quart côté FFE Telemat où le
    HTML contient un `<sup>` après le chiffre — texte concaténé donne
    "1 er" qu'on veut juste "1" pour pouvoir trier numériquement.

    Cas spécial "Éliminé" → "-1" : le cheval a couru mais été disqualifié.
    On encode comme entier négatif pour que la colonne reste numérique
    (sortable, statistiquement traitable) tout en restant distinct de 0
    (qui est le fill pour donnée manquante depuis le passage récent).
    "El.", "EL", "Eliminé", "Élim", "Disq" et variantes sont reconnus.
    """
    if not val:
        return val
    s = val.strip()
    if _ELIMINATED_RE.match(s):
        return '-1'
    m = _ORDINAL_RE.match(s)
    return m.group(1) if m else val


# Marqueurs d'élimination "actif" : le cheval a couru mais a été
# disqualifié, éliminé ou a abandonné en cours d'épreuve. La row reste
# dans le dataset, juste avec rang/points = -1.
# `np` (non placé) et `nc` (non classé) → ambiguës en équitation : on
# les traite comme éliminés (a participé mais sans rang final).
_ELIMINATED_RE = re.compile(
    r'^(?:el|eli|elim|élim|eliminé|éliminé|elimine|disq|dsq|dq'
    r'|abandon|abd|nc|np)(?:\.|\s|$)',
    re.IGNORECASE,
)

# Marqueurs de NON-PARTICIPATION : forfait, non-partant, hors-concours.
# La row est SUPPRIMÉE du dataset — pas de score, pas de classement,
# le cheval/cavalier n'a pas couru du tout. Encoder en -1 fausserait
# les stats (un forfait n'est pas un mauvais résultat, c'est l'absence
# de résultat). Détecté n'importe où dans la row par _explode_tr_to_row.
_DID_NOT_PARTICIPATE_RE = re.compile(
    r'^(?:'
    r'non[\s\-]*partant\b'
    r'|forfait\b'
    r'|hors[\s\-]*concours\b'
    r'|fft\b'
    r'|hc\b'
    # Épreuves annulées : la row n'a pas de sens dans le dataset
    # (l'épreuve elle-même n'a pas eu lieu). Match "annulé(e)" /
    # "épreuve annulée" en début de cellule. Tolère É/e en initiale.
    r'|(?:[ée]preuve\s+)?annul(?:é|ée|ee|e)\b'
    r')',
    re.IGNORECASE,
)


def _is_did_not_participate(val: str) -> bool:
    """True si la valeur signale une non-participation (forfait, non-
    partant, hors-concours). Utilisé pour filtrer la row complète."""
    if not val:
        return False
    return bool(_DID_NOT_PARTICIPATE_RE.match(val.strip()))


def _first_meaningful_class(el) -> str:
    """
    Retourne la première classe CSS "sémantique" d'un élément (filtre
    les classes utilitaires Bootstrap/Tailwind). Utilisée pour nommer
    les colonnes du CSV d'après le nom de classe de l'élément matché
    par data_selector, plutôt que de tout entasser dans "text".
    """
    if not isinstance(el, Tag):
        return ''
    classes = el.get('class') or []
    for c in classes:
        if not c or len(c) < 3:
            continue
        if c in _UTILITY_CLASS_EXACT:
            continue
        if any(c.startswith(p) for p in _UTILITY_CLASS_PREFIXES):
            continue
        cleaned = ' '.join(c.split())
        return cleaned[:80]
    return ''


def _is_css_selector(selector: str) -> bool:
    """Détecte si la chaîne est un sélecteur CSS (contient . # , [ > ~ +)."""
    return bool(re.search(r'[.#,\[\]>~+:]', selector))


def _class_selector(class_name: str):
    """
    Construit le sélecteur BeautifulSoup pour la classe.
    Accepte 'my-class' ou '.my-class' ou 'class1 class2' (multi-classe).
    Note : pour les sélecteurs CSS complexes (.a, .b), utiliser soup.select().
    """
    name = class_name.lstrip('.')
    # Multi-classe : "foo bar" → doit avoir les deux classes
    if ' ' in name:
        parts = name.split()
        return re.compile(r'(?<!\S)' + r'(?=.*\b'.join(re.escape(p) for p in parts))
    return re.compile(r'(?:^|\s)' + re.escape(name) + r'(?:\s|$)')


def _now() -> str:
    from datetime import datetime
    return datetime.now().isoformat(timespec='seconds')


# ── CLI ────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Web Crawler Toolbox — extrait le contenu d\'une classe CSS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--url',  help='URL unique à crawler')
    group.add_argument('--urls', type=Path, help='Fichier texte (une URL par ligne)')

    p.add_argument('--class',    dest='class_name', required=True, help='Nom de la classe CSS cible')
    p.add_argument('--output',   default='',        help='Slug du dossier de sortie (optionnel)')
    p.add_argument('--dynamic',  action='store_true', help='Forcer playwright (défaut: auto-detect)')
    p.add_argument('--raw',      action='store_true', help='Sauvegarder le HTML brut')
    p.add_argument('--download-media', action='store_true', help='Télécharger les médias trouvés')
    p.add_argument('--delay',    type=float, default=1.0, help='Délai entre requêtes batch (sec, défaut: 1)')
    p.add_argument('--debug',    action='store_true', help='Activer les logs DEBUG')

    return p


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    log_level = LogLevel.DEBUG if args.debug else LogLevel.INFO

    # Le fichier log va dans crawlresult/ — on le crée après avoir le slug
    from output.file_writer import CRAWLRESULT_ROOT, _slugify
    slug    = args.output or (args.url or str(args.urls))
    log_dir = CRAWLRESULT_ROOT / _slugify(slug)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / 'crawler.log'

    # Reset pour que le logger prenne la bonne config de cette session
    CrawlerLogger.reset()
    CrawlerLogger.get_instance(min_level=log_level, log_file=log_file)

    crawler = Crawler(
        force_dynamic=args.dynamic,
        save_raw=args.raw,
        download_media=args.download_media,
        log_level=log_level,
        log_file=log_file,
    )

    if args.url:
        crawler.crawl(args.url, args.class_name)

    elif args.urls:
        if not args.urls.exists():
            print(f'[ERREUR] Fichier introuvable : {args.urls}', file=sys.stderr)
            sys.exit(1)
        urls = [line.strip() for line in args.urls.read_text().splitlines() if line.strip()]
        results: list[CrawlResult] = []
        crawler.crawl_batch(urls, args.class_name, results, delay=args.delay)


if __name__ == '__main__':
    main()
