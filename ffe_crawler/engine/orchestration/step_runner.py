# ============================================================
# © 2026 Louis Guillory (Silence Wolf) — Tous droits réservés
# Licence : PROPRIETARY — voir fichier LICENSE
# ============================================================

# StepRunner : exécute UNE CrawlStep.
# Appelle Crawler.crawl() tel quel (composition, pas héritage), route
# les rows vers data.csv ou urls.csv selon step.url_export_columns, et
# publie les événements au fur et à mesure sur l'EventBus pour le streaming
# WebSocket côté front.

from __future__ import annotations

import re
import time
import warnings
from dataclasses import dataclass, field
from datetime    import datetime
from pathlib     import Path
from typing      import Any, Callable

from bs4 import BeautifulSoup, Tag
from bs4 import XMLParsedAsHTMLWarning

# Certaines pages FFE servent un flux XML (SIF/XHR) qui fait aboyer
# BeautifulSoup quand on l'ouvre avec 'html.parser'. On ignore le
# warning : le parser HTML reste le bon choix pour notre pipeline
# (on veut <tr>/<td> même quand ils sont wrappés en XHTML strict).
warnings.filterwarnings('ignore', category=XMLParsedAsHTMLWarning)

from log     import CrawlerLogger
from crawler import Crawler
from extractors.recursive_extractor import RecursiveExtractor
from output  import CsvWriter, UrlCsvWriter

from detection.language_detector import LanguageDetector
from session.step import CrawlStep
from orchestration.url_filters import (
    build_url_with_filters as _shared_build_url,
    fetch_filtered_html,
)

# Cache module-level du dossier CLI de chaque étape racine. Les étapes
# enfants (répétitions) y écrivent aussi leur <step_id>.csv, de sorte
# que toute la chaîne partage le même dossier d'inspection CLI-style.
# Survit entre instances StepRunner (ChainRunner en crée une nouvelle
# par étape) mais reste scopé au processus sidecar.
_cli_dirs_by_step: dict[str, Path] = {}

from .form_submit import FormSubmitter


class _PreloadedCrawler(Crawler):
    """
    Subclass légère de Crawler qui retourne du HTML pré-fetché pour
    certaines URLs au lieu de refaire un GET. Utilisé après une
    soumission de formulaire — on a déjà la réponse, pas besoin de
    re-fetcher (ce qui re-fait un GET sans le POST et donc perd les
    filtres).
    """

    def __init__(self, preloaded: dict[str, str], **kwargs) -> None:
        super().__init__(**kwargs)
        self._preloaded = preloaded

    def _fetch(self, url: str) -> str:
        if url in self._preloaded:
            return self._preloaded[url]
        return super()._fetch(url)

    # _process_url passe par _fetch_static (puis bascule éventuellement
    # sur Playwright via _is_js_heavy). Si on n'override que _fetch, le
    # HTML préchargé (formulaire soumis, AJAX rendu via Playwright dans
    # url_filters) est ignoré et remplacé par une refetch GET — qui rate
    # les contenus dynamiques (#t_engts tbody tr chargé en AJAX, etc.).
    def _fetch_static(self, url: str) -> str:
        if url in self._preloaded:
            return self._preloaded[url]
        return super()._fetch_static(url)

    # _is_js_heavy aussi bypassé pour ne pas re-déclencher Playwright
    # alors qu'on a déjà le HTML rendu en main : on a confiance dans le
    # contenu préchargé, pas besoin de re-évaluer.
    def _is_js_heavy(self, html: str) -> bool:
        return False


# Type d'une fonction publish(event, session_id). On injecte l'EventBus
# de cette façon pour garder le StepRunner testable sans FastAPI.
PublishFn = Callable[[dict[str, Any], str], None]


@dataclass
class StepResult:
    """Résultat d'une étape de crawl. Sert aussi d'input pour l'étape suivante."""
    step_id:   str
    data_rows: list[dict] = field(default_factory=list)
    url_rows:  list[dict] = field(default_factory=list)


def _has_value(v) -> bool:
    """
    True si la valeur est considérée comme réellement présente dans le
    dataset. Traite les sentinels de remplissage (NONE, -1, chaîne
    vide, None) comme "pas de valeur" — on veut juger la MATIÈRE, pas
    la présence du placeholder.
    """
    if v is None:
        return False
    s = str(v).strip()
    if not s:
        return False
    if s in ('NONE', '-1'):
        return False
    return True


class StepRunner:
    """
    Exécute une CrawlStep en composition sur un Crawler existant.
    N'hérite pas du Crawler et ne le modifie pas.
    """

    def __init__(
        self,
        crawler:   Crawler | None = None,
        publish:   PublishFn | None = None,
        session_dir: Path | None = None,
        auth_id:   str | None = None,
    ) -> None:
        # Crawler par défaut si aucun injecté (utile en scratch/test)
        self._base_crawler = crawler or Crawler()
        self._publish      = publish or (lambda evt, sid: None)
        self._session_dir  = session_dir
        self._log          = CrawlerLogger.get_instance()
        # auth_id opaque retourné par /crawl/auth/login. Si défini,
        # _process_one_url chargera les cookies depuis auth_store et
        # les injectera dans FormSubmitter + Playwright pool.
        self._auth_id      = auth_id

    def _auth_cookies_dict(self) -> dict[str, str] | None:
        """Cookies (name → value) depuis l'auth_store, ou None si pas
        d'auth active. Format dict pour requests.Session."""
        if not self._auth_id:
            return None
        from .auth_store import get_store
        sess = get_store().get(self._auth_id)
        if sess is None:
            return None
        return sess.raw_cookies_dict

    def _auth_cookies_pw(self) -> list[dict] | None:
        """Cookies au format Playwright depuis l'auth_store."""
        if not self._auth_id:
            return None
        from .auth_store import get_store
        sess = get_store().get(self._auth_id)
        if sess is None:
            return None
        return sess.cookies

    def run(self, step: CrawlStep, session_id: str) -> StepResult:
        """
        Joue une étape complète et retourne ses résultats.
        Publie step_started, row/url_row, step_finished (ou error) sur le bus.
        """
        self._log.separator()
        self._log.info(f'▶ Step {step.step_id} — {len(step.entry_urls)} URL(s) en entrée')

        # Le cache module-level _cli_dirs_by_step persiste entre runs —
        # si on ne l'évinçait pas, un re-run écrirait dans le dossier
        # timestamp du run PRÉCÉDENT (confusion + s1.csv écrasé). On
        # expire l'entrée de cette étape : un root recalcule un nouveau
        # timestamp via _ensure_cli_dir ; un child re-hérite du dossier
        # que le parent vient de créer ce run-ci.
        _cli_dirs_by_step.pop(step.step_id, None)

        self._publish({
            'type':       'step_started',
            'step_id':    step.step_id,
            'entry_urls': step.entry_urls,
            'timestamp':  datetime.now().isoformat(timespec='seconds'),
        }, session_id)

        # Garde-fou : si la step n'a NI entry_urls NI in_site_search,
        # le run serait silencieux (0 row, 0 log utile). On publie une
        # erreur explicite pour que le front affiche quelque chose.
        if not step.entry_urls and step.in_site_search is None:
            msg = (
                f'Step {step.step_id} sans source : entry_urls est vide '
                f'et aucun in_site_search. Parent={step.parent_step_id!r}. '
                f'Utilise "Répéter sur les liens" pour hériter des URLs '
                f'd\'une étape précédente, ou remplis l\'URL dans le wizard.'
            )
            self._log.error(msg)
            self._publish({
                'type':    'error',
                'step_id': step.step_id,
                'message': msg,
            }, session_id)
            self._publish({
                'type':      'step_finished',
                'step_id':   step.step_id,
                'row_count': 0,
                'url_count': 0,
                'timestamp': datetime.now().isoformat(timespec='seconds'),
            }, session_id)
            return StepResult(step_id=step.step_id)

        # Le crawler effectif dépend de la profondeur récursive demandée
        crawler = self._build_crawler(step)

        data_rows: list[dict] = []
        url_rows:  list[dict] = []

        try:
            # ── Mode in-site search ────────────────────────────────────
            # Si un InSiteSearchSpec est fourni, on prend les valeurs d'une
            # colonne d'un CSV précédent et on les injecte dans le form de
            # recherche du site cible. Chaque valeur produit une ou plusieurs
            # rows. entry_urls est ignoré dans ce mode — on itère sur la
            # colonne source.
            if step.in_site_search is not None:
                from .in_site_search import InSiteSearchRunner
                search_runner = InSiteSearchRunner(
                    crawler = crawler,
                    publish = self._publish,
                )
                rows = search_runner.run(step.in_site_search, step, session_id)
                for row in rows:
                    # step_id déjà ajouté par InSiteSearchRunner
                    has_url = self._is_url_row(row, step.url_export_columns)
                    data_rows.append(row)
                    if has_url:
                        url_rows.append(row)

            # ── Mode URLs directes ─────────────────────────────────────
            else:
                lang_det = LanguageDetector()
                total_urls = len(step.entry_urls)

                # Parallélise le fetch + extraction sur N workers — gros
                # gain pour les répétitions (200 URLs séquentielles =
                # 5 min, 4 workers ≈ 1 min). Délai entre submissions
                # pour ne pas hammer le serveur.
                from concurrent.futures import ThreadPoolExecutor, as_completed
                import threading

                index_lock = threading.Lock()
                progress_counter = {'value': 0}

                def _process_one(idx_url):
                    idx, url = idx_url
                    # Shutdown coopératif : on sort tout de suite si l'app
                    # est en arrêt — pas de nouvelle requête envoyée.
                    import crawler as _cr
                    if _cr.is_shutdown_requested():
                        return [], []
                    # Publie le step_progress AVANT le fetch
                    with index_lock:
                        progress_counter['value'] += 1
                        cur = progress_counter['value']
                    self._publish({
                        'type':        'step_progress',
                        'step_id':     step.step_id,
                        'current_url': url,
                        'index':       cur,
                        'total':       total_urls,
                    }, session_id)
                    try:
                        return self._process_one_url(
                            step, url, crawler, lang_det,
                        )
                    except Exception as exc:
                        self._log.warn(f'  worker URL a échoué ({url[:60]}) — {exc}')
                        return [], []

                # Parallélisme & stagger : 3 workers + 0.25s + jitter
                # pour rester sous le seuil FFE (~5 req/s) et éviter
                # d'avoir l'air d'un bot trop régulier. Le crawler.py
                # ajoute un circuit breaker global : si on encaisse
                # 3 × 429 en 30s, tous les workers se mettent en pause
                # 90s — c'est ce qui évite les bannissements d'IP.
                MAX_WORKERS    = 3
                STAGGER_SEC    = 0.25
                STAGGER_JITTER = 0.15   # ajoute 0-150ms aléatoires

                indexed = list(enumerate(step.entry_urls, start=1))

                # ── Circuit breaker step-level ────────────────────
                # Si on enchaîne N pages consécutives à 0 résultats,
                # c'est qu'un problème systémique nous bloque (auth
                # expirée qui redirige tout vers la modal de login,
                # selector qui ne match plus suite à une mise à jour
                # FFE, etc.). Plutôt que de continuer 600 fetches
                # inutiles, on abandonne l'étape proprement.
                ZERO_STREAK_ABORT = 30      # 30 pages d'affilée à 0 → abandon
                consecutive_zeros = 0
                aborted = False

                # Reset le shutdown event au début de chaque step — un
                # crawl précédent qui s'est terminé proprement peut avoir
                # laissé l'event set (pas typique, mais safety).
                import crawler as _cr_mod
                _cr_mod.reset_shutdown()

                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
                    futures = []
                    import random as _r
                    for item in indexed:
                        # Avant chaque submit, on check le shutdown. Si
                        # l'app s'arrête entre 2 submits, on stoppe sec
                        # au lieu de pousser 500 URLs dans la queue.
                        if _cr_mod.is_shutdown_requested():
                            self._log.warn(f'  ⛔ shutdown — {len(indexed) - len(futures)} URL(s) jamais soumises.')
                            break
                        futures.append(exe.submit(_process_one, item))
                        # stagger + jitter pour ne pas avoir un pattern
                        # parfaitement régulier (= bot évident côté FFE)
                        time.sleep(STAGGER_SEC + _r.uniform(0, STAGGER_JITTER))

                    for f in as_completed(futures):
                        # Shutdown coopératif : si l'app s'arrête, on
                        # cancel les futures restantes et on sort. Évite
                        # d'attendre 15s × N pour qu'elles timeout.
                        if _cr_mod.is_shutdown_requested():
                            cancelled = sum(1 for ff in futures if ff.cancel())
                            self._log.warn(f'  ⛔ shutdown — {cancelled} future(s) annulée(s).')
                            break

                        try:
                            d_rows, u_rows = f.result()
                        except Exception as exc:
                            self._log.warn(f'  future exception : {exc}')
                            continue

                        # Tracking 0-results pour le circuit breaker
                        if not d_rows:
                            consecutive_zeros += 1
                            if not aborted and consecutive_zeros >= ZERO_STREAK_ABORT:
                                aborted = True
                                self._log.warn(
                                    f'  ⛔ {ZERO_STREAK_ABORT} pages consécutives à 0 '
                                    f'résultat sur step {step.step_id} ("{step.data_selector}") — '
                                    f'abandon de l\'étape. Vérifie l\'auth FFE ou que '
                                    f'le sélecteur cible toujours la bonne page.'
                                )
                                self._publish({
                                    'type':    'error',
                                    'step_id': step.step_id,
                                    'message': (
                                        f'{ZERO_STREAK_ABORT} pages d\'affilée vides — '
                                        f'auth FFE expirée ou page modifiée ? '
                                        f'Étape arrêtée pour ne pas perdre 1h de fetches inutiles.'
                                    ),
                                }, session_id)
                                # On annule les futures restantes (peu d'effet
                                # sur celles déjà en cours, mais évite que les
                                # nouvelles partent).
                                for ff in futures:
                                    if not ff.done():
                                        ff.cancel()
                                # On break la collecte — le crawl continue
                                # avec les rows déjà accumulées (peut-être 0).
                                break
                        else:
                            consecutive_zeros = 0     # reset au moindre succès

                        data_rows.extend(d_rows)
                        url_rows.extend(u_rows)
                        # Publish des rows pour le streaming live
                        for row in d_rows:
                            self._publish({
                                'type':    'row',
                                'step_id': step.step_id,
                                'row':     row,
                            }, session_id)
                        for row in u_rows:
                            self._publish({
                                'type':    'url_row',
                                'step_id': step.step_id,
                                'row':     row,
                            }, session_id)
                            self._publish({
                                'type':    'url_row',
                                'step_id': step.step_id,
                                'row':     row,
                            }, session_id)

            # Dédup cross-URL : quand la boucle a itéré sur plusieurs
            # entry_urls, les mêmes liens/textes apparaissent souvent
            # dupliqués (ex : header/footer partagé entre pages). La
            # dédup par Crawler._process_url ne couvre QUE une URL à la
            # fois, pas l'agrégat. On dédup ici sur (href, text, content).
            data_rows = self._dedup_rows(data_rows)
            url_rows  = self._dedup_rows(url_rows)

            # Uniformise le dataset si l'utilisateur l'a explicitement
            # demandé via le flag step.auto_uniformize. Désactivé par
            # défaut : le filtre peut faire disparaître des rows
            # légitimes (ex: 4e-9e places sans Pts qualif → drop si
            # majorité a des Pts), surprise désagréable. L'utilisateur
            # a maintenant FilterColumnsPanel pour faire ça à la main
            # avec retour visuel immédiat.
            if step.auto_uniformize:
                data_rows = self._uniformize_by_source(data_rows)
                url_rows  = self._uniformize_by_source(url_rows)
            else:
                self._log.info(
                    f'  [uniformize] désactivé (step.auto_uniformize=False) '
                    f'→ {len(data_rows)} rows conservées telles quelles'
                )

            # Publie la version FINALE (post-dédup + post-uniformize)
            # pour que le front remplace son tampon live et n'affiche
            # plus les rows qui ont été filtrées.
            self._publish({
                'type':      'step_finalized',
                'step_id':   step.step_id,
                'data_rows': data_rows,
                'url_rows':  url_rows,
            }, session_id)

            # Sauvegarde CSV côté disque — séparation data / urls,
            # et overwrite du CSV CLI-style avec la version enrichie.
            # target_url/selector viennent de la DERNIÈRE itération de la
            # boucle (suffisant pour retrouver le dossier crawlresult/<slug>/
            # que Crawler.crawl a créé pendant cette exécution).
            last_target_url = locals().get('target_url', '')
            last_selector   = locals().get('selector', '')
            self._save_csvs(
                step, data_rows, url_rows,
                target_url = last_target_url,
                selector   = last_selector,
            )

            self._publish({
                'type':      'step_finished',
                'step_id':   step.step_id,
                'row_count': len(data_rows),
                'url_count': len(url_rows),
                'timestamp': datetime.now().isoformat(timespec='seconds'),
            }, session_id)
            self._log.success(
                f'✔ Step {step.step_id} — {len(data_rows)} données, '
                f'{len(url_rows)} URL(s) exportées'
            )

        except Exception as exc:
            import traceback
            self._log.error(f'Step {step.step_id} a échoué : {exc}')
            self._publish({
                'type':    'error',
                'step_id': step.step_id,
                'message': str(exc),
                'trace':   traceback.format_exc(),
            }, session_id)
            raise

        return StepResult(
            step_id   = step.step_id,
            data_rows = data_rows,
            url_rows  = url_rows,
        )

    # ── Internals ──────────────────────────────────────────────────────────

    def _prefetch_soup(
        self,
        crawler: Crawler,
        url:     str,
    ) -> BeautifulSoup | None:
        """
        Parse la page cible UNE FOIS pour permettre à la fois la
        détection de langue globale et la recherche per-row de drapeaux
        voisins. Retourne None si le fetch plante — on laisse le
        crawler principal gérer l'erreur, on se contentera de page_lang=''.
        """
        try:
            html = crawler._fetch(url)
            return BeautifulSoup(html, 'html.parser')
        except Exception as exc:
            self._log.debug(f'  prefetch soup échoué sur {url} : {exc}')
            return None

    # Pattern pour extraire l'URL cible d'un onclick AJAX jQuery :
    #   onclick="$.ajax({ ... url: './?cs=4.XXX' ... })"
    # Cas FFE SIF pour la pagination Liste.
    _AJAX_URL_RE = re.compile(r"url\s*:\s*['\"]([^'\"]+)['\"]", re.I)

    # Noms de mois FR + EN minuscules → numéro (1..12). Pour parser les
    # titres de table de calendrier type "Avril 2026" ou "April 2026".
    _MONTH_NAMES: dict[str, int] = {
        'janvier': 1, 'february': 2, 'février': 2, 'fevrier': 2,
        'january': 1, 'mars': 3, 'march': 3,
        'avril': 4, 'april': 4,
        'mai': 5, 'may': 5,
        'juin': 6, 'june': 6,
        'juillet': 7, 'july': 7,
        'août': 8, 'aout': 8, 'august': 8,
        'septembre': 9, 'september': 9,
        'octobre': 10, 'october': 10,
        'novembre': 11, 'november': 11,
        'décembre': 12, 'decembre': 12, 'december': 12,
    }

    # Pattern "<mois_nom> <année>" ou "<année> <mois_nom>" avec mois
    # français ou anglais (case-insensitive). Ex : "Avril 2026", "2026 Avril".
    _MONTH_YEAR_RE = re.compile(
        r'\b('
        r'janvier|february|f[eé]vrier|january|mars|march|avril|april|mai|may'
        r'|juin|june|juillet|july|ao[uû]t|august|septembre|september'
        r'|octobre|october|novembre|november|d[eé]cembre|december'
        r')\b[\s,]*\b(20\d{2}|19\d{2})\b',
        re.I,
    )
    # Variante "année d'abord" : "2026 avril"
    _YEAR_MONTH_RE = re.compile(
        r'\b(20\d{2}|19\d{2})\b[\s,]*\b('
        r'janvier|february|f[eé]vrier|january|mars|march|avril|april|mai|may'
        r'|juin|june|juillet|july|ao[uû]t|august|septembre|september'
        r'|octobre|october|novembre|november|d[eé]cembre|december'
        r')\b',
        re.I,
    )

    def _resolve_calendar_date(
        self,
        row:  dict,
        soup: BeautifulSoup | None,
    ) -> str:
        """
        Quand la row vient d'une cellule de calendrier (un <a> à l'intérieur
        d'un <td> d'une <table>), reconstruit la date complète (YYYY-MM-DD)
        en combinant :
          - le JOUR (texte du <a>, ex : "16")
          - le MOIS/ANNÉE du titre de la table parent (caption, th colspan,
            heading précédent, ex : "Avril 2026")

        Retourne chaîne vide si on ne peut pas reconstruire (pas de table
        parent, pas de titre de mois trouvé, texte pas numérique, etc.).
        """
        if soup is None:
            return ''

        text = str(row.get('text') or row.get('content') or '').strip()
        if not text or not text.isdigit():
            return ''
        day = int(text)
        if not (1 <= day <= 31):
            return ''

        # Retrouve le <a> d'origine dans le soup (même texte).
        # On préfère celui qui est dans un <td class="on"> (ou similaire)
        # pour éviter de capter un chiffre isolé hors calendrier.
        candidates: list[Tag] = []
        for a in soup.find_all('a'):
            if a.get_text(' ', strip=True) == text and a.find_parent('td'):
                candidates.append(a)

        for a in candidates:
            table = a.find_parent('table')
            if table is None:
                continue
            month_year = self._find_calendar_heading(table)
            if not month_year:
                continue
            parsed = self._parse_month_year(month_year)
            if parsed is None:
                continue
            month, year = parsed
            try:
                return f'{year:04d}-{month:02d}-{day:02d}'
            except Exception:
                continue
        return ''

    def _find_calendar_heading(self, table: Tag) -> str:
        """
        Cherche le titre "Mois Année" d'une table de calendrier :
          1. <caption> de la table
          2. <th colspan="..."> dans le <thead> (pattern classique)
          3. sibling précédent de la table : h1-h6, ou div avec texte
             court ressemblant à "Avril 2026"
          4. parent direct : premier enfant text / h1-h6
        """
        # 1. <caption>
        caption = table.find('caption')
        if isinstance(caption, Tag):
            text = caption.get_text(' ', strip=True)
            if text:
                return text

        # 2. <th colspan> dans thead (souvent utilisé pour titrer un mois)
        thead = table.find('thead')
        if isinstance(thead, Tag):
            for th in thead.find_all('th'):
                text = th.get_text(' ', strip=True)
                if text and self._MONTH_YEAR_RE.search(text):
                    return text

        # 3. Sibling précédent (h1-h6 ou div titrant la table)
        sib = table.find_previous_sibling()
        hops = 0
        while sib is not None and hops < 4:
            if isinstance(sib, Tag):
                if sib.name in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
                    text = sib.get_text(' ', strip=True)
                    if text:
                        return text
                # div / span qui contient juste un titre de mois
                if sib.name in ('div', 'span', 'p'):
                    text = sib.get_text(' ', strip=True)
                    if text and len(text) <= 40 and self._MONTH_YEAR_RE.search(text):
                        return text
            sib = sib.previous_sibling
            hops += 1

        # 4. Ancêtre proche qui titrerait (card-header, etc.)
        parent = table.parent
        hops = 0
        while parent is not None and hops < 3:
            if isinstance(parent, Tag):
                # heading direct dans le parent
                for h in parent.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'], recursive=False):
                    text = h.get_text(' ', strip=True)
                    if text and self._MONTH_YEAR_RE.search(text):
                        return text
            parent = parent.parent
            hops += 1

        return ''

    def _parse_month_year(self, text: str) -> tuple[int, int] | None:
        """
        Parse un titre type "Avril 2026" ou "2026 Avril" → (month, year).
        Retourne None si aucun pattern reconnu.
        """
        m = self._MONTH_YEAR_RE.search(text)
        if m:
            month_name = m.group(1).lower()
            year_str   = m.group(2)
        else:
            m = self._YEAR_MONTH_RE.search(text)
            if not m:
                return None
            year_str   = m.group(1)
            month_name = m.group(2).lower()

        month = self._MONTH_NAMES.get(month_name)
        if month is None:
            return None
        try:
            year = int(year_str)
        except ValueError:
            return None
        return (month, year)

    def _process_one_url(
        self,
        step:     CrawlStep,
        url:      str,
        crawler:  Crawler,
        lang_det: LanguageDetector,
    ) -> tuple[list[dict], list[dict]]:
        """Fetch + extraction + enrichissement pour UNE URL — pensé
        pour être appelé en parallèle depuis un ThreadPoolExecutor."""
        # Fetch (filtré si root, brut si enfant)
        # Sélecteur que Playwright attend dans le DOM après goto. On
        # passe le data_selector entier : si c'est `#t_engts tbody tr`,
        # ça attend qu'au moins un <tr> dans `#t_engts tbody` existe.
        # Couvre les pages qui injectent la table cible via AJAX après
        # onReady (FFE Telemat) — sans cette attente, page.content()
        # capture le DOM avant injection et on récupère 0 row.
        wait_sel = (step.data_selector or '').strip() or None

        if step.parent_step_id:
            target_url = url
            sel = (step.data_selector or '').lower()
            # Si le sélecteur cible un tableau, on tape direct en
            # Playwright avec attente networkidle : sur FFE le <tbody>
            # est injecté en AJAX après onReady, un simple GET HTTP
            # renvoie 12 k octets stripped sans <tr>. Mieux vaut
            # prendre les 3-4s Playwright que 0 row.
            use_pw_primary = bool(sel) and ('tr' in sel or 'table' in sel)
            filtered_html = ''
            # Cookies post-login (si auth_id défini sur le runner) :
            # propagés à TOUS les fetchers — Playwright pool ET fetch
            # statique via FormSubmitter — pour que le serveur cible
            # voit l'utilisateur connecté à chaque requête.
            pw_auth_cookies   = self._auth_cookies_pw()
            http_auth_cookies = self._auth_cookies_dict()
            if use_pw_primary:
                try:
                    from orchestration.playwright_pool import fetch as _pw_fetch
                    filtered_html = _pw_fetch(
                        url,
                        wait_until        = 'networkidle',
                        timeout_ms        = 25_000,
                        wait_for_selector = wait_sel,
                        cookies           = pw_auth_cookies,
                    )
                except Exception as exc:
                    self._log.warn(f'  Playwright primary a échoué — {exc}')
            if not filtered_html:
                # Fallback HTTP : soit sélecteur non-tabulaire, soit
                # Playwright a planté. Permet les cas rapides (listings
                # statiques) + sert de filet de secours.
                # IMPORTANT : on passe les cookies d'auth — sans ça,
                # un site qui exige le login (FFE Telemat fiches) sert
                # la page modal de connexion au lieu du contenu réel.
                try:
                    filtered_html = crawler._fetch(url, cookies=http_auth_cookies)
                except Exception as exc:
                    self._log.warn(f'  GET direct a échoué — {exc}')
                    filtered_html = ''
                # Si encore vide de <tr> malgré un fetch HTTP OK et
                # que le sélecteur est tabulaire, on retry Playwright.
                if filtered_html and use_pw_primary:
                    try:
                        sniff = BeautifulSoup(filtered_html, 'html.parser')
                        if not sniff.select('table tbody tr'):
                            from orchestration.playwright_pool import fetch as _pw_fetch
                            self._log.info(f'  page sans <tr> sur {url[:60]}… → retry Playwright')
                            filtered_html = _pw_fetch(
                                url,
                                wait_until        = 'networkidle',
                                timeout_ms        = 25_000,
                                wait_for_selector = wait_sel,
                                cookies           = pw_auth_cookies,
                            )
                    except Exception as exc:
                        self._log.warn(f'  retry Playwright child a échoué — {exc}')
        else:
            pw_auth_cookies   = self._auth_cookies_pw()
            http_auth_cookies = self._auth_cookies_dict()
            try:
                filtered_html, target_url = fetch_filtered_html(
                    url, step.field_values,
                    force_dynamic = step.force_dynamic,
                    auth_cookies  = http_auth_cookies,
                )
            except Exception as exc:
                self._log.warn(f'  fetch_filtered_html a échoué, fallback GET — {exc}')
                filtered_html = crawler._fetch(url, cookies=http_auth_cookies)
                target_url    = url
            # Même retry Playwright que pour les enfants : si le
            # sélecteur cible un tableau mais qu'on n'a aucun <tr>,
            # c'est probablement un chargement AJAX post-form-submit —
            # on rejoue avec Playwright qui exécute le JS.
            sel = (step.data_selector or '').lower()
            if filtered_html and ('tr' in sel or 'table' in sel):
                try:
                    sniff = BeautifulSoup(filtered_html, 'html.parser')
                    if not sniff.select('table tbody tr'):
                        from orchestration.playwright_pool import fetch as _pw_fetch
                        self._log.info(f'  page sans <tr> sur {target_url[:60]}… → retry Playwright')
                        filtered_html = _pw_fetch(
                            target_url,
                            wait_for_selector = wait_sel,
                            cookies           = pw_auth_cookies,
                        )
                except Exception as exc:
                    self._log.warn(f'  retry Playwright root a échoué — {exc}')

        active_crawler = _PreloadedCrawler(
            preloaded     = {target_url: filtered_html},
            extractors    = crawler._extractors,
            force_dynamic = crawler._force_dynamic,
        )

        soup      = BeautifulSoup(filtered_html, 'html.parser')
        page_lang = lang_det.detect_page(soup) if soup else ''
        selector  = step.data_selector or step.link_selector or 'body'

        # On utilise _process_url dans TOUS les cas pour éviter
        # Crawler._save qui crée un data.csv parasite. Pour les étapes
        # racines on crée manuellement le dossier CLI via
        # _ensure_cli_dir (idempotent : un seul dir pour toute l'étape).
        result = active_crawler._process_url(target_url, selector)
        if not step.parent_step_id:
            self._ensure_cli_dir(step.step_id, target_url, selector)

        d_rows: list[dict] = []
        u_rows: list[dict] = []
        for row in result.rows:
            row['step_id'] = step.step_id
            resolved = self._resolve_ajax_href(row, soup, target_url)
            if resolved:
                row['href'] = resolved
            cal_date = self._resolve_calendar_date(row, soup)
            if cal_date:
                row['date'] = cal_date
            if not row.get('language'):
                row['language'] = self._detect_row_language(
                    row, soup, page_lang, lang_det,
                )
            d_rows.append(row)
            if self._is_url_row(row, step.url_export_columns):
                u_rows.append(row)
        return d_rows, u_rows

    def _resolve_ajax_href(
        self,
        row:        dict,
        soup:       BeautifulSoup | None,
        base_url:   str,
    ) -> str:
        """
        Si le href de la row est `javascript:void(0)` (ou vide), tente de
        retrouver le `<a>` correspondant dans le soup via son texte et
        d'extraire l'URL réelle depuis l'attribut onclick.

        Retourne une URL absolue si trouvée, sinon chaîne vide.
        """
        if soup is None:
            return ''

        href = str(row.get('href') or '').strip()
        # Cas nominal : le href est déjà utilisable
        if href and not href.lower().startswith('javascript'):
            return ''

        # Cherche le <a> par son texte (LinkExtractor capture ce champ)
        text = str(row.get('text') or row.get('content') or '').strip()
        if not text:
            return ''

        from urllib.parse import urljoin
        for a in soup.find_all('a'):
            if a.get_text(' ', strip=True) != text:
                continue
            onclick = a.get('onclick') or ''
            if not onclick:
                continue
            m = self._AJAX_URL_RE.search(onclick)
            if m:
                extracted = m.group(1).strip()
                if extracted:
                    # Résout en URL absolue contre la page courante
                    return urljoin(base_url, extracted)
        return ''

    def _detect_row_language(
        self,
        row:       dict,
        soup:      BeautifulSoup | None,
        page_lang: str,
        detector:  LanguageDetector,
    ) -> str:
        """
        Enrichissement per-row : cherche le <a> ou l'élément source dans
        le soup pré-parsé pour appliquer detect_near (drapeau voisin).
        Fallback sur page_lang si on ne peut pas localiser l'élément.
        """
        if soup is None:
            return page_lang

        # Cas 1 : row vient d'un <a> — on retrouve l'élément par href
        href = row.get('href') or ''
        if not href and isinstance(row.get('content'), str) and row['content'].startswith('http'):
            href = row['content']

        if href:
            try:
                link_el = soup.find('a', href=href)
            except Exception:
                link_el = None
            if link_el is not None:
                found = detector.detect(link_el, page_lang)
                if found:
                    return found

        # Cas 2 : on n'a pas d'ancre utilisable → langue de la page
        return page_lang

    def _build_crawler(self, step: CrawlStep) -> Crawler:
        """
        Si la step demande une extraction récursive, on fabrique un Crawler
        avec RecursiveExtractor en extracteur unique (opt-in).
        Sinon on retourne le Crawler de base avec ses DEFAULT_EXTRACTORS.
        """
        if step.recursive_depth > 0:
            return Crawler(
                extractors=[RecursiveExtractor(
                    max_depth            = step.recursive_depth,
                    capture_headers_from = step.capture_headers_from,
                )],
                force_dynamic = step.force_dynamic,
            )

        if step.force_dynamic and not self._base_crawler._force_dynamic:
            # Clone léger avec dynamic forcé sans muter le crawler de base
            return Crawler(force_dynamic=True)

        return self._base_crawler

    def _split_field_values(
        self,
        field_values: dict[str, str],
    ) -> tuple[dict[str, str], dict[str, str]]:
        """
        Sépare les field_values en deux paquets :
          - url_values : valeurs qui RESSEMBLENT à des URLs/chemins
            (commencent par '?', '/', 'http'). Typiquement le résultat
            d'un dropdown Bootstrap dont chaque option est un <a href>
            qui navigue vers une URL filtrée.
          - form_values : valeurs scalaires (texte, date, nombre).
            Destinées à être soumises à un <form> du site cible (POST
            avec token CSRF préservé).

        Permet au StepRunner d'appliquer les deux mécaniques en
        séquence : d'abord URL replacement, puis form submission sur
        l'URL résultante.
        """
        url_values:  dict[str, str] = {}
        form_values: dict[str, str] = {}
        for key, value in field_values.items():
            if not value:
                continue
            v_str = str(value).strip()
            if not v_str:
                continue
            if v_str.startswith(('?', '/')) or v_str.startswith('http://') or v_str.startswith('https://'):
                url_values[key] = v_str
            else:
                form_values[key] = v_str
        return url_values, form_values

    def _build_url_with_filters(self, url: str, field_values: dict[str, str]) -> str:
        """Délègue à orchestration.url_filters.build_url_with_filters
        (factorisé pour être réutilisé par les routes de détection,
        /detect-content-classes et /fetch-preview notamment)."""
        return _shared_build_url(url, field_values)

    # Noms de colonnes qui indiquent une URL exportable (pas source_url qui est interne)
    _URL_COL_NAMES = {'href', 'link', 'url', 'uri', 'page_url', 'canonical'}

    def _detect_url_columns(self, row: dict) -> list[str]:
        """Auto-détecte les colonnes URL par leur nom uniquement."""
        return [key for key in row if key.lower() in self._URL_COL_NAMES]

    def _is_url_row(self, row: dict, url_columns: list[str]) -> bool:
        """
        Une row est considérée "URL" si elle contient un vrai lien (href avec http).
        Toutes les rows vont dans data. Les rows avec href vont AUSSI dans urls.
        """
        if url_columns:
            return any(col in row and row[col] for col in url_columns)
        # Auto : une row avec href est une URL row
        href = row.get('href', '')
        return bool(href and isinstance(href, str) and href.startswith('http'))

    def _save_csvs(
        self,
        step:      CrawlStep,
        data_rows: list[dict],
        url_rows:  list[dict],
        target_url: str = '',
        selector:   str = '',
    ) -> None:
        """
        Écrit data.csv et urls.csv :
          - dans le dossier de session (crawlresult/runs/<sid>/<step>/)
            — consommé par l'API /crawl/sessions/{id}/*.csv
          - ET dans le dossier CLI-style (crawlresult/<slug>/<ts>/)
            — path facile à trouver pour l'inspection manuelle ; overwrite
            le CSV que le Crawler.crawl a produit en interne (sans
            enrichissement language/date), avec la version enrichie.

        Merge parent : si step.parent_step_id est défini, on écrit
        dans le dossier du PARENT (même data.csv / urls.csv, avec
        dedup). Cas typique : étape de répétition sur les liens
        collectés par le parent — même format de données, un seul
        fichier est plus lisible que N fragments.

        data.csv contient TOUTES les colonnes dynamiques, y compris href
        (les URLs sont ainsi présentes à la fois dans data.csv pour le
        contexte complet et dans urls.csv pour un listing dédié).
        """
        if self._session_dir is None:
            return

        # Dossier de session : on écrit à la racine du dossier de la
        # session, pas par sous-dossier — un seul fichier par étape
        # (s1.csv, s2.csv, ...). Plus simple à lire et à partager.
        step_dir = self._session_dir
        step_dir.mkdir(parents=True, exist_ok=True)

        # 1. Path sidecar (crawlresult/runs/<sid>/<step_id>.csv)
        # Un fichier PAR étape. Strictement SANS les colonnes URL :
        # les liens vont exclusivement dans urls_<step_id>.csv. Reste
        # dans <step_id>.csv uniquement les données "intrinsèques"
        # (text, date, language, etc.) — plus propre pour l'analyse.
        # On GARDE les colonnes URL dans le CSV unique de chaque étape :
        # source_url (provenance) + href (destination) sur chaque ligne.
        # Plus de fichier urls_<step_id>.csv séparé — tout est dans
        # <step_id>.csv (utilisé par le dataset builder pour la jointure).
        _META = {'step_id', 'extractor', 'type', 'tag', 'extra'}
        if data_rows:
            cleaned = [
                {k: v for k, v in row.items() if k not in _META}
                for row in data_rows
            ]
            cleaned = [r for r in cleaned if any(str(v).strip() for v in r.values())]
            CsvWriter().write(cleaned, step_dir / f'{step.step_id}.csv')

        # 2. Overwrite du CSV CLI-style — crawlresult/<slug>/<timestamp>/
        # Crawler.crawl() a créé ce dossier avec un data.csv SANS enrichissement
        # (pas de language ni de date). On le remplace par notre version
        # enrichie (href + date + language + toutes les colonnes dynamiques).
        # Résolution du dossier CLI :
        #   - étape enfant : réutilise le dir du parent (toute la chaîne
        #     partage le même dossier d'inspection)
        #   - étape racine : utilise le dir créé par _ensure_cli_dir
        #     (peuplé dans _cli_dirs_by_step au moment du premier worker)
        #   - fallback : recherche disque via slug + timestamp récent
        cli_dir = None
        if step.parent_step_id and step.parent_step_id in _cli_dirs_by_step:
            cli_dir = _cli_dirs_by_step[step.parent_step_id]
        elif step.step_id in _cli_dirs_by_step:
            cli_dir = _cli_dirs_by_step[step.step_id]
        elif target_url and selector:
            cli_dir = self._find_cli_output_dir(target_url, selector)

        if cli_dir is not None:
            _cli_dirs_by_step[step.step_id] = cli_dir
            if data_rows:
                cleaned = [
                    {k: v for k, v in row.items() if k not in _META}
                    for row in data_rows
                ]
                cleaned = [r for r in cleaned if any(str(v).strip() for v in r.values())]
                if cleaned:
                    CsvWriter().write(cleaned, cli_dir / f'{step.step_id}.csv')

            # Nettoyage TOUJOURS — même sans data_rows. data.csv créé
            # par Crawler._save n'a plus sa place dans le nouveau format.
            for stale in (cli_dir / 'data.csv', cli_dir / f'urls_{step.step_id}.csv'):
                try:
                    if stale.exists():
                        stale.unlink()
                except Exception as exc:
                    self._log.warn(f'  suppression {stale.name} échouée : {exc}')

    # Seuils du filtre de majorité — configurables ici pour tests.
    _UNIFORM_MIN_SAMPLE    = 5     # en dessous on ne filtre pas (stats pas fiables)
    # Seuils volontairement très conservateurs : on ne filtre que les
    # OUTLIERS extrêmes. Avant on était à 60/40 → pour FFE (et tout site
    # similaire), une colonne remplie sur 7/9 rows était traitée comme
    # "obligatoire" et droppait les 2 rows sans → perte massive de
    # données légitimes (ex: 4e-9e places sans points qualif). Le user
    # a maintenant FilterColumnsPanel pour choisir lui-même les
    # colonnes obligatoires — l'uniformize n'a plus besoin d'être
    # agressif, juste de virer le bruit clair.
    _UNIFORM_PRESENT_RATIO = 0.95  # ≥95% → colonne "présente" → on drop les manquants (rares cas)
    _UNIFORM_MISSING_RATIO = 0.95  # ≥95% absente → colonne "absente" → on drop les outliers (rares cas)

    def _uniformize_by_source(self, rows: list[dict]) -> list[dict]:
        """
        Homogénéise le dataset GLOBAL de l'étape. Pour chaque colonne
        on calcule le ratio de présence sur l'ensemble des rows :
          - ≥ _UNIFORM_PRESENT_RATIO (60 %) → colonne "attendue" →
            DROP les rows où elle est vide (NONE / -1 / "")
          - ≤ 1 - _UNIFORM_MISSING_RATIO (40 %) → colonne "absente" →
            DROP les rows où elle est remplie (bruit/outlier)
          - sinon → indécis, on laisse

        Une analyse globale (au lieu de par source_url) est plus fiable
        car beaucoup de groupes n'ont qu'1-2 rows, ce qui rendait les
        stats inutilisables en dessous du seuil min sample.

        Les drops sont logués avec la raison précise (colonne,
        ratio attendu, valeur réelle) pour pouvoir auditer le filtrage.
        """
        if not rows:
            return rows

        _META = {'source_url', 'step_id', 'extractor', 'type', 'tag', 'extra'}
        n = len(rows)
        if n < self._UNIFORM_MIN_SAMPLE:
            self._log.info(
                f'  [uniformize] {n} row(s) — sous le seuil min de '
                f'{self._UNIFORM_MIN_SAMPLE}, pas de filtrage'
            )
            return rows

        # Stats de présence par colonne (toutes valeurs non-sentinelles)
        all_cols = {k for r in rows for k in r.keys() if k not in _META}
        present_ratio: dict[str, float] = {}
        for col in all_cols:
            present = sum(1 for r in rows if _has_value(r.get(col)))
            present_ratio[col] = present / n

        # Classifie chaque colonne selon les seuils
        expected: dict[str, float]   = {}  # cols dont la majorité a une valeur
        unexpected: dict[str, float] = {}  # cols dont la majorité est vide
        for col, ratio in present_ratio.items():
            if ratio >= self._UNIFORM_PRESENT_RATIO:
                expected[col] = ratio
            elif ratio <= (1 - self._UNIFORM_MISSING_RATIO):
                unexpected[col] = ratio
        self._log.info(
            f'  [uniformize] sur {n} rows · attendues={list(expected)} · '
            f'absentes={list(unexpected)} · indécises={[c for c in all_cols if c not in expected and c not in unexpected]}'
        )

        # Filtre + log des raisons
        out: list[dict] = []
        drop_counts: dict[str, int] = {}
        for r in rows:
            reason = ''
            for col, ratio in expected.items():
                if not _has_value(r.get(col)):
                    reason = (
                        f"colonne '{col}' attendue ({ratio:.0%} des rows "
                        f"l'ont) mais vide ici"
                    )
                    break
            if not reason:
                for col, ratio in unexpected.items():
                    if _has_value(r.get(col)):
                        reason = (
                            f"colonne '{col}' rare ({ratio:.0%}) mais "
                            f"remplie ici → outlier"
                        )
                        break
            if reason:
                drop_counts[reason] = drop_counts.get(reason, 0) + 1
            else:
                out.append(r)

        # Résumé compact des raisons (un seul log par cause × N rows)
        for reason, count in sorted(drop_counts.items(), key=lambda kv: -kv[1]):
            self._log.info(f'  [uniformize] dropped {count} row(s) : {reason}')

        kept = len(out)
        self._log.info(f'  [uniformize] résultat : {kept}/{n} rows gardées ({n - kept} filtrées)')
        return out

    def _dedup_rows(self, rows: list[dict]) -> list[dict]:
        """
        Déduplique une liste de rows en préservant l'ordre d'insertion.
        Clé : tuple trié de tous les couples (colonne, valeur) sauf les
        champs internes/de provenance (source_url, step_id, etc.). Une
        clé identique = même contenu utile = doublon.

        Avant on dédup'ait sur (href, text, content), mais ces champs
        n'existent plus depuis le passage au format "colonnes nommées"
        (Date, Discipline, Épreuve…) — ce qui faisait perdre la data.
        """
        _IGNORED_KEYS = {
            'source_url', 'step_id', 'extractor', 'type', 'tag', 'extra',
        }
        seen: set[tuple] = set()
        out: list[dict] = []
        for r in rows:
            key = tuple(sorted(
                (k, str(v)) for k, v in r.items()
                if k not in _IGNORED_KEYS and v not in (None, '')
            ))
            if not key:
                continue
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
        return out

    def _ensure_cli_dir(self, step_id: str, url: str, selector: str):
        """
        Crée crawlresult/<slug>/<timestamp>/ comme l'aurait fait
        Crawler._save, MAIS sans écrire de data.csv. Enregistre aussi
        le path dans _cli_dirs_by_step pour que _save_csvs le retrouve
        directement (sans dépendre de target_url/selector qui sont
        scopés au worker en mode parallèle).
        """
        from datetime import datetime
        from output.file_writer import CRAWLRESULT_ROOT, _slugify
        # Réutilise un dir déjà créé pour ce step (cas N URLs en
        # parallèle : on veut UN seul dossier, pas N timestamps).
        if step_id in _cli_dirs_by_step:
            return _cli_dirs_by_step[step_id]
        slug      = _slugify(f'{url}_{selector}')
        timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        path      = CRAWLRESULT_ROOT / slug / timestamp
        path.mkdir(parents=True, exist_ok=True)
        _cli_dirs_by_step[step_id] = path
        return path

    def _read_csv_rows(self, path) -> list[dict]:
        """Lit un CSV et retourne ses rows en list[dict]. Retourne []
        si le fichier n'existe pas. Utilisé pour le merge parent."""
        import csv as _csv
        if not path.exists():
            return []
        try:
            with path.open('r', encoding='utf-8-sig', newline='') as f:
                reader = _csv.DictReader(f)
                return [dict(row) for row in reader]
        except Exception as exc:
            self._log.warn(f'  lecture CSV existant a échoué ({path.name}) : {exc}')
            return []

    def _find_cli_output_dir(self, url: str, selector: str):
        """
        Trouve le dossier CLI-style que Crawler._save vient de créer.
        Le slug est calculé de la même façon que dans Crawler.crawl
        (f'{url}_{class_name}' passé à _slugify).
        """
        from output.file_writer import CRAWLRESULT_ROOT, _slugify
        slug = _slugify(f'{url}_{selector}')
        slug_dir = CRAWLRESULT_ROOT / slug
        if not slug_dir.exists():
            return None
        # Dossier de timestamp le plus récent (créé par Crawler.crawl
        # pendant cette exécution)
        candidates = [p for p in slug_dir.iterdir() if p.is_dir()]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)
