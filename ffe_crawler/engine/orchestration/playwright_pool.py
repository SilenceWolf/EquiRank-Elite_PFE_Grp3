# ============================================================
# © 2026 Louis Guillory (Silence Wolf) — Tous droits réservés
# Licence : PROPRIETARY — voir fichier LICENSE
# ============================================================

# Pool Playwright persistant — garde un Chromium headless vivant
# entre les requêtes pour éviter le coût de relance (~3-5s par call).
#
# Utilisé par fetch_filtered_html quand on détecte qu'un fallback
# Playwright est nécessaire (contenu AJAX qu'on ne voit pas en
# static fetch). Crawler._fetch_dynamic continue d'ouvrir/fermer sa
# propre instance — ce pool est réservé au chemin "preview rapide"
# du sidecar.
#
# Archi : lazy singleton. Au premier appel, lance Chromium + un
# Browser Context partagé (cookies persistent entre fetches de la
# même "session" utilisateur). Un lock protège la concurrence si
# plusieurs requêtes arrivent en parallèle (FastAPI sync).

from __future__ import annotations

import atexit
import queue
import threading

from log import CrawlerLogger

_log  = CrawlerLogger.get_instance()

# Playwright sync API est thread-affine : toute interaction avec les
# objets Playwright DOIT se faire depuis le thread qui a appelé
# sync_playwright().start(). Si un worker d'un ThreadPoolExecutor
# appelle fetch() alors que le pool a été démarré par un autre thread,
# on obtient "cannot switch to a different thread" et tout plante.
#
# Solution : un unique "pool thread" dédié qui possède Playwright. Les
# appelants (n'importe quel thread) envoient des jobs dans une queue,
# le pool thread les exécute séquentiellement et renvoie le résultat
# via une second queue. On garde les avantages d'un browser persistant
# sans jamais traverser les threads.
_job_queue:    queue.Queue | None = None
_pool_thread:  threading.Thread | None = None
_start_lock    = threading.Lock()

# PID du subprocess Chromium — capturé au boot, utilisé pour le kill
# brutal au shutdown (sinon `pw.stop()` attend la fin des opérations
# en cours, ce qui peut laisser fuiter des requêtes vers FFE et te
# faire bannir l'IP).
_chromium_pid: int | None = None

_default_ua = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
    'AppleWebKit/537.36 (KHTML, like Gecko) '
    'Chrome/124.0.0.0 Safari/537.36'
)


def _read_proxy_env() -> str | None:
    """
    Récupère un proxy depuis l'environnement. Format accepté :
      - http://host:port
      - http://user:pass@host:port
      - socks5://127.0.0.1:9050   (pour Tor)

    On lit en priorité EQUIRANK_PROXY (spécifique à notre crawler) ;
    fallback sur HTTPS_PROXY / HTTP_PROXY (standards).
    """
    import os
    for key in ('EQUIRANK_PROXY', 'HTTPS_PROXY', 'HTTP_PROXY',
                'https_proxy',  'http_proxy'):
        v = os.environ.get(key)
        if v:
            return v
    return None


def _pool_worker(inbox: queue.Queue) -> None:
    """Thread dédié qui possède Playwright + browser + context.
    Consomme des jobs (dict) et renvoie le résultat via job['reply']."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        _log.fatal('playwright non installé — pip install playwright && playwright install chromium')
        return

    global _chromium_pid

    proxy_url = _read_proxy_env()
    launch_kwargs: dict = {'headless': True}
    if proxy_url:
        launch_kwargs['proxy'] = {'server': proxy_url}
        _log.info(f'[pool] proxy actif : {proxy_url}')

    _log.info('[pool] lancement Chromium headless (thread dédié)')
    pw      = sync_playwright().start()
    browser = pw.chromium.launch(**launch_kwargs)

    # Capture le PID du subprocess Chromium pour pouvoir le tuer
    # brutalement au shutdown (cf. shutdown() en bas du fichier).
    try:
        bp = getattr(browser, 'process', None) or getattr(browser, '_process', None)
        if bp is not None:
            _chromium_pid = bp.pid
            _log.debug(f'[pool] Chromium PID capturé : {_chromium_pid}')
    except Exception:
        _chromium_pid = None

    context = browser.new_context(
        user_agent         = _default_ua,
        extra_http_headers = {'Accept-Language': 'fr-FR,fr;q=0.9'},
    )

    try:
        while True:
            job = inbox.get()
            if job is None:
                break  # signal d'arrêt
            reply: queue.Queue = job['reply']
            try:
                # Nettoyage cookies avant chaque fetch (même thread → safe).
                try:
                    context.clear_cookies()
                except Exception as exc:
                    _log.warn(f'[pool] clear_cookies a échoué : {exc}')
                if job.get('cookies'):
                    try:
                        context.add_cookies(job['cookies'])
                    except Exception as exc:
                        _log.warn(f'[pool] add_cookies a échoué : {exc}')
                page = context.new_page()
                try:
                    page.goto(
                        job['url'],
                        wait_until = job.get('wait_until', 'domcontentloaded'),
                        timeout    = job.get('timeout_ms', 20_000),
                    )
                    # Si l'appelant fournit un sélecteur cible, on attend
                    # qu'il apparaisse. Indispensable pour les pages qui
                    # injectent leur contenu via AJAX après onReady (ex:
                    # FFE Telemat charge #t_engts par $.ajax post-load).
                    # Sans ce wait, page.content() capture le DOM avant
                    # que la table soit injectée → 0 row.
                    wait_sel = job.get('wait_for_selector')
                    if wait_sel:
                        try:
                            page.wait_for_selector(
                                wait_sel,
                                timeout = job.get('wait_timeout_ms', 15_000),
                                state   = 'attached',
                            )
                        except Exception as exc:
                            # Beaucoup de pages FFE (épreuves annulées,
                            # épreuves sans engagés, etc.) n'ont jamais
                            # de table d'engagements — pas la peine de
                            # crier "warn" pour chacune, on log en debug.
                            _log.debug(
                                f'[pool] wait_for_selector "{wait_sel}" '
                                f'a expiré : {exc} — extraction quand même'
                            )
                    try:
                        page.wait_for_timeout(500)
                    except Exception:
                        pass
                    html = page.content()
                    reply.put(('ok', html))
                finally:
                    page.close()
            except Exception as exc:
                reply.put(('err', exc))
    finally:
        try:
            context.close()
            browser.close()
            pw.stop()
        except Exception:
            pass


def _ensure_started() -> None:
    """Lance le thread du pool au premier appel."""
    global _job_queue, _pool_thread
    with _start_lock:
        if _pool_thread is not None and _pool_thread.is_alive():
            return
        _job_queue = queue.Queue()
        _pool_thread = threading.Thread(
            target = _pool_worker,
            args   = (_job_queue,),
            name   = 'playwright-pool',
            daemon = True,
        )
        _pool_thread.start()
        atexit.register(shutdown)


def fetch(
    url:               str,
    wait_until:        str           = 'domcontentloaded',
    timeout_ms:        int           = 20_000,
    cookies:           list[dict] | None = None,
    wait_for_selector: str | None    = None,
    wait_timeout_ms:   int           = 15_000,
) -> str:
    """
    Récupère le HTML final d'une URL via le Chromium du pool.
    Appelable depuis n'importe quel thread : la requête est routée
    vers le thread dédié qui possède Playwright.

    `wait_for_selector` : sélecteur CSS optionnel attendu après
    `goto`. Sert quand le contenu cible est injecté via AJAX — sans
    attente, `page.content()` capture le DOM avant l'injection.
    L'attente est best-effort : un timeout est loggé mais on
    extrait quand même.
    """
    _ensure_started()
    assert _job_queue is not None
    reply: queue.Queue = queue.Queue()
    _job_queue.put({
        'url':               url,
        'wait_until':        wait_until,
        'timeout_ms':        timeout_ms,
        'cookies':           cookies,
        'wait_for_selector': wait_for_selector,
        'wait_timeout_ms':   wait_timeout_ms,
        'reply':             reply,
    })
    status, payload = reply.get()
    if status == 'err':
        raise payload
    return payload


def cookies_from_requests_session(session) -> list[dict]:
    """
    Convertit les cookies d'une requests.Session en format Playwright.
    Préserve name, value, domain, path et attributs de sécurité quand
    ils sont connus — suffisant pour que Chromium se comporte comme
    la session requests qui a posé les cookies.
    """
    out: list[dict] = []
    for c in session.cookies:
        cookie = {
            'name':   c.name,
            'value':  c.value,
            'domain': c.domain,
            'path':   c.path or '/',
        }
        if c.expires:
            cookie['expires'] = c.expires
        if c.secure:
            cookie['secure'] = True
        out.append(cookie)
    return out


def shutdown(timeout: float = 4.0) -> None:
    """
    Arrête le pool et libère Chromium. Ordre critique pour ne PAS
    fuiter des requêtes HTTP au moment du shutdown (= cause #1 de
    bannissement IP FFE) :

      1. Drain de la queue Playwright (les jobs pending ne partiront jamais)
      2. **KILL BRUTAL** du subprocess Chromium via son PID (immédiat,
         abandon des connexions TCP en cours, plus une seule requête
         n'est envoyée)
      3. Signal de fin au pool thread
      4. Join avec timeout — au-delà, on laisse mourir avec le process

    Le kill du subprocess Chromium est ce qui empêche les bursts de
    requêtes en attente au Ctrl+C : tant que Chromium est vivant, il
    peut continuer à charger des ressources des pages en cours.
    """
    global _job_queue, _pool_thread, _chromium_pid
    try:
        # 1. Drain queue
        if _job_queue is not None:
            drained = 0
            while True:
                try:
                    pending = _job_queue.get_nowait()
                except queue.Empty:
                    break
                drained += 1
                if isinstance(pending, dict) and 'reply' in pending:
                    try:
                        pending['reply'].put(('err', RuntimeError('pool shutting down')))
                    except Exception:
                        pass
            if drained:
                _log.info(f'[pool] shutdown : {drained} job(s) en attente jetés.')

        # 2. KILL BRUTAL Chromium — abandon immédiat de TOUTES les
        # connexions HTTP en cours, donc plus rien ne fuit vers FFE.
        if _chromium_pid:
            _kill_chromium(_chromium_pid)
            _chromium_pid = None

        # 3. Signal de fin au worker
        if _job_queue is not None:
            _job_queue.put(None)

        # 4. Join avec timeout
        if _pool_thread is not None:
            _pool_thread.join(timeout = timeout)
            if _pool_thread.is_alive():
                _log.warn(f'[pool] thread toujours actif après {timeout}s — process exit forcera.')
    except Exception:
        pass
    finally:
        _job_queue   = None
        _pool_thread = None


def _kill_chromium(pid: int) -> None:
    """
    Tue le subprocess Chromium et tous ses descendants. Cross-platform :
      - Windows : `taskkill /F /T /PID` (force + arbre)
      - Unix    : `os.kill(pid, SIGKILL)` + `pgrep` pour les enfants
    """
    import os, subprocess, signal as _sig
    try:
        if os.name == 'nt':
            # /F = forcer ; /T = tuer l'arbre (Chromium spawn des enfants)
            subprocess.run(
                ['taskkill', '/F', '/T', '/PID', str(pid)],
                capture_output = True, timeout = 3,
            )
        else:
            try:
                os.killpg(os.getpgid(pid), _sig.SIGKILL)
            except (ProcessLookupError, PermissionError, AttributeError):
                try:
                    os.kill(pid, _sig.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        _log.info(f'[pool] Chromium PID {pid} tué — fin immédiate des requêtes.')
    except Exception as exc:
        _log.warn(f'[pool] kill Chromium PID {pid} a échoué : {exc}')
