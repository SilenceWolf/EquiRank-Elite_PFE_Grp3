# ============================================================
# © 2026 PFE EquiRank Elite — Tous droits réservés
# Équipe : Louis Guillory · Karlotta Martin · Mathéo Isidoro
#         · Mattéo D'Andrea · Melvin Bandeira · Yohan Cana
# Licence : PROPRIETARY — voir fichier LICENSE
# ============================================================

"""
Serveur FastAPI — point d'entrée web unique du PFE.

Sert deux pages HTML adossées au même backend :

  GET  /                          Page Equirank Elite (prédiction)
  GET  /crawl                     Page de crawler FFE (collecte)
  GET  /static/...                Assets (logo.png, etc.)

API JSON :

  Prédiction (utilise models/*.joblib + data/dataset.csv) :
    GET  /api/disciplines               disciplines connues du dataset
    GET  /api/suggest/cheval?q=...      auto-complétion cheval
    GET  /api/suggest/cavalier?q=...    auto-complétion cavalier
    POST /api/predict                   prédiction d'un duo

  Crawler (utilise ffe_crawler/) :
    GET  /api/crawl/disciplines              liste FFE supportée (in/out)
    POST /api/crawl/start                    démarre un crawl, renvoie job_id
    GET  /api/crawl/status/{job_id}          polling : état + events depuis cursor
    GET  /api/crawl/download/{job_id}        télécharge le dataset_brut_v2 produit
    POST /api/crawl/write_to_data/{job_id}   écrit dans data/dataset_brut_v2.csv

Lancement :
    uvicorn equirank.server:app --reload --port 8000
"""

from __future__ import annotations

from pathlib import Path

from fastapi               import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses     import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles   import StaticFiles
from pydantic              import BaseModel, Field

from .predictor  import getDefaultPredictor
from . import crawl_jobs
from . import future_jobs
from . import snapshots
from . import prediction_history


_STATIC_DIR = Path(__file__).resolve().parent / 'static'


from contextlib import asynccontextmanager


@asynccontextmanager
async def _lifespan(_app):
    """
    Startup : capture data/+models/ en snapshot 'default' au 1er run.
    Shutdown : ferme Chromium puis force la fin du process. Sans le
    force-exit, le subprocess Chromium (orphelin du pool thread) garde
    le process Python en vie sur Windows pendant ~30s.
    """
    snapshots.ensureDefaultSnapshot()
    yield
    # ── Shutdown ───────────────────────────────────────────────
    print('🛑 Arrêt — signal aux workers en cours…', flush = True)

    # 1. Signal coopératif : tous les workers qui sont sur le point
    #    d'envoyer une requête HTTP s'arrêtent. Évite la rafale de N
    #    requêtes pending qui partiraient en burst (= bannissement IP).
    try:
        import sys, importlib
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'ffe_crawler' / 'engine'))
        _crawler_mod = importlib.import_module('crawler')
        _crawler_mod.request_shutdown()
        print('   shutdown coopératif demandé aux workers.', flush = True)
    except Exception as exc:
        print(f'   request_shutdown a échoué : {exc}', flush = True)

    # 2. Fermeture pool Playwright (Chromium)
    print('   fermeture du pool Playwright (Chromium)…', flush = True)
    pool_clean = False
    try:
        _pp = importlib.import_module('orchestration.playwright_pool')
        _pp.shutdown(timeout = 3.0)
        # Si le thread du pool est toujours là, c'est Playwright qui
        # bloque (page.goto en cours, subprocess Chromium qui ne meurt
        # pas, etc.). Inutile d'attendre plus longtemps : on a déjà
        # essayé proprement.
        import threading
        for t in threading.enumerate():
            if t.name == 'playwright-pool' and t.is_alive():
                print('   ⚠ thread playwright-pool encore actif — force exit.', flush = True)
                break
        else:
            pool_clean = True
            print('   pool Playwright fermé proprement.', flush = True)
    except Exception as exc:
        print(f'   pool shutdown a échoué : {exc}', flush = True)

    # ── Force-exit en dernier recours ────────────────────────────
    # Si le pool n'a pas pu fermer proprement, le subprocess Chromium
    # tient le process Python en vie. On force la fin avec os._exit(0)
    # ET on signale le process parent (= reloader uvicorn quand on
    # est en --reload) pour qu'il quitte aussi — sinon le terminal
    # reste captif du reloader même après que le worker soit mort.
    if not pool_clean:
        import os, signal, threading as _th
        def _hard_exit():
            print('   👋 force exit pour libérer le terminal.', flush = True)
            try:
                parent_pid = os.getppid()
                if parent_pid and parent_pid > 1 and parent_pid != os.getpid():
                    # Signal Ctrl+C au reloader (worker uvicorn --reload).
                    # SIGINT existe partout ; sous Windows, c'est traduit
                    # en CTRL_C_EVENT pour le process group.
                    try:
                        os.kill(parent_pid, signal.SIGINT)
                    except (ProcessLookupError, PermissionError, OSError):
                        pass
            except Exception:
                pass
            os._exit(0)
        _th.Timer(1.5, _hard_exit).start()


app = FastAPI(
    title       = 'EquiRank Elite — PFE',
    description = (
        'Frontend unique du PFE : prédiction LightGBM (page /) et '
        'crawler FFE (page /crawl).'
    ),
    version     = '1.2.0',
    lifespan    = _lifespan,
)

app.mount('/static', StaticFiles(directory = str(_STATIC_DIR)), name = 'static')


# ──────────────────────────────────────────────
# Schémas Pydantic
# ──────────────────────────────────────────────
class PredictRequest(BaseModel):
    cheval:     str = Field(..., description='Nom du cheval (insensible à la casse)')
    cavalier:   str = Field('',  description='Nom du cavalier — optionnel')
    discipline: str = Field(..., description='Famille de discipline — voir /api/disciplines')
    hauteur:    str = Field('',  description='Hauteur en cm — optionnel, voir /api/distances')


class CrawlStartRequest(BaseModel):
    deb:        str = Field(..., description='Date de début YYYY-MM-DD')
    fin:        str = Field(..., description='Date de fin YYYY-MM-DD')
    discipline: str = Field('Toutes disciplines', description='Discipline FFE')
    username:   str = Field('', description='Identifiant FFE (optionnel)')
    password:   str = Field('', description='Mot de passe FFE (optionnel)')
    entry_url:  str = Field('', description="URL d'entrée s1 — si l'utilisateur a pré-filtré via les dropdowns du calendrier FFE")


class FutureStartRequest(BaseModel):
    deb:        str = Field(..., description='Date de début YYYY-MM-DD (doit être future)')
    fin:        str = Field(..., description='Date de fin YYYY-MM-DD (doit être future)')
    discipline: str = Field('Toutes disciplines', description='Discipline FFE')
    entry_url:  str = Field('', description="URL d'entrée s1 (cf. /api/crawl/filters)")
    username:   str = Field('', description="Identifiant FFE — sans auth, les noms sont masqués 'Voir sa fiche'")
    password:   str = Field('', description='Mot de passe FFE (optionnel)')


# ──────────────────────────────────────────────
# Pages HTML
# ──────────────────────────────────────────────
@app.get('/', include_in_schema = False)
def serveIndex() -> FileResponse:
    return FileResponse(_STATIC_DIR / 'index.html')


@app.get('/crawl', include_in_schema = False)
def serveCrawlPage() -> FileResponse:
    return FileResponse(_STATIC_DIR / 'crawl.html')


@app.get('/crawl-predict', include_in_schema = False)
def serveCrawlPredictPage() -> FileResponse:
    return FileResponse(_STATIC_DIR / 'crawl_future.html')


@app.get('/crawl-future', include_in_schema = False)
def serveCrawlFuturePageLegacy() -> RedirectResponse:
    """Ancienne URL — redirige vers /crawl-predict pour les bookmarks."""
    return RedirectResponse(url = '/crawl-predict', status_code = 301)


@app.get('/predictions', include_in_schema = False)
def servePredictionsPage() -> FileResponse:
    return FileResponse(_STATIC_DIR / 'predictions.html')


@app.get('/datasets', include_in_schema = False)
def serveDatasetsPage() -> FileResponse:
    return FileResponse(_STATIC_DIR / 'datasets.html')


@app.get('/history', include_in_schema = False)
def serveHistoryPage() -> FileResponse:
    return FileResponse(_STATIC_DIR / 'history.html')


@app.get('/history/{filename}', include_in_schema = False)
def serveHistoryDetailPage(filename: str) -> FileResponse:
    """
    Page détail d'un fichier d'historique — le filename est lu côté JS
    depuis window.location.pathname (cf. history_detail.html).
    """
    resp = FileResponse(_STATIC_DIR / 'history_detail.html')
    resp.headers['X-History-File'] = filename
    return resp


@app.get('/predictions/{concours_id}', include_in_schema = False)
def serveConcoursDetailPage(concours_id: str) -> FileResponse:
    # La page lit `concours_id` depuis l'URL côté JS (window.location.pathname).
    # On le ré-expose en header pour faciliter le debugging (curl + grep).
    resp = FileResponse(_STATIC_DIR / 'concours_detail.html')
    resp.headers['X-Concours-Id'] = concours_id
    return resp


@app.get('/logo.png', include_in_schema = False)
def serveLogo() -> FileResponse:
    return FileResponse(_STATIC_DIR / 'logo.png')


@app.get('/favicon.ico', include_in_schema = False)
def serveFavicon() -> FileResponse:
    """Le navigateur demande /favicon.ico automatiquement — on sert
    le logo PFE pour éviter le 404 dans la console."""
    return FileResponse(
        _STATIC_DIR / 'logo.png',
        media_type = 'image/png',
        headers    = {'Cache-Control': 'public, max-age=86400'},
    )


# ──────────────────────────────────────────────
# API JSON — Prédiction
# ──────────────────────────────────────────────
@app.get('/api/disciplines')
def listDisciplines() -> dict:
    return {'disciplines': getDefaultPredictor().listDisciplines()}


@app.get('/api/distances')
def listDistances(discipline: str = Query('', max_length = 80)) -> dict:
    """Hauteurs (cm) disponibles pour la discipline donnée."""
    return {
        'discipline': discipline,
        'distances':  getDefaultPredictor().listDistances(discipline),
    }


@app.get('/api/stats')
def stats() -> dict:
    """Stats live (modèle + dataset) pour le hero de la page /."""
    try:
        out = dict(getDefaultPredictor().getStats())
    except FileNotFoundError as exc:
        raise HTTPException(status_code = 503, detail = str(exc))
    # On joint le snapshot actif pour que la page / puisse afficher
    # "Modèle servi : <label>" et un bouton de bascule rapide.
    active = snapshots.getActiveSnapshot()
    out['active_snapshot'] = active.to_dict() if active else None
    return out


@app.get('/api/suggest/cheval')
def suggestCheval(q: str = Query('', max_length = 80)) -> dict:
    return {'suggestions': getDefaultPredictor().suggestCheval(q, limit = 8)}


@app.get('/api/suggest/cavalier')
def suggestCavalier(q: str = Query('', max_length = 80)) -> dict:
    return {'suggestions': getDefaultPredictor().suggestCavalier(q, limit = 8)}


@app.post('/api/predict')
def predict(req: PredictRequest) -> JSONResponse:
    try:
        result = getDefaultPredictor().predict(
            cheval     = req.cheval,
            cavalier   = req.cavalier,
            discipline = req.discipline,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code = 503, detail = str(exc))

    if isinstance(result, dict):
        return JSONResponse(status_code = 422, content = result)

    payload = result.to_dict()

    # Journalise la prédiction réussie dans l'historique persistant
    # (data/prediction_history.jsonl). Permet à l'utilisateur de
    # consulter et exporter ses précédentes prédictions via /history.
    try:
        active = snapshots.getActiveSnapshot()
        # Estimation du classement individuel (rang dans une épreuve
        # typique de cette discipline). Le modèle PFE ne prédit pas un
        # rang direct, on l'inverse depuis la proba — voir
        # EquirankPredictor.estimateRank() pour la formule.
        rank = getDefaultPredictor().estimateRank(
            payload.get('proba', 0), req.discipline,
        )
        prediction_history.addEntry(
            cheval            = req.cheval,
            cavalier          = req.cavalier,
            discipline        = req.discipline,
            hauteur           = req.hauteur,
            classement_estime = rank,
            proba             = payload.get('proba', 0),
            verdict           = payload.get('verdict', '') or (
                'cold-start' if payload.get('is_cold_start') else ''
            ),
            model_name        = payload.get('model_name', ''),
            is_cold_start     = bool(payload.get('is_cold_start', False)),
            snapshot          = active.snapshot_id if active else '',
        )
    except Exception:
        # L'historique est best-effort — on ne casse pas la prédiction
        # si l'écriture disque échoue (disque plein, permissions, etc.).
        pass

    return JSONResponse(content = payload)


# ──────────────────────────────────────────────
# API JSON — Crawler FFE
# ──────────────────────────────────────────────
@app.get('/api/crawl/disciplines')
def crawlDisciplines() -> dict:
    return {'disciplines': crawl_jobs.listDisciplines()}


@app.get('/api/crawl/filters')
def crawlFilters(url: str = Query('', max_length = 500)) -> dict:
    """
    Renvoie les 4 dropdowns du calendrier FFE (région, département,
    discipline, championnat) pour `url` (par défaut DEFAULT_ENTRY_URL).
    Le front rappelle cet endpoint quand l'utilisateur change un filtre,
    pour reflèter la cascade côté Telemat (ex : choisir une région
    restreint la liste des départements).
    """
    try:
        return crawl_jobs.fetchFilterOptions(url or None)
    except Exception as exc:
        raise HTTPException(status_code = 502, detail = f'FFE injoignable : {exc}')


@app.post('/api/crawl/start')
def crawlStart(req: CrawlStartRequest) -> dict:
    try:
        job_id = crawl_jobs.startJob(
            deb        = req.deb,
            fin        = req.fin,
            discipline = req.discipline,
            username   = req.username or None,
            password   = req.password or None,
            entry_url  = req.entry_url or None,
        )
    except ValueError as exc:
        raise HTTPException(status_code = 400, detail = str(exc))
    return {'job_id': job_id}


@app.post('/api/crawl/train/{job_id}')
def crawlTrain(job_id: str) -> dict:
    """
    Lance le pipeline post-crawl : écrit data/dataset_brut_v2.csv,
    puis chaîne adapt_dataset_v2 + train.py, et recharge le predictor
    en mémoire. Renvoie un pipeline_id à poller via /api/crawl/train/status.
    """
    try:
        pid = crawl_jobs.startTrainingPipeline(job_id)
    except ValueError as exc:
        raise HTTPException(status_code = 404, detail = str(exc))
    return {'pipeline_id': pid}


@app.get('/api/crawl/train/status/{pipeline_id}')
def crawlTrainStatus(pipeline_id: str) -> dict:
    snap = crawl_jobs.getPipelineSnapshot(pipeline_id)
    if snap.get('state') == 'unknown':
        raise HTTPException(
            status_code = 404,
            detail = f'Pipeline {pipeline_id} introuvable',
        )
    return snap


# ──────────────────────────────────────────────
# API JSON — Crawler FUTUR (concours pas encore disputés)
# ──────────────────────────────────────────────
@app.post('/api/future/start')
def futureStart(req: FutureStartRequest) -> dict:
    try:
        job_id = future_jobs.startFutureJob(
            deb        = req.deb,
            fin        = req.fin,
            discipline = req.discipline,
            entry_url  = req.entry_url or None,
            username   = req.username  or None,
            password   = req.password  or None,
        )
    except ValueError as exc:
        raise HTTPException(status_code = 400, detail = str(exc))
    return {'job_id': job_id}


@app.get('/api/future/status/{job_id}')
def futureStatus(job_id: str, cursor: int = 0) -> dict:
    snap = future_jobs.getFutureSnapshot(job_id, cursor = cursor)
    if snap.get('state') == 'unknown':
        raise HTTPException(status_code = 404, detail = f'Job {job_id} introuvable')
    return snap


@app.get('/api/future/download/{job_id}')
def futureDownload(job_id: str) -> Response:
    payload = future_jobs.downloadPredictionsCsv(job_id)
    if payload is None:
        raise HTTPException(
            status_code = 404,
            detail = 'Job inconnu ou non terminé.',
        )
    filename = f'predictions_futures_{job_id[:8]}.csv'
    return Response(
        content = payload,
        media_type = 'text/csv; charset=utf-8',
        headers = {'Content-Disposition': f'attachment; filename="{filename}"'},
    )


# ──────────────────────────────────────────────
# API JSON — Liste / détail / recherche des prédictions
# ──────────────────────────────────────────────
@app.get('/api/predictions/concours')
def predictionsListConcours() -> dict:
    return {'concours': future_jobs.listConcours()}


@app.get('/api/predictions/search')
def predictionsSearch(q: str = Query('', max_length = 100)) -> dict:
    return {'results': future_jobs.search(q)}


@app.get('/api/predictions/download')
def predictionsDownloadAll() -> Response:
    """CSV concaténé de toutes les prédictions de tous les jobs done."""
    payload = future_jobs.downloadAllPredictionsCsv()
    if payload is None:
        raise HTTPException(status_code = 404, detail = 'Aucune prédiction disponible.')
    return Response(
        content    = payload,
        media_type = 'text/csv; charset=utf-8',
        headers    = {'Content-Disposition': 'attachment; filename="predictions_futures.csv"'},
    )


@app.get('/api/predictions/download/{concours_id}')
def predictionsDownloadConcours(concours_id: str) -> Response:
    """CSV d'un seul concours (tous jobs confondus)."""
    payload = future_jobs.downloadConcoursCsv(concours_id)
    if payload is None:
        raise HTTPException(
            status_code = 404,
            detail = f'Aucune prédiction pour le concours {concours_id}.',
        )
    return Response(
        content    = payload,
        media_type = 'text/csv; charset=utf-8',
        headers    = {'Content-Disposition': f'attachment; filename="predictions_{concours_id}.csv"'},
    )


@app.get('/api/predictions/{concours_id}')
def predictionsGetConcours(concours_id: str) -> dict:
    detail = future_jobs.getConcoursDetail(concours_id)
    if detail is None:
        raise HTTPException(
            status_code = 404,
            detail = f'Concours {concours_id} introuvable (pas encore crawlé ?)',
        )
    return detail


@app.get('/api/crawl/status/{job_id}')
def crawlStatus(job_id: str, cursor: int = 0) -> dict:
    snap = crawl_jobs.getJobSnapshot(job_id, cursor = cursor)
    if snap.get('state') == 'unknown':
        raise HTTPException(status_code = 404, detail = f'Job {job_id} introuvable')
    return snap


@app.get('/api/crawl/download/{job_id}')
def crawlDownload(job_id: str) -> Response:
    payload = crawl_jobs.getJobDatasetCsv(job_id)
    if payload is None:
        raise HTTPException(
            status_code = 404,
            detail = 'Job inconnu ou non terminé — réessaie quand state=done.',
        )
    filename = f'dataset_brut_v2_{job_id[:8]}.csv'
    return Response(
        content = payload,
        media_type = 'text/csv; charset=utf-8',
        headers = {'Content-Disposition': f'attachment; filename="{filename}"'},
    )


@app.post('/api/crawl/write_to_data/{job_id}')
def crawlWriteToData(job_id: str) -> dict:
    path = crawl_jobs.writeJobToData(job_id)
    if path is None:
        raise HTTPException(
            status_code = 404,
            detail = 'Job inconnu ou non terminé.',
        )
    return {'path': str(path)}


# ──────────────────────────────────────────────
# API JSON — Historique des prédictions unitaires
# ──────────────────────────────────────────────
@app.get('/api/history')
def historyList(q: str = Query('', max_length = 100), limit: int = 200) -> dict:
    """Liste les prédictions (les plus récentes en haut). Filtre optionnel `q`."""
    return {'entries': prediction_history.listEntries(limit = max(1, min(limit, 1000)), query = q)}


@app.get('/api/history/file/{name}/download')
def historyDownloadOneFile(name: str) -> Response:
    """Télécharge un fichier CSV tel quel (zéro transformation)."""
    target = (prediction_history._HISTORY_DIR / name).resolve()
    # garde-fou : éviter une traversée de chemin (../)
    if target.parent != prediction_history._HISTORY_DIR.resolve() or not target.exists():
        raise HTTPException(status_code = 404, detail = f'Fichier {name} introuvable')
    return FileResponse(
        path        = target,
        media_type  = 'text/csv; charset=utf-8',
        filename    = name,
    )


@app.delete('/api/history')
def historyClear() -> dict:
    """Vide complètement l'historique."""
    n = prediction_history.clearAll()
    return {'deleted': n}


@app.delete('/api/history/entry/{entry_id}')
def historyDeleteOne(entry_id: str) -> dict:
    ok = prediction_history.deleteEntry(entry_id)
    if not ok:
        raise HTTPException(status_code = 404, detail = f'Entrée {entry_id} introuvable')
    return {'deleted': entry_id}


# ── Gestion des fichiers CSV du dossier data/history/ ──
@app.get('/api/history/files')
def historyFiles() -> dict:
    """Métadonnées de chaque CSV du dossier `data/history/`."""
    return {'files': prediction_history.listFiles()}


@app.get('/api/history/file/{name}')
def historyOneFile(name: str, q: str = Query('', max_length = 100), sort: str = 'proba') -> dict:
    """Entrées d'un seul fichier — pas de fusion. Trié classement par défaut."""
    entries = prediction_history.listEntriesFromFile(name, query = q, sort = sort)
    if entries == []:
        # Distinguer "fichier inconnu" de "fichier vide"
        files = {f['name'] for f in prediction_history.listFiles()}
        if name not in files:
            raise HTTPException(status_code = 404, detail = f'Fichier {name} introuvable')
    return {'filename': name, 'entries': entries}


@app.post('/api/history/import')
async def historyImport(file: UploadFile = File(...)) -> dict:
    """
    Upload d'un CSV existant — copié tel quel dans `data/history/`.
    L'utilisateur peut aussi déposer ses CSV directement dans ce dossier
    sans passer par l'UI ; cet endpoint sert juste de raccourci.
    """
    content  = await file.read()
    filename = file.filename or 'import.csv'
    res = prediction_history.importCsv(content, filename = filename)
    # On retourne toujours 200 (le fichier est sur disque même si le
    # parsing a échoué — l'utilisateur peut l'éditer puis recharger).
    return res


@app.delete('/api/history/files/{name}')
def historyDeleteFile(name: str) -> dict:
    try:
        ok = prediction_history.deleteFile(name)
    except ValueError as exc:
        raise HTTPException(status_code = 400, detail = str(exc))
    if not ok:
        raise HTTPException(status_code = 404, detail = f'Fichier {name} introuvable')
    return {'deleted': name}


# ──────────────────────────────────────────────
# API JSON — Snapshots de datasets
# ──────────────────────────────────────────────
@app.get('/api/datasets')
def datasetsList() -> dict:
    """Liste tous les snapshots (default + crawls archivés)."""
    return {'snapshots': [s.to_dict() for s in snapshots.listSnapshots()]}


@app.get('/api/datasets/active')
def datasetsActive() -> dict:
    s = snapshots.getActiveSnapshot()
    return {'snapshot': s.to_dict() if s else None}


@app.post('/api/datasets/activate/{snapshot_id}')
def datasetsActivate(snapshot_id: str) -> dict:
    """Copie le snapshot dans data/+models/ et marque-le comme actif."""
    try:
        s = snapshots.activateSnapshot(snapshot_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code = 404, detail = str(exc))
    return {'snapshot': s.to_dict()}


@app.post('/api/datasets/restore-default')
def datasetsRestoreDefault() -> dict:
    s = snapshots.restoreDefault()
    return {'snapshot': s.to_dict()}


@app.delete('/api/datasets/{snapshot_id}')
def datasetsDelete(snapshot_id: str) -> dict:
    try:
        snapshots.deleteSnapshot(snapshot_id)
    except ValueError as exc:
        raise HTTPException(status_code = 400, detail = str(exc))
    return {'deleted': snapshot_id}


# ──────────────────────────────────────────────
# Health-check
# ──────────────────────────────────────────────
@app.get('/api/health', include_in_schema = False)
def health() -> dict:
    return {'status': 'ok'}
