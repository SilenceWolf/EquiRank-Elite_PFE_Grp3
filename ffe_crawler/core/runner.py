# ============================================================
# © 2026 PFE EquiRank Elite — Tous droits réservés
# Équipe : Louis Guillory · Karlotta Martin · Mathéo Isidoro
#         · Mattéo D'Andrea · Melvin Bandeira · Yohan Cana
# Licence : PROPRIETARY — voir fichier LICENSE
# ============================================================

"""
Lance la chaîne FFE figée et collecte les résultats des 5 étapes.

Le crawler générique (engine/) reste responsable du fetch, du parsing,
de l'authentification et du chaînage parent → enfant. Ici on se contente
d'instancier la chaîne et de capter les step_results pour les passer
au dataset_builder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path
from typing      import Any, Callable

# Force l'injection du sys.path → engine/ avant les imports moteur
from .ffe_chain import buildFfeChain, DEFAULT_ENTRY_URL  # noqa: F401

from crawler                    import Crawler
from orchestration.chain_runner import ChainRunner
from orchestration.auth_store   import get_store as _getAuthStore
from orchestration.form_submit  import FormSubmitter
from log                        import CrawlerLogger


# URL du formulaire de login FFE Telemat — sert à fois de page cible
# pour récupérer les hidden fields (cs/redir CSRF) et de point d'envoi
# du POST. Le FormSubmitter détecte automatiquement le <form>.
FFE_LOGIN_URL = 'https://www.telemat.org/FFE/sif/'


@dataclass
class FfeRunResult:
    """Snapshot d'une exécution complète de la chaîne FFE."""

    chain_id:   str
    total_rows: int = 0
    total_urls: int = 0
    # step_id → liste de dict (les data_rows). On garde des dicts bruts
    # pour laisser dataset_builder.py choisir comment les agréger.
    step_rows:  dict[str, list[dict]] = field(default_factory=dict)


PublishFn = Callable[[dict[str, Any], str], None]


def _loginFfe(username: str, password: str) -> tuple[str | None, str]:
    """
    Authentifie l'utilisateur auprès du portail FFE Telemat et stocke
    la session post-login dans l'auth_store du moteur.

    Renvoie un tuple `(auth_id, reason)` :
      * auth_id : token opaque à passer à ChainRunner pour que les fetchs
                  héritent des cookies post-login. None si l'auth a échoué.
      * reason  : code court pour le frontend (ok | empty_credentials |
                  form_get_failed | no_cookies | credentials_rejected).
    """
    if not username or not password:
        return (None, 'empty_credentials')

    # Même mécanique que api/routes_auth.py : FormSubmitter retrouve le
    # form, injecte login/password (+ hidden CSRF) et conserve les cookies
    # post-login dans submitter.session ; on remet le tout dans le store.
    submitter = FormSubmitter()
    result    = submitter.submit(
        FFE_LOGIN_URL,
        {
            'login':  username,
            'passwd': password,
        },
    )
    if result is None:
        return (None, 'form_get_failed')
    if len(submitter.session.cookies) == 0:
        return (None, 'no_cookies')

    # Heuristique d'échec : si la réponse contient encore un input type
    # password, le form de login a été re-render → identifiants refusés.
    html_lower = (result.html or '').lower()
    if 'type="password"' in html_lower and 'name="passwd"' in html_lower:
        return (None, 'credentials_rejected')

    return (_getAuthStore().store(FFE_LOGIN_URL, username, submitter.session), 'ok')


def _publishAuth(publish: PublishFn | None, reason: str) -> None:
    """Surface le résultat du login dans le flux d'events du job, pour
    que le frontend puisse afficher "✓ Login FFE OK" ou un message
    précis en cas d'échec (au lieu d'un silencieux "Voir sa fiche").
    """
    if publish is None:
        return
    msg_by_reason = {
        'ok':                     "✓ Login FFE réussi — noms cheval/cavalier accessibles.",
        'empty_credentials':      "ℹ Pas d'identifiants FFE — noms masqués par 'Voir sa fiche'.",
        'form_get_failed':        "✗ Login FFE : impossible d'atteindre le formulaire (réseau / proxy / Tor bloqué ?).",
        'no_cookies':             "✗ Login FFE : FFE n'a renvoyé aucun cookie — proxy/Tor probablement filtré.",
        'credentials_rejected':   "✗ Login FFE : identifiants refusés — vérifie login + mot de passe.",
    }
    publish({
        'type':    'auth_status',
        'reason':  reason,
        'ok':      reason == 'ok',
        'message': msg_by_reason.get(reason, f'Login FFE : {reason}'),
    }, 'auth')


def runFfeChain(
    deb:        str,
    fin:        str,
    discipline: str = 'Toutes disciplines',
    username:   str | None = None,
    password:   str | None = None,
    publish:    PublishFn | None = None,
    session_dir: Path | None = None,
    entry_url:  str | None = None,
) -> FfeRunResult:
    """
    Exécute la chaîne FFE figée et collecte les résultats par étape.

    Args:
        deb        : date début YYYY-MM-DD
        fin        : date fin   YYYY-MM-DD
        discipline : valeur du radio Discipline
        username   : identifiant FFE (optionnel — sinon s5 sera vide)
        password   : mot de passe FFE
        publish    : callback events
        session_dir: dossier où le moteur peut écrire les CSV intermédiaires
        entry_url  : override de l'URL d'entrée s1. Si l'utilisateur a
                     pré-filtré la page calendrier FFE via les dropdowns
                     (région / département / discipline / championnat),
                     on passe ici l'URL résultante pour que s1 parte
                     directement du bon scope.
    """

    log = CrawlerLogger.get_instance(context = 'ffe-crawler')
    log.banner(f'FFE crawl — {deb} → {fin} — {discipline}')

    chainArgs: dict = {
        'deb':        deb,
        'fin':        fin,
        'discipline': discipline,
    }
    if entry_url:
        chainArgs['entry_url'] = entry_url
    chainArgs['include_s5'] = True
    chain   = buildFfeChain(**chainArgs)
    crawler = Crawler()

    # Surface l'entry_url effectif côté frontend — laisse l'utilisateur
    # vérifier d'un coup d'œil que ses filtres FFE (région/dept/discipline)
    # sont bien encodés dans l'URL d'entrée.
    if publish:
        publish({
            'type':       'entry_url',
            'url':        entry_url or DEFAULT_ENTRY_URL,
            'discipline': discipline,
            'deb':        deb,
            'fin':        fin,
        }, 'crawl-start')

    authId:    str | None = None
    authReason: str       = 'empty_credentials'
    if username and password:
        authId, authReason = _loginFfe(username, password)
    _publishAuth(publish, authReason)
    if authId is None and (username or password):
        log.warn(
            f'Échec login FFE ({authReason}) — s5 (fiche équidé) ne pourra pas être crawlé.'
        )

    runner = ChainRunner(
        crawler     = crawler,
        publish     = publish,
        session_dir = session_dir,
        auth_id     = authId,
    )

    chainResult = runner.run(chain, session_id = chain.chain_id)

    # On ré-emballe les step_results en structures simples pour
    # dataset_builder. data_rows = lignes "métier", url_rows = liens
    # internes utilisés pour le chaînage (rarement utile en aval).
    stepRows: dict[str, list[dict]] = {}
    for stepId, stepResult in chainResult.step_results.items():
        stepRows[stepId] = list(stepResult.data_rows)

    return FfeRunResult(
        chain_id   = chainResult.chain_id,
        total_rows = chainResult.total_rows,
        total_urls = chainResult.total_urls,
        step_rows  = stepRows,
    )


def runFfeChainFuture(
    deb:        str,
    fin:        str,
    discipline: str = 'Toutes disciplines',
    username:   str | None = None,
    password:   str | None = None,
    publish:    PublishFn | None = None,
    session_dir: Path | None = None,
    entry_url:  str | None = None,
) -> FfeRunResult:
    """
    Variante "futur" : pas de s5 (la fiche détaillée n'apporte rien
    pour des concours pas encore disputés). En revanche, on peut
    fournir username/password : sans auth, FFE masque les noms de
    chevaux / cavaliers par "Voir sa fiche" — avec auth, on a accès
    aux vrais noms, ce qui rend les prédictions bien plus utiles.
    """
    log = CrawlerLogger.get_instance(context = 'ffe-future-crawler')
    log.banner(f'FFE crawl FUTUR — {deb} → {fin} — {discipline}')

    chainArgs: dict = {
        'deb':        deb,
        'fin':        fin,
        'discipline': discipline,
        'include_s5': False,
        'chain_name': 'FFE futur (s1-s4)',
    }
    if entry_url:
        chainArgs['entry_url'] = entry_url
    chain = buildFfeChain(**chainArgs)

    if publish:
        publish({
            'type':       'entry_url',
            'url':        entry_url or DEFAULT_ENTRY_URL,
            'discipline': discipline,
            'deb':        deb,
            'fin':        fin,
        }, 'crawl-start')

    authId:    str | None = None
    authReason: str       = 'empty_credentials'
    if username and password:
        authId, authReason = _loginFfe(username, password)
    _publishAuth(publish, authReason)
    if authId is None and (username or password):
        log.warn(
            f'Échec login FFE ({authReason}) — '
            f'les noms cheval/cavalier seront masqués par "Voir sa fiche".'
        )

    runner = ChainRunner(
        crawler     = Crawler(),
        publish     = publish,
        session_dir = session_dir,
        auth_id     = authId,
    )
    chainResult = runner.run(chain, session_id = chain.chain_id)

    stepRows: dict[str, list[dict]] = {}
    for stepId, stepResult in chainResult.step_results.items():
        stepRows[stepId] = list(stepResult.data_rows)

    return FfeRunResult(
        chain_id   = chainResult.chain_id,
        total_rows = chainResult.total_rows,
        total_urls = chainResult.total_urls,
        step_rows  = stepRows,
    )
