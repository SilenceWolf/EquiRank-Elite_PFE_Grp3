# ============================================================
# © 2026 Louis Guillory (Silence Wolf) — Tous droits réservés
# Licence : PROPRIETARY — voir fichier LICENSE
# ============================================================

# FormSubmitter — soumet un <form> HTML d'une page cible en respectant
# son action / method, en transportant TOUS les <input type="hidden">
# (token CSRF, session id, etc.) et en ajoutant les valeurs saisies
# par l'utilisateur.
#
# Cas d'usage type FFE SIF :
#   Le formulaire de recherche du calendrier a method="post" et un
#   <input name="cs" type="hidden" value="4.XXX..."> qui est un token
#   CSRF. Sans ce token, le serveur rejette la requête. Sans submit
#   réel, les filtres date n'ont aucun effet — le crawl retombe sur
#   la page non filtrée.
#
# Architecture : maintient une requests.Session pour préserver les
# cookies entre le GET initial (récupération du form + cookies) et le
# POST de soumission. Sinon le serveur perd la session.

from __future__ import annotations

from dataclasses import dataclass
from typing      import Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, Tag

from log import CrawlerLogger


# Headers réalistes — certains serveurs bloquent les UA robots évidents.
# Identiques à ceux du Crawler principal pour cohérence.
_DEFAULT_HEADERS = {
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


@dataclass
class FormSubmitResult:
    """Résultat d'une soumission de formulaire."""
    final_url: str        # URL finale après follow_redirects (= action si pas de redir)
    html:      str        # HTML brut de la réponse, encodage déjà décodé
    method:    str        # 'POST' ou 'GET' — method effective utilisée
    fields_sent: dict[str, str]   # ce qu'on a réellement envoyé (hidden + user)


class FormSubmitter:
    """
    Trouve un <form> HTML dans une page et le soumet avec les valeurs
    fournies par l'utilisateur, en préservant cookies + hidden tokens.

    L'instance maintient une requests.Session() — réutilisable pour
    plusieurs soumissions sur le même domaine si nécessaire.
    """

    def __init__(
        self,
        headers: dict[str, str] | None = None,
        cookies: dict[str, str] | None = None,
    ) -> None:
        self._session = requests.Session()
        self._session.headers.update(headers or _DEFAULT_HEADERS)
        # Proxy : si l'utilisateur a défini EQUIRANK_PROXY / HTTPS_PROXY
        # (ex. Tor en SOCKS5), on l'applique à la session — sinon le
        # login FFE partirait depuis l'IP locale même quand le reste
        # du crawler passe par Tor.
        try:
            from ..crawler import _get_proxy_dict
            _proxies = _get_proxy_dict()
            if _proxies:
                self._session.proxies.update(_proxies)
        except Exception:
            pass
        # Pool de connexions agrandi : avec 8 workers en parallèle qui
        # tapent tous le même domaine, le pool par défaut (10) suffit
        # mais on monte à 20 pour absorber un pic et éviter le warning
        # "Connection pool is full". Réutilisation des connexions TCP
        # (HTTP keep-alive) → -150-200ms par requête sur les chaînes
        # longues vers le même hôte.
        from requests.adapters import HTTPAdapter
        adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=0)
        self._session.mount('https://', adapter)
        self._session.mount('http://',  adapter)
        # Cookies initiaux : sert à injecter une session post-login
        # (auth_store) → le FormSubmitter voit le site comme l'utili-
        # sateur connecté plutôt que comme un visiteur anonyme. Sans
        # ça, FFE Telemat sert les pages anonymisées même si l'auth
        # a été faite par un endpoint séparé.
        if cookies:
            for name, value in cookies.items():
                self._session.cookies.set(name, value)
        self._log = CrawlerLogger.get_instance()

    @property
    def session(self) -> requests.Session:
        """Expose la session (cookies + headers) pour que l'appelant
        puisse continuer à naviguer sur le site en conservant l'état."""
        return self._session

    # ── API publique ──────────────────────────────────────────────────────

    def submit(
        self,
        page_url:    str,
        field_values: dict[str, str],
    ) -> Optional[FormSubmitResult]:
        """
        Pipeline complet :
          1. GET la page (récupère cookies + le HTML qui contient le form)
          2. Trouve le form qui contient au moins un input dont le name
             matche field_values
          3. Construit le body : hidden inputs + user values
          4. POST (ou GET selon form.method) sur l'action absolue
          5. Retourne (final_url, html, method, fields_sent)

        Retourne None si aucun form approprié n'est trouvé — l'appelant
        peut alors retomber sur le comportement GET classique.
        """
        try:
            # (connect, read) timeout étendu : les back-office FFE et autres
            # intranets sont lents à générer leur HTML. Retry 1x sur ReadTimeout.
            try:
                initial = self._session.get(page_url, timeout=(15, 60))
            except requests.exceptions.ReadTimeout:
                self._log.warn(f'  form-submit : ReadTimeout sur GET initial, retry')
                initial = self._session.get(page_url, timeout=(15, 120))
            initial.raise_for_status()
            initial.encoding = initial.apparent_encoding or 'utf-8'
        except requests.RequestException as exc:
            self._log.warn(f'  form-submit : GET initial échoué sur {page_url} : {exc}')
            return None

        soup = BeautifulSoup(initial.text, 'html.parser')
        form = self._find_matching_form(soup, field_values)
        if form is None:
            return None

        action_url = self._resolve_action(form, page_url)
        method     = (form.get('method') or 'GET').upper()
        body       = self._build_body(form, field_values)

        self._log.info(
            f'  form-submit : {method} → {action_url} '
            f'({len(body)} champs, dont {sum(1 for k in body if k in field_values)} user)'
        )

        try:
            # Read timeout plus long pour le POST : traitement serveur
            # (requête SQL, génération de vue filtrée) + retour HTML.
            # Retry sur ReadTimeout ET ConnectionError (FFE ferme parfois
            # la connexion — "Remote end closed without response").
            resp = self._try_submit(method, action_url, body)
            if resp is None:
                return None
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding or 'utf-8'
        except requests.RequestException as exc:
            self._log.error(f'  form-submit : soumission échouée : {exc}')
            return None

        return FormSubmitResult(
            final_url   = resp.url,
            html        = resp.text,
            method      = method,
            fields_sent = body,
        )

    # ── Internals ──────────────────────────────────────────────────────────

    def _try_submit(
        self,
        method:     str,
        action_url: str,
        body:       dict[str, str],
    ) -> requests.Response | None:
        """
        Tente le POST/GET avec retry sur ReadTimeout et ConnectionError
        (FFE ferme parfois la connexion pendant le traitement — elle
        déclenche une `RemoteDisconnected` / `ConnectionError`).
        Retourne None si tous les essais échouent.
        """
        # Tuples (connect_timeout, read_timeout) pour chaque tentative.
        attempts = [(15, 90), (15, 180)]

        for i, (ct, rt) in enumerate(attempts, start=1):
            try:
                if method == 'POST':
                    return self._session.post(
                        action_url, data=body, timeout=(ct, rt),
                        allow_redirects=True,
                    )
                else:
                    return self._session.get(
                        action_url, params=body, timeout=(ct, rt),
                        allow_redirects=True,
                    )
            except (
                requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError,
            ) as exc:
                if i == len(attempts):
                    self._log.error(
                        f'  form-submit : {method} échoué après {i} tentatives : {exc}'
                    )
                    return None
                self._log.warn(
                    f'  form-submit : {method} échec (tentative {i}/{len(attempts)}) → {exc}'
                )
        return None

    def _find_matching_form(
        self,
        soup:         BeautifulSoup,
        field_values: dict[str, str],
    ) -> Tag | None:
        """
        Trouve le <form> qui contient au moins un input/select/textarea
        dont le name est dans field_values. S'il y en a plusieurs, prend
        celui qui matche le PLUS de field_values (probablement le bon).
        """
        wanted = {k for k, v in field_values.items() if v}
        if not wanted:
            return None

        best:        Tag | None = None
        best_score:  int        = 0

        for form in soup.find_all('form'):
            score = 0
            for inp in form.find_all(['input', 'select', 'textarea']):
                name = inp.get('name')
                if name and name in wanted:
                    score += 1
            if score > best_score:
                best       = form
                best_score = score

        return best

    def _resolve_action(self, form: Tag, page_url: str) -> str:
        """
        Résout l'attribut `action` du form en URL absolue.
        Cas particuliers :
          - action manquante ou vide → soumet sur la même URL
          - action = '-' (FFE) → idem
          - action absolue ou relative → urljoin standard
        """
        action = (form.get('action') or '').strip()
        if not action or action == '-' or action == '?':
            return page_url
        return urljoin(page_url, action)

    def _build_body(
        self,
        form:         Tag,
        field_values: dict[str, str],
    ) -> dict[str, str]:
        """
        Construit le dict de soumission. Règles :
          - hidden : toujours inclus avec leur valeur d'origine (CSRF, etc.)
          - input/select/textarea avec name : valeur user si fournie,
            sinon valeur par défaut de l'élément (évite d'envoyer une
            valeur vide qui pourrait casser la validation côté serveur)
          - submit/button/reset/image/file : ignorés (on simule pas
            l'upload, on ne sait pas quel bouton a été "cliqué" donc
            on n'envoie aucune submit value)
        """
        body: dict[str, str] = {}

        for inp in form.find_all('input'):
            name      = inp.get('name')
            inp_type  = (inp.get('type') or 'text').lower()
            if not name:
                continue
            if inp_type in ('submit', 'button', 'reset', 'image', 'file'):
                continue
            if inp_type == 'hidden':
                body[name] = inp.get('value', '') or ''
            elif name in field_values:
                body[name] = field_values[name]
            elif inp_type in ('checkbox', 'radio'):
                # Préserve si coché par défaut
                if inp.has_attr('checked'):
                    body[name] = inp.get('value', 'on') or 'on'
            else:
                # Valeur par défaut pour les autres
                default = inp.get('value', '') or ''
                if default:
                    body[name] = default

        for sel in form.find_all('select'):
            name = sel.get('name')
            if not name:
                continue
            if name in field_values:
                body[name] = field_values[name]
            else:
                # Première option sélectionnée par défaut, sinon première option
                selected = sel.find('option', selected=True) or sel.find('option')
                if selected:
                    body[name] = selected.get('value', selected.get_text(strip=True))

        for ta in form.find_all('textarea'):
            name = ta.get('name')
            if not name:
                continue
            if name in field_values:
                body[name] = field_values[name]

        return body
