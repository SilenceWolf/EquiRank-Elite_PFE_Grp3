# ============================================================
# © 2026 Louis Guillory (Silence Wolf) — Tous droits réservés
# Licence : PROPRIETARY — voir fichier LICENSE
# ============================================================

# ChainRunner : exécute une CrawlChain complète en séquence.
# Résout les dépendances parent/enfant entre étapes : si une step a un
# parent_step_id, ses entry_urls sont remplies depuis les url_rows du parent.
# Pattern inspiré de la boucle de pipeline.py (IFCE) mais sans aucune
# dépendance spécifique au domaine.

from __future__ import annotations

from dataclasses import dataclass, field
from datetime    import datetime
from pathlib     import Path
from typing      import Any, Callable

from log     import CrawlerLogger
from crawler import Crawler

from session.chain import CrawlChain

from .step_runner import StepRunner, StepResult


PublishFn = Callable[[dict[str, Any], str], None]


@dataclass
class ChainResult:
    """Résultats agrégés d'une exécution de chain complète."""
    chain_id:     str
    step_results: dict[str, StepResult] = field(default_factory=dict)
    total_rows:   int = 0
    total_urls:   int = 0


class ChainRunner:
    """
    Joue une CrawlChain entière :
      - exécute chaque step dans l'ordre
      - résout les URLs héritées des étapes parentes
      - agrège les résultats
      - publie un event 'done' en fin de chaîne
    """

    def __init__(
        self,
        crawler:     Crawler | None = None,
        publish:     PublishFn | None = None,
        session_dir: Path | None = None,
        auth_id:     str | None = None,
    ) -> None:
        self._crawler     = crawler or Crawler()
        self._publish     = publish or (lambda evt, sid: None)
        self._session_dir = session_dir
        self._log         = CrawlerLogger.get_instance()
        # auth_id propagé à chaque StepRunner créé dans la chaîne →
        # tous les steps héritent de la session post-login.
        self._auth_id     = auth_id

    def run(self, chain: CrawlChain, session_id: str) -> ChainResult:
        self._log.banner(
            f'Chain run — {chain.name}  ({len(chain.steps)} étape(s))'
        )

        step_results: dict[str, StepResult] = {}
        total_rows = 0
        total_urls = 0

        try:
            for step in chain.steps:
                # Hérite les URLs du parent si demandé et pas déjà renseigné
                if step.parent_step_id and not step.entry_urls:
                    parent = step_results.get(step.parent_step_id)
                    if parent is None:
                        self._log.warn(
                            f'  step {step.step_id} : parent {step.parent_step_id} '
                            f'introuvable, étape skippée'
                        )
                        continue
                    # On préfère 'href' (rempli par tous les extractors sur <a>)
                    # à 'content' (qui peut être le TEXTE du lien — ex. ".on"
                    # sur le calendrier FFE où content="16" et href="…cs=4.xxx").
                    # Et on résout systématiquement contre `source_url` :
                    # FFE Telemat sert des href relatifs (`./?cs=4.…`) que le
                    # filtre `startswith('http')` rejetterait sinon.
                    from urllib.parse import urljoin as _urljoin
                    seen: set[str] = set()
                    hrefs: list[str] = []
                    for r in parent.url_rows:
                        raw = r.get('href') or r.get('content') or ''
                        if not isinstance(raw, str) or not raw:
                            continue
                        if raw.startswith('http'):
                            url = raw
                        else:
                            src = r.get('source_url') or ''
                            if not src:
                                continue
                            url = _urljoin(src, raw)
                            if not url.startswith('http'):
                                continue
                        if url not in seen:
                            seen.add(url)
                            hrefs.append(url)
                    step.entry_urls = hrefs
                    self._log.info(
                        f'  step {step.step_id} : {len(step.entry_urls)} URL(s) héritées '
                        f'de {step.parent_step_id}'
                    )

                runner = StepRunner(
                    crawler     = self._crawler,
                    publish     = self._publish,
                    session_dir = self._session_dir,
                    auth_id     = self._auth_id,
                )
                result = runner.run(step, session_id)
                step_results[step.step_id] = result
                total_rows += len(result.data_rows)
                total_urls += len(result.url_rows)

            self._publish({
                'type':       'done',
                'session_id': session_id,
                'chain_id':   chain.chain_id,
                'total_rows': total_rows,
                'total_urls': total_urls,
                'timestamp':  datetime.now().isoformat(timespec='seconds'),
            }, session_id)
            self._log.success(
                f'Chain terminée — {total_rows} lignes, {total_urls} URL(s)'
            )

        except Exception as exc:
            import traceback
            self._log.error(f'Chain {chain.chain_id} a échoué : {exc}')
            self._publish({
                'type':     'error',
                'chain_id': chain.chain_id,
                'message':  str(exc),
                'trace':    traceback.format_exc(),
            }, session_id)
            raise

        return ChainResult(
            chain_id     = chain.chain_id,
            step_results = step_results,
            total_rows   = total_rows,
            total_urls   = total_urls,
        )
