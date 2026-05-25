# 📐 FFE Crawler — Architecture interne

> © 2026 PFE EquiRank Elite — Licence PROPRIETARY
> **Équipe** : Louis Guillory · Karlotta Martin · Mathéo Isidoro
> · Mattéo D'Andrea · Melvin Bandeira · Yohan Cana.

Document à destination des contributeurs de l'équipe.
Le `README.md` couvre l'usage ; ce fichier explique les **choix
techniques** et les pièges à ne pas refaire.

---

## 1. Pourquoi un dérivé figé ?

Le crawler générique de `outils/crawler/` est puissant mais demande
à l'utilisateur de tout (re)configurer à chaque session : sélecteurs,
chaînage entre étapes, filtres, colonnes URL… Pour le PFE, on a
**déjà** convergé sur une recette qui marche pour FFE Telemat. Plutôt
que d'imposer aux utilisateurs PFE de relancer le wizard à chaque
fois, on encode cette recette en dur dans `core/ffe_chain.py`.

Le seul axe de paramétrage reste métier : **plage de dates,
discipline, identifiants**. Tout le reste — les `data_selector`, les
colonnes URL de répétition (`href`, `Équidé_url`), le mapping
`required_columns_by_step` — est volontairement immuable.

## 2. Découplage UI / moteur

L'arbre `engine/` est une copie **strictement passive** du crawler
générique : aucun import inverse, aucune modification. Si on veut
remonter une amélioration, il faut la rebaser dans
`outils/crawler/` puis re-synchroniser ici via `cp -r`. Cette
duplication est assumée — c'est le prix à payer pour avoir un outil
autonome qui ne casse pas quand on déplace les autres projets.

L'injection de `engine/` dans `sys.path` est faite **une seule fois**
au tout début de `core/ffe_chain.py` (qui est importé en premier par
`__init__.py`). Tous les autres modules `core/` héritent
implicitement de ce path.

## 3. Mécanique d'authentification

`core/runner.py::_loginFfe` reproduit la logique de
`api/routes_auth.py::login` mais sans FastAPI :

1. `FormSubmitter()` GET la page de login, repère le `<form>`,
   capture les hidden fields (`cs` = CSRF FFE, `redir`)
2. Re-POST avec `login=…` + `passwd=…` + les hiddens captés
3. Heuristique d'échec : si la réponse contient encore un
   `<input type="password" name="passwd">`, le serveur a re-render
   le form → mauvais identifiants → on renvoie `None`
4. Sinon on pousse la session HTTP dans `auth_store` qui rend un
   `auth_id` opaque
5. Ce `auth_id` est passé à `ChainRunner`, qui le propage à chaque
   `StepRunner` — au moment de fetch s5, les cookies de session
   sont injectés dans `requests.Session` ET dans Playwright

Aucun mot de passe n'est persisté disque : `auth_store` est purement
en RAM, et les credentials ne quittent jamais le worker thread.

## 4. Reconstruction du dataset

`core/dataset_builder.py` part de **s4 comme table de fait** (1 row
= 1 cheval engagé sur 1 épreuve). On y rattache :

- **s5 via `s4.Équidé_url == s5.source_url`** — détails du cheval
- **s3 via `s4.source_url == s3.href`**     — discipline / épreuve
- **s2 via `s3.source_url == s2.href`**     — numéro du concours

Si un lookup échoue, on remplit la cellule avec le sentinel `'NONE'`
(c'est ce que fait le dataset_brut_v2 historique). Pas d'exception,
pas de filtre silencieux — l'utilisateur verra les `NONE` dans le
preview Streamlit et pourra décider quoi en faire.

## 5. UI Streamlit — pourquoi un thread + queue ?

Un crawl FFE complet peut prendre 10-30 minutes. Lancer
`runFfeChain` en synchrone dans le main thread Streamlit gèle l'UI.
On utilise donc :

- **Worker thread** : `threading.Thread` daemon qui appelle
  `runFfeChain` avec un `publish` qui pousse les events dans une
  `queue.Queue`
- **Main thread Streamlit** : drain la queue à chaque rerun pour
  mettre à jour `events_log` et l'état `crawl_state`
- **Polling** : tant que `crawl_state == 'running'`, on `time.sleep(2)`
  puis `st.rerun()` — assez réactif pour l'œil, sans saturer le CPU

L'état complet vit dans `st.session_state` car Streamlit ré-exécute
tout le script à chaque interaction.

## 6. Points d'extension

- **Sélecteurs FFE évoluent** : changer dans `core/ffe_chain.py`.
  Pas besoin de toucher Streamlit ni le moteur.
- **Nouvelle colonne dans le dataset** : ajouter dans
  `core/dataset_builder.py::DATASET_COLUMNS` + la branche qui la
  remplit. Penser à ajouter le lookup correspondant si elle vient
  d'une étape pas encore jointe.
- **Discipline supplémentaire** : ajouter dans
  `SUPPORTED_DISCIPLINES`. La valeur est ré-utilisée verbatim
  côté Telemat → vérifier qu'elle correspond exactement au label
  servi par le `<select>`.
