# 🐴 FFE Crawler — Outil interne PFE

> © 2026 PFE EquiRank Elite — Licence PROPRIETARY (voir `LICENSE`).
> **Équipe** : Louis Guillory · Karlotta Martin · Mathéo Isidoro
> · Mattéo D'Andrea · Melvin Bandeira · Yohan Cana.
>
> Le **moteur** `engine/` est une copie du crawler générique de
> Louis Guillory (Silence Wolf), repris tel quel pour le PFE. La
> chaîne FFE figée (`core/`), l'intégration FastAPI et l'UI sont,
> elles, le travail de l'équipe.

Outil **dédié au projet PFE** : collecte automatique du dataset
`dataset_brut_v2.csv` depuis le portail Telemat/SIF FFE
(Fédération Française d'Équitation).


1. **Calendrier** — dates de début / fin (`deb`, `fin`)
2. **Discipline** — radio (`Toutes disciplines` par défaut)
3. **Identifiants FFE** — uniquement pour l'étape `s5` (fiche
   détaillée de l'équidé : robe, sexe, taille, père, mère)


---

## 🚀 Lancement rapide

Depuis la racine du projet PFE (`PFE_v2/PFE/`) :

```bash
pip install -r ffe_crawler/requirements.txt
playwright install chromium

uvicorn equirank.server:app --reload --port 8000   # ouvre /crawl
```

L'app s'ouvre sur `http://localhost:8501` :

1. Choisis les dates et la discipline
2. Saisis tes identifiants FFE (laisse vide pour un dry-run sans `s5`)
3. Clique sur **🚀 Lancer le crawl FFE**
4. À la fin, télécharge ou écris directement `data/dataset_brut_v2.csv`

---

## 📂 Architecture

```
ffe_crawler/
├── __init__.py
├── core/
│   ├── ffe_chain.py      ← chaîne 5 étapes figée (s1 → s5)
│   ├── runner.py         ← wrapper login + ChainRunner
│   └── dataset_builder.py← join s2/s3/s4/s5 → dataset_brut_v2
├── engine/               ← moteur copié depuis outils/crawler/
│   ├── crawler.py
│   ├── extractors/  detection/  orchestration/
│   ├── output/  session/  storage/
│   └── log/  log_sinks/
├── requirements.txt
├── LICENSE
└── README.md
```

L'UI vit dans `../equirank/` (page `crawl.html`) et appelle le moteur
ci-dessus via le gestionnaire de jobs `equirank/crawl_jobs.py`.

`engine/` est une **copie embarquée** du crawler générique : on n'a
pas de dépendance directe sur `PROJET_PERSO/outils/crawler`. Le tool
est entièrement autonome dans le repo PFE.

---

## 🔄 Pipeline de la chaîne FFE (figé)

| Step | Rôle | `data_selector` | Auth | Colonne URL → step suivant |
|------|------|-----------------|------|-----------------------------|
| s1   | Calendrier (jours actifs) | `.on` | non | `href` |
| s2   | Liste des concours | `td[data-label="Numéro"]` | non | `href` |
| s3   | Liste des épreuves | `td[data-label="Date|Discipline|Épreuve"]` | non | `href` |
| s4   | Engagements (chevaux) | `#t_engts tr` | non | `Équidé_url` |
| s5   | Fiche détail équidé | `.fieldset .label` | **oui** | — |

Le `required_columns_by_step` est verrouillé à
`{'s3': ['href'], 's4': ['Équidé_url']}` pour rester fidèle à la
session de référence.

---

## 📊 Colonnes finales (`dataset_brut_v2.csv`)

```
s2_Numéro,
s3_Discipline, s3_Épreuve,
s4_Clt., s4_Cavalier, s4_Club engageur, s4_Équidé,
s4_Pts qualif. Chpt, s4_Équipe,
s5_Robe, s5_Sexe, s5_Taille, s5_Père, s5_Mère
```

Ordre identique à `data/dataset_brut_v2.csv` historique, prêt à être
ingéré par `adapt_dataset_v2.py` puis `train.py`.

---

## 🌐 Rotation d'IP / proxy (anti-ban FFE)

Le portail Telemat rate-limite agressivement : à partir d'~5 req/s on
encaisse des `429`, puis l'IP source est gelée pendant plusieurs
heures (parfois plusieurs jours) sur **toutes les connexions sortant
par cette IP** — WiFi domestique, partage de connexion mobile, et même
certains VPN partagés.

Pour passer outre sans changer le code, il suffit de définir une
variable d'environnement **avant** de lancer `uvicorn`. Le crawler
(`requests` + Playwright pool + `_fetch_dynamic`) s'en sert
automatiquement.

Variables reconnues, par priorité :

| Variable | Notes |
|----------|-------|
| `EQUIRANK_PROXY` | Spécifique au crawler PFE — gagne sur tout le reste |
| `HTTPS_PROXY`    | Standard `requests` / `curl` |
| `HTTP_PROXY`     | Fallback |

Formats acceptés :

```text
http://host:port
http://user:pass@host:port
socks5://127.0.0.1:9050        ← Tor (requiert `pip install requests[socks]`)
```

### Exemples

**Windows PowerShell — proxy HTTP classique** :
```powershell
$env:EQUIRANK_PROXY = "http://user:pass@proxy.example.com:8080"
uvicorn equirank.server:app --port 8000
```

**Tor (gratuit, IP qui change à chaque circuit)** :
```powershell
# 1. Installer Tor Browser ou Tor expert bundle → service tor en local
# 2. Installer le support SOCKS pour requests
pip install "requests[socks]"
# 3. Pointer le crawler vers le SOCKS de Tor
$env:EQUIRANK_PROXY = "socks5://127.0.0.1:9050"
uvicorn equirank.server:app --port 8000
```

Pour **changer d'IP Tor à la volée** entre deux crawls, redémarre le
service Tor (ou envoie `NEWNYM` via le port de contrôle 9051).

### Vérifier que le proxy est actif

Au boot du pool Playwright, on log la ligne :

```
[pool] proxy actif : socks5://127.0.0.1:9050
```

Si absente, c'est que la variable d'environnement n'est pas vue par
le process Python (relancer le shell après l'avoir définie).

### ⚠️ Anti-ban : shutdown propre

Au `Ctrl+C` pendant un crawl, le serveur :

1. **Met le `_abort_event` à 1** → toutes les requêtes en attente
   `requests.get()` lèvent `RuntimeError('crawl aborted')` sans
   partir sur le réseau ;
2. **Drain la queue Playwright** → les jobs pending sont jetés ;
3. **Kill brutal du subprocess Chromium** (`taskkill /F /T /PID` sous
   Windows, `os.killpg(SIGKILL)` sous Unix) → plus une seule connexion
   TCP vers FFE ne peut partir, même celles en cours de chargement.

C'est ce qui empêche le "burst final" de 600 requêtes qui faisait
bannir l'IP à chaque arrêt brutal.

---

## 🛠️ Usage en bibliothèque

```python
from ffe_crawler import runFfeChain, buildDatasetBrutV2

result  = runFfeChain(
    deb        = '2025-01-01',
    fin        = '2026-01-01',
    discipline = 'Equifun',
    username   = 'monlogin',
    password   = 'monpass',
)
rows = buildDatasetBrutV2(
    step_rows   = result.step_rows,
    output_path = 'data/dataset_brut_v2.csv',
)
print(f'{len(rows)} lignes générées')
```
