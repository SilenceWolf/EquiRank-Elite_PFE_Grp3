# 🐴 PFE — EquiRank Elite

> © 2026 PFE EquiRank Elite — Projet IA 2025/2026.
> **Équipe** : Louis Guillory · Karlotta Martin · Mathéo Isidoro
> · Mattéo D'Andrea · Melvin Bandeira · Yohan Cana.
> Prédiction de performances en concours FFE de bout en bout :
> collecte des données, adaptation, entraînement, et interface web.

---

## 📁 Structure du projet

```
PFE/
├── data/                      ← Datasets (brut + adapté)
│   ├── dataset_brut_v2.csv    ← Sortie brute du crawler FFE
│   └── dataset.csv            ← Sortie de adapt_dataset_v2.py
├── src/
│   ├── preprocessing_v2.py    ← Feature engineering + encodage
│   ├── eda.py                 ← Analyse exploratoire
│   └── model_training.py      ← Entraînement, évaluation, SHAP
├── models/                    ← Modèles sauvegardés (.joblib)
│   ├── LightGBM.joblib        ← Modèle utilisé en prod
│   ├── encoders.joblib        ← LabelEncoders / SimpleImputer
│   └── feature_cols.joblib    ← Ordre des 19 features attendu
├── outputs/
│   ├── eda/                   ← Graphiques EDA
│   └── models/                ← ROC, importance, SHAP, metrics_*.json
│
├── ffe_crawler/               ← 🕷️ Moteur de collecte FFE (dérivé crawler)
│   ├── core/                  ← chaîne FFE figée + builder dataset_brut_v2
│   └── engine/                ← moteur crawler embarqué
│
├── equirank/                  ← 🎯 Frontend web unique (prédiction + crawl + futur)
│   ├── static/
│   │   ├── equirank.css       ← Feuille de style partagée (dark + gold)
│   │   ├── index.html         ← Page de prédiction (Equirank Elite)
│   │   ├── crawl.html         ← Page crawl passé (entraîne le modèle)
│   │   ├── crawl_future.html  ← Page /crawl-predict (prédit les engagements)
│   │   ├── predictions.html   ← Liste des concours futurs + recherche
│   │   ├── concours_detail.html ← Détail d'UN concours futur (table participants)
│   │   └── logo.png
│   ├── predictor.py           ← Pont saisie utilisateur → modèle LightGBM (+ predictBatch)
│   ├── crawl_jobs.py          ← Jobs de crawl PASSÉ (entraînement)
│   ├── future_jobs.py         ← Jobs de crawl predict + store prédictions
│   ├── server.py              ← FastAPI : sert toutes les pages + APIs
│   └── liaison.py             ← Banc d'essai Streamlit du predictor
│
├── adapt_dataset_v2.py        ← dataset_brut_v2 → dataset.csv (features)
├── train.py                   ← Entraînement principal
├── generate_sample_data.py    ← Générateur de données synthétiques
└── requirements.txt
```

---

## 🚀 Pipeline complet (de A à Z)

```bash
# 0. Installation
pip install -r requirements.txt
playwright install chromium    # uniquement si on relance ffe_crawler

# 1. Lance le serveur web — un seul process pour les 4 pages
uvicorn equirank.server:app --reload --port 8000
#   → http://localhost:8000              Prédiction unique
#   → http://localhost:8000/crawl        Crawl PASSÉ (entraîne le modèle)
#   → http://localhost:8000/crawl-predict Crawl PREDICT (prédit les engagements + auto-save historique)
#   → http://localhost:8000/predictions  Liste + recherche dans les prédictions

# 2. /crawl — concours déjà disputés (résultats connus) → entraînement
#    Date de fin bornée à aujourd'hui. À la fin : bouton "⚡ Adapter +
#    ré-entraîner" qui chaîne adapt_dataset_v2 + train.py + reload du predictor.

# 3. /crawl-predict — concours pas encore disputés (sans résultats)
#    Date de début bornée à demain. Skip s5 (pas d'auth nécessaire).
#    À la fin : prédiction batch de chaque engagement avec le modèle
#    courant. Téléchargement CSV avec les colonnes `proba` + `verdict` en plus.

# 4. /predictions — vue agrégée
#    Cartes par concours → clic → table des participants triée par proba.
#    Recherche unique : tape un nom de cheval, cavalier ou n° concours.

# Alternative CLI (équivalent du bouton "⚡ Adapter + ré-entraîner")
python adapt_dataset_v2.py --input data/dataset_brut_v2.csv --output data/dataset.csv
python train.py --data data/dataset.csv --model LightGBM
```

---

## 🧅 Lancer le crawler derrière Tor (anti-ban IP FFE)

Le portail FFE Telemat bannit l'IP source dès qu'on dépasse ~5 req/s.
Une fois ban, **toutes** les connexions sortant par cette IP sont
gelées (WiFi maison, partage 4G, certains VPN). Pour contourner sans
toucher au code, on route tout le crawler (`requests` + Playwright)
par **Tor** — chaque circuit fournit une IP différente, c'est gratuit.

### Setup (une fois)

```powershell
# 1. Tor — Windows : télécharger Tor Expert Bundle
#    https://www.torproject.org/download/tor/
#    Lancer `tor.exe` → écoute SOCKS5 sur 127.0.0.1:9050
#    (alternative simple : ouvrir Tor Browser, ça suffit aussi)

# 2. Dépendances Python (le support SOCKS est déjà dans requirements.txt)
pip install -r requirements.txt
playwright install chromium
```

### Lancement avec Tor activé

```powershell
# Pointer le crawler vers le SOCKS local de Tor
$env:EQUIRANK_PROXY = "socks5://127.0.0.1:9050"

# Lancer l'app normalement — le crawler utilisera Tor automatiquement
uvicorn equirank.server:app --reload --port 8000
```

Tu dois voir dans les logs au premier crawl :
```
[pool] proxy actif : socks5://127.0.0.1:9050
```

### Changer d'IP entre deux crawls

Redémarre simplement le process `tor.exe`, ou envoie `NEWNYM` via le
port de contrôle 9051 si tu l'as activé dans `torrc`.

### Désactiver Tor

```powershell
Remove-Item Env:EQUIRANK_PROXY
```

Pour les détails complets (autres formats de proxy, vérification,
shutdown propre anti-burst), voir
[ffe_crawler/README.md](ffe_crawler/README.md#-rotation-dip--proxy-anti-ban-ffe).

---

## 🎯 Interface de prédiction Equirank Elite

`equirank/` contient la page web qui restitue les prédictions du
modèle LightGBM entraîné en (3). La page (`static/index.html`) est
servie par FastAPI (`server.py`) qui expose :

| Endpoint                          | Méthode | Rôle                                          |
|-----------------------------------|---------|-----------------------------------------------|
| `/`                                   | GET     | Page Equirank Elite (prédiction)              |
| `/crawl`                              | GET     | Page de crawl FFE (même style)                |
| `/api/disciplines`                    | GET     | Disciplines connues du dataset                |
| `/api/suggest/cheval?q=...`           | GET     | Auto-complétion noms de chevaux               |
| `/api/suggest/cavalier?q=...`         | GET     | Auto-complétion noms de cavaliers             |
| `/api/predict`                        | POST    | Prédiction `{cheval, cavalier, discipline}`   |
| `/api/crawl/disciplines`              | GET     | Disciplines supportées par le crawler         |
| `/api/crawl/filters?url=...`          | GET     | Dropdowns FFE (région/dept/disc/championnat)  |
| `/api/crawl/start`                    | POST    | Démarre un crawl → renvoie `job_id`           |
| `/api/crawl/status/{job_id}?cursor=N` | GET     | Polling : état + events depuis cursor         |
| `/api/crawl/download/{job_id}`        | GET     | Télécharge le `dataset_brut_v2.csv` produit   |
| `/api/crawl/write_to_data/{job_id}`   | POST    | Écrit le dataset dans `data/`                 |
| `/api/crawl/train/{job_id}`           | POST    | Lance adapt + train + reload predictor        |
| `/api/crawl/train/status/{pipeline_id}` | GET   | Polling du pipeline d'apprentissage           |
| **/crawl-predict**                    | GET     | Page de crawl predict (auto-save historique)   |
| `/api/future/start`                   | POST    | Démarre un crawl predict → job_id              |
| `/api/future/status/{job_id}?cursor=N` | GET    | Polling (events crawl + step "predict")        |
| `/api/future/download/{job_id}`       | GET     | CSV des engagements + colonnes proba/verdict   |
| **/predictions**                      | GET     | Page liste + recherche                         |
| **/predictions/{concours_id}**        | GET     | Page détail d'un concours                      |
| `/api/predictions/concours`           | GET     | Liste des concours futurs crawlés              |
| `/api/predictions/search?q=...`       | GET     | Recherche cheval / cavalier / n° concours      |
| `/api/predictions/{concours_id}`      | GET     | Détail d'un concours (engagements + prédict.)  |
| `/api/health`                         | GET     | Health-check                                  |

Le predictor reconstruit les 19 features attendues par le modèle à
partir de l'historique du cheval / cavalier dans `data/dataset.csv`
— le lien entre la page (3 saisies) et le modèle (19 features) est
opéré dans `equirank/predictor.py`.

### Banc d'essai Streamlit (optionnel)

Pour tester rapidement la logique sans navigateur :

```bash
streamlit run equirank/liaison.py
```

---

## 🕷️ Collecte FFE — page `/crawl`

La page recueille dates, discipline et identifiants FFE, déclenche un job en
background via `equirank/crawl_jobs.py` et restitue la progression en
temps réel (rail des 5 étapes + barre + journal d'événements). À la
fin, deux actions : télécharger le CSV, ou l'écrire directement dans
`data/dataset_brut_v2.csv`.


---

## 🤖 Modèles disponibles

| Modèle              | Rapide | Performant | Interprétable |
|---------------------|:------:|:----------:|:-------------:|
| LogisticRegression  | ✅     | ➖         | ✅            |
| DecisionTree        | ✅     | ➖         | ✅✅          |
| RandomForest        | ➖     | ✅         | ✅            |
| XGBoost             | ➖     | ✅✅       | ✅            |
| **LightGBM**        | ✅     | ✅✅       | ✅            |

LightGBM est sélectionné automatiquement par `equirank.predictor`
s'il est présent dans `models/` ; sinon on essaie dans l'ordre :
XGBoost → RandomForest → DecisionTree → LogisticRegression.

---

## 📈 Sorties générées

Après `train.py`, on retrouve dans `outputs/` :

- `eda/` : EDA visuel (missing values, distributions, heatmap, perf par
  catégorie, etc.)
- `models/` : matrices de confusion, courbes ROC/PR, importance des
  features, SHAP, et `metrics_{MODEL}.json`.

---

## 🎯 Cible & métriques

**Cible binaire** : `1` si classement ≤ 3 OU points qualif. ≥ 8.

**Métriques** : F1-score (principal), AUC-ROC, précision, rappel,
SHAP pour l'interprétabilité.

---

## 📦 Dépendances clés

- `pandas`, `numpy`, `scikit-learn` — manipulation + modèles de base
- `xgboost`, `lightgbm` — gradient boosting
- `shap` — interprétabilité
- `streamlit` — apps internes (crawler + liaison)
- `fastapi`, `uvicorn`, `pydantic` — backend Equirank
- `requests`, `beautifulsoup4`, `playwright` — crawler FFE
