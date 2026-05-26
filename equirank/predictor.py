# ============================================================
# © 2026 PFE EquiRank Elite — Tous droits réservés
# Équipe : Louis Guillory · Karlotta Martin · Mathéo Isidoro
#         · Mattéo D'Andrea · Melvin Bandeira · Yohan Cana
# Licence : PROPRIETARY — voir fichier LICENSE
# ============================================================

"""
Pont entre les saisies de la page Equirank et le modèle PFE entraîné.

Le modèle (LightGBM, models/LightGBM.joblib) attend 19 features fortement
typées — la page Equirank, elle, n'expose qu'un duo cheval / cavalier /
discipline (+ distance optionnelle). On comble l'écart ici :

  1. On retrouve toutes les rows historiques du cheval et du cavalier
     dans data/dataset.csv (la version adaptée par adapt_dataset_v2.py).
  2. On agrège ces rows en valeurs ponctuelles (mode pour les catégories,
     moyenne pour les numériques, count pour les win_rate cumulés).
  3. On encode les colonnes catégorielles avec models/encoders.joblib
     (LabelEncoder) — strictement les mêmes que le modèle a vu en train.
  4. On envoie le vecteur unique au modèle qui renvoie predict_proba ;
     on retourne aussi les stats brutes pour les res-stats de la page.

Si le cheval OU le cavalier est inconnu, on renvoie un dict d'erreur que
la page sait afficher. Pas d'exception qui remonte au navigateur.
"""

from __future__ import annotations

import os
import joblib
import threading
import warnings
from dataclasses import dataclass
from pathlib     import Path
from typing      import Any

import numpy as np
import pandas as pd


# Chemins par défaut — relatifs à la racine PFE/. On les résout dans
# le constructeur pour rester portable (l'utilisateur peut lancer
# uvicorn depuis n'importe où).
_DEFAULT_ROOT = Path(__file__).resolve().parent.parent


# Ordre préférentiel des modèles à charger — on prend le premier qui
# existe sur disque. Cohérent avec ce que app.py historique faisait.
_MODEL_PRIORITY = ('LightGBM', 'XGBoost', 'RandomForest', 'DecisionTree', 'LogisticRegression')


@dataclass
class PredictionResult:
    """Sortie sérialisable de predictor.predict()."""
    proba:                 float       # ∈ [0, 1] — probabilité du résultat positif
    cheval_race:           str         # plus proche de "race" disponible (robe au format dataset PFE)
    cheval_age:            int         # NOTE : pas dans dataset_brut_v2 → expérience cumulée à la place
    cheval_victoires:      int
    cheval_courses:        int
    jockey_victoires:      int
    jockey_courses:        int
    duo_victoires:         int
    duo_courses:           int
    forme:                 str
    discipline:            str
    model_name:            str
    cheval_clt_moyen:      float | None = None
    cavalier_clt_moyen:    float | None = None
    cheval_placements:     list[dict[str, Any]] | None = None
    cavalier_placements:   list[dict[str, Any]] | None = None
    is_cold_start:         bool = False     # cheval et/ou cavalier absents de la base

    def to_dict(self) -> dict[str, Any]:
        # On floor à l'unité côté entiers pour matcher le JSON attendu
        # par le JS de la page Equirank (Math.round(proba), etc.).
        return {
            'proba':            int(round(self.proba * 100)),
            'cheval_race':      self.cheval_race,
            'cheval_age':       int(self.cheval_age),
            'cheval_victoires': int(self.cheval_victoires),
            'cheval_courses':   int(max(self.cheval_courses, self.cheval_victoires)),
            'jockey_victoires': int(self.jockey_victoires),
            'jockey_courses':   int(max(self.jockey_courses, self.jockey_victoires)),
            'duo_victoires':    int(self.duo_victoires),
            'duo_courses':      int(max(self.duo_courses, self.duo_victoires)),
            'forme':            self.forme,
            'discipline':       self.discipline,
            'model_name':       self.model_name,
            'cheval_clt_moyen':    (round(self.cheval_clt_moyen, 1)   if self.cheval_clt_moyen   is not None else None),
            'cavalier_clt_moyen':  (round(self.cavalier_clt_moyen, 1) if self.cavalier_clt_moyen is not None else None),
            'cheval_placements':   self.cheval_placements   or [],
            'cavalier_placements': self.cavalier_placements or [],
            'is_cold_start':    bool(self.is_cold_start),
        }


class EquirankPredictor:
    """
    Garde le modèle + dataset chargés en mémoire pour répondre à plusieurs
    requêtes successives sans relire les fichiers à chaque appel. Thread-safe
    via un lock autour de l'accès au modèle (LightGBM n'est pas garanti
    sûr en multi-thread pour predict_proba).
    """

    def __init__(
        self,
        root_dir:    Path | str | None = None,
        dataset_csv: Path | str | None = None,
    ) -> None:
        root = Path(root_dir) if root_dir else _DEFAULT_ROOT
        self._root        = root
        self._dataset_csv = Path(dataset_csv) if dataset_csv else root / 'data' / 'dataset.csv'
        self._models_dir  = root / 'models'

        self._lock = threading.Lock()
        self._loaded = False

        # Champs remplis par _load() — typés "déclaratifs" pour l'IDE
        self._model:      Any           = None
        self._model_name: str           = ''
        self._encoders:   dict          = {}
        self._feat_cols:  list[str]     = []
        self._df:         pd.DataFrame  = pd.DataFrame()
        self._disciplines: list[str]    = []

    # ──────────────────────────────────────────────
    # Chargement paresseux (pas au constructeur — uvicorn aime que le
    # __init__ soit rapide ; on charge à la première prédiction).
    # ──────────────────────────────────────────────
    def _load(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return

            for name in _MODEL_PRIORITY:
                path = self._models_dir / f'{name}.joblib'
                if path.exists():
                    # Versions sklearn parfois divergentes (entraîné en 1.8,
                    # ré-ouvert ici en 1.6) — on étouffe le warning car on
                    # ne réentraîne pas, on prédit seulement.
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        self._model = joblib.load(path)
                        self._encoders  = joblib.load(self._models_dir / 'encoders.joblib')
                        self._feat_cols = joblib.load(self._models_dir / 'feature_cols.joblib')
                    self._model_name = name
                    break

            if self._model is None:
                raise FileNotFoundError(
                    f'Aucun modèle .joblib trouvé dans {self._models_dir}. '
                    f'Lance d\'abord : python train.py'
                )

            # Le dataset adapté sert à fois pour les stats agrégées
            # (cheval_victoires, etc.) ET pour les valeurs canoniques
            # à passer en input du modèle (taille, robe, etc.).
            if not self._dataset_csv.exists():
                raise FileNotFoundError(
                    f'Dataset manquant : {self._dataset_csv}. '
                    f'Lance d\'abord : python adapt_dataset_v2.py'
                )
            self._df = pd.read_csv(self._dataset_csv)

            # Normalise les noms en MAJUSCULES strip pour matching tolérant
            # (la page Equirank renvoie tel quel ce que l'utilisateur tape).
            self._df['_cheval_key']   = self._df['cheval_nom'].astype(str).str.strip().str.upper()
            self._df['_cavalier_key'] = self._df['cavalier_nom'].astype(str).str.strip().str.upper()

            self._disciplines = sorted(self._df['discipline_famille'].dropna().unique().tolist())
            self._loaded = True

    # ──────────────────────────────────────────────
    # API publique
    # ──────────────────────────────────────────────
    def listDisciplines(self) -> list[str]:
        self._load()
        return list(self._disciplines)

    def estimateRank(self, proba: int | float | None, discipline: str = '') -> int:
        """
        Estime le classement (rang prédit) d'un participant individuel
        à partir de sa proba de réussite. Le modèle PFE ne prédit pas
        directement un rang (il fait une classification binaire « top 3
        ou pts qualif ≥ 8 ») — on dérive donc le rang de la proba en
        l'inversant sur la distribution réelle des tailles d'épreuves
        observées dans la discipline ciblée.

        Méthode :
          - On regarde le nombre médian de participants par épreuve
            pour cette discipline (ex. CSO ≈ 25 partants typiques,
            Equifun ≈ 12)
          - rang_estime = round((1 - proba/100) × (n - 1)) + 1
          - proba 100% → rang 1   (favori absolu)
          - proba  50% → médian   (~ n/2)
          - proba   0% → rang n   (dernier)

        Retourne 0 si on n'a pas assez d'info pour estimer.
        """
        self._load()
        if proba is None or proba == '':
            return 0
        try:
            p = max(0, min(100, int(proba)))
        except (TypeError, ValueError):
            return 0

        df = self._df
        if 'nombre_participants' in df.columns:
            sub = df[df['discipline_famille'] == discipline] if discipline else df
            if not sub.empty:
                med = sub['nombre_participants'].median()
                n_avg = int(med) if med and not np.isnan(med) else 12
            else:
                n_avg = 12
        else:
            n_avg = 12
        n_avg = max(2, n_avg)         # garde-fou minimum (au moins 2 participants)

        rank = round((100 - p) / 100 * (n_avg - 1)) + 1
        return max(1, min(rank, n_avg))

    def listDistances(self, discipline: str = '') -> list[int]:
        """
        Hauteurs (en cm) effectivement vues dans le dataset pour la
        discipline donnée. Sert au <select> de la page / pour ne
        proposer que des distances "réelles" et pas un input libre.

        Si `discipline` est vide, on renvoie l'union de toutes les
        hauteurs présentes dans le dataset.
        Les disciplines sans hauteur (Dressage, Attelage, Western…)
        retournent une liste vide → le frontend désactive le select.
        """
        self._load()
        df = self._df
        if 'hauteur_cm' not in df.columns:
            return []
        sub = df if not discipline else df[df['discipline_famille'] == discipline]
        if sub.empty:
            return []
        vals = sub['hauteur_cm'].dropna().unique().tolist()
        out = sorted({int(v) for v in vals if v and v > 0})
        return out

    def getStats(self) -> dict[str, Any]:
        """
        Stats globales pour l'affichage du hero — récupérées en direct
        depuis les artefacts (pas hard-codées) :

          - métriques du modèle chargé (outputs/models/metrics_{MODEL}.json)
          - cardinalités du dataset adapté
        """
        self._load()

        # Métriques d'évaluation — produites par train.py
        metrics: dict = {}
        metricsPath = self._root / 'outputs' / 'models' / f'metrics_{self._model_name}.json'
        if metricsPath.exists():
            import json
            try:
                metrics = json.loads(metricsPath.read_text(encoding = 'utf-8'))
            except Exception:
                metrics = {}

        # Cardinalités du dataset
        df = self._df
        nChevaux   = int(df['cheval_nom'].nunique())   if 'cheval_nom'   in df.columns else 0
        nCavaliers = int(df['cavalier_nom'].nunique()) if 'cavalier_nom' in df.columns else 0
        # Un duo = un couple (cheval, cavalier) — ngroups est exact
        nDuos      = int(df.groupby(['cheval_nom', 'cavalier_nom']).ngroups) \
                     if {'cheval_nom','cavalier_nom'}.issubset(df.columns) else 0

        return {
            'model_name':    self._model_name,
            'accuracy':      float(metrics.get('accuracy', 0) or 0),
            'f1':            float(metrics.get('f1', 0) or 0),
            'roc_auc':       float(metrics.get('roc_auc', 0) or 0),
            'n_engagements': int(len(df)),
            'n_chevaux':     nChevaux,
            'n_cavaliers':   nCavaliers,
            'n_duos':        nDuos,
            'n_disciplines': len(self._disciplines),
        }

    def suggestCheval(self, prefix: str, limit: int = 10) -> list[str]:
        """Auto-complétion pour le champ 'Nom du cheval' de la page."""
        self._load()
        p = (prefix or '').strip().upper()
        if not p:
            return []
        mask  = self._df['_cheval_key'].str.startswith(p)
        names = self._df.loc[mask, 'cheval_nom'].drop_duplicates().head(limit)
        return names.tolist()

    def suggestCavalier(self, prefix: str, limit: int = 10) -> list[str]:
        self._load()
        p = (prefix or '').strip().upper()
        if not p:
            return []
        mask  = self._df['_cavalier_key'].str.startswith(p)
        names = self._df.loc[mask, 'cavalier_nom'].drop_duplicates().head(limit)
        return names.tolist()

    def predict(
        self,
        cheval:        str,
        cavalier:      str,
        discipline:    str,
        allow_unknown: bool = False,
    ) -> PredictionResult | dict[str, str]:
        """
        Renvoie un PredictionResult si le couple est trouvable, sinon un
        dict {'error': '...'} que le frontend sait afficher dans .error-msg.

        Si `allow_unknown=True`, on prédit MÊME quand le cheval et/ou le
        cavalier sont introuvables dans le dataset historique : on
        construit alors un vecteur de features "rookie" (win_rate
        baseline 0.3, 0 participation) et on laisse le modèle faire avec
        la discipline + nombre de participants + niveau d'épreuve.
        Indispensable pour les concours futurs où une bonne partie des
        engagés peut être nouvelle.
        """
        self._load()

        chevalKey   = (cheval   or '').strip().upper()
        cavalierKey = (cavalier or '').strip().upper()
        discKey     = (discipline or '').strip()

        # Cavalier optionnel : certaines épreuves (équipe non nominative,
        # demi-fond, attelage à plusieurs personnes…) n'ont pas de
        # cavalier identifié. Champ vide → on bascule en cold-start côté
        # cavalier sans pour autant exiger allow_unknown=True.
        cavalierOptional = (cavalierKey == '')

        if not chevalKey or not discKey:
            return {'error': 'Cheval et discipline sont requis.'}

        chevalRows   = self._df[self._df['_cheval_key']   == chevalKey]
        cavalierRows = (
            self._df[self._df['_cavalier_key'] == cavalierKey]
            if cavalierKey else self._df.iloc[0:0]      # DataFrame vide quand cavalier absent
        )

        if chevalRows.empty and not allow_unknown:
            return {'error': f"Cheval « {cheval} » introuvable dans la base."}
        # Cavalier inconnu ET pas optionnel ET pas allow_unknown → erreur.
        # Cavalier optionnel (champ laissé vide) → on accepte → cold-start.
        if cavalierRows.empty and not allow_unknown and not cavalierOptional:
            return {'error': f"Cavalier « {cavalier} » introuvable dans la base."}

        if discKey not in self._disciplines:
            # Fallback : prend la discipline la plus probable du cheval
            if not chevalRows.empty:
                discFallback = chevalRows['discipline_famille'].mode()
                if not discFallback.empty:
                    discKey = str(discFallback.iloc[0])
            # Si toujours inconnue : on retombe sur la discipline modale
            # globale (= ce que le modèle voit le plus souvent).
            if discKey not in self._disciplines:
                fallback = self._df['discipline_famille'].mode()
                discKey = str(fallback.iloc[0]) if not fallback.empty else self._disciplines[0]

        # Rows historiques du duo (intersection) — sert au compteur duo
        duoRows = self._df[
            (self._df['_cheval_key']   == chevalKey) &
            (self._df['_cavalier_key'] == cavalierKey)
        ]

        features = self._buildFeatureVector(chevalRows, cavalierRows, discKey)

        with self._lock:
            try:
                proba = float(self._model.predict_proba(features)[0][1])
            except Exception as exc:
                return {'error': f"Erreur du modèle pendant la prédiction : {exc}"}

        chevalVictoires   = int(chevalRows['resultat_binaire'].sum())   if not chevalRows.empty   else 0
        chevalCourses     = int(len(chevalRows))
        cavalierVictoires = int(cavalierRows['resultat_binaire'].sum()) if not cavalierRows.empty else 0
        cavalierCourses   = int(len(cavalierRows))
        duoVictoires      = int(duoRows['resultat_binaire'].sum())      if not duoRows.empty      else 0
        duoCourses        = int(len(duoRows))

        # Cold-start = au moins l'un des deux n'est pas dans la base
        isColdStart = bool(chevalRows.empty or cavalierRows.empty)

        # Pas d'âge dans le dataset FFE → on substitue par le nombre de
        # participations cheval, capé à 12 pour rester cohérent avec ce
        # que le JS de la page attend (entier entre 3 et 12).
        chevalAge = max(3, min(12, chevalCourses or 3))

        # Robe — c'est la valeur la plus proche d'une "race" disponible
        # dans le dataset PFE (qui n'a pas la race au sens strict).
        chevalRace = (self._safeMode(chevalRows['robe_cheval']) if not chevalRows.empty else '') \
                     or 'Inconnue'

        forme = self._composeForme(chevalRows, cavalierRows, duoRows)

        chevalCltMoyen   = self._meanClassement(chevalRows)
        cavalierCltMoyen = self._meanClassement(cavalierRows)
        chevalPlacements   = self._placementsList(chevalRows)
        cavalierPlacements = self._placementsList(cavalierRows)

        return PredictionResult(
            proba                = proba,
            cheval_race          = chevalRace,
            cheval_age           = chevalAge,
            cheval_victoires     = chevalVictoires,
            cheval_courses       = chevalCourses,
            jockey_victoires     = cavalierVictoires,
            jockey_courses       = cavalierCourses,
            duo_victoires        = duoVictoires,
            duo_courses          = duoCourses,
            forme                = forme,
            discipline           = discKey,
            model_name           = self._model_name,
            cheval_clt_moyen     = chevalCltMoyen,
            cavalier_clt_moyen   = cavalierCltMoyen,
            cheval_placements    = chevalPlacements,
            cavalier_placements  = cavalierPlacements,
            is_cold_start        = isColdStart,
        )

    # ──────────────────────────────────────────────
    # Construction du vecteur de features
    # ──────────────────────────────────────────────
    def _buildFeatureVector(
        self,
        chevalRows:   pd.DataFrame,
        cavalierRows: pd.DataFrame,
        disc:         str,
    ) -> pd.DataFrame:
        """
        Reconstruit une row au format `feature_cols` du modèle. On agrège
        l'historique du cheval / cavalier avec des heuristiques simples :

          - catégoriel → mode (la valeur la plus représentée dans son
            historique — ex : la robe ne change pas, la taille non plus)
          - numérique  → moyenne ou dernière valeur (pour les win_rate
            cumulés on prend la moyenne qui est plus stable que `.iloc[-1]`)
        """

        # Filtre l'historique cheval sur la même discipline si possible,
        # sinon on garde toutes les rows (mieux que zéro signal). Quand
        # le cheval est inconnu (rows vides), on retombe sur le sous-
        # ensemble dataset filtré sur la discipline cible — ce qui donne
        # un signal "moyen de la discipline" plutôt que "moyen global".
        if chevalRows.empty:
            sameDisc   = self._df[self._df['discipline_famille'] == disc]
            chevalHist = sameDisc if not sameDisc.empty else self._df
        else:
            chevalDisc = chevalRows[chevalRows['discipline_famille'] == disc]
            chevalHist = chevalDisc if len(chevalDisc) >= 2 else chevalRows

        if cavalierRows.empty:
            sameDisc     = self._df[self._df['discipline_famille'] == disc]
            cavalierHist = sameDisc if not sameDisc.empty else self._df
        else:
            cavalierDisc = cavalierRows[cavalierRows['discipline_famille'] == disc]
            cavalierHist = cavalierDisc if len(cavalierDisc) >= 2 else cavalierRows

        # Valeurs canoniques du cheval (la dernière row vue est souvent
        # la plus à jour ; pour les catégorielles on prend le mode).
        sexe   = self._safeMode(chevalHist['sexe_cheval'])
        robe   = self._safeMode(chevalHist['robe_cheval'])
        taille = self._safeMode(chevalHist['taille_cheval'])

        # Valeurs canoniques de l'épreuve — on extrapole depuis les
        # rows passées du cheval dans cette discipline ; à défaut on
        # prend la médiane globale du dataset (cf. SimpleImputer
        # entraîné sur la base complète).
        hauteur  = float(chevalHist['hauteur_cm'].median()
                         if not chevalHist['hauteur_cm'].isna().all()
                         else self._df['hauteur_cm'].median())
        niveau   = self._safeMode(chevalHist['niveau_epreuve']) or \
                   self._safeMode(self._df['niveau_epreuve'])
        typeEpr  = self._safeMode(chevalHist['type_epreuve']) or \
                   self._safeMode(self._df['type_epreuve'])
        isEquipe = int(round(chevalHist['is_equipe'].mean()
                             if not chevalHist['is_equipe'].isna().all() else 0))
        nbPart   = int(round(chevalHist['nombre_participants'].mean()
                             if not chevalHist['nombre_participants'].isna().all()
                             else self._df['nombre_participants'].mean()))
        niveauN  = float(chevalHist['niveau_num'].mean()
                         if not chevalHist['niveau_num'].isna().all()
                         else self._df['niveau_num'].mean())

        # Win-rates cumulés — moyenne sur l'historique. Pour un cheval
        # ou cavalier inconnu, on retombe sur la baseline 0.3 utilisée
        # par adapt_dataset_v2 (`rolling_winrate` initialisé à 0.3).
        horseWR  = float(chevalRows['horse_win_rate'].mean())   if not chevalRows.empty   else 0.3
        horsePr  = int(chevalRows['horse_participations'].max() or 0) if not chevalRows.empty else 0
        riderWR  = float(cavalierRows['rider_win_rate'].mean()) if not cavalierRows.empty else 0.3
        riderPr  = int(cavalierRows['rider_participations'].max() or 0) if not cavalierRows.empty else 0
        clubWR   = float(chevalRows['club_win_rate'].mean()) \
                   if (not chevalRows.empty and not chevalRows['club_win_rate'].isna().all()) else 0.3

        synergy   = horseWR * riderWR
        expRatio  = horsePr / (hauteur / 10 + 1) if hauteur else horsePr
        ptsQ      = float(chevalRows['pts_qualification'].mean()) \
                    if (not chevalRows.empty and not chevalRows['pts_qualification'].isna().all()) else 0.0

        # competition_density n'est pas dans dataset.csv : c'est calculé
        # à l'entraînement (cf. preprocessing_v2.py). On la met à la
        # valeur moyenne du dataset — l'encoder catégoriel s'en occupe
        # ensuite. Si l'encoder ne connaît pas la valeur, on retombe sur
        # une catégorie "fréquente" (handle_unknown ne marche pas pour
        # LabelEncoder, donc on choisit une valeur typique en amont).
        competitionDensity = self._mostCommonEncoded('competition_density')

        # Assemblage du dict — clés DOIVENT matcher feat_cols exactement
        rowDict = {
            'sexe_cheval':           sexe   or 'Hongre',
            'robe_cheval':           robe   or 'Bai',
            'taille_cheval':         taille or self._safeMode(self._df['taille_cheval']),
            'discipline_famille':    disc,
            'hauteur_cm':            hauteur,
            'niveau_epreuve':        niveau,
            'type_epreuve':          typeEpr,
            'is_equipe':             isEquipe,
            'nombre_participants':   nbPart,
            'niveau_num':            niveauN,
            'horse_win_rate':        horseWR,
            'horse_participations':  horsePr,
            'rider_win_rate':        riderWR,
            'rider_participations':  riderPr,
            'club_win_rate':         clubWR,
            'couple_synergy':        synergy,
            'experience_ratio':      expRatio,
            'pts_qualification':     ptsQ,
            'competition_density':   competitionDensity,
        }

        df = pd.DataFrame([rowDict])

        # Encodage catégoriel — strictement les LabelEncoder utilisés
        # à l'entraînement. Les colonnes "imp_*" du dict d'encoders
        # sont des SimpleImputer (pour les numériques) — pas besoin de
        # les appliquer ici car on a déjà rempli les NaN au-dessus.
        for col, encoder in self._encoders.items():
            if col not in df.columns:
                continue
            classes = list(getattr(encoder, 'classes_', []))
            val     = df[col].iloc[0]
            if val not in classes:
                # Fallback : la classe la plus fréquente (= index 0 souvent)
                df[col] = classes[0] if classes else val
            df[col] = encoder.transform(df[col].astype(str))

        # Garde l'ordre exact attendu par le modèle
        df = df[self._feat_cols]
        return df

    def _mostCommonEncoded(self, col: str) -> str:
        """Retourne la classe la plus fréquente connue de l'encoder."""
        enc = self._encoders.get(col)
        if enc is None or not hasattr(enc, 'classes_') or len(enc.classes_) == 0:
            return ''
        return str(enc.classes_[0])

    @staticmethod
    def _safeMode(series: pd.Series) -> str:
        """Mode d'une Series — chaîne vide si tout NaN."""
        s = series.dropna()
        if s.empty:
            return ''
        m = s.mode()
        return str(m.iloc[0]) if not m.empty else ''

    @staticmethod
    def _meanClassement(rows: pd.DataFrame) -> float | None:
        """Classement moyen sur l'historique (None si pas exploitable)."""
        if rows.empty or 'classement' not in rows.columns:
            return None
        vals = pd.to_numeric(rows['classement'], errors = 'coerce').dropna()
        if vals.empty:
            return None
        return float(vals.mean())

    @staticmethod
    def _placementsList(rows: pd.DataFrame, limit: int = 30) -> list[dict[str, Any]]:
        """
        Renvoie l'historique des placements pour le détail dépliable côté UI.
        On garde les `limit` plus récents (ordre d'apparition dans le dataset).
        """
        if rows.empty:
            return []
        sub = rows.tail(limit)
        out: list[dict[str, Any]] = []
        for _, r in sub.iterrows():
            clt = pd.to_numeric(r.get('classement'), errors = 'coerce')
            hauteur = r.get('hauteur_cm')
            try:
                hauteurStr = f"{int(hauteur)} cm" if hauteur and not pd.isna(hauteur) else ''
            except (TypeError, ValueError):
                hauteurStr = ''
            out.append({
                'discipline':  str(r.get('discipline_famille') or ''),
                'niveau':      str(r.get('niveau_epreuve') or ''),
                'hauteur':     hauteurStr,
                'clt':         (int(clt) if not pd.isna(clt) else None),
                'participants': (int(r.get('nombre_participants'))
                                 if r.get('nombre_participants') and not pd.isna(r.get('nombre_participants'))
                                 else None),
            })
        return out

    @staticmethod
    def _composeForme(
        chevalRows:   pd.DataFrame,
        cavalierRows: pd.DataFrame,
        duoRows:      pd.DataFrame,
    ) -> str:
        """Phrase courte humaine pour le bloc 'Forme récente' de la page."""

        recent = chevalRows.tail(5)
        winsRecent = int(recent['resultat_binaire'].sum())
        nRecent    = int(len(recent))

        if not duoRows.empty:
            duoWinrate = duoRows['resultat_binaire'].mean()
            tone = "solide" if duoWinrate >= 0.5 else "irrégulière" if duoWinrate >= 0.2 else "fragile"
            return (
                f"Duo {tone} — {len(duoRows)} sorties communes, "
                f"{int(duoRows['resultat_binaire'].sum())} réussites. "
                f"Cheval : {winsRecent}/{nRecent} sur ses 5 dernières."
            )

        if nRecent == 0:
            return "Aucune sortie récente exploitable, prédiction basée sur les caractéristiques."
        if winsRecent >= 3:
            return f"En grande forme : {winsRecent} réussites sur les {nRecent} dernières sorties."
        if winsRecent == 0:
            return f"En manque de résultats : 0 réussite sur les {nRecent} dernières sorties."
        return f"Forme correcte : {winsRecent}/{nRecent} réussites récentes."


    # ──────────────────────────────────────────────
    # Prédiction batch (concours futurs)
    # ──────────────────────────────────────────────
    def predictBatch(self, engagements: list[dict]) -> list[dict]:
        """
        Prend une liste de rows au format dataset_brut_v2 (typiquement
        produites par buildDatasetFuture) et renvoie pour chacune un
        dict enrichi avec `proba`, `verdict` et `model_name`.

        Tolérant aux chevaux/cavaliers absents de la base d'entraînement :
        on prédit quand même en mettant des valeurs neutres pour les
        win_rate (heuristique du dataset adapté : 0.3 = baseline).

        Args:
            engagements : list de dicts avec au moins les clés
                          s4_Équidé, s4_Cavalier, s3_Discipline
        """
        self._load()
        out: list[dict] = []

        for row in engagements:
            cheval     = str(row.get('s4_Équidé')   or '').strip()
            cavalier   = str(row.get('s4_Cavalier') or '').strip()
            discipline = str(row.get('s3_Discipline') or '').strip()

            # allow_unknown=True : on prédit MÊME pour les cheval/cavalier
            # absents du dataset historique. Le verdict 'cold-start' permet
            # à l'UI de signaler que la proba est moins fiable.
            res = self.predict(
                cheval        = cheval,
                cavalier      = cavalier,
                discipline    = discipline,
                allow_unknown = True,
            )

            enriched = dict(row)
            if isinstance(res, dict):
                # Cas qui ne devrait plus arriver avec allow_unknown=True,
                # sauf saisie vide ou erreur modèle. On garde le fallback.
                enriched['proba']      = ''
                enriched['verdict']    = 'inconnu'
                enriched['model_name'] = ''
                enriched['note']       = res.get('error', '')
                enriched['is_cold_start'] = True
            else:
                d = res.to_dict()
                enriched['proba']         = d['proba']
                enriched['verdict']       = ('cold-start' if d['is_cold_start']
                                              else _verdictFromProba(d['proba']))
                enriched['model_name']    = d['model_name']
                enriched['is_cold_start'] = d['is_cold_start']
                enriched['note']          = ('Cheval ou cavalier absent de la base — proba estimée à partir du contexte (discipline, niveau).'
                                              if d['is_cold_start'] else '')
                enriched['cheval_courses']    = d['cheval_courses']
                enriched['cheval_victoires']  = d['cheval_victoires']
                enriched['jockey_courses']    = d['jockey_courses']
                enriched['jockey_victoires']  = d['jockey_victoires']
                enriched['duo_courses']       = d['duo_courses']
                enriched['duo_victoires']     = d['duo_victoires']
                enriched['forme']             = d['forme']
            out.append(enriched)

        return out


def _verdictFromProba(proba: int) -> str:
    """Mappe la proba (0-100) au verdict utilisé côté UI."""
    if proba == '' or proba is None:
        return 'inconnu'
    if proba >= 60: return 'favori'
    if proba >= 35: return 'outsider+'
    return 'outsider'


# ──────────────────────────────────────────────
# Singleton process-wide — uvicorn instancie le module une fois
# au démarrage, on évite donc de recharger model + dataset à chaque
# request HTTP.
# ──────────────────────────────────────────────
_default_predictor: EquirankPredictor | None = None


def getDefaultPredictor() -> EquirankPredictor:
    global _default_predictor
    if _default_predictor is None:
        _default_predictor = EquirankPredictor()
    return _default_predictor
