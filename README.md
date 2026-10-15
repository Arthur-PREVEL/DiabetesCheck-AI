# DiabetesCheck AI : dépistage du diabète expliqué (IA hybride)

**Auteurs** : Arthur PREVEL et Wiame BOURJILA
**Cadre** : mini-projet *IA Hybride*, ISIS Castres, 5e année DSIA (2026), **sujet 1 : dépistage du diabète expliqué**
**Dépôt** : https://github.com/Arthur-PREVEL/DiabetesCheck-AI

Question clinique : à partir de quelques mesures simples, estimer le risque de diabète d'une patiente et l'expliquer d'une façon
qu'un médecin puisse vérifier. L'approche est **hybride** : un modèle d'apprentissage (régression logistique, XGBoost) combiné à des
**règles médicales sourcées**, placées en entrée (variables expertes) et en sortie (règles qui encadrent la prédiction).

## Contenu du dépôt

| Fichier | Rôle |
|---|---|
| `DiabetesCheck_Hybride_v2.ipynb` | **Code principal** : notebook complet (code, résultats et conclusions), à exécuter de haut en bas |
| `diabetes.csv` | Jeu de données Pima Indians Diabetes (768 patientes, 8 variables + étiquette) |
| `requirements.txt` | Dépendances Python |
| `README.md` | Ce fichier |
| `Soutenance_DiabetesCheck.pptx` | Diaporama de la soutenance |
| `archive/` | **Hors périmètre** : fichiers du premier projet (notebook initial, application Streamlit, brouillons), conservés pour mémoire |

Pour évaluer le projet, il suffit de lire ce README, le `requirements.txt` et le notebook `DiabetesCheck_Hybride_v2.ipynb`.
Tout le reste, y compris l'application Streamlit, relève de l'archive et n'a pas été mis à jour avec les résultats de la version 2.

## Réexécuter le projet

```bash
git clone https://github.com/Arthur-PREVEL/DiabetesCheck-AI.git && cd DiabetesCheck-AI
python -m venv .venv && source .venv/bin/activate     # Windows : .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook DiabetesCheck_Hybride_v2.ipynb         # puis « Run All »
```

- Le notebook lit `diabetes.csv` s'il est dans le même dossier ; sinon il télécharge le jeu depuis OpenML
  (`fetch_openml("diabetes", version=1)`, connexion Internet nécessaire). Les résultats sont identiques avec les deux sources.
- Toutes les graines aléatoires sont fixées (42). L'exécution complète prend de l'ordre de deux minutes dans notre environnement de test
  (validation croisée répétée, sous-échantillons, bootstrap) ; ce temps dépend de la machine.
- Environnement testé : Python 3.12 ; les versions minimales du `requirements.txt` n'ont pas toutes été vérifiées.
- Aucun LLM n'est utilisé dans le projet : les explications destinées à la patiente sont produites par **gabarit** (texte déterministe, vérifiable).

## Données

Source : **Kaggle**, *Pima Indians Diabetes Database* (https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database), jeu issu du National Institute of Diabetes and
Digestive and Kidney Diseases et diffusé par le dépôt UCI. 768 femmes d'origine Pima, d'au moins 21 ans ; 8 variables (grossesses, glycémie 2 h après charge en glucose,
pression diastolique, pli cutané, insulinémie, IMC, score d'antécédents familiaux, âge) et une étiquette diabète oui/non. Dans plusieurs colonnes, un 0 est une valeur
manquante (par exemple 49 % pour l'insulinémie), traitée comme telle.

## Structure du notebook

| Sections | Contenu |
|---|---|
| 1-6 | Étape 1 : règle de qualité (0 = valeur manquante), comparaison des imputations (médiane, KNN, MICE), modèles de référence, **choix du seuil sur le train (hors-pli)**, SMOTE contre seuil à rappel égal, tranches d'âge |
| 7-11 | Étape 2 : règles médicales et sources, règles en entrée, règles en sortie, ablation sur le test puis **ablation robuste** (validation croisée 5×5, intervalles de confiance) |
| 12-14 | Cohérence, performance par tranche d'âge, cas hors distribution |
| 15 | **Calibration** : ECE, courbes de fiabilité, recalibration par température |
| 16 | **Cohérence** sur 768 patientes (R2, R6, monotonie) et XGBoost sous contrainte de monotonie |
| 17 | Cas difficile : **peu de données** (50 à 614 patientes) |
| 18 | **Explication patiente** : SHAP + règles déclenchées + texte par gabarit |
| 19 | **Analyse d'erreurs** attribuée à chaque brique |
| 20-21 | Coût de calcul, conclusion générale (grille de dimensions) |

Le protocole commun de la fiche est couvert : référence non hybride, ablations, cas difficiles (peu de données, règle applicable, hors distribution),
analyse d'erreurs par brique, et sept dimensions de la grille (performance, cohérence, fidélité, calibration, explicabilité, robustesse, coût).

## Résultats principaux

- **Performance** : AUC ≈ 0,84 (régression logistique) et 0,83 (XGBoost) en validation croisée sur 768 patientes ; aucune version hybride n'est démontrablement meilleure.
- **Seuil** : choisi sur le train (rappel cible 80 %), et non sur le test ; SMOTE n'apporte rien de plus qu'un seuil bas à rappel égal.
- **Valeur des règles** : garde-fou hors distribution (XGBoost sous-estime une glycémie de 250 mg/dL, la règle R1 corrige), traçabilité de chaque décision, explication vérifiable.
- **Cohérence** : XGBoost viole la monotonie (le risque baisse quand la glycémie monte pour 20 % des patientes) ; une contrainte de monotonie corrige l'essentiel et améliore l'AUC.
- **Limite principale** : jeu restreint aux femmes Pima d'au moins 21 ans, résultats non généralisables.

## Sources des règles médicales

| Règle | Condition | Source |
|---|---|---|
| R1 | Glucose ≥ 200 mg/dL (2 h, HGPO 75 g) : diabète | American Diabetes Association, *Standards of Care in Diabetes*, section 2, Diagnosis and Classification (édition 2024 utilisée ; seuil inchangé dans l'édition 2026) |
| R2 | 140 ≤ Glucose < 200 mg/dL : intolérance au glucose | ADA, *Standards of Care*, section 2 ; OMS/IDF 2006, *Definition and diagnosis of diabetes mellitus and intermediate hyperglycaemia* |
| R3 | IMC ≥ 30 : obésité | OMS, fiche d'information *Obesity and overweight* |
| R4 | IMC ≥ 25 et âge ≥ 35 ans : dépistage recommandé | USPSTF, *Screening for Prediabetes and Type 2 Diabetes*, JAMA 2021;326(8):736-743 (adultes de 35 à 70 ans en surpoids ou obèses) |
| R5 | Pression diastolique ≥ 90 mmHg : hypertension | ESC/ESH 2018 ; facteur de risque listé par l'ADA |
| R6 | Glucose < 140, IMC < 25, âge < 35 : aucun critère de dépistage | **Règle dérivée** de USPSTF 2021 et ADA ; plafond de risque 0,1 fixé empiriquement (1 diabétique sur 59 dans le train) |

Définition de l'étiquette : critères de l'OMS (glycémie à 2 h ≥ 200 mg/dL), d'après la description du jeu (OpenML/UCI) ; Smith et al. (1988), *Using the ADAP
learning algorithm to forecast the onset of diabetes mellitus*, Proc. Symp. Computer Applications in Medical Care, p. 261-265.
Les sources sont à citer dans leur édition la plus récente ; le seuil de la règle R5 dépend de l'édition de recommandation retenue (130/80 ou 140/90).

## Usage d'assistants de code

Nous avons utilisé **Claude Code** (Anthropic) pour assister la réalisation du code (notebook) et la création des diapositives de la soutenance.
Les choix de conception, la relecture, la vérification des résultats et leur interprétation relèvent des auteurs.

## Avertissement

Cet outil est un exercice pédagogique d'aide au dépistage. **Ce n'est pas un dispositif médical et il ne fournit pas de diagnostic.**
