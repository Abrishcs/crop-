# Team Qiyas — Ethiopian Smallholder Crop-Yield Challenge

## Team

- (names here)

## Summary

Predicts Ethiopian smallholder crop yield (tons/ha) from plot-level survey data,
regional weather, and crop prices. Notebook 01 cleans and joins the three raw sources
(fixing inconsistent region/crop labels, two different missing-value sentinels, and a
price unit mix-up) into master feature tables using a 4-month growing-season weather
window. Notebooks 02-03 analyze the cleaned data and produce 12 figures. Notebook 04
compares several model families, tunes and evaluates the best one, and fills in the
leaderboard submission. `app/app.py` is a Streamlit demo that predicts yield and
revenue for a single plot, looking up weather and price automatically.

## Setup (Anaconda / Jupyter)

```bash
conda create -n qiyas-hackathon python=3.10 -y
conda activate qiyas-hackathon
pip install -r requirements.txt
```

Launch notebooks with JupyterLab (via Anaconda Navigator, or from this environment):

```bash
jupyter lab
```

## Run order

Open and run each notebook top to bottom, in this order, from JupyterLab:

1. `notebooks/01_cleaning_and_integration.ipynb`
2. `notebooks/02_analysis_report.ipynb`
3. `notebooks/03_visualizations.ipynb`
4. `notebooks/04_modeling_and_evaluation.ipynb`
5. `streamlit run app/app.py` (demo — run from an Anaconda Prompt / terminal with the
   `qiyas-hackathon` environment active, not from inside Jupyter)

## Testing checkpoints (one per build phase)

Run these in order; each should pass before moving to the next.

1. **Phase 1 — `01_cleaning_and_integration.ipynb`**: run top to bottom. Check: no
   exceptions, the A7 integrity-checks cell prints all `[PASS]`, and
   `data/processed/master_train.csv`, `master_test.csv`, `data_dictionary_master.csv`,
   `cleaning_stats.json` exist, plus `app/assets/weather_clean.csv` and `price_clean.csv`.
2. **Phase 2 — `02_analysis_report.ipynb` then `03_visualizations.ipynb`**: run top to
   bottom. Check: `figures/` has 9 PNGs (`fig01_...png` through
   `fig09_revenue_by_crop_region.png`) and `figures/figure_captions.md` has those 9 entries
   filled in. Figures 10-12 (model comparison, residuals, feature importance) are generated
   by notebook 04 instead, since they need the trained model.
3. **Phase 3 — `04_modeling_and_evaluation.ipynb`**: run top to bottom. Check:
   `models/final_model.joblib` exists, `submission/team_qiyas_submission.csv` has all
   3,750 rows filled with numeric predictions, `figures/` now has all 12 PNGs, and
   `figures/figure_captions.md` has all 12 entries filled in.
4. **Phase 4 — demo**: `streamlit run app/app.py`, enter a plot's details, confirm you get
   a predicted yield and a revenue estimate with no crash.

## Exporting the written reports (A, B, D)

Each report deliverable is just the matching notebook exported, run after that notebook
executes cleanly:

```bash
jupyter nbconvert --to html notebooks/01_cleaning_and_integration.ipynb --output-dir reports --output A_cleaning_and_integration
jupyter nbconvert --to html notebooks/02_analysis_report.ipynb --output-dir reports --output B_analysis_report
jupyter nbconvert --to html notebooks/04_modeling_and_evaluation.ipynb --output-dir reports --output D_model_evaluation
```

## Where each deliverable lives

| Deliverable | Location |
|---|---|
| Prediction score | `submission/team_qiyas_submission.csv` |
| A — Pipeline | `notebooks/01_...`, `reports/A_...`, `data/processed/` |
| B — Analysis | `reports/B_analysis_report.*`, `notebooks/02_...` |
| C — Visualizations | `figures/`, `notebooks/03_...` |
| D — Modeling | `reports/D_...`, `notebooks/04_...`, `models/` |
| E — Demo | `app/`, URL below |
| F — Slides | `presentation/` |
| G — Structure | project root |

## Final validation score

(fill in from notebook 04's D9 cell: RMSE, MAE, and percent-of-mean-yield, once run)

## Demo link

(fill in, or note: run locally with `streamlit run app/app.py`)
