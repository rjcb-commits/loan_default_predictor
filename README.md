# Loan Default Predictor

A LightGBM model that estimates default probability for Lending Club personal loans, served as an interactive Streamlit app. Portfolio demonstration using historical public data; not validated for lending decisions.

**Live demo:** https://loandefaultpredictor-rayjackcb.streamlit.app

## What it does

Given a set of borrower features (loan amount, FICO score, debt-to-income, employment length, etc.), the model returns the probability that a loan will be charged off rather than fully repaid. Trained on the public Lending Club historical loan dataset.

## Stack

- Python, LightGBM, scikit-learn, pandas
- Streamlit for the interactive UI
- matplotlib for diagnostics

## Local setup

```bash
python -m venv .venv
.venv/Scripts/activate     # PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Download the Lending Club loan dataset from Kaggle and place `loan.csv` (or whichever main file you grabbed) in `data/`. Anything in `data/` is gitignored.

## Train the model

```bash
python train.py
```

Reads from `data/loan.csv`, writes `model.pkl` and metadata into `artifacts/`. Runs on a 10% stratified subsample; observed runtime is a couple of minutes on a modern laptop.

## Run the app

```bash
streamlit run app.py
```

Opens at http://localhost:8501.

## Feature engineering

- Filtered to settled loans (`Fully Paid` or `Charged Off`); excluded in-flight loans
- Parsed text fields: `term`, `int_rate`, `revol_util`, `emp_length`
- Dropped post-hoc fields (`total_pymnt`, `recoveries`, `last_pymnt_*`, etc.) that leak the target
- Twenty originate-time features kept

## Modeling notes

- LightGBM with 500 trees, early stopping on validation AUC
- Categorical features handled natively by LightGBM, no manual encoding
- 80/20 train/test split, stratified by default outcome
- 10% stratified subsample of the full 2.2M-row dataset keeps training fast; the deployed model reaches ~0.71 test AUC on 26,907 held-out loans

## Project layout

```
data/                 (gitignored, place loan.csv here)
artifacts/            model.pkl, feature_meta.json, metrics.json, plots/
notebooks/            optional EDA
train.py              training script
app.py                Streamlit app
```

## License

All rights reserved.
