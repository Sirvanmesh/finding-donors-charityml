# Finding Donors for CharityML

This project predicts whether a person earns more than $50,000 a year from 1994 U.S. Census data, so that CharityML can focus on the people most likely to donate.
It compares three supervised learning models (Gradient Boosting, Random Forest and Logistic Regression) and tunes the best one with `GridSearchCV`.

## Final results (test set)

Chosen model: `GradientBoostingClassifier` (tuned: `learning_rate=0.1`, `n_estimators=400`).

| Model | Accuracy | F-score (beta = 0.5) |
| :-- | :--: | :--: |
| Naive predictor (everyone > $50K) | 0.2478 | 0.2917 |
| Gradient Boosting, unoptimized | 0.8630 | 0.7395 |
| Gradient Boosting, optimized | 0.8708 | 0.7531 |
| Optimized model, top 5 features only | 0.8588 | 0.7249 |

## Setup

Python 3.10:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install numpy==1.26.3 pandas==2.1.4 matplotlib==3.8.2 seaborn==0.13.2 scikit-learn==1.4.0 scipy==1.12.0 jupyter ipykernel nbconvert
```

The dataset `census.csv` is not included in this repo. Download it from the `starter` folder of
https://github.com/udacity/cd0025-supervised-learning and put it in `finding_donors/`.

## How to run

```bash
jupyter notebook finding_donors/finding_donors.ipynb
```

## Main outputs

- `finding_donors/finding_donors.ipynb`: the completed notebook with all code, outputs and answers
- `finding_donors/report.html`: HTML export of the notebook with all plots and answers
