# House Prices: Advanced Regression Techniques

**Author:** Tatev Poghosyan
**Course:** Picsart AI/ML Course: Classical Machine Learning, final project
**Competition:** [Kaggle: House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

The task is to predict the sale price of 1459 houses in Ames, Iowa, from 79 features. The metric is RMSE between the log of the predicted price and the log of the real price.

## Project structure

```
house-prices-tatev-surname/
├── data/
│   ├── raw/                  4 original Kaggle files (never edited)
│   └── processed/            train_processed.csv, test_processed.csv (made by 01_EDA)
├── notebooks/
│   ├── 01_EDA.ipynb          Part B: explore, clean, new features, encode
│   └── 02_Models.ipynb       Part C: models, tuning, comparison, error analysis, submission
├── results/
│   ├── figures/              saved plots (.png)
│   └── model_comparison.csv  all models and their CV scores
├── submissions/
│   ├── submission_average.csv
│   └── kaggle_score.png
└── README.md
```

## How to run

1. Python 3.10+ and the libraries:
   ```bash
   pip install numpy pandas scipy matplotlib seaborn scikit-learn xgboost lightgbm jupyter
   ```
2. Put the 4 Kaggle files in `data/raw/` (they are already there).
3. Open the `notebooks/` folder and run **in order**:
   - `01_EDA.ipynb` creates `data/processed/*.csv` and figures
   - `02_Models.ipynb` creates `results/model_comparison.csv`, figures and `submissions/submission_average.csv`

All randomness uses `random_state=42`. The full run takes about 3 minutes.

## What I did

**01_EDA**
- Target: `SalePrice` is right-skewed (skew 1.88), so models learn `log1p(SalePrice)` (skew 0.12).
- Missing values: 34 columns. NA usually means "none" (no pool, no garage…) → `"None"` / `0`; `LotFrontage` → neighborhood median; `GarageYrBlt` → `YearBuilt`; a few gaps → most frequent value. Fill values come from train only.
- Outliers: removed 2 train houses with `GrLivArea > 4000` and price < 300k (partial sales, recommended by the dataset author).
- New features: `TotalSF`, `TotalBath`, `HouseAge`, `RemodAge`, `TotalPorchSF`, `QualSF`. They correlate with price better than the original columns, and the Ridge CV RMSE dropped from 0.1346 to 0.1298.
- Encoding: quality scales → numbers (Po=1 … Ex=5), log1p of 22 skewed columns, one-hot encoding on train+test together → 262 features.

**02_Models**
- 5-fold CV (shuffle, `random_state=42`), RMSE on log price, baseline = predict the mean.
- 8 models: Linear Regression, Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting, XGBoost, LightGBM.
- `GridSearchCV` tuning for Ridge, Lasso and Gradient Boosting.
- Winner: the **average of Lasso + Gradient Boosting + XGBoost + LightGBM** (`VotingRegressor`).
- Error analysis on out-of-fold predictions: the largest errors are abnormal or family sales and cheap old houses.

## Results

| Model | 5-fold CV RMSE (log) |
|---|---|
| Baseline (mean) | 0.3993 |
| Random Forest | 0.1285 |
| Linear Regression | 0.1266 |
| LightGBM | 0.1177 |
| Ridge tuned (alpha=20) | 0.1135 |
| XGBoost | 0.1123 |
| ElasticNet | 0.1119 |
| Lasso tuned (alpha=0.0005) | 0.1116 |
| Gradient Boosting tuned | 0.1110 |
| **Average (Lasso+GB+XGB+LGBM)** | **0.1085** |

Full table: `results/model_comparison.csv`.

**Kaggle Public Score:** 0.12543 (file `submissions/submission_average.csv`, screenshot `submissions/kaggle_score.png`)

## Sources

- Kaggle competition page and `data_description.txt`
- Dean De Cock, *Ames, Iowa: Alternative to the Boston Housing Data* (Journal of Statistics Education, 2011): outlier recommendation
- scikit-learn, XGBoost and LightGBM documentation
- Course materials (Picsart AI/ML course)
