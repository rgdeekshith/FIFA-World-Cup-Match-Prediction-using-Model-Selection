# FIFA World Cup Match Outcome Prediction (1930–2022)

This project uses machine learning to predict the outcome of FIFA World Cup matches (Home win `H`, Draw `D`, Away win `A`) using historical match data from 1930 to 2022.

The main goal is to compare several classification models and understand which one best balances performance across all three classes, with special attention to the harder classes **Draw (D)** and **Away win (A)**.

---

## Dataset

- Source: FIFA World Cup matches from 1930 to 2022 (single CSV file).
- Target variable: match result with three classes:
  - `H` – Home team win  
  - `D` – Draw  
  - `A` – Away team win
- Example features:
  - Year of the World Cup  
  - Home team and away team  
  - Host country  
  - Stage (group stage vs knockout)  
  - Other match-related categorical features

All categorical variables are encoded for use in scikit‑learn models.

---

## Models

The following models were trained and evaluated on the same train/test split:

1. **Logistic Regression (Multinomial)**
2. **Random Forest Classifier** (baseline tree ensemble, using bagging)
3. **GradientBoostingClassifier** (scikit‑learn implementation of gradient boosting)
4. **XGBoost Classifier** (gradient boosting trees with external `xgboost` library)

For each model, the project reports:

- Accuracy  
- Macro F1‑score  
- Per‑class precision, recall, and F1‑score (for `H`, `D`, `A`)  
- Confusion matrix on the test set

---

## Key Results

A summary table of the main metrics is created in the notebook (see `df_metrics_round`):

- **Random Forest** achieves the best macro F1‑score and the most balanced F1 across all three classes (`H`, `D`, `A`).
- **Gradient Boosting (sklearn)** and **XGBoost** obtain similar or slightly higher accuracy for the majority class `H`, but their F1‑scores for `D` (draw) and `A` (away win) are lower than Random Forest.
- **Logistic Regression** is the weakest model overall, with the lowest macro F1‑score and difficulty capturing draws and away wins.

Because the problem is imbalanced and draws/away wins are important, the project selects **Random Forest with the `Year` feature included** as the best model.

---

## Model Interpretation

The notebook also explores model interpretability, focusing on the Random Forest:

- **Feature Importance (Gini importance)** to see which features the forest uses most.
- **Permutation Importance** to more reliably measure the impact of individual features (especially `Year`).
- An experiment removing the `Year` feature:
  - Removing `Year` slightly reduces macro F1, confirming that it carries some useful signal (likely capturing era effects), even if it is not a direct causal factor.

Basic hyperparameter changes (e.g., `min_samples_leaf`) are tested to see their impact on performance and overfitting.

The below plans were devised for Random Forest(RF) Model tuning:

- Plan 1 – Retrain RF without Year and save results
- Plan 2 – Permutation importance for Year on original model, we keep the original RF (with Year), and only measure how important Year is.
- Plan 3 – If “no Year” model looks better, inspect and tweak it, assuming Plan 1 showed RF without Year is better and you decide to keep rf_no_year:
         - 3a. Inspect trees and depth
         - 3b. Class balance of predictions (H/D/A)
         - 3c. Try a small hyperparameter tweak – min_samples_leaf
         
---

## Project Structure

- `notebooks/` – Jupyter notebook(s) with data exploration, model training, and evaluation.
- `data/` – FIFA World Cup dataset (`fifa_worldcup_1930_2022.csv`).
- `models/` – Saved models (e.g., `rf_worldcup.joblib`) created with `joblib.dump`.
- `README.md` – This file.


---

## How to Run

1. Clone the repository and navigate into it.
2. Install dependencies (example using `pip`):

   ```bash
   pip install -r requirements.txt
   ```

   Typical libraries:
   - `pandas`, `numpy`
   - `scikit-learn`
   - `xgboost`
   - `matplotlib`, `seaborn` (for plots)

3. Open the main notebook:

   ```bash
   jupyter notebook
   ```

   or use your preferred environment (VS Code, Colab, etc.).

4. Run the notebook cells in order to:
   - Load and preprocess the data  
   - Train all models  
   - Generate classification reports, confusion matrices, and comparison table  
   - Save the best model with `joblib` (Random Forest)

---

## Future Work

I may try the following extensions if possible in future:

- Try additional gradient boosting libraries such as **LightGBM** and **CatBoost**.
- Add more match-level features (e.g., betting odds, team rankings, player strength).
- Perform more systematic hyperparameter tuning (e.g., `RandomizedSearchCV` or `GridSearchCV`).
- Calibrate predicted probabilities and evaluate with proper scoring rules.
- Fixing the Imbalance: Learn techniques (like SMOTE or class weights) to mathematically force your models to care more about Draws.
- Tuning the Engine: Explore how adjusting the hyperparameters of Gradient Boosting can make it less "greedy" and more balanced.
- Exploring Feature Engineering: Discuss how creating new data points from your existing columns might give the models better clues to predict a Draw.

This project demonstrates a complete end‑to‑end comparison of several classical machine learning models on multi‑class football match outcome prediction.
