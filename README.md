# Supervised Machine Learning — Practice Projects

This repository documents my hands-on learning journey in supervised machine learning, working through classic classification problems end-to-end: from raw data to model comparison and interpretation.

## Projects

### 1. Heart Disease Prediction
Binary classification predicting the presence of heart disease from clinical features (UCI Heart Disease dataset, via `ucimlrepo`).

- **Preprocessing:** missing value imputation (`SimpleImputer`), feature scaling (`StandardScaler`)
- **Target handling:** binarized the original multiclass target (severity 0–4) into disease / no disease
- **Models compared:** Logistic Regression, Random Forest, SVM
- **Evaluation:** Accuracy, F1-score, ROC-AUC, Confusion Matrix, 5-fold Cross-Validation, ROC curve comparison

### 2. Titanic Survival Prediction
Binary classification predicting passenger survival (Titanic dataset, via `seaborn`).

- **Preprocessing:** separate imputation strategies for numeric (mean) vs categorical (most frequent) features, one-hot encoding
- **Data leakage check:** identified and removed the `alive` column (a direct duplicate of the target) and redundant encoded columns (`class`, `who`, `adult_male`, `embark_town`, `deck`)
- **Models compared:** Logistic Regression, Random Forest, SVM
- **Evaluation:** same metrics as above, plus Random Forest feature importance analysis
- **Key finding:** `fare` and `pclass` are correlated (r = -0.55), which explains why `fare` ranks above `pclass` in feature importance despite both representing socio-economic status

### 3. Wine Quality Classification
Multiclass classification (6 classes, quality scores 3–8) predicting wine quality from physicochemical features (UCI Wine Quality dataset).

- **Preprocessing:** no missing values, all-numeric features — feature scaling only, stratified train/test split (`stratify=y`) to preserve class proportions across a severely imbalanced target
- **Multiclass-adapted metrics:** `f1_score(average='weighted')`, `roc_auc_score(multi_class='ovr')` instead of the binary defaults
- **Models compared:** Logistic Regression, Random Forest, SVM — Random Forest won across all metrics
- **Class imbalance investigation:** 82% of samples fall into just 2 of 6 classes (quality 5–6), with the rarest class (quality 3) having only 10 total samples. Systematically tested and compared three mitigation strategies:
  1. **Hyperparameter tuning** (`GridSearchCV`) — minimal impact (F1 ~0.660 → ~0.660)
  2. **`class_weight='balanced'`** — no meaningful improvement on minority classes
  3. **SMOTE oversampling** — best result (F1 = 0.671), improved detection on moderately rare classes (4, 7), but still failed on the most extreme minority class (quality 3, only 8 real training samples)
- **Key finding:** class regrouping (3/6 categories) was tested and rejected — grouping classes 5+6 into "medium" still left 82.5% of data in one bucket, proving the regrouping only relabels the imbalance rather than resolving it. No resampling or weighting technique can compensate for a genuine lack of real data in the rarest class.

## Methodology (applied to all projects)

1. Exploratory Data Analysis (EDA)
2. Data cleaning & missing value imputation
3. Categorical encoding (where applicable)
4. Feature scaling
5. Train/test split (stratified for imbalanced targets)
6. Model training (Logistic Regression, Random Forest, SVM)
7. Evaluation (Accuracy, F1, ROC-AUC, Confusion Matrix) — adapted for binary or multiclass
8. 5-fold Cross-Validation for robust comparison
9. ROC curve visualization
10. Feature importance / model interpretation
11. Hyperparameter tuning (`GridSearchCV`)
12. Class imbalance handling (`class_weight`, SMOTE) — where relevant

## Key takeaways

- A single train/test split can give overly optimistic (or just noisy) results — cross-validation gives a more honest performance estimate, especially on smaller datasets.
- Feature importance can be misleading when features are correlated (collinearity) — always sanity-check surprising rankings against domain knowledge.
- False negatives matter differently depending on context: a missed diagnosis in a medical dataset is actionable and consequential, while a misclassification on a historical dataset like Titanic is not — the reasoning still transfers to real deployment scenarios.
- Overall accuracy can mask complete failure on minority classes — always inspect the confusion matrix per class, not just the aggregate score.
- Hyperparameter tuning, class weighting, and oversampling (SMOTE) each address imbalance differently, and none can substitute for a genuine lack of real data in an extremely rare class.

## Tools

Python · pandas · scikit-learn · matplotlib · seaborn

## Author

Learning project — feedback and suggestions welcome!
