# Mozilla Perfherder Regression Detection - Final Report

## 1. Project Overview
The objective of this project was to build a machine learning model to automatically classify performance alerts from Mozilla's Perfherder system as either "Regression" or "Noise/Improvement". The goal is to reduce the manual triage burden on developers by accurately identifying real regressions.

## 2. Data Exploration & Preprocessing
**Notebooks:** `01_explore_data.ipynb`, `02_preprocess_features.ipynb`

*   **Dataset:** The dataset consists of performance alerts with various features including alert magnitude, t-statistic, platform details, and test suite information.
*   **Target:** `single_alert_is_regression` (Binary: 1 for Regression, 0 for Noise/Improvement).
*   **Key Findings:**
    *   The classes are relatively balanced but slightly skewed towards non-regressions.
    *   `single_alert_amount_abs` (absolute magnitude of change) showed a strong correlation with the target.
    *   Missing values were handled (e.g., filling with -1 or 0 where appropriate).
    *   Categorical features like `repository`, `platform`, and `suite` were encoded.

## 3. Model Development
**Notebook:** `03_train_models.ipynb`

*   **Models Evaluated:** Logistic Regression, Random Forest, XGBoost.
*   **Performance:**
    *   **XGBoost** emerged as the top performer with near-perfect accuracy (>99%) and ROC-AUC (~1.0).
    *   Random Forest also performed very well.
    *   Logistic Regression was significantly weaker, indicating non-linear relationships in the data.
*   **Selected Model:** XGBoost (optimized for GPU).

## 4. Feature Importance Analysis
**Notebook:** `04_feature_importance_analysis.ipynb`

*   **Methods:** XGBoost Gain, Permutation Importance, SHAP values.
*   **Top Features:**
    1.  `single_alert_amount_abs` (Absolute magnitude of the alert) - **Dominant Feature**
    2.  `single_alert_amount_pct` (Percentage change)
    3.  `single_alert_t_value` (Statistical significance)
*   **Insight:** The model primarily looks at *how big* the change is. Large changes are almost always regressions in this dataset context.

## 5. Experiments

### E2: Cross-Repository Generalization
**Notebook:** `05_cross_repository_experiment.ipynb`

*   **Setup:** Train on `autoland` repo, Test on `mozilla-central` and `mozilla-beta`.
*   **Results:**
    *   **In-Distribution (Autoland):** Accuracy 0.9963, F1 0.9959.
    *   **Cross-Repository:** Accuracy 0.9958, F1 0.9935.
*   **Conclusion:** The model generalizes exceptionally well. It learns universal patterns of regression (likely magnitude-based) rather than overfitting to specific repository artifacts.

### E3: Feature Ablation Study
**Notebook:** `06_feature_ablation_experiment.ipynb`

*   **Setup:** Train on subsets of features: Magnitude only, Context only, Workflow only, etc.
*   **Results:**
    *   **Magnitude Only:** Accuracy ~99.1%
    *   **Context Only:** Accuracy ~57% (Near random guess)
    *   **Magnitude + Context:** Accuracy ~99.8%
*   **Conclusion:** Magnitude features provide the vast majority of the predictive power. Context features (platform, suite) add a small marginal gain but are insufficient on their own.

## 6. Conclusion & Phase 2 Motivation
We have successfully built a highly accurate regression detection model. The key takeaways are:
1.  **High Performance:** The XGBoost model achieves >99% accuracy and F1-score.
2.  **Robustness:** It works well across different repositories.
3.  **Simplicity:** The decision logic is dominated by the magnitude of the performance change (`amount_abs`), making the model interpretable and reliable.

**Motivation for Phase 2:**
While the Phase 1 model is highly effective using metadata alone, its heavy reliance on magnitude suggests it may struggle with subtle regressions or complex "Downstream" patterns that don't manifest as simple magnitude spikes. Phase 2 will introduce time-series features to capture these temporal dependencies and subtle shifts, potentially improving detection for the edge cases where metadata falls short.

## 7. Artifacts
*   **Final Model:** `results/models/best_xgb_gpu_model.json`
*   **Preprocessed Data:** `results/data/preprocessed_alerts.csv`
*   **Plots:** See `results/figures/*.png` for feature importance and experiment visualizations.
