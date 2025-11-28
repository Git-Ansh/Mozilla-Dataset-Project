# Mozilla Perfherder Regression Detection

## Project Overview
This project aims to automate the triage of performance alerts in Mozilla's **Perfherder** system. Developers currently face a high volume of performance alerts, many of which are noise or improvements rather than actual regressions. 

The goal of this machine learning system is to classify these alerts as either **"Regression"** or **"Noise/Improvement"** based on alert metadata, thereby reducing the manual burden on developers and speeding up the identification of critical performance issues.

## Key Concepts & Insights

### 1. The Core Problem: Signal vs. Noise
Performance data is noisy. A test might run slightly slower due to infrastructure variance rather than a code change. The challenge is to identify statistically significant regressions that require human attention.

### 2. The Solution: XGBoost Classifier
We implemented a gradient boosting model (**XGBoost**) optimized for GPU. This model analyzes features such as:
*   **Magnitude**: How large is the performance change? (`amount_abs`, `amount_pct`)
*   **Significance**: Is it statistically significant? (`t_value`)
*   **Context**: Which platform, suite, or repository did this occur on?

### 3. Primary Insight: "Magnitude is King"
Our analysis (SHAP, Permutation Importance) revealed that the **absolute magnitude of the change** is the single most predictive feature. 
*   Large spikes are almost always regressions.
*   Contextual features (like which OS the test ran on) provide marginal gains but are insufficient on their own.

---

## Experimental Results

We conducted three key experiments to validate the model's robustness and understand its decision-making process.

### Experiment E1: Baseline Model Selection
*   **Objective**: Compare different algorithms to find the best baseline.
*   **Result**: **XGBoost** outperformed Logistic Regression and Random Forest, achieving **>99% Accuracy and F1-Score**.
*   **Notebook**: `03_train_models.ipynb`

### Experiment E2: Cross-Repository Generalization
*   **Objective**: Determine if a model trained on one repository (`autoland`) can generalize to others (`mozilla-central`, `mozilla-beta`).
*   **Result**: **Yes.** The model achieved **99.58% Accuracy** on unseen repositories.
*   **Takeaway**: The patterns of regression (e.g., "big jump = bad") are universal across Mozilla's development branches.
*   **Notebook**: `05_cross_repository_experiment.ipynb`

### Experiment E3: Feature Ablation Study
*   **Objective**: Quantify the contribution of different feature groups.
*   **Configurations Tested**:
    1.  **Magnitude Only**: Statistical signals (amount, t-value).
    2.  **Context Only**: Metadata (platform, suite, repo).
    3.  **Magnitude + Context**: Combined.
    4.  **All Features**: Including workflow states.
*   **Result**: 
    *   *Magnitude Only* achieved **~99.1%** accuracy.
    *   *Context Only* failed with **~57%** accuracy (near random guess).
*   **Takeaway**: Metadata alone is not enough; the statistical signal is the primary driver of classification.
*   **Notebook**: `06_feature_ablation_experiment.ipynb`

---

## Detailed Notebook Guide

Here is a deep dive into the logic and content of each notebook in the pipeline.

### 1. Data Exploration (`01_explore_data.ipynb`)
*   **Purpose**: To understand the raw dataset structure, quality, and class distribution.
*   **Key Steps**:
    *   Loads the raw `alerts_data.csv`.
    *   Visualizes the target variable balance (Regressions vs. Non-Regressions).
    *   Analyzes missing values and data types.
    *   Plots distributions of key numerical features like `amount_abs` and `t_value`.
*   **Outcome**: Confirmed that the dataset is slightly imbalanced and identified `amount_abs` as a potentially strong predictor.

### 2. Feature Engineering (`02_preprocess_features.ipynb`)
*   **Purpose**: To transform raw data into a machine-learning-ready format.
*   **Key Steps**:
    *   **Imputation**: Fills missing values (e.g., -1 for missing numericals, 'Unknown' for categoricals).
    *   **Encoding**: Converts categorical features (Repository, Platform, Suite) into numerical representations.
    *   **Cleaning**: Removes irrelevant columns (IDs, timestamps) that could cause overfitting or leakage.
*   **Outcome**: Produces `results/data/preprocessed_alerts.csv`, a clean feature table ready for training.

### 3. Model Training & Tuning (`03_train_models.ipynb`)
*   **Purpose**: To establish a baseline and train the final high-performance classifier (Experiment E1).
*   **Key Steps**:
    *   **Baseline Comparison**: Trains Logistic Regression, Random Forest, and XGBoost (default settings).
    *   **Optimization**: Uses GridSearchCV to tune XGBoost hyperparameters (learning rate, max depth, etc.) with GPU acceleration.
    *   **Evaluation**: Calculates Accuracy, Precision, Recall, F1-Score, and ROC-AUC on a held-out test set.
*   **Outcome**: Saves the best model (`best_xgb_gpu_model.json`) which achieves >99% accuracy.

### 4. Interpretability Analysis (`04_feature_importance_analysis.ipynb`)
*   **Purpose**: To open the "black box" and understand *why* the model makes its decisions.
*   **Key Steps**:
    *   **Native Importance**: Plots XGBoost's internal Gain metric.
    *   **Permutation Importance**: Shuffles features one by one to measure their impact on model accuracy.
    *   **SHAP Analysis**: Uses Game Theory to calculate the exact contribution of each feature to individual predictions.
*   **Outcome**: Proves that `single_alert_amount_abs` (magnitude) is the dominant driver of the model's predictions.

### 5. Cross-Repository Generalization (`05_cross_repository_experiment.ipynb`)
*   **Purpose**: To verify if the model can detect regressions on repositories it has never seen before (Experiment E2).
*   **Key Steps**:
    *   **Split by Repo**: Uses `autoland` data for training.
    *   **Out-of-Sample Test**: Uses `mozilla-central` and `mozilla-beta` data exclusively for testing.
    *   **Comparison**: Compares performance metrics between the training repo and the unseen test repos.
*   **Outcome**: Demonstrates robust generalization, confirming the model learns universal performance regression patterns.

### 6. Feature Ablation Study (`06_feature_ablation_experiment.ipynb`)
*   **Purpose**: To determine which groups of features are actually necessary (Experiment E3).
*   **Key Steps**:
    *   **Grouping**: Categorizes features into "Magnitude" (statistical), "Context" (metadata), and "Workflow" (process state).
    *   **Combinatorial Training**: Trains and evaluates models on every subset of these groups (e.g., Magnitude Only, Context Only).
*   **Outcome**: Reveals that Context features alone are non-predictive (~57% accuracy), while Magnitude features alone achieve near-perfect performance (~99%), validating the "Magnitude is King" hypothesis.

---

## How to Run the Project

Follow these steps to reproduce the experiments and results.

### Prerequisites
*   Python 3.8+
*   Jupyter Notebook
*   Libraries: `pandas`, `numpy`, `xgboost`, `scikit-learn`, `matplotlib`, `seaborn`, `shap`

### Step 1: Setup & Launch
Navigate to the project directory and start Jupyter:

```powershell
cd "...\Mozilla Dataset Project"
jupyter notebook
```

### Step 2: Execution Order
For a full reproduction, run the notebooks in the following order:

1.  **Data Prep**: Run `01_explore_data.ipynb` and `02_preprocess_features.ipynb` to generate the clean dataset.
2.  **Model Training**: Run `03_train_models.ipynb` to train the core XGBoost model.
3.  **Analysis**: Run `04_feature_importance_analysis.ipynb` to generate SHAP plots.
4.  **Experiments**: 
    *   Run `05_cross_repository_experiment.ipynb` for E2.
    *   Run `06_feature_ablation_experiment.ipynb` for E3.

### Step 3: View Results
After execution, check the `results/` folder:
*   **Figures**: `results/figures/` contains all plots (SHAP, Confusion Matrices, etc.).
*   **Metrics**: `results/metrics/` contains raw CSVs of model performance.
*   **Reports**: `results/reports/final_report.md` contains the detailed scientific conclusion.
