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

## Project Structure

*   `notebooks/`:
    *   `01_explore_data.ipynb`: Initial EDA and understanding the dataset.
    *   `02_preprocess_features.ipynb`: Cleaning and feature engineering.
    *   `03_train_models.ipynb`: Training and tuning the XGBoost model (E1).
    *   `04_feature_importance_analysis.ipynb`: Deep dive into what the model learned (SHAP/Permutation).
    *   `05_cross_repository_experiment.ipynb`: Generalization tests (E2).
    *   `06_feature_ablation_experiment.ipynb`: Feature group analysis (E3).
*   `src/`:
    *   `utils.py`: Shared utility functions for data loading and model initialization.
*   `results/`:
    *   `data/`: Processed datasets and markdown tables.
    *   `models/`: Saved model artifacts.
    *   `figures/`: Visualizations for reports.
    *   `reports/`: Final written analysis.

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
cd "c:\Users\anshj\OneDrive\Documents\Anitgravity\Mozilla Dataset Project"
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
