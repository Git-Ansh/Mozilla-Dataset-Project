# Mozilla Dataset Project

This repository contains the Mozilla Dataset Project.

## How to Run the Experiments

All features have been implemented. Follow these steps to reproduce the results.

### Step 1: Open Jupyter

Navigate to the project directory and start Jupyter:

```powershell
cd "c:\Users\anshj\OneDrive\Documents\Anitgravity\Mozilla Dataset Project"
jupyter notebook
```

### Step 2: Run These Notebooks

Run them in order:

1. **04_feature_importance_analysis.ipynb**
   - Contains SHAP analysis and permutation importance
   - Generates feature importance visualizations

2. **05_cross_repository_experiment.ipynb**
   - Tests cross-repository generalization (Experiment E2)

3. **06_feature_ablation_experiment.ipynb**
   - Feature ablation study (Experiment E3)
   - Tests 4 feature combinations as required

### Step 3: Check Results

After running, verify these files are created in `results/`:

**From Notebook 04:**
- `shap_importance.png`
- `shap_summary_detailed.png`
- `permutation_importance.png`
- `permutation_importance.csv`

**From Notebook 05:**
- `repository_distribution.png`
- `cross_repository_comparison.png`
- `cross_repository_results.csv`
- `per_repository_results.csv`

**From Notebook 06:**
- `ablation_study_results.csv`
- `ablation_study_report.txt`
- `ablation_performance_comparison.png`
- `ablation_f1_rocauc_comparison.png`

## Implemented Features

*   **SHAP Analysis**: Included in notebook 04
*   **Permutation Importance**: Included in notebook 04
*   **Experiment E2 (Cross-Repository)**: Complete in notebook 05
*   **Experiment E3 (Feature Ablation)**: Complete in notebook 06 with 4 combinations

## The 4 Feature Combinations (Experiment E3)

Notebook 06 tests these combinations as specified in requirements:

1.  **Only Magnitude**: Statistical signals only (amount, t-value, etc.)
2.  **Only Context**: Platform metadata only (repository, framework, platform, suite)
3.  **Magnitude + Context**: Combined magnitude and context
4.  **All Features**: Everything including workflow hints (status, manually_created, etc.)
