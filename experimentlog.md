# Experiment Log: Baseline Model Establishment
**Date:** April 22, 2026

## 1. Experiment Objective
To establish a rigorous predictive benchmark using a simplified linear pipeline. The goal is to determine the predictive power of real-time behavioral features (browsing activity) while removing seasonal temporal bias by excluding the `Month` variable.

## 2. Configuration & Methodology
* **Model:** Logistic Regression (L2 Penalty, max_iter=1000)
* **Data Split:** 64% Training / 16% Validation / 20% Validation (Stratified)
* **Features Included:** 10 Numerical (Scaled), 6 Categorical (One-Hot Encoded)
* **Key Exclusion:** `Month` (Removed to isolate behavioral intent from seasonal trends)

## 3. Results (Quantitative)
The model achieved high overall accuracy but revealed a significant struggle with minority class detection.

| Metric | Result |
| :--- | :--- |
| **ROC AUC Score** | **0.8793** |

## 4. Next Steps for AutoResearch Agent
The baseline is now locked at **0.8793 AUC**. The next research cycle will focus on:
1.  **Non-linear Modeling:** Implementing Gradient Boosted Trees (XGBoost/LightGBM) to capture interactions between `ExitRates` and `ProductRelated_Duration`.
2.  **Synthetic Balancing:** Testing SMOTE or class-weight adjustments to address the 34% Recall floor.
3.  **Feature Interaction:** Investigating if specific combinations of `TrafficType` and `PageValues` yield higher signal.



---

# Experiment Log: AutoResearch Iterations
**Date:** April 28, 2026

## 1. Experiment Objective
Run three autoresearch iterations on `model.py` to improve validation ROC AUC while preserving a valid sklearn-compatible pipeline and keeping runtime practical on CPU.

## 2. Configuration & Methodology
* **Scope Constraint:** Modified `model.py` only
* **Evaluation Data/Metric:** Existing frozen split and metrics from `prepare.py` (Validation ROC AUC, Validation F1)
* **Approach:** Tested 3 candidate model configurations and retained the highest-AUC result

## 3. Results (Quantitative)

| Iteration | Model Summary | Validation AUC | Validation F1 |
| :--- | :--- | :---: | :---: |
| **Iter 1** | Logistic Regression (`class_weight='balanced'`) | **0.891852** | 0.640327 |
| **Iter 2** | Random Forest (`n_estimators=350`, `min_samples_leaf=2`, `class_weight='balanced_subsample'`) | **0.915401** | **0.655008** |
| **Iter 3** | HistGradientBoosting + dense one-hot preprocessing | 0.915381 | 0.635548 |

## 4. Outcome
Best configuration from today: **Iteration 2 (Random Forest)** with **AUC = 0.915401**.

This replaces the prior `model.py` configuration because it achieved the highest validation AUC across the three iterations.

## 5. Next Steps
1. Tune Random Forest depth/leaf constraints and `max_features` around the winning setup.
2. Try probability calibration (`CalibratedClassifierCV`) to potentially improve threshold-sensitive performance.
3. Compare with a tuned gradient-boosted tree under the same preprocessing and logging protocol.

---

# Experiment Log: Ten Additional AutoResearch Iterations
**Date:** May 6, 2026

## 1. Experiment Objective
Run 10 additional iterations (runs #4-#13) to improve validation ROC AUC beyond the prior best model, while continuing automatic tracking in `results.tsv`, `metric_over_time.png`, and `errors.log`.

## 2. Configuration & Methodology
* **Run Range:** #4 through #13
* **Model Families Tested:** RandomForest, HistGradientBoosting, LogisticRegression
* **Selection Rule:** Keep any run that improves best-so-far validation AUC
* **Tracking:** Per-run metrics and metadata appended to `results.tsv`; AUC-over-time plot regenerated after each run

## 3. Results (Quantitative)

| Run | Iteration Label | Validation AUC | Validation F1 | Runtime (s) | Status |
| :--- | :--- | :---: | :---: | :---: | :--- |
| 4 | iter4_rf_500_leaf2 | 0.915823 | 0.657143 | 2.323 | keep |
| 5 | iter5_rf_500_leaf1 | 0.912484 | 0.600000 | 3.255 | discard |
| 6 | iter6_rf_700_leaf2 | 0.915588 | 0.657143 | 4.121 | discard |
| 7 | iter7_rf_entropy | **0.917215** | 0.653846 | 2.934 | keep |
| 8 | iter8_rf_log2 | 0.914768 | 0.660465 | 2.483 | discard |
| 9 | iter9_rf_depth20 | 0.915210 | 0.656347 | 2.613 | discard |
| 10 | iter10_rf_leaf3 | 0.915818 | **0.683432** | 2.533 | discard |
| 11 | iter11_hgb_400 | 0.916240 | 0.633274 | 5.837 | discard |
| 12 | iter12_hgb_500 | 0.915865 | 0.649030 | 4.784 | discard |
| 13 | iter13_logreg_c08 | 0.891786 | 0.642177 | 0.091 | discard |

## 4. Outcome
New best model after this cycle: **run #7 (`iter7_rf_entropy`)** with **AUC = 0.917215**.

`model.py` now reflects this best configuration.

## 5. Error Taxonomy Update
No new errors occurred in runs #4-#13. `errors.log` has been updated to explicitly record this.

## 6. Forward Note
Future iterations can change feature count by modifying feature selection lists in `model.py` (not only hyperparameters/model family).

---

# Experiment Log: Additional Ten Iterations (Feature-Count Variants Included)
**Date:** May 6, 2026

## 1. Experiment Objective
Run another 10 iterations (runs #14-#23) while continuing full artifact updates after each run, including experiments with changed feature counts.

## 2. Configuration & Methodology
* **Run Range:** #14 through #23
* **Main Families:** RandomForest (majority) + HistGradientBoosting (1 run)
* **New Scope:** Included feature-selection variants (63, 64, and 65 transformed features)
* **Tracking:** `results.tsv`, `metric_over_time.png`, and `errors.log` updated during the cycle

## 3. Results (Quantitative)

| Run | Iteration Label | Validation AUC | Validation F1 | Runtime (s) | Features | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| 14 | iter14_rf_entropy_700 | 0.917528 | 0.659236 | 3.828 | 65 | keep |
| 15 | iter15_rf_entropy_leaf3 | 0.916584 | 0.683358 | 3.686 | 65 | discard |
| 16 | iter16_rf_gini_leaf4 | 0.916761 | 0.692308 | 3.199 | 65 | discard |
| 17 | iter17_rf_entropy_half_features | **0.918170** | 0.643952 | 6.796 | 65 | keep |
| 18 | iter18_rf_entropy_no_bounce_exit | 0.910689 | 0.670623 | 2.368 | 63 | discard |
| 19 | iter19_rf_entropy_add_depth18 | 0.915863 | 0.670732 | 2.748 | 65 | discard |
| 20 | iter20_hgb_featdrop2 | 0.913286 | 0.606061 | 5.307 | 63 | discard |
| 21 | iter21_rf_entropy_no_specialday | 0.914544 | 0.646104 | 2.686 | 64 | discard |
| 22 | iter22_rf_entropy_leaf2_depth24 | 0.916490 | 0.660348 | 2.839 | 65 | discard |
| 23 | iter23_rf_entropy_ccp | 0.917219 | 0.653846 | 2.790 | 65 | discard |

## 4. Outcome
New best model after this cycle: **run #17 (`iter17_rf_entropy_half_features`)** with **AUC = 0.918170**.

`model.py` now reflects this best configuration.

## 5. Error Taxonomy Update
No new errors occurred in runs #14-#23. `errors.log` was updated to explicitly state this.

---

# Experiment Log: Single Random Forest Iteration
**Date:** May 7, 2026

## 1. Experiment Objective
Run one additional autoresearch iteration focused on a Random Forest variant and update all tracking artifacts.

## 2. Configuration & Methodology
* **Run Number:** #24
* **Model:** RandomForestClassifier (`n_estimators=700`, `criterion='entropy'`, `max_features=0.5`, `max_depth=22`, `min_samples_leaf=2`)
* **Tracking Updated:** `results.tsv`, `metric_over_time.png`, `errors.log`

## 3. Results (Quantitative)
| Run | Iteration Label | Validation AUC | Validation F1 | Runtime (s) | Features | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| 24 | iter24_rf_entropy_700_depth22 | 0.916839 | 0.652027 | 4.290 | 65 | discard |

## 4. Outcome
Run #24 did not beat the best-so-far AUC (0.918170), so `model.py` was reverted to the previous best configuration (run #17 winner).

## 5. Error Taxonomy Update
No runtime error occurred in run #24.

---

# Experiment Log: Ten Iterations Focused on Feature Selection + Class Imbalance
**Date:** May 12, 2026

## 1. Experiment Objective
Run 10 additional iterations (runs #25-#34) emphasizing reduced feature sets and class-imbalance handling, including SMOTE pipelines as a targeted mitigation strategy.

## 2. Configuration & Methodology
* **Run Range:** #25 through #34
* **Focus Areas:**
  - Feature reduction (62-65 transformed features)
  - Class imbalance mitigation via `class_weight` and SMOTE variants
* **Model Families:** RandomForestClassifier and SMOTE+RandomForest (imblearn pipeline)
* **Artifact Updates:** `results.tsv`, `metric_over_time.png`, `errors.log`

## 3. Results (Quantitative)

| Run | Iteration Label | Validation AUC | Validation F1 | Runtime (s) | Features | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| 25 | iter25_rf_entropy_feat9 | 0.916388 | 0.648464 | 7.508 | 64 | discard |
| 26 | iter26_rf_entropy_feat8 | **0.918567** | 0.646465 | 8.763 | 63 | keep |
| 27 | iter27_rf_entropy_feat10_depth18 | 0.916466 | 0.655791 | 8.088 | 65 | discard |
| 28 | iter28_rf_entropy_feat7 | 0.918544 | 0.676101 | 7.290 | 62 | discard |
| 29 | iter29_rf_entropy_feat10_minsplit4 | 0.918117 | 0.641638 | 7.861 | 65 | discard |
| 30 | iter30_smote_rf_feat10 | 0.915111 | 0.653266 | 7.369 | 65 | discard |
| 31 | iter31_smote_rf_feat8 | 0.916663 | 0.644182 | 8.756 | 63 | discard |
| 32 | iter32_smote_rf_feat10_depth16 | 0.917504 | 0.650000 | 8.968 | 65 | discard |
| 33 | iter33_smote_rf_feat9 | 0.918192 | 0.654545 | 9.753 | 64 | discard |
| 34 | iter34_smote_rf_feat10_light | 0.915579 | 0.653333 | 6.853 | 65 | discard |

## 4. Outcome
New best model after this cycle: **run #26 (`iter26_rf_entropy_feat8`)** with **AUC = 0.918567**.

`model.py` now reflects this best configuration.

## 5. Error Taxonomy Update
No runtime errors occurred in runs #25-#34. `errors.log` was updated to explicitly state this.

---

# Final Locked Test Set Evaluation
**Date:** May 26, 2026

## 1. Objective
Evaluate the best-performing model found during autoresearch on the locked test set (`online_shoppers_LOCKED.csv`) without modifying model architecture or hyperparameters.

## 2. Procedure
* Used the current best saved `model.py` configuration unchanged.
* Trained on the existing training split from the WORK dataset.
* Evaluated on the LOCKED test set instead of validation.
* Appended results to `results.tsv` as final test row and regenerated `metric_over_time.png`.

## 3. Final Test Result
| Run | Dataset | AUC | F1 | Features | Runtime (s) | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| 35 | LOCKED test set | **0.901138** | **0.640327** | 63 | 8.444 | final_test |

## 4. Note
This row is the final locked test-set performance entry and is labeled as `FINAL_LOCKED_TEST_RESULT` in `results.tsv`.
