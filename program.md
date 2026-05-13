# AutoResearch Agent Instructions

## Objective

Maximize **AUC Score** on the Consumer Purchase Behavior Classification task.

## Rules

1. **Modify ONLY `model.py`:** Focus on the `build_model()` function.
2. **Frozen Infrastructure:** `prepare.py` and `run.py` are strictly off-limits.
3. **Return a Pipeline:** `build_model()` must return an sklearn-compatible Pipeline.
4. **Efficiency:** Training and evaluation must complete in **under 60 seconds** on CPU.
5. **No Peeking:** Do not use external data or hard-code validation logic.

## Workflow & Strategy

1. **Incremental Modification:** Propose and implement changes to **only 1–3 variables at a time**. Avoid "huge" structural overhauls in a single run; small, measurable steps are required to maintain a clear audit trail.
2. **Execute:** Run `python run.py "specific description of changes"`.
3. **Evaluate:** * If **AUC improves**: `git add model.py && git commit -m "feat: <description>"`
* If **Performance drops**: `git checkout model.py` (revert and try a different direction).


4. **Error Taxonomy:** If a run fails, consult `errors.log` to categorize the issue before attempting the next modification.

## Ideas to Explore

* **Model Selection (The "Model Zoo"):** Do not stick to one algorithm. Systematically test different architectures including `RandomForestClassifier`, `HistGradientBoostingClassifier`, `SVC` (with probability enabled), and `KNeighborsClassifier`.
* **Hyperparameter Sensitivity:** When a model shows promise, perform a "Micro-Search" on its specific parameters. For example, in tree-based models, experiment with `n_estimators`, `max_depth`, and `min_samples_split` in small increments.
* **Feature Selection:** You are not required to use all available features. Experiment with dropping low-variance features to reduce noise.
* **Class Imbalance:** This dataset is highly imbalanced. Prioritize techniques like `class_weight='balanced'`, SMOTE (if available in your environment), or adjusting probability thresholds to improve minority class recall.
* **Ensemble Tuning:** Fine-tune `RandomForestClassifier` or `HistGradientBoostingClassifier` with a focus on `max_depth` and `min_samples_leaf` to prevent overfitting.
* **Scaling & Transforms:** Explore `RobustScaler` for duration-based features or `QuantileTransformer` for skewed distributions.

## What NOT to do

* **No "Black Box" Changes:** Do not change 10 hyperparameters at once. We must know *why* a model improved.
* **No Data Leakage:** Ensure `Month` remains excluded to focus on behavioral intent rather than seasonality.
* **Do Not Change Signatures:** Keep the `build_model()` function name and return type consistent.
