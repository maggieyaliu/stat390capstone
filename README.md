# Consumer Purchase Behavior Prediction: Baseline to AutoResearch Optimization

This repository contains the end-to-end machine learning pipeline for predicting online session-level purchase intent (`Revenue`) using the [UCI Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset).

Through an automated research loop comprising 35 tracked experiments, we optimized a baseline model to handle severe class imbalance and feature selection constraints.

## 1. Project Setup & Infrastructure

To run the code, you will need **Python 3.8+** and the following libraries installed:

* `pandas`
* `numpy`
* `scikit-learn`

Install the dependencies via your terminal:

```bash
pip install pandas numpy scikit-learn

```

Note: Due to the automated nature of the experiment loop, using pinned environment dependencies is highly recommended to avoid runtime instability.

## 2. Experimental Protocol & Data Split

The dataset consists of 12,330 single-user sessions with a severe 84.5% majority-class imbalance of non-purchasers. Following project rules, the feature `Month` is explicitly excluded from all pipelines to isolate behavioral metrics from seasonal bias.

* **Development Set:** A stratified train/validation split generated from `online_shoppers_WORK.csv`.


* **Final Holdout Set:** A completely held-out, evaluation-ready `online_shoppers_LOCKED.csv`.



## 3. Methodology & Search Space

We transitioned from a basic linear model to a fully automated research loop, editing only `model.py` across execution cycles:

1. **Baseline Phase:** A standard `LogisticRegression` pipeline with robust scaling (`StandardScaler`) and categorical encoding (`OneHotEncoder`). Established an initial validation performance of **~0.88 AUC**.


2. **AutoResearch Loop:** 35 tracked sequential experiments primarily exploring Random Forest hyperparameter configurations, boosting variants, and feature counts.


3. **Ablation Focus:** Evaluated feature-count optimization (narrowing down to 62–65 transformed features) alongside explicit class-imbalance interventions comparing synthetic oversampling (`SMOTE`) against algorithmic class-weighting pipelines.



## 4. Tracking Artifacts

Every execution cycle in the loop strictly updates and logs the following artifacts to preserve reproducibility:

* `results.tsv`: Tracks run ID, ROC AUC, F1 score, active feature counts, and model runtime metadata.


* `metric_over_time.png`: A live visualization mapping the validation AUC trajectory against the original baseline.


* `errors.log`: A runtime taxonomy error log used to guide agent adjustment strategies.



## 5. Final Key Results

### Best Validation Configuration (Run #26)

The highest performing validation setup was a tuned **Random Forest Classifier** utilizing an entropy criterion, 800 estimators, and a `balanced_subsample` class-weighting framework over a restricted subset of **63 features**.

* **Validation AUC:** **0.918567** 


* **Validation F1:** 0.646465 



### Locked Test Generalization (Run #35)

Evaluating the champion configuration unchanged on the locked test set yielded strong out-of-sample stability:

* **Locked Test AUC:** **0.901138** 


* **Locked Test F1:** 0.640327 


* **Test Prediction Inference Runtime:** 8.444 seconds 



## 6. Main Takeaways

* **Ensembles Win:** Random Forest architectures significantly outpaced linear benchmarks when charting complex web analytics behavior.


* **Simplicity over Complexity:** Algorithmic class-weighting (`balanced_subsample`) consistently yielded a higher AUC ceiling than heavier, computationally intense SMOTE pipelines.


* **Feature Pruning:** Trimming irrelevant dimensions to a tight 63-feature space optimized the model's predictive ceiling without destroying metric representations.
