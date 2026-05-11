# Python Machine Learning
### Preprocessing Pipelines | Supervised Learning | Model Evaluation & Optimization

**Author:** Kevin Victor
**Scope:** Consolidated reference across three laboratory modules
**Status:** Educational & Applied

---

## Module Overview

This module covers the complete supervised machine learning workflow — from raw data through preprocessing, model training, evaluation, and systematic optimization — alongside unsupervised clustering and principled hyperparameter tuning.

| Module | Focus Area |
|---|---|
| **Module 7** | ML Preprocessing Pipelines — Imputation, Feature Engineering, Scaling, Leakage Prevention, Linear Regression |
| **Module 8** | Supervised Learning — Linear Regression, Logistic Regression, K-Nearest Neighbours, Classification Evaluation |
| **Module 9** | Model Evaluation & Optimization — Confusion Matrix, K-Means Clustering, Silhouette Scoring, GridSearchCV |

The unifying principle: a model is only as reliable as the pipeline that prepared its data, the rigor of its evaluation, and the discipline of its optimization.

---

## Module 7 — Complete ML Preprocessing Pipelines

### Core Concepts

**DataFrame Inspection** establishes data structure before any processing. `df.head()` provides a visual sample; `df.columns` returns column names for programmatic access; `df.info()` reveals the most critical structural information — row count, dtype per column, and non-null count per column. Any column where non-null count falls short of total rows contains missing values.

**SimpleImputer** is Scikit-learn's unified missing value handler. It supports four strategies: `'mean'` (for symmetric numerical distributions), `'median'` (for skewed distributions or those with outliers), `'most_frequent'` (for categorical columns — the only arithmetically valid strategy for string data), and `'constant'`. It follows the standard fit-transform pattern: fitted on training data, applied to test data — the same leakage discipline as scalers. After `fit_transform()`, all columns are returned as `object` dtype; explicit `.astype()` reconversion is required for numerical columns.

**Feature Engineering** creates derived columns from existing ones to surface implicit relationships as explicit model inputs. Representative patterns include ratios (Order Fulfillment Ratio = packages prepared / orders received), sums (Total Stock = perishable + non-perishable inventory), differences (Stock Gap = reorder level − current stock), and composite indices (Process Efficiency = layers processed / total process time). The `+ 1` guard in denominators prevents division-by-zero. Feature engineering precedes imputation because derived features inherit missing values from their source columns.

**Data Leakage Prevention** is the critical discipline governing the fit-transform boundary. Any transformer — imputer, scaler, encoder — fitted on the full dataset before the train-test split incorporates statistical information from test observations into the training process. The correct sequence is: split first, then fit all transformers on training data only, then apply (`.transform()`) to test data. A scaler fitted on test data does not re-learn; it applies training-derived statistics to new observations — precisely what production inference requires.

**The Correct Preprocessing Order** is: impute missing values → then scale. The technical reason is definitive: `StandardScaler` computes column means and standard deviations. Any `NaN` in a column propagates through NumPy's statistics operations, producing undefined scaler parameters. Analytically, imputed complete columns produce more accurate normalization statistics than columns with gaps.

**Linear Regression Evaluation** uses two complementary metrics. MSE (Mean Squared Error) measures average squared prediction error, penalizing large errors disproportionately. R² (Coefficient of Determination) measures the proportion of target variance explained by the model — dimensionless, range [0, 1], directly comparable across datasets. Domain-calibrated metrics — such as the proportion of predictions within ±10 marks — translate statistical measures into operationally meaningful assessments. Regression line visualization requires sorting test observations by the X variable before plotting to produce a smooth, continuous line.

### Industrial Use Cases

| Domain | Pattern Applied |
|---|---|
| Educational Analytics | `SimpleImputer('most_frequent')` on mixed numerical/categorical student data |
| Smart Buildings / Energy | Data leakage prevention in HVAC energy prediction models |
| Warehouse / Logistics | End-to-end pipeline with feature engineering + imputation + split + scaling |
| Sports Science / Biometrics | StandardScaler on height-weight for distance-based models |
| Retail / Supply Chain | Demand-supply gap feature + profit margin ratio before regression |
| Semiconductor Manufacturing | Impute-before-scale mandate for 8-orders-of-magnitude feature ranges |

---

## Module 8 — Supervised Learning Algorithms

### Core Concepts

**Linear Regression** fits Y = β₀ + β₁X₁ + … + βₙXₙ by minimizing sum of squared residuals (OLS). After training, `model.coef_` gives one coefficient per feature — each expressing the change in the predicted target per unit increase in that feature, holding others constant. Sign indicates direction; magnitude (after scaling) indicates relative importance. `model.intercept_` is baseline predicted value when all scaled features equal zero. Coefficient display is the primary form of model interpretability for linear models.

**Logistic Regression** maps the linear combination of features to a class probability via the sigmoid function: P(Y=1|X) = 1 / (1 + e^−(β₀ + β₁X₁ + …)). `predict()` returns class labels at the 0.5 threshold; `predict_proba()` returns the probability — operationally more informative, as a prediction at 0.51 should be treated very differently from one at 0.97. For multi-class problems, `multi_class='multinomial'` uses the softmax function across all classes simultaneously, producing probabilities that sum to one. `np.select(conditions, choices, default)` implements ordered priority logic for target variable construction, where conditions are evaluated in sequence and earlier conditions take precedence.

**K-Nearest Neighbours (KNN)** classifies a new observation by finding the K most similar training observations (by Euclidean distance) and assigning the majority class. KNN has no explicit training phase — it stores the training data and computes distances at inference time. K is a critical hyperparameter: small K risks overfitting; large K smooths the decision boundary but may miss local patterns. **KNN requires feature scaling** — unscaled features with large numerical ranges dominate Euclidean distance and override features with smaller ranges regardless of predictive relevance.

**Model Evaluation — Classification** uses three complementary tools. Accuracy (correct / total) is intuitive but misleading for imbalanced classes. The confusion matrix reveals which error types the model is making: False Positives (false alarm) and False Negatives (missed detection) — whose relative cost is domain-dependent and determines threshold strategy. The classification report provides per-class precision (of all predicted positives, how many were truly positive), recall (of all actual positives, how many were detected), and F1 score (harmonic mean of both). `cm.ravel()` deconstructs a 2×2 matrix into (TN, FP, FN, TP) for named variable assignment.

**Feature Engineering for Classification** constructs domain-informed composite inputs that directly inform the decision boundary. Examples: `stability_index = |voltage − 230| + |frequency − 50|` (combined grid deviation from nominal); `route_efficiency = speed × LiDAR_accuracy / (traffic_density + 0.1)` (multi-factor composite increasing with favourable routing conditions); `risk_factor = weather_severity × road_complexity × (1 − LiDAR_accuracy)` (product of simultaneously unfavourable conditions). Engineered weights encode domain knowledge directly — a feature given a higher weight in a risk score reflects the engineer's judgement about relative severity.

**Explainability and Safety Layers** translate model outputs into operational justifications. Feature contribution analysis (feature value × coefficient, ranked by magnitude) identifies the inputs most responsible for each prediction. Rule-based explanation engines inspect raw feature values against domain thresholds to generate human-readable reasons. Fallback checks override model recommendations when critical safety thresholds are independently exceeded — separating model recommendations from safety authority.

### Industrial Use Cases

| Domain | Pattern Applied |
|---|---|
| Smart Grid / Demand Forecasting | Linear regression coefficients for renewable offset interpretation |
| Smart Grid / Status Classification | Logistic regression with feature attribution — mirrors grid operator alert systems |
| Autonomous Vehicles | Logistic regression vs. KNN direct comparison for real-time route classification |
| Air Traffic Control | Multi-class logistic regression + explainability + fallback safety override |
| Cybersecurity / Spam Detection | Logistic regression with class balancing + rule-based reason engine |
| Healthcare / Cardiology | KNN multi-class classification for rare clinical condition detection |
| HR Analytics | Linear regression coefficient display for pay equity and compensation benchmarking |

---

## Module 9 — Model Evaluation & Optimization

### Core Concepts

**Confusion Matrix — Error Anatomy** provides a complete picture of where a classifier fails. True Positives and True Negatives are correct classifications; False Positives (model raises an alert that was unwarranted) and False Negatives (model fails to detect a real event) represent the two distinct failure modes. The relative operational cost of FP versus FN is domain-specific — in clinical classification, FN (missed disease) is more dangerous than FP (unnecessary follow-up), which justifies class weighting (`class_weight={0: 1.0, 1: 5.0}`) and threshold reduction below 0.5 to improve recall at the cost of precision.

**K-Means Clustering** partitions unlabeled data into K clusters by iteratively assigning observations to the nearest centroid (by Euclidean distance) and recomputing centroids as cluster means. Convergence is guaranteed but may reach a local optimum. K-Means requires feature scaling and requires K to be specified. It assumes spherical, approximately equal-sized clusters. `fit_predict()` fits the model and assigns cluster labels simultaneously. Cluster centroids are interpreted by inspecting their values against domain thresholds to characterize each discovered group and formulate segment-specific strategies.

**Silhouette Score** evaluates clustering quality without ground-truth labels. For each observation: a(i) = mean intra-cluster distance (cohesion); b(i) = mean distance to nearest other cluster (separation). Silhouette coefficient = (b − a) / max(a, b), ranging from −1 (misassigned) to +1 (well-matched). `silhouette_score()` returns the mean across all observations. Evaluating silhouette score across K = 2 to K_max and selecting the maximizing K is the standard, objective method for optimal cluster count selection — superior to the elbow method, which requires subjective identification of a bend in an inertia curve.

**VarianceThreshold** removes features whose variance falls below a defined threshold before clustering. Low-variance features carry minimal discriminative information — all observations have nearly identical values — and add noise to distance computations without improving cluster quality.

**PCA for Cluster Visualization** projects high-dimensional cluster assignments to two dimensions for visualization. `explained_variance_ratio_` quantifies how much of the original variance is retained. If the sum of PC1 and PC2 explained variance exceeds 0.7, the 2D plot faithfully represents the high-dimensional cluster structure. Cluster centroids can be projected into PCA space via `pca.transform(centroids)` to show centroid positions in the visualization.

**GridSearchCV** evaluates every combination in a defined hyperparameter grid using K-fold cross-validation. For logistic regression, key parameters include: `C` (regularization strength — smaller means stronger regularization, simpler model), `solver` (optimization algorithm), `penalty` (L1 or L2), and `max_iter`. `n_jobs=-1` parallelizes across all CPU cores. `grid.best_params_` returns the optimal combination; `grid.best_estimator_` returns the model refitted on the full training set. Before-and-after accuracy comparison on the same test set quantifies the performance gain from tuning.

**Polynomial Feature Expansion** generates pairwise interaction terms and squared terms from original features using `PolynomialFeatures(degree=2)`, enabling logistic regression (a linear classifier) to learn non-linear decision boundaries. For n original features, expansion produces n + n(n−1)/2 + n features — dimensionality grows quadratically, increasing overfitting risk. For complex non-linear problems, tree-based ensembles or RBF kernel SVMs are often more robust alternatives.

**Dual-Model Pipelines** train regression and classification models on the same preprocessed dataset using different target variables — e.g., medical cost (continuous) for regression and readmission (binary) for classification. Using `random_state=42` for both splits ensures both models are evaluated on the same held-out observations, making their performance directly comparable. The `y = x` reference line on regression scatter plots represents perfect prediction; systematic deviation from this line indicates model bias.

### Industrial Use Cases

| Domain | Pattern Applied |
|---|---|
| Meteorology / Agriculture | Confusion matrix FN analysis — mirrors IMD and NOAA operational forecast evaluation |
| Retail / CRM | K-Means customer segmentation + silhouette validation — mirrors commercial segmentation platforms |
| Warehouse Automation | GridSearchCV on robot priority classifier — mirrors WMS hyperparameter optimization |
| Healthcare / Insurance | Dual-model pipeline + class weighting — mirrors CMS readmission penalty compliance models |
| Materials Science / Fusion | Silhouette + PCA on plasma-facing material clusters — mirrors ITER materials qualification |
| Battery Technology | Complete GridSearchCV pipeline with tanh-stabilized features — mirrors Materials Project screening |
| Quantum / Bioinformatics | Polynomial + FFT + Laplacian features with GridSearchCV — domain-specific signal processing for classification |

---

## Future Industry-Grade Extensions

The following upgrade paths apply across all three modules and represent standard engineering investments for production ML systems.

**Advanced Imputation:** Replace `SimpleImputer` with `IterativeImputer` (MICE — models each missing feature as a function of all others) for context-aware imputation when inter-feature correlations exist. Use `KNNImputer` for neighbor-based estimation. For time-series sensor data, forward-fill (`df.ffill()`) is more appropriate than global column statistics.

**Scikit-learn Pipeline API:** Chain imputers, encoders, scalers, and models into a single `Pipeline` object with `ColumnTransformer`. This makes data leakage structurally impossible — all transformers are fitted on `X_train` during `pipeline.fit()` and applied to `X_test` during `pipeline.predict()`. Serialize the complete pipeline with `joblib.dump()` for deployment.

**Regularization:** For linear and logistic regression with many features, Ridge (L2 — shrinks all coefficients) and Lasso (L1 — drives some coefficients to exactly zero, performing feature selection) regularization prevent overfitting. Scikit-learn's `LogisticRegression` applies L2 by default via the `C` parameter, which should be tuned via cross-validation rather than left at default.

**Advanced Scaling:** Use `RobustScaler` (median + IQR based) when outliers are legitimate and should not distort normalization. Apply `PowerTransformer` (Yeo-Johnson) to make right-skewed features more Gaussian before StandardScaler. Use `QuantileTransformer` to completely eliminate outlier and heavy-tail effects.

**Class Imbalance Handling:** Replace `replace=True` oversampling with SMOTE (`imbalanced-learn` library), which generates synthetic minority class samples by interpolation. Apply `class_weight='balanced'` in logistic regression and KNN. Adjust classification threshold below 0.5 to improve recall for high-cost minority classes. Evaluate with AUPRC and AUC-ROC rather than accuracy.

**Model Evaluation Extensions:** Replace single train-test splits with K-fold `cross_val_score` for stable performance estimates. Add precision-recall curves and ROC curves with threshold optimization. Use Matthews Correlation Coefficient (MCC) for imbalanced binary classification. Perform residual analysis for regression — plotting residuals against predicted values reveals systematic model misspecification.

**Advanced Clustering:** For non-spherical clusters or datasets with outliers, use DBSCAN (discovers clusters of arbitrary shape, labels noise points automatically, no K required) or Gaussian Mixture Models (probabilistic soft assignments, elliptical clusters). Use hierarchical clustering for dendrogram-based K selection.

**Extended Hyperparameter Search:** Replace exhaustive GridSearchCV with `RandomizedSearchCV` for large parameter spaces (samples random combinations at a defined budget). Use Bayesian optimization (`Optuna`, `Hyperopt`) for sample-efficient search. Apply stratified K-fold to maintain class proportions across folds for imbalanced data.

**Non-Linear Models:** Move from polynomial expansion to `GradientBoostingClassifier` or `XGBRegressor` — scale-invariant, handles missing values, provides native feature importance, and outperforms linear models on most tabular datasets. For classification with non-linear boundaries, `SVC(kernel='rbf')` implicitly maps to high-dimensional space without explicit feature expansion.

**Explainability — SHAP and LIME:** Replace rule-based explanation engines with model-agnostic attribution methods. SHAP (SHapley Additive exPlanations) computes, per prediction, each feature's contribution to the deviation from the mean prediction — theoretically grounded and consistent across model types. LIME fits a locally interpretable linear model around each prediction using input perturbations. Calibrate probability outputs with `CalibratedClassifierCV` for reliable confidence estimates.

**Production Deployment:** Serve trained pipelines as REST API endpoints via FastAPI or Flask, accepting JSON feature inputs and returning predictions and probabilities. Monitor for data drift (feature distribution shift) and concept drift (label relationship shift) using Evidently AI or Nannyml. Track experiments, model versions, and metrics with MLflow. Use A/B testing before full rollout of GridSearchCV-tuned model replacements.

---

## Concept-to-Production Mapping

| Demonstrated Concept | Production Equivalent |
|---|---|
| `df.info()` structural inspection | Schema validation in data ingestion pipeline |
| `SimpleImputer(strategy='mean')` | `IterativeImputer` / `KNNImputer` in Scikit-learn pipeline |
| Manual feature engineering | `featuretools` automated relational feature generation |
| Fit-on-train, transform-test | `Pipeline` + `ColumnTransformer` — leakage-impossible by construction |
| `model.coef_` coefficient display | SHAP feature importance for non-linear models |
| `predict_proba()` at 0.5 threshold | Threshold optimization via precision-recall curve |
| `accuracy_score()` | AUC-ROC + AUPRC + MCC for imbalanced evaluation |
| `cm.ravel()` confusion matrix | Operational FN/FP cost analysis for threshold selection |
| K-Means with scatter plot | DBSCAN / GMM for non-spherical clusters |
| Silhouette score K selection | Hierarchical clustering dendrogram for K-free selection |
| GridSearchCV on parameter grid | Bayesian optimization with Optuna for large search spaces |
| Polynomial feature expansion | XGBoost / RBF-SVM for non-linear classification |
| Rule-based explanation engine | SHAP / LIME for model-agnostic prediction attribution |
| In-script model training | `joblib.dump()` + FastAPI REST endpoint for production inference |

---

## Summary

These three modules cover the complete machine learning workflow from a data engineering and algorithmic perspective. Module 7 establishes that the correctness of a model is determined before training — by the quality of imputation, feature engineering, and leakage-free preprocessing. Module 8 demonstrates how three foundational algorithms — linear regression, logistic regression, and KNN — each embody a distinct approach to learning from labeled data, and how their outputs must be made interpretable and safe for operational use. Module 9 addresses how models are rigorously evaluated beyond accuracy, how natural structure in unlabeled data is discovered and validated, and how performance is systematically improved through principled hyperparameter search. Taken together, these modules constitute the core engineering discipline of applied machine learning — not the theory of algorithms, but the practice of building systems that can be trusted.

---
