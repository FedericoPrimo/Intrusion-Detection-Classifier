# Network Intrusion Detection with Machine Learning

End-to-end **binary and multiclass network intrusion detection** on the KDD99 dataset, designed and implemented independently in Python.

## Results at a glance

| Metric | Result |
| --- | --- |
| Multiclass macro-F1 | **0.9754** on a held-out test set |
| Input features | **41** per network connection |
| Models compared | **3**: Random Forest, Logistic Regression, Linear SVM |
| Model comparison | **5-fold** stratified cross-validation |
| Random Forest tuning | **8** hyperparameter combinations × **3** folds = **24 fits** |

## Approach

1. **Preprocess:** encode categorical variables, log-transform skewed features, and scale numerical values.
2. **Handle imbalance:** apply **SMOTENC** within training folds; use model-specific feature selection (feature importance or RFE).
3. **Compare and tune:** evaluate three classifiers using macro-F1, then optimize Random Forest with grid search.
4. **Evaluate:** test the selected pipeline on a separate held-out set; analyze confusion matrices and per-class F1-scores.

The optimized **Random Forest** achieved **0.9754 multiclass macro-F1**, outperforming the tested linear baselines. The experiments used a deduplicated dataset and kept the held-out test set separate from model selection.

## Limitations

**KDD99 is a historical benchmark** with known redundancy and class-imbalance issues. Some rare attack classes were excluded from parts of the evaluation; these results do **not** establish performance on modern production traffic.

For methodology, plots, and detailed evaluation, see the project's **Project Documentation.pdf**.
