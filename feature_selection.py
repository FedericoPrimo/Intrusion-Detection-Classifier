from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE

class FeatureSelectorByModel(BaseEstimator, TransformerMixin):
  def __init__(self, model_type='rf', n_features=20, threshold=0.3, random_state=42, n_rf_fit=1):
    self.model_type = model_type
    self.n_features = n_features
    self.threshold = threshold
    self.random_state = random_state
    self.selected_features_ = None
    self.n_rf_fit = n_rf_fit

  def fit(self, X, y):
    X = pd.DataFrame(X)

    ## Random Forest ##
    if self.model_type == 'rf':
      rf_scores_list = []
      ## Fit multiple Random Forest models to average feature importances
      for i in range(self.n_rf_fit):
        model = RandomForestClassifier(random_state=self.random_state + i)
        model.fit(X, y)
        rf_scores_list.append(model.feature_importances_)

        rf_importance = np.mean(rf_scores_list, axis=0) 
        rf_scaled = MinMaxScaler().fit_transform(rf_importance.reshape(-1, 1)).flatten()
        df_rank = pd.DataFrame({
          'Feature': X.columns,
          'RF': rf_scaled
        })
        df_rank = df_rank.sort_values(by='RF', ascending=False)
        self.selected_features_ = df_rank.head(self.n_features)['Feature'].tolist()

    ## SVM ##
    elif self.model_type == 'svm':
      estimator = LinearSVC(dual=False, max_iter=2000, random_state=self.random_state)
      rfe = RFE(estimator, n_features_to_select=self.n_features)
      rfe.fit(X, y)
      self.selected_features_ = X.columns[rfe.support_].tolist()

    ## Logistic Regression ##
    elif self.model_type == 'logreg':
      estimator = LogisticRegression(max_iter=1000, random_state=self.random_state)
      rfe = RFE(estimator, n_features_to_select=self.n_features)
      rfe.fit(X, y)
      self.selected_features_ = X.columns[rfe.support_].tolist()

    else:
      raise ValueError("model_type must be 'rf', 'svm' o 'logreg'")
    return self

  def transform(self, X):
    if self.selected_features_ is None:
      raise RuntimeError("Transformer hasn't been fitted yet.")
    
    return pd.DataFrame(X)[self.selected_features_]
