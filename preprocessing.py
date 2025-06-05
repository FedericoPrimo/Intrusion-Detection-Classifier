from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import constant as const
import numpy as np

class KDD99Preprocessor(BaseEstimator, TransformerMixin):
  def __init__(self):
    ## DEFINE COLUMNS
    self.categorical_cols = const.CATEGORICAL_COLS
    self.boolean_cols = const.BOOLEAN_COLS
    self.continuous_cols = const.CONTINUOUS_COLS
    self.percentile_cols = const.PERCENTILES_COLS
    self.transform_to_log = const.TRANSFORM_TO_LOG
    self.ordinal_encoder = OrdinalEncoder(dtype='int', handle_unknown='use_encoded_value', unknown_value=-1)
    self.scalers = {}

  def fit(self, X, y=None):
    dataframe = X.copy()

    # Log transform for some continuous features
    for col in self.transform_to_log:
      if col in dataframe.columns:
        dataframe[col] = dataframe[col].fillna(0)
        dataframe[col] = np.log1p(dataframe[col].clip(lower=0))

    for col in self.continuous_cols:
      scaler = MinMaxScaler()
      scaler.fit(dataframe[[col]])
      self.scalers[col] = scaler
    

    dataframe[self.categorical_cols] = dataframe[self.categorical_cols].fillna("missing")
    self.ordinal_encoder.fit(dataframe[self.categorical_cols])
  
    return self

  def transform(self, X):
    dataframe = X.copy()

    # Log transform for some continuous features
    for col in self.transform_to_log:
      if col in dataframe.columns:
        dataframe[col] = dataframe[col].fillna(0)
        dataframe[col] = np.log1p(dataframe[col].clip(lower=0))

    # Using MinMaxScaler for continuous features
    for col in self.continuous_cols:
      dataframe[col] = self.scalers[col].transform(dataframe[[col]])

    for bcol in self.boolean_cols:
      if bcol in dataframe.columns:
        dataframe[bcol] = dataframe[bcol].astype(int)

    dataframe[self.categorical_cols] = dataframe[self.categorical_cols].fillna("missing")
    dataframe[self.categorical_cols] = self.ordinal_encoder.transform(dataframe[self.categorical_cols])

    return dataframe
