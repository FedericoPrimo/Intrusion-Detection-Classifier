from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

class KDD99Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.categorical_cols = ['protocol_type', 'service', 'flag']
        self.boolean_cols = ['land', 'logged_in', 'lroot_shell', 'lsu_attempted', 'is_host_login', 'is_guest_login']
        self.transform_to_log = [
            'duration', 'src_bytes', 'dst_bytes',
            'count', 'srv_count', 'dst_host_count', 'dst_host_srv_count'
        ]
        self.encoder_continuous = MinMaxScaler()
        self.continuous_cols = None  # Verrà settato in fit

    def fit(self, X, y=None):
        dataframe = X.copy()
        # Identifica le colonne continue (dopo aver rimosso quelle da log)
        continuous_cols = dataframe.select_dtypes(include=['int64', 'float64']).columns.tolist()
        continuous_cols = [col for col in continuous_cols if col not in self.transform_to_log]
        self.continuous_cols = continuous_cols
        # Fit del MinMaxScaler SOLO su train (importantissimo!)
        self.encoder_continuous.fit(dataframe[continuous_cols])
        return self

    def transform(self, X):
        dataframe = X.copy()

        # Categorical
        dataframe[self.categorical_cols] = dataframe[self.categorical_cols].astype('category')
        # Boolean
        dataframe[self.boolean_cols] = dataframe[self.boolean_cols].astype(bool)

        # Log transform
        for col in self.transform_to_log:
            if col in dataframe.columns:
                dataframe[col] = dataframe[col].fillna(0)
                dataframe[col + '_log'] = np.log1p(dataframe[col].clip(lower=0))
        dataframe.drop(columns=self.transform_to_log, inplace=True)

        # Normalizza
        dataframe[self.continuous_cols] = self.encoder_continuous.transform(dataframe[self.continuous_cols])

        # One-hot encoding
        X_cat_onehot = pd.get_dummies(dataframe[self.categorical_cols], drop_first=True)
        X_transf = pd.concat([dataframe, X_cat_onehot], axis=1).drop(columns=self.categorical_cols)

        return X_transf
