import constant as const

# Utility functions for data handling
categorical_cols = const.CATEGORICAL_COLS
boolean_cols = const.BOOLEAN_COLS
percentile_cols = const.PERCENTILES_COLS

def showFeatures(features, dataframe, model_type):
  print(f"{model_type} selected features:")
  print(features)
  print(dataframe[features].head())

def save_features(feature_list, filename):
  import json
  with open(filename, 'w') as f:
    json.dump(feature_list, f)

def load_features(filename):
  import json
  with open(filename, 'r') as f:
    features = json.load(f)
  return features

def continuous_feature_descriptor(X):
  import pandas as pd

  df = X.copy()
  continuous_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
  # Remove boolean and percentile columns
  continuous_features = [col for col in continuous_features if col not in boolean_cols + percentile_cols]
  continuous_features.remove('label') if 'label' in continuous_features else None
  print("=== CONTINUOUS FEATURES STATISTICAL ANALYSIS ===\n")
  print(f"Number of continuous features: {len(continuous_features)}")
  print(f"Continuous features: {continuous_features}\n")

  # General statistical description
  print("=== DESCRIPTIVE STATISTICS ===")
  print(df[continuous_features].describe())

  # Outlier check (values beyond 3 standard deviations)
  print("\n=== OUTLIER CHECK (>3σ) ===")
  for col in continuous_features[:10]:  # First 10 to avoid overload
    mean_val = df[col].mean()
    std_val = df[col].std()
    outliers = df[(df[col] > mean_val + 3*std_val) | (df[col] < mean_val - 3*std_val)]
    print(f"{col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.2f}%)")

  # Check null values and distribution
  print("\n=== NULL VALUES AND DISTRIBUTION ===")
  missing_info = pd.DataFrame({
    'Missing': df[continuous_features].isnull().sum(),
    'Missing%': (df[continuous_features].isnull().sum() / len(df) * 100).round(2),
    'Unique': df[continuous_features].nunique(),
    'Min': df[continuous_features].min(),
    'Max': df[continuous_features].max()
  })
  print(missing_info)

def save_gridsearch_results(grid, prefix="rf", suffix="bin"):
    import json
    if grid is not None and hasattr(grid, "best_params_"):
        # Save best_params_
        with open(f"best_{prefix}_params_{suffix}.json", "w") as f:
            json.dump(grid.best_params_, f)
    else:
        print("Grid search not recalculated and results not found. Skipping save operation.")

def load_gridsearch_results(prefix="rf", suffix="bin"):
    import json
    with open(f"best_{prefix}_params_{suffix}.json", "r") as f:
        best_params = json.load(f)
    return best_params

def check_skew_and_log_effect(df, col):
  from scipy.stats import skew
  import numpy as np
  original = df[col].dropna()
  logged = np.log1p(original.clip(lower=0))

  print(f"== {col} ==")
  print(f"Skew original: {skew(original):.3f}")
  print(f"Skew log1p   : {skew(logged):.3f}")

  if skew(original) >= 1.0 and skew(logged) < 1.0:
    return True
  return False