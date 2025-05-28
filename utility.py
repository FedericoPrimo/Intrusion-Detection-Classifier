# Utility functions for data handling

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


def save_gridsearch_results(grid, prefix="rf"):
    import json
    import numpy as np
    if grid is not None and hasattr(grid, "best_params_"):
        # Salva best_params_
        with open(f"best_{prefix}_params.json", "w") as f:
            json.dump(grid.best_params_, f)

        # Salva best_score_
        with open(f"best_{prefix}_score.json", "w") as f:
            json.dump(float(grid.best_score_), f)

        # Salva cv_results_ (convertendo tipi numpy)
        cv_results_safe = {}
        for k, v in grid.cv_results_.items():
            if isinstance(v, np.ndarray):
                cv_results_safe[k] = v.tolist()
            else:
                cv_results_safe[k] = v
        with open(f"best_{prefix}_cv_results.json", "w") as f:
            json.dump(cv_results_safe, f)
        print("Grid search results saved.")
    else:
        print("Grid search not recalculated and results not found. Skipping save operation.")


def load_gridsearch_results(prefix="rf"):
    import json
    with open(f"best_{prefix}_params.json", "r") as f:
        best_params = json.load(f)
    with open(f"best_{prefix}_score.json", "r") as f:
        best_score = json.load(f)
    with open(f"best_{prefix}_cv_results.json", "r") as f:
        cv_results = json.load(f)
    return best_params, best_score, cv_results