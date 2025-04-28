def FeatureSelection(X_encoded, target):
  from sklearn.preprocessing import MinMaxScaler
  import numpy as np
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.feature_selection import mutual_info_classif, chi2
  import pandas as pd
  
  # Converte eventuali feature binarie float in int
  for col in X_encoded.columns:
      unique_vals = X_encoded[col].dropna().unique()
      if X_encoded[col].dtype == 'float' and all(val.is_integer() for val in unique_vals):
          X_encoded[col] = X_encoded[col].astype(int)
  # Rigenera la maschera delle discrete
  discrete_mask = [pd.api.types.is_integer_dtype(X_encoded[col]) and X_encoded[col].nunique() <= 10
                   for col in X_encoded.columns]
  # Calcola la MI e Chi2
  mi_scores = mutual_info_classif(X_encoded, target, discrete_features=discrete_mask)
  # Random Forest
  model = RandomForestClassifier()
  model.fit(X_encoded, target)
  rf_importance = model.feature_importances_
  # Scaling dei punteggi
  scaler = MinMaxScaler()
  df_rank = pd.DataFrame({
      'Feature': X_encoded.columns,
      'MI': scaler.fit_transform(mi_scores.reshape(-1, 1)).flatten(),
      'RF': scaler.fit_transform(rf_importance.reshape(-1, 1)).flatten()
  })
  df_rank['MeanScore'] = df_rank[['MI', 'RF']].mean(axis=1)
  df_rank = df_rank.sort_values(by='MeanScore', ascending=False)
  return df_rank

def compare_models_on_encodings(X_ord_final, X_oh_final, y, task_name="Classificazione"):
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import f1_score, classification_report
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.linear_model import LogisticRegression
  from sklearn.svm import SVC
  import matplotlib.pyplot as plt
  import seaborn as sns
  import pandas as pd
  results = []
  datasets = {
    'Ordinal':{
      'dataset': X_ord_final,
      'models': [
        RandomForestClassifier(random_state=42)
      ]
    },
    'OneHot': {
      'dataset': X_oh_final,
      'models': [
        LogisticRegression(max_iter=1000, class_weight='balanced'),
        SVC(class_weight='balanced')
      ]
    }
  }
  
  for ds_name, ds_info in datasets.items():
    X_data = ds_info['dataset']
    models = ds_info['models']
    X_train, X_test, y_train, y_test = train_test_split(
      X_data, y, test_size=0.3, stratify=y, random_state=42
    )

    for model in models:
      model_name = model.__class__.__name__
      model.fit(X_train, y_train)
      y_pred = model.predict(X_test)
      f1 = f1_score(y_test, y_pred, average='macro')
      results.append({
        'Dataset': ds_name,
        'Model': model_name,
        'F1-macro': f1
      })
      print(f"\n📊 {task_name} - {ds_name} - {model_name}")
      print(classification_report(y_test, y_pred))
  # Confronto tabellare
  df_results = pd.DataFrame(results)
  
  # Barplot
  plt.figure(figsize=(10, 6))
  sns.barplot(data=df_results, x='Model', y='F1-macro', hue='Dataset')
  plt.title(f'Confronto F1-macro - {task_name}')
  plt.ylim(0, 1)
  plt.grid(True, axis='y', linestyle='--', alpha=0.5)
  plt.legend(title='Dataset')
  plt.tight_layout()
  plt.show()
  
  return df_results
