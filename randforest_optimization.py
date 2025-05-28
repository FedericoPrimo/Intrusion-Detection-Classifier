def random_forest_multiclass_optimization(
    X, y, param_grid=None, test_size=0.2, random_state=42, verbose=True
):
    """
    Esegue split, bilanciamento, tuning iperparametri e confronto tra
    Random Forest base e ottimizzata (multiclasse).
    
    - X: DataFrame feature selezionate per RF (già encodate e pulite)
    - y: target multiclass originale (Series o array)
    - param_grid: dizionario per GridSearchCV
    - test_size: proporzione test set
    - random_state: seed
    - verbose: se True stampa dettagli
    
    Ritorna: df_scores (tabella risultati), rf_base, rf_best (modelli allenati)
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score, f1_score
    from imblearn.over_sampling import SMOTE

    # Step 1: Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    if verbose:
        print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        print("Distribuzione y_train:")
        print(pd.Series(y_train).value_counts())

    # Step 2: Bilanciamento SOLO su training set
    sm = SMOTE(random_state=random_state)
    X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
    if verbose:
        print("\nDistribuzione dopo SMOTE (solo train):")
        print(pd.Series(y_train_bal).value_counts())

    # Step 3: Grid Search tuning RF (o param_grid default)
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'class_weight': [None, 'balanced']
        }
    rf = RandomForestClassifier(random_state=random_state)
    grid = GridSearchCV(rf, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
    grid.fit(X_train_bal, y_train_bal)
    rf_best = grid.best_estimator_
    if verbose:
        print("\nMigliori parametri trovati:", grid.best_params_)

    # Step 4: Modello base (senza tuning né SMOTE)
    rf_base = RandomForestClassifier(random_state=random_state)
    rf_base.fit(X_train, y_train)
    y_pred_base = rf_base.predict(X_test)

    # Step 5: Modello ottimizzato (con tuning + SMOTE)
    y_pred_best = rf_best.predict(X_test)

    # Step 6: Valutazione
    if verbose:
        print("\n=== RANDOM FOREST BASE ===")
        print(classification_report(y_test, y_pred_base))
        print("Accuracy:", accuracy_score(y_test, y_pred_base))
        print("F1 macro:", f1_score(y_test, y_pred_base, average='macro'))

        print("\n=== RANDOM FOREST OTTIMIZZATO + SMOTE ===")
        print(classification_report(y_test, y_pred_best))
        print("Accuracy:", accuracy_score(y_test, y_pred_best))
        print("F1 macro:", f1_score(y_test, y_pred_best, average='macro'))

    # Step 7: Barplot confronto
    scores = {
        "RF Base": {
            "accuracy": accuracy_score(y_test, y_pred_base),
            "f1_macro": f1_score(y_test, y_pred_base, average='macro')
        },
        "RF Ottimizzata": {
            "accuracy": accuracy_score(y_test, y_pred_best),
            "f1_macro": f1_score(y_test, y_pred_best, average='macro')
        }
    }
    df_scores = pd.DataFrame(scores).T
    df_scores.plot(kind='bar', figsize=(8,5))
    plt.title("Random Forest Multiclasse: Prima vs Dopo Ottimizzazione")
    plt.ylim(0,1)
    plt.ylabel("Score")
    plt.show()

    return df_scores, rf_base, rf_best
