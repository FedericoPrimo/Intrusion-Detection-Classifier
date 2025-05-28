def compare_binary_models_split(models, X_dict, y_dict, model_names=None, test_size=0.2, random_state=42):
    """
    Allena e valuta ogni modello su una train/test split.
    Restituisce un DataFrame con le metriche.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve, f1_score, accuracy_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split
    import pandas as pd
    import numpy as np

    results = []
    plt.figure(figsize=(10, 7))
    for name in (model_names or models.keys()):
        model = models[name]
        X = X_dict[name]
        y = y_dict[name]
        # SPLIT train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:,1]
        else:
            y_score = model.decision_function(X_test)
            y_score = (y_score - y_score.min())/(y_score.max() - y_score.min())
        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_score)
        plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc_score(y_test, y_score):.2f})')
        # Metrics
        results.append({
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_score)
        })
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Classificazione Binaria (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    # Barplot metriche
    df_results = pd.DataFrame(results)
    df_plot = df_results.set_index("model")[["accuracy", "f1", "precision", "recall", "roc_auc"]]
    df_plot.plot(kind="bar", figsize=(12,6))
    plt.title("Metriche di confronto tra modelli (binario, test set)")
    plt.ylabel("Score")
    plt.ylim(0,1)
    plt.tight_layout()
    plt.show()
    return df_results

def compare_multiclass_models_split(
    models,
    X_dict,
    y_dict,
    model_names=None,
    test_size=0.2,
    random_state=42
):
    """
    Confronta modelli multiclass usando train/test split. 
    models: dict {nome: modello istanziato}
    X_dict: dict {nome: X_preprocessed (con features selezionate)}
    y: vettore target multiclass (stringhe o numeri)
    model_names: lista opzionale per ordinamento
    test_size: proporzione del test set
    random_state: seed per riproducibilità
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    from sklearn.model_selection import train_test_split
    import pandas as pd
    import numpy as np
    import seaborn as sns

    results = []
    for name in (model_names or models.keys()):
        model = models[name]
        X = X_dict[name]
        y = y_dict[name]
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        results.append({
            "model": name,
            "accuracy": acc,
            "f1_macro": f1_macro
        })
        # Confusion matrix heatmap
        plt.figure(figsize=(8,6))
        cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
        sns.heatmap(cm, annot=False, cmap="Blues", 
                    xticklabels=np.unique(y), yticklabels=np.unique(y))
        plt.title(f"Confusion Matrix - {name} (test set)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.show()

    df_results = pd.DataFrame(results)
    df_plot = df_results.set_index("model")[["accuracy", "f1_macro"]]
    df_plot.plot(kind="bar", figsize=(10,5))
    plt.title("Metriche di confronto tra modelli (multiclasse, test set)")
    plt.ylabel("Score")
    plt.ylim(0,1)
    plt.tight_layout()
    plt.show()
    return df_results

