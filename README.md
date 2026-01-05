# Intrusion Detection Classifier

End-to-end machine learning pipeline for intrusion detection, supporting both binary and multiclass classification.

## 🎯 Project Goal
The goal of this project is to design and evaluate a machine learning system capable of:
- detecting network intrusions
- classifying different types of attacks
- comparing binary vs multiclass classification strategies

The project was developed for the Data Mining and Machine Learning course at the University of Pisa.

## 🧩 Pipeline Overview
The system follows a structured pipeline:
1. **Preprocessing**
   - data cleaning
   - feature encoding
   - normalization
2. **Feature Selection**
   - dimensionality reduction
   - selection of relevant attributes
3. **Model Training**
   - binary classifier (normal vs intrusion)
   - multiclass classifier (attack categories)
4. **Hyperparameter Optimization**
   - best parameters stored for reproducibility
5. **Evaluation**
   - performance comparison across tasks

## 📂 Project Structure
- `datasets/` – Input datasets
- `preprocessing.py` – Data preprocessing logic
- `feature_selection.py` – Feature selection methods
- `rf_pipeline_binary.joblib` – Trained binary classifier pipeline
- `rf_pipeline_multiclass.joblib` – Trained multiclass classifier pipeline
- `best_rf_params_*.json` – Optimized hyperparameters
- `gui.py` – Simple GUI for running the classifier
- `progetto.ipynb` – Experimental notebook
- `Project Documentation.pdf` – Detailed report

## 🔍 Design Notes
The project emphasizes:
- separation between experimentation and production-ready code
- reproducible ML experiments
- explicit comparison between different classification strategies

Saved pipelines allow inference without retraining models.
