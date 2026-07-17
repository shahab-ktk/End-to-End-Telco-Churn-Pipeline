# FILE LOCATION: src/utils.py
import os
import sys
import joblib
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, recall_score
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    """Serializes and saves a python object (transformer/model) to disk using joblib."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(obj, file_path)
        logging.info(f"Artifact successfully saved to: {file_path}")
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """Deserializes and loads a saved pipeline asset back into active memory using joblib."""
    try:
        logging.info(f"Attempting to load artifact from: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Target object file missing at path: {file_path}")
            
        return joblib.load(file_path)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_single_model(X_train, y_train, X_test, y_test, model):
    """Fits and evaluates a single definitive model instance with no heavy loops."""
    try:
        logging.info(f"Training definitive production instance: {type(model).__name__}")
        
        # 1. Train the single model
        model.fit(X_train, y_train)
        
        # 2. Generate predictions and probabilities
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1]
        
        # 3. Compute our core metrics
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')
        test_recall = recall_score(y_test, y_test_pred)
        test_auc = roc_auc_score(y_test, y_test_proba)
        
        logging.info(f"Final Model Results -> Accuracy: {test_accuracy:.4f} | F1: {test_f1:.4f} | Recall: {test_recall:.4f} | ROC-AUC: {test_auc:.4f}")
        
        return test_auc, test_accuracy
        
    except Exception as e:
        raise CustomException(e, sys)
