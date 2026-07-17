import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    """Config dataclass locking the output destination path of the serialized final model binary."""
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl") 
 
class ModelTrainer:
    def __init__(self):
        # Automatically instantiates configuration paths on runtime initialization
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """
        Ingests processed multi-dimensional arrays, separates variables from 
        the target column, and trains a production instance of Logistic Regression.
        """
        try:
            logging.info("Splitting transformed training and testing input data arrays into features and target")
            
            # Slice the final vertical column (-1) out to isolate your Churn target vector labels
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )
            
            logging.info("Initializing the definitive winning Logistic Regression architecture")
            # Production hyperparameter configuration established during our EDA benchmark evaluations
            model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                C=1.0
            )
            
            logging.info("Fitting the production classifier instance fully on training matrices")
            model.fit(X_train, y_train)
            
            # Generate predictions and raw probability continuous scores on the unseen testing set
            y_test_pred = model.predict(X_test)
            y_test_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate classification metrics to verify real-world project health
            test_accuracy = accuracy_score(y_test, y_test_pred)
            test_f1 = f1_score(y_test, y_test_pred, average='weighted')
            test_precision = precision_score(y_test, y_test_pred, zero_division=0)
            test_recall = recall_score(y_test, y_test_pred, zero_division=0)
            test_roc_auc = roc_auc_score(y_test, y_test_proba)
            
            logging.info(f"=== COMPONENT RESULTS FOR LOGISTIC REGRESSION ===")
            logging.info(f" - Accuracy:      {test_accuracy:.4f}")
            logging.info(f" - F1-Score:      {test_f1:.4f}")
            logging.info(f" - Precision:     {test_precision:.4f}")
            logging.info(f" - Recall:        {test_recall:.4f} (Core Retention Catch Rate)")
            logging.info(f" - ROC-AUC Score: {test_roc_auc:.4f}")
            logging.info(f"==================================================")
            
            # Safeguard Gate: Ensures that if data drops unexpectedly during scaling, it flags before saving
            if test_roc_auc < 0.60:
                raise CustomException("Model predictive performance is below acceptable business thresholds.")
                
            logging.info(f"Model successfully passed validation thresholds. Saving artifact...")
            
            # Persist your trained classification file directly into the artifacts directory
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )
            
            # Return the leading score metric back to your orchestration layer
            return test_roc_auc, test_accuracy

        except Exception as e:
            logging.error("Exception occurred during the model training pipeline component execution")
            raise CustomException(e, sys)
