# FILE LOCATION: train.py (Put this in your root project folder)
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.Model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

def run_training_pipeline():
    try:
        logging.info("==================================================")
        logging.info("🎬 INITIATING COMPLETE TELCO CHURN PRODUCTION RUN")
        logging.info("==================================================")
        
        # 1. Component Step A: Ingest and split raw files safely
        ingestion = DataIngestion()
        train_csv_path, test_csv_path = ingestion.initiate_data_ingestion()
        
        # 2. Component Step B: Clean features and build preprocessor matrix
        transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = transformation.initiate_data_tranformation(
            train_path=train_csv_path, 
            test_path=test_csv_path
        )
        logging.info(f"Preprocessor tracking artifact successfully saved at: {preprocessor_path}")

        
        # 3. Component Step C: Train and validate your final Logistic Regression model
        trainer = ModelTrainer()
        final_roc_auc, final_accuracy = trainer.initiate_model_trainer(
            train_array=train_arr, 
            test_array=test_arr)
        

        logging.info("🏁 PRODUCTION PIPELINE EXECUTION SUCCESSFUL!")
        logging.info(f"   ↳ Serialized Artifacts Locked in 'artifacts/'")
        logging.info(f"   ↳ Definitive Model Validation AUC: {final_roc_auc:.4f}")

        
    except Exception as e:
        logging.error("CRITICAL CRASH: Pipeline workflow failed at runtime.")
        raise CustomException(e, sys)

if __name__ == "__main__":
    run_training_pipeline()
