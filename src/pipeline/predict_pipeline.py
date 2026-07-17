# FILE LOCATION: src/pipeline/predict_pipeline.py
import sys
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        # Cache file paths to avoid I/O performance bottlenecks
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        self.model_path = os.path.join("artifacts", "model.pkl")

    def predict_live_input(self, features_df: pd.DataFrame) -> tuple:
        """
        Loads preprocessor and model binaries, executes data transformation,
        and calculates customer churn probability scores.
        """
        try:
            logging.info("Loading production preprocessor and model binaries into RAM...")
            preprocessor = load_object(file_path=self.preprocessor_path)
            model = load_object(file_path=self.model_path)
            
            logging.info("Executing parallel data transformation workflows...")
            processed_data = preprocessor.transform(features_df)
            
            logging.info("Calculating final customer attrition probability metrics...")
            prediction = model.predict(processed_data)
            probability = model.predict_proba(processed_data)[:, 1]
            
            return prediction, probability
            
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    Maps user interface inputs into a structured 17-column DataFrame.
    Note: customerID, PhoneService, and gender are completely removed.
    """
    def __init__(self, SeniorCitizen: int, Partner: str, Dependents: str, tenure: int, 
                 MultipleLines: str, InternetService: str, OnlineSecurity: str, 
                 OnlineBackup: str, DeviceProtection: str, TechSupport: str, 
                 StreamingTV: str, StreamingMovies: str, Contract: str, 
                 PaperlessBilling: str, PaymentMethod: str, MonthlyCharges: float, 
                 TotalCharges: float):
        
        # Demographic Profiles
        self.SeniorCitizen = SeniorCitizen
        self.Partner = Partner
        self.Dependents = Dependents
        
        # Account Longevity & Core Service Matrix
        self.tenure = tenure
        self.MultipleLines = MultipleLines
        self.InternetService = InternetService
        
        # Digital Ecosystem Add-ons
        self.OnlineSecurity = OnlineSecurity
        self.OnlineBackup = OnlineBackup
        self.DeviceProtection = DeviceProtection
        self.TechSupport = TechSupport
        self.StreamingTV = StreamingTV
        self.StreamingMovies = StreamingMovies
        
        # Contract Billing Infrastructure Parameters
        self.Contract = Contract
        self.PaperlessBilling = PaperlessBilling
        self.PaymentMethod = PaymentMethod
        self.MonthlyCharges = MonthlyCharges
        self.TotalCharges = TotalCharges

    def get_data_as_data_frame(self) -> pd.DataFrame:
        """Assembles variables into the clean 17-column DataFrame layout required by the preprocessor."""
        try:
            custom_data_input_dict = {
                "SeniorCitizen": [self.SeniorCitizen],
                "Partner": [self.Partner],
                "Dependents": [self.Dependents],
                "tenure": [self.tenure],
                "MultipleLines": [self.MultipleLines],
                "InternetService": [self.InternetService],
                "OnlineSecurity": [self.OnlineSecurity],
                "OnlineBackup": [self.OnlineBackup],
                "DeviceProtection": [self.DeviceProtection],
                "TechSupport": [self.TechSupport],
                "StreamingTV": [self.StreamingTV],
                "StreamingMovies": [self.StreamingMovies],
                "Contract": [self.Contract],
                "PaperlessBilling": [self.PaperlessBilling],
                "PaymentMethod": [self.PaymentMethod],
                "MonthlyCharges": [self.MonthlyCharges],
                "TotalCharges": [self.TotalCharges]
            }
            return pd.DataFrame(custom_data_input_dict)
            
        except Exception as e:
            raise CustomException(e, sys)


# =====================================================================
# INTEGRATED TERMINAL STANDALONE TEST BENCH
# =====================================================================
if __name__ == "__main__":
    print("🎬 Initiating Terminal Test of the Prediction Pipeline...")
    
    # Check if artifacts exist before trying to load them
    if not os.path.exists(os.path.join("artifacts", "model.pkl")):
        print("❌ Error: Could not find 'artifacts/model.pkl'. Run 'python -m src.pipeline.train_pipeline' first!")
    else:
        try:
            # 1. Instantiate the object with exactly 17 matching parameters
            simulated_customer = CustomData(
                SeniorCitizen=0, 
                Partner="No", 
                Dependents="No",
                tenure=2, 
                MultipleLines="No", 
                InternetService="Fiber optic",
                OnlineSecurity="No", 
                OnlineBackup="No", 
                DeviceProtection="No", 
                TechSupport="No",
                StreamingTV="Yes", 
                StreamingMovies="Yes", 
                Contract="Month-to-month", 
                PaperlessBilling="Yes", 
                PaymentMethod="Electronic check", 
                MonthlyCharges=89.50, 
                TotalCharges=179.00
            )
            
            # 2. Build the structured test row layout DataFrame
            raw_df = simulated_customer.get_data_as_data_frame()
            
            # 3. Trigger pipeline prediction execution
            pipeline = PredictPipeline()
            predicted_class, predicted_probability = pipeline.predict_live_input(raw_df)
            
            # 4. Render output block cleanly
            print("\n" + "="*50)
            print("🔮 LIVE MODEL PREDICTION OUTPUT RESULT")
            print("="*50)
            print(f"📊 Raw Predicted Class array: {predicted_class}")
            print(f"📈 Raw Attrition Probability:  {predicted_probability[0]*100:.2f}%")
            
            status = "🚨 HIGH CHURN RISK" if predicted_class[0] == 1 else "✅ STABLE CUSTOMER"
            print(f"📌 Risk Classification:       {status}")
            print("="*50 + "\n")
            
        except Exception as e:
            print(f"❌ Execution crashed with error: {e}")
