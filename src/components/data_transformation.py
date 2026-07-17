import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from sklearn.base import BaseEstimator, TransformerMixin
from src.logger import logging
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OrdinalEncoder,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.utils._set_output import _SetOutputMixin
from src.utils import save_object
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")
class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin,_SetOutputMixin):
    def __init__(self):
        super().__init__() 
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            df_eng = X.copy()
            # 1.Custom data reduction
            noise_columns = ["customerID","gender","PhoneService"]
            df_eng = df_eng.drop(columns =noise_columns, errors = "ignore")
            # 2.Fix missing values in TotalCharge column
            if 'TotalCharges' in df_eng.columns:
                df_eng['TotalCharges'] = pd.to_numeric(df_eng['TotalCharges'], errors='coerce')
                df_eng['TotalCharges'] = df_eng['TotalCharges'].fillna(0)
            
            # 3.Dynamic Feature Synthesis
    
            # Feature 1: Financial Burden Ratio (Monthly cost footprint over life of account)
            df_eng['MonthlyToTotalRatio'] = df_eng['MonthlyCharges'] / (df_eng['TotalCharges'] + 1e-5)

            # Feature 2: Tech Ecosystem Adoption Count (Number of sticky security layers active)
            security_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
            existing_services = [col for col in security_services if col in df_eng.columns]
            if existing_services:
                df_eng['EcosystemFeaturesCount'] = df_eng[existing_services].apply(
                    lambda row: (row == 'Yes').sum(), axis=1
                )
            else:
                df_eng['EcosystemFeaturesCount'] = 0

            # Feature 3: High-Risk Persona Flag (Isolates the premium month-to-month pricing wall)
            if 'Contract' in df_eng.columns and 'TechSupport' in df_eng.columns:
                df_eng['Is_High_Risk_Persona'] = (
                    (df_eng['Contract'] == 'Month-to-month') &
                    (df_eng['TechSupport'] == 'No') &
                    (df_eng['MonthlyCharges'] >= 70)
                ).astype(int)
            else:
                df_eng["Is_High_Risk_Persona"]=0
            return df_eng

    
        except Exception as e :
            raise CustomException(e,sys)

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    def get_data_transformation_obj(self,X_train_sample):
        try:
            feature_step = FeatureEngineeringTransformer()
            df_feature_sample = feature_step.transform(X_train_sample)
            contract_categories = ['Month-to-month', 'One year', 'Two year']
            all_categorical = [
                'Partner', 'Dependents', 'PaperlessBilling', 'InternetService', 
                'PaymentMethod', 'OnlineSecurity', 'OnlineBackup', 'MultipleLines',
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
            ]
            categorical_columns = [col for col in all_categorical if col in df_feature_sample.columns]
            all_numerical = [
                'tenure', 'MonthlyCharges', 'TotalCharges', 
                'MonthlyToTotalRatio', 'EcosystemFeaturesCount'
            ]
            numerical_columns = [col for col in all_numerical if col in df_feature_sample.columns]
             # 2. Parallel processing structure
            preprocessor_lanes = ColumnTransformer(
                transformers=[
                    ("Contract_Ordinal",OrdinalEncoder(
                        categories=[contract_categories], handle_unknown='use_encoded_value', unknown_value=-1),
                    ['Contract'] if 'Contract' in df_feature_sample.columns else []),
                    ("Categorical_OHE",OneHotEncoder(
                        drop='first',handle_unknown='ignore',sparse_output=False),categorical_columns),
                    ("Numerical_scaler",StandardScaler(),numerical_columns)                    
                ],
                remainder='passthrough'
            )
            # 3. Ultimate Integration Pipeline
            final_pipeline = Pipeline(steps=[
                ("Feature_Engineering",feature_step),
                ("Data_preprocessor",preprocessor_lanes)
            ])#.set_output(transform='pandas')
            
            return final_pipeline
        except Exception as e:
            raise CustomException(e,sys)
    def initiate_data_tranformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed within transformation module")
            target_column_name = "Churn"
            # Segregate target variables explicitly using manual map logic
            y_train = train_df[target_column_name].map({'Yes': 1, 'No': 0})
            X_train = train_df.drop(columns=[target_column_name], errors='ignore')
            
            y_test = test_df[target_column_name].map({'Yes': 1, 'No': 0})
            X_test = test_df.drop(columns=[target_column_name], errors='ignore')
            preprocessing_pipeline_obj = self.get_data_transformation_obj(X_train)
            logging.info("Applying full pipeline preprocessing transformation on training data and training data")
            X_train_processed = preprocessing_pipeline_obj.fit_transform(X_train)
            X_test_processed = preprocessing_pipeline_obj.transform(X_test)
             # Stitch matrices together back into optimized numpy runtime arrays cleanly
            train_arr = np.c_[X_train_processed, np.array(y_train)]
            test_arr = np.c_[X_test_processed, np.array(y_test)]
            logging.info("Saved Preprocessing Object.")
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_pipeline_obj
            )
            return (train_arr,
            test_arr,
            self.data_transformation_config.preprocessor_obj_file_path)
        except Exception as e:
            raise CustomException(e, sys)
        
    
        

                
            