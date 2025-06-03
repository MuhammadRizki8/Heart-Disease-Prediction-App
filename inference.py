"""
Heart Disease Prediction Inference System
==========================================
Sistem inferensi untuk prediksi penyakit jantung menggunakan model yang telah dilatih.
Mendukung berbagai model (RandomForest, LogisticRegression, SVM) dan logging untuk monitoring.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Setup logging untuk monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('inference.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HeartDiseasePredictor:
    """
    Kelas untuk prediksi penyakit jantung
    """
    
    def __init__(self, model_dir: str = "outputs"):
        """
        Inisialisasi predictor
        
        Args:
            model_dir: Direktori tempat model dan artefak disimpan
        """
        self.model_dir = model_dir
        self.models = {}
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = []
        self.preprocessing_params = {}
        
        # Load all necessary components
        self._load_components()
        logger.info("HeartDiseasePredictor initialized successfully")
    
    def _load_components(self):
        """Load semua komponen yang diperlukan untuk inferensi"""
        try:
            # Load models
            model_files = {
                'RandomForest': 'model_randomforest.joblib',
                'LogisticRegression': 'model_logisticregression.joblib',
                'SVM': 'model_svm.joblib'
            }
            
            for model_name, filename in model_files.items():
                model_path = os.path.join(self.model_dir, filename)
                if os.path.exists(model_path):
                    self.models[model_name] = joblib.load(model_path)
                    logger.info(f"Loaded {model_name} model")
                else:
                    logger.warning(f"Model file {filename} not found")
            
            # Load scaler
            scaler_path = os.path.join(self.model_dir, 'scaler.joblib')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info("Loaded scaler")
            
            # Load label encoders
            encoders_path = os.path.join(self.model_dir, 'label_encoders.joblib')
            if os.path.exists(encoders_path):
                self.label_encoders = joblib.load(encoders_path)
                logger.info("Loaded label encoders")
            
            # Load feature names
            features_path = os.path.join(self.model_dir, 'feature_names.json')
            if os.path.exists(features_path):
                with open(features_path, 'r') as f:
                    self.feature_names = json.load(f)
                logger.info(f"Loaded {len(self.feature_names)} feature names")
            
            # Load preprocessing parameters
            params_path = os.path.join(self.model_dir, 'preprocessing_params.json')
            if os.path.exists(params_path):
                with open(params_path, 'r') as f:
                    self.preprocessing_params = json.load(f)
                logger.info("Loaded preprocessing parameters")
                
        except Exception as e:
            logger.error(f"Error loading components: {str(e)}")
            raise
    
    def _validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validasi input data
        
        Args:
            input_data: Dictionary berisi data input
            
        Returns:
            bool: True jika valid, False jika tidak
        """
        required_fields = [
            'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
            'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 
            'Oldpeak', 'ST_Slope'
        ]
        
        # Cek apakah semua field required ada
        missing_fields = [field for field in required_fields if field not in input_data]
        if missing_fields:
            logger.error(f"Missing required fields: {missing_fields}")
            return False
        
        # Validasi range nilai
        validations = {
            'Age': (28, 77),
            'RestingBP': (94, 200),
            'Cholesterol': (85, 603),
            'MaxHR': (60, 202),
            'Oldpeak': (-2.6, 6.2),
            'FastingBS': (0, 1)
        }
        
        for field, (min_val, max_val) in validations.items():
            if field in input_data:
                value = input_data[field]
                if not (min_val <= value <= max_val):
                    logger.error(f"{field} value {value} is out of range [{min_val}, {max_val}]")
                    return False
        
        # Validasi kategorikal
        categorical_validations = {
            'Sex': ['M', 'F'],
            'ChestPainType': ['ATA', 'NAP', 'ASY', 'TA'],
            'RestingECG': ['Normal', 'ST', 'LVH'],
            'ExerciseAngina': ['Y', 'N'],
            'ST_Slope': ['Up', 'Flat', 'Down']
        }
        
        for field, valid_values in categorical_validations.items():
            if field in input_data:
                value = input_data[field]
                if value not in valid_values:
                    logger.error(f"{field} value '{value}' not in valid values: {valid_values}")
                    return False
        
        return True
    
    def _preprocess_input(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocess input data sesuai dengan training pipeline
        
        Args:
            input_data: Dictionary berisi data input
            
        Returns:
            pd.DataFrame: Data yang sudah dipreprocess
        """
        # Convert to DataFrame
        df = pd.DataFrame([input_data])
        
        # One-hot encoding untuk kategorikal multi-class
        df_encoded = pd.get_dummies(
            df, 
            columns=['ChestPainType', 'RestingECG', 'ST_Slope'],
            prefix=['ChestPain', 'RestingECG', 'ST_Slope']
        )
        
        # Label encoding untuk binary variables
        binary_columns = ['Sex', 'ExerciseAngina']
        for col in binary_columns:
            if col in df_encoded.columns and col in self.label_encoders:
                df_encoded[col] = self.label_encoders[col].transform(df_encoded[col])
        
        # Pastikan semua kolom training ada
        for col in self.feature_names:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        
        # Reorder kolom sesuai training
        df_encoded = df_encoded[self.feature_names]
        
        # Scale numerical features
        numerical_columns = self.preprocessing_params.get('numerical_columns', 
                                                         ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak'])
        
        if self.scaler is not None:
            df_scaled = df_encoded.copy()
            df_scaled[numerical_columns] = self.scaler.transform(df_encoded[numerical_columns])
            return df_scaled
        
        return df_encoded
    
    def predict(self, input_data: Dict[str, Any], model_name: str = 'RandomForest') -> Dict[str, Any]:
        """
        Melakukan prediksi penyakit jantung
        
        Args:
            input_data: Dictionary berisi data pasien
            model_name: Nama model yang digunakan
            
        Returns:
            Dictionary berisi hasil prediksi dan probabilitas
        """
        start_time = datetime.now()
        
        try:
            # Validasi input
            if not self._validate_input(input_data):
                raise ValueError("Invalid input data")
            
            # Cek apakah model tersedia
            if model_name not in self.models:
                available_models = list(self.models.keys())
                raise ValueError(f"Model '{model_name}' not available. Available models: {available_models}")
            
            model = self.models[model_name]
            
            # Preprocess input
            processed_data = self._preprocess_input(input_data)
            
            # Prediksi
            prediction = model.predict(processed_data)[0]
            
            # Probabilitas (jika model mendukung)
            probability = None
            if hasattr(model, 'predict_proba'):
                prob_array = model.predict_proba(processed_data)[0]
                probability = {
                    'No Heart Disease': float(prob_array[0]),
                    'Heart Disease': float(prob_array[1])
                }
            
            # Hitung waktu inferensi
            inference_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'prediction': int(prediction),
                'prediction_label': 'Heart Disease' if prediction == 1 else 'No Heart Disease',
                'probability': probability,
                'model_used': model_name,
                'inference_time_seconds': inference_time,
                'timestamp': datetime.now().isoformat()
            }
            
            # Log hasil prediksi
            logger.info(f"Prediction completed - Model: {model_name}, "
                       f"Result: {result['prediction_label']}, "
                       f"Time: {inference_time:.4f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise
    
    def predict_batch(self, batch_data: list, model_name: str = 'RandomForest') -> list:
        """
        Prediksi batch untuk multiple samples
        
        Args:
            batch_data: List berisi dictionary data pasien
            model_name: Nama model yang digunakan
            
        Returns:
            List berisi hasil prediksi untuk setiap sample
        """
        results = []
        for i, data in enumerate(batch_data):
            try:
                result = self.predict(data, model_name)
                result['sample_id'] = i
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing sample {i}: {str(e)}")
                results.append({
                    'sample_id': i,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Mendapatkan informasi tentang model yang tersedia
        
        Returns:
            Dictionary berisi informasi model
        """
        model_results_path = os.path.join(self.model_dir, 'model_results.json')
        model_info = {
            'available_models': list(self.models.keys()),
            'feature_count': len(self.feature_names),
            'feature_names': self.feature_names
        }
        
        if os.path.exists(model_results_path):
            with open(model_results_path, 'r') as f:
                model_results = json.load(f)
                model_info['model_performance'] = model_results
        
        return model_info

def main():
    """
    Fungsi utama untuk testing dan contoh penggunaan
    """
    print("="*60)
    print("HEART DISEASE PREDICTION INFERENCE SYSTEM")
    print("="*60)
    
    # Initialize predictor
    try:
        predictor = HeartDiseasePredictor()
        print("✅ Predictor initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing predictor: {e}")
        return
    
    # Print model info
    model_info = predictor.get_model_info()
    print(f"\n📊 Available Models: {model_info['available_models']}")
    print(f"📊 Feature Count: {model_info['feature_count']}")
    
    # Test cases
    test_cases = [
        {
            'name': 'High Risk Patient',
            'data': {
                'Age': 65,
                'Sex': 'M',
                'ChestPainType': 'ASY',
                'RestingBP': 140,
                'Cholesterol': 250,
                'FastingBS': 1,
                'RestingECG': 'ST',
                'MaxHR': 120,
                'ExerciseAngina': 'Y',
                'Oldpeak': 2.0,
                'ST_Slope': 'Flat'
            }
        },
        {
            'name': 'Low Risk Patient',
            'data': {
                'Age': 35,
                'Sex': 'F',
                'ChestPainType': 'ATA',
                'RestingBP': 120,
                'Cholesterol': 200,
                'FastingBS': 0,
                'RestingECG': 'Normal',
                'MaxHR': 180,
                'ExerciseAngina': 'N',
                'Oldpeak': 0.0,
                'ST_Slope': 'Up'
            }
        }
    ]
    
    # Test dengan berbagai model
    for model_name in predictor.models.keys():
        print(f"\n🔬 TESTING WITH {model_name.upper()}")
        print("-" * 50)
        
        for test_case in test_cases:
            try:
                result = predictor.predict(test_case['data'], model_name)
                
                print(f"\n👤 {test_case['name']}:")
                print(f"   Prediction: {result['prediction_label']}")
                if result['probability']:
                    print(f"   Confidence: {result['probability']['Heart Disease']:.3f}")
                print(f"   Inference Time: {result['inference_time_seconds']:.4f}s")
                
            except Exception as e:
                print(f"❌ Error predicting {test_case['name']}: {e}")
    
    # Test batch prediction
    print(f"\n🔄 TESTING BATCH PREDICTION")
    print("-" * 50)
    
    batch_data = [case['data'] for case in test_cases]
    batch_results = predictor.predict_batch(batch_data, 'RandomForest')
    
    for i, result in enumerate(batch_results):
        if 'error' not in result:
            print(f"Sample {i}: {result['prediction_label']} "
                  f"(Time: {result['inference_time_seconds']:.4f}s)")
        else:
            print(f"Sample {i}: Error - {result['error']}")

if __name__ == "__main__":
    main()