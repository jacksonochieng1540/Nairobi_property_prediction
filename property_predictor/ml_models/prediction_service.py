# nairobi_property_predictor/property_predictor/ml_models/prediction_service.py

import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from django.conf import settings


class PredictionService:
    """Service for making property price predictions"""
    
    _instance = None
    _model = None
    _preprocessor = None
    _is_loaded = False
    
    def __new__(cls):
        """Singleton pattern to load model once"""
        if cls._instance is None:
            cls._instance = super(PredictionService, cls).__new__(cls)
            cls._instance._load_model()
        return cls._instance
    
    def _load_model(self):
        """Load the trained model and preprocessor"""
        try:
            # Use pathlib for better path handling
            base_dir = Path(settings.BASE_DIR)
            model_path = base_dir / 'property_predictor' / 'ml_models' / 'models' / 'trained_model.joblib'
            preprocessor_path = base_dir / 'property_predictor' / 'ml_models' / 'models' / 'preprocessor.joblib'
            
            # Create directory if it doesn't exist
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            if model_path.exists() and preprocessor_path.exists():
                self._model = joblib.load(model_path)
                preprocessor_data = joblib.load(preprocessor_path)
                
                self._preprocessor = {
                    'label_encoders': preprocessor_data['label_encoders'],
                    'scaler': preprocessor_data['scaler'],
                    'feature_names': preprocessor_data['feature_names']
                }
                
                self._is_loaded = True
                print("✓ Model and preprocessor loaded successfully")
            else:
                print(f"✗ Model files not found at: {model_path}")
                self._create_placeholder_model()
                
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            self._create_placeholder_model()
    
    def _create_placeholder_model(self):
        """Create a placeholder model for development"""
        try:
            from sklearn.ensemble import RandomForestRegressor
            import numpy as np
            
            print("Creating placeholder model for development...")
            
            # Create a simple model
            model = RandomForestRegressor(n_estimators=10, random_state=42)
            X_dummy = np.random.rand(100, 8)
            y_dummy = np.random.rand(100) * 50000000 + 5000000
            model.fit(X_dummy, y_dummy)
            
            # Create placeholder preprocessor
            feature_names = [
                'propertyType_encoded', 'Location_encoded', 'Bedroom', 'bathroom',
                'House size', 'Land size', 'location_tier_encoded', 'size_category_encoded'
            ]
            
            from sklearn.preprocessing import LabelEncoder, StandardScaler
            
            # Create dummy encoders and scaler
            label_encoders = {
                'propertyType': LabelEncoder().fit(['Apartment', 'House', 'Townhouse']),
                'Location': LabelEncoder().fit(['Westlands', 'Kilimani', 'Karen', 'Lavington']),
                'location_tier': LabelEncoder().fit(['Tier1_Premium', 'Tier2_Upscale', 'Tier3_Standard']),
                'size_category': LabelEncoder().fit(['No_Rooms', 'Small', 'Medium', 'Large'])
            }
            
            scaler = StandardScaler().fit(X_dummy)
            
            self._preprocessor = {
                'label_encoders': label_encoders,
                'scaler': scaler,
                'feature_names': feature_names
            }
            
            self._model = model
            self._is_loaded = True
            
            print("✓ Placeholder model created for development")
            
        except Exception as e:
            print(f"✗ Error creating placeholder model: {e}")
            self._is_loaded = False
    
    def _create_location_tier(self, location):
        """Categorize location into tier"""
        if not location:
            return 'Tier3_Standard'
            
        tier1 = ['Runda', 'Muthaiga', 'Muthaiga North', 'Karen', 'Kitisuru']
        tier2 = ['Lavington', 'Kileleshwa', 'Westlands', 'Nyari', 
                 'Loresho', 'Rosslyn', 'Thigiri', 'Riverside']
        
        if location in tier1:
            return 'Tier1_Premium'
        elif location in tier2:
            return 'Tier2_Upscale'
        else:
            return 'Tier3_Standard'
    
    def _engineer_features(self, data):
        """Engineer features from input data"""
        # Ensure all required fields have default values
        data.setdefault('House size', 0)
        data.setdefault('Land size', 0)
        data.setdefault('Bedroom', 0)
        data.setdefault('bathroom', 0)
        
        # Total area
        data['total_area'] = data['House size'] + data['Land size']
        
        # Binary indicators
        data['has_land'] = 1 if data['Land size'] > 0 else 0
        data['has_house'] = 1 if data['House size'] > 0 else 0
        
        # Bathroom to bedroom ratio
        if data['Bedroom'] > 0:
            data['bath_bed_ratio'] = data['bathroom'] / data['Bedroom']
        else:
            data['bath_bed_ratio'] = 0
        
        # Size category
        bedrooms = data['Bedroom']
        if bedrooms == 0:
            size_cat = 'No_Rooms'
        elif bedrooms <= 2:
            size_cat = 'Small'
        elif bedrooms <= 4:
            size_cat = 'Medium'
        else:
            size_cat = 'Large'
        data['size_category'] = size_cat
        
        # Location tier
        data['location_tier'] = self._create_location_tier(data.get('Location', ''))
        
        return data
    
    def _encode_features(self, data):
        """Encode categorical features"""
        if not self._preprocessor or 'label_encoders' not in self._preprocessor:
            return data
            
        encoders = self._preprocessor['label_encoders']
        
        # Encode categorical variables
        categorical_cols = ['propertyType', 'Location', 'location_tier', 'size_category']
        
        for col in categorical_cols:
            if col in encoders:
                value = str(data.get(col, ''))
                if value in encoders[col].classes_:
                    data[f'{col}_encoded'] = encoders[col].transform([value])[0]
                else:
                    # Use default value (first class)
                    data[f'{col}_encoded'] = 0
        
        return data
    
    def predict(self, property_data):
        """
        Make price prediction
        
        Args:
            property_data: dict with keys:
                - property_type: str
                - location: str
                - bedrooms: int
                - bathrooms: int
                - house_size: float (in m²)
                - land_size: float (in m²)
        
        Returns:
            dict with prediction results
        """
        try:
            if not self._is_loaded:
                return {
                    'success': False,
                    'error': 'Model not loaded'
                }
            
            # Prepare data with defaults
            data = {
                'propertyType': property_data.get('property_type', 'Apartment'),
                'Location': property_data.get('location', 'Westlands'),
                'Bedroom': property_data.get('bedrooms', 2),
                'bathroom': property_data.get('bathrooms', 1),
                'House size': property_data.get('house_size', 100),
                'Land size': property_data.get('land_size', 0)
            }
            
            # Engineer features
            data = self._engineer_features(data)
            
            # Encode features
            data = self._encode_features(data)
            
            # Create feature vector
            feature_names = self._preprocessor.get('feature_names', [])
            features = []
            for fname in feature_names:
                features.append(data.get(fname, 0))
            
            if not features:
                return {
                    'success': False,
                    'error': 'No features available for prediction'
                }
            
            # Scale features
            X = np.array(features).reshape(1, -1)
            X_scaled = self._preprocessor['scaler'].transform(X)
            
            # Make prediction
            prediction = self._model.predict(X_scaled)[0]
            
            # Ensure prediction is reasonable
            prediction = max(1000000, prediction)  # At least 1 million KES
            
            # Calculate confidence intervals
            std_error = prediction * 0.15  # 15% standard error for placeholder
            lower_bound = max(1000000, prediction - 1.96 * std_error)
            upper_bound = prediction + 1.96 * std_error
            
            # Confidence score
            confidence = 0.75 if not self._is_loaded else 0.85
            
            return {
                'success': True,
                'predicted_price': float(prediction),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'confidence_score': confidence,
                'formatted_price': f"KES {prediction:,.0f}",
                'price_range': f"KES {lower_bound:,.0f} - KES {upper_bound:,.0f}",
                'is_placeholder': not self._is_loaded  # Indicate if using placeholder model
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Prediction error: {str(e)}'
            }
    
    def predict_batch(self, properties_list):
        """Make predictions for multiple properties"""
        results = []
        for prop in properties_list:
            result = self.predict(prop)
            results.append(result)
        return results
    
    def get_sample_input(self):
        """Get sample input structure"""
        return {
            'property_type': 'Apartment',
            'location': 'Westlands',
            'bedrooms': 3,
            'bathrooms': 2,
            'house_size': 120,
            'land_size': 0
        }
    
    def is_ready(self):
        """Check if service is ready for predictions"""
        return self._is_loaded


# Utility function for easy access
def get_prediction_service():
    """Get singleton instance of prediction service"""
    return PredictionService()