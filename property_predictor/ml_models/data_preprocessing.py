import pandas as pd
import numpy as np
import re
import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')


class PropertyDataPreprocessor:
    """Comprehensive data preprocessing for Nairobi property prices"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def clean_price(self, price_str):
        """Extract numeric price from string format"""
        if pd.isna(price_str):
            return np.nan
        
        # Remove 'KSh', 'Ksh', spaces, and commas
        price_str = str(price_str).replace('KSh', '').replace('Ksh', '')
        price_str = price_str.replace(' ', '').replace(',', '')
        
        try:
            return float(price_str)
        except:
            return np.nan
    
    def clean_area(self, area_str):
        """Extract numeric area from string format"""
        if pd.isna(area_str) or area_str == '':
            return np.nan
        
        area_str = str(area_str)
        
        # Extract number from patterns like "230 m²", "0.5 acres"
        numbers = re.findall(r'[\d.]+', area_str)
        
        if not numbers:
            return np.nan
        
        value = float(numbers[0])
        
        # Convert acres to square meters (1 acre = 4046.86 m²)
        if 'acre' in area_str.lower():
            value = value * 4046.86
        
        return value
    
    def fix_property_type_typos(self, prop_type):
        """Fix common typos in property types"""
        if pd.isna(prop_type):
            return 'Unknown'
        
        prop_type = str(prop_type).strip()
        
        # Fix typos
        typo_map = {
            'Townhuse': 'Townhouse',
            'townhouse': 'Townhouse',
            'apartment': 'Apartment',
        }
        
        return typo_map.get(prop_type, prop_type)
    
    def create_location_tier(self, location):
        """Categorize locations into tiers based on prestige"""
        if pd.isna(location):
            return 'Tier3'
        
        location = str(location).strip()
        
        # Premium locations
        tier1 = ['Runda', 'Muthaiga', 'Muthaiga North', 'Karen', 'Kitisuru']
        tier2 = ['Lavington', 'Kileleshwa', 'Westlands', 'Nyari', 
                 'Loresho', 'Rosslyn', 'Thigiri', 'Riverside']
        
        if location in tier1:
            return 'Tier1_Premium'
        elif location in tier2:
            return 'Tier2_Upscale'
        else:
            return 'Tier3_Standard'
    
    def engineer_features(self, df):
        """Create additional features from existing data"""
        df = df.copy()
        
        # Price per square meter (for properties with house size)
        df['price_per_sqm'] = df.apply(
            lambda row: row['Price'] / row['House size'] 
            if pd.notna(row['House size']) and row['House size'] > 0 
            else np.nan, 
            axis=1
        )
        
        # Total area (combine house and land)
        df['total_area'] = df[['House size', 'Land size']].sum(axis=1)
        df['has_land'] = (df['Land size'] > 0).astype(int)
        df['has_house'] = (df['House size'] > 0).astype(int)
        
        # Bathroom to bedroom ratio
        df['bath_bed_ratio'] = df.apply(
            lambda row: row['bathroom'] / row['Bedroom'] 
            if pd.notna(row['Bedroom']) and row['Bedroom'] > 0 
            else np.nan,
            axis=1
        )
        
        # Property size category
        df['size_category'] = pd.cut(
            df['Bedroom'].fillna(0), 
            bins=[-1, 0, 2, 4, 10], 
            labels=['No_Rooms', 'Small', 'Medium', 'Large']
        )
        
        # Location tier
        df['location_tier'] = df['Location'].apply(self.create_location_tier)
        
        return df
    
    def find_data_file(self, csv_path):
        """Find the data file in various possible locations"""
        possible_paths = [
            csv_path,
            Path(csv_path),
            Path('data/raw/nairobi_properties.csv'),
            Path('../data/raw/nairobi_properties.csv'),
            Path('../../data/raw/nairobi_properties.csv'),
            Path(__file__).parent.parent.parent / 'data' / 'raw' / 'nairobi_properties.csv',
        ]
        
        for path in possible_paths:
            if isinstance(path, str):
                path = Path(path)
            if path.exists():
                print(f"Found data file at: {path}")
                return str(path)
        
        # If no file found, create directory and provide instructions
        data_dir = Path('data/raw')
        data_dir.mkdir(parents=True, exist_ok=True)
        
        raise FileNotFoundError(f"""
        Cannot find nairobi_properties.csv!
        
        Expected locations:
        - data/raw/nairobi_properties.csv
        - {Path(__file__).parent.parent.parent / 'data' / 'raw' / 'nairobi_properties.csv'}
        
        Please ensure your CSV file exists in one of these locations.
        Current working directory: {os.getcwd()}
        """)
    
    def preprocess_data(self, csv_path, is_training=True):
        """Complete preprocessing pipeline"""
        # Find and validate data file
        actual_csv_path = self.find_data_file(csv_path)
        
        # Load data
        df = pd.read_csv(actual_csv_path)
        
        print(f"Original data shape: {df.shape}")
        
        # Clean price
        df['Price'] = df['Price'].apply(self.clean_price)
        
        # Clean areas
        df['House size'] = df['House size'].apply(self.clean_area)
        df['Land size'] = df['Land size'].apply(self.clean_area)
        
        # Fix property type typos
        df['propertyType'] = df['propertyType'].apply(self.fix_property_type_typos)
        
        # Fill missing values
        df['Bedroom'] = df['Bedroom'].fillna(0)
        df['bathroom'] = df['bathroom'].fillna(0)
        df['House size'] = df['House size'].fillna(0)
        df['Land size'] = df['Land size'].fillna(0)
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Remove rows with missing prices (target variable)
        df = df.dropna(subset=['Price'])
        
        # Remove outliers (prices beyond reasonable range)
        df = df[(df['Price'] >= 1_000_000) & (df['Price'] <= 2_000_000_000)]
        
        print(f"Cleaned data shape: {df.shape}")
        
        if is_training:
            # Encode categorical variables
            categorical_cols = ['propertyType', 'Location', 'location_tier', 'size_category']
            
            for col in categorical_cols:
                if col in df.columns:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
            
            # Select features for training
            feature_cols = [
                'propertyType_encoded', 'Location_encoded', 'Bedroom', 'bathroom',
                'House size', 'Land size', 'total_area', 'has_land', 'has_house',
                'bath_bed_ratio', 'location_tier_encoded', 'size_category_encoded'
            ]
            
            # Remove columns with all NaN
            feature_cols = [col for col in feature_cols if col in df.columns]
            
            X = df[feature_cols].fillna(0)
            y = df['Price']
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            X = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
            
            self.feature_names = feature_cols
            
            return X, y, df
        else:
            # For prediction, use fitted encoders
            categorical_cols = ['propertyType', 'Location', 'location_tier', 'size_category']
            
            for col in categorical_cols:
                if col in df.columns and col in self.label_encoders:
                    # Handle unseen categories
                    df[f'{col}_encoded'] = df[col].apply(
                        lambda x: self.label_encoders[col].transform([str(x)])[0] 
                        if str(x) in self.label_encoders[col].classes_ 
                        else 0
                    )
            
            X = df[self.feature_names].fillna(0)
            X_scaled = self.scaler.transform(X)
            X = pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)
            
            return X, df
    
    def save_preprocessor(self, path='data/models/preprocessor.joblib'):
        """Save preprocessor for later use"""
        # Create directory if it doesn't exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump({
            'label_encoders': self.label_encoders, 
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }, path)
        print(f"Preprocessor saved to {path}")
    
    def load_preprocessor(self, path='data/models/preprocessor.joblib'):
        """Load saved preprocessor"""
        data = joblib.load(path)
        self.label_encoders = data['label_encoders']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        print(f"Preprocessor loaded from {path}")


# Usage example
if __name__ == "__main__":
    preprocessor = PropertyDataPreprocessor()
    
    try:
        X, y, df = preprocessor.preprocess_data('data/raw/nairobi_properties.csv')
        
        print("\nFeatures shape:", X.shape)
        print("Target shape:", y.shape)
        print("\nFeature names:", X.columns.tolist())
        print("\nPrice statistics:")
        print(y.describe())
        
        # Save preprocessor
        preprocessor.save_preprocessor()
        
    except FileNotFoundError as e:
        print(e)
        print("\nCreating sample data file for testing...")
        
        # Create sample data directory and file
        sample_data_dir = Path('data/raw')
        sample_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample data
        sample_data = {
            'Location': ['Westlands', 'Kilimani', 'Karen', 'Lavington', 'Kileleshwa'],
            'Bedroom': [3, 2, 4, 3, 2],
            'bathroom': [2, 2, 3, 2, 1],
            'House size': [120, 90, 200, 110, 85],
            'Land size': [0, 0, 500, 0, 0],
            'Price': [15000000, 12000000, 25000000, 14000000, 10000000],
            'propertyType': ['Apartment', 'Apartment', 'House', 'Apartment', 'Apartment']
        }
        
        sample_df = pd.DataFrame(sample_data)
        sample_df.to_csv(sample_data_dir / 'nairobi_properties.csv', index=False)
        print("Sample data created at: data/raw/nairobi_properties.csv")
        print("Please replace with your actual data and run again.")


def advanced_feature_engineering(self, df):
    """Extract more meaningful features"""
    
    # 1. Geospatial features (if lat/long available)
    df['distance_to_cbd'] = self.calculate_distance_to_cbd(df)
    df['distance_to_airport'] = self.calculate_distance_to_airport(df)
    df['nearby_amenities_score'] = self.count_nearby_amenities(df)
    
    # 2. Time-based features
    df['days_since_listing'] = (datetime.now() - df['created_at']).dt.days
    df['is_weekend'] = df['created_at'].dt.dayofweek >= 5
    df['month'] = df['created_at'].dt.month
    df['quarter'] = df['created_at'].dt.quarter
    
    # 3. Property value indicators
    df['price_per_bedroom'] = df['Price'] / df['Bedroom'].clip(lower=1)
    df['price_per_bathroom'] = df['Price'] / df['bathroom'].clip(lower=1)
    df['room_efficiency'] = df['House size'] / (df['Bedroom'] + df['bathroom'])
    
    # 4. Location interaction features
    df['location_type_interaction'] = (
        df['location_tier_encoded'] * df['propertyType_encoded']
    )
    
    # 5. Polynomial features for important variables
    df['bedrooms_squared'] = df['Bedroom'] ** 2
    df['house_size_sqrt'] = np.sqrt(df['House size'])
    
    # 6. Market segment features
    df['is_luxury'] = ((df['location_tier'] == 'Tier1_Premium') & 
                       (df['Bedroom'] >= 4)).astype(int)
    df['is_affordable'] = (df['Bedroom'] <= 2).astype(int)
    
    return df
