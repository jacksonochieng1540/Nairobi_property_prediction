import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class PropertyPriceModel:
    """Ensemble model for property price prediction"""
    
    def __init__(self):
        self.models = {}
        self.ensemble_model = None
        self.best_model = None
        self.metrics = {}
        
    def initialize_models(self):
        """Initialize various regression models"""
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=7,
                min_samples_split=5,
                random_state=42
            ),
            'xgboost': XGBRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=7,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            ),
            'lightgbm': LGBMRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=7,
                num_leaves=31,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'catboost': CatBoostRegressor(
                iterations=200,
                learning_rate=0.1,
                depth=7,
                random_state=42,
                verbose=False
            ),
            'ridge': Ridge(alpha=10.0),
            'lasso': Lasso(alpha=10.0)
        }
    
    def train_individual_models(self, X_train, y_train, X_test, y_test):
        """Train each model individually and evaluate"""
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Metrics
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            
            results[name] = {
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_r2': train_r2,
                'test_r2': test_r2
            }
            
            print(f"  Train MAE: KSh {train_mae:,.0f}")
            print(f"  Test MAE: KSh {test_mae:,.0f}")
            print(f"  Test R²: {test_r2:.4f}")
        
        return results
    
    def create_ensemble(self, X_train, y_train):
        """Create voting ensemble from top models"""
        # Select top 3 models based on individual performance
        top_models = [
            ('xgboost', self.models['xgboost']),
            ('lightgbm', self.models['lightgbm']),
            ('catboost', self.models['catboost'])
        ]
        
        self.ensemble_model = VotingRegressor(
            estimators=top_models,
            n_jobs=-1
        )
        
        print("\nTraining ensemble model...")
        self.ensemble_model.fit(X_train, y_train)
        
    def evaluate_ensemble(self, X_train, y_train, X_test, y_test):
        """Evaluate ensemble model"""
        y_pred_train = self.ensemble_model.predict(X_train)
        y_pred_test = self.ensemble_model.predict(X_test)
        
        ensemble_metrics = {
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test)
        }
        
        print("\n" + "="*50)
        print("ENSEMBLE MODEL PERFORMANCE")
        print("="*50)
        print(f"Train MAE: KSh {ensemble_metrics['train_mae']:,.0f}")
        print(f"Test MAE: KSh {ensemble_metrics['test_mae']:,.0f}")
        print(f"Train RMSE: KSh {ensemble_metrics['train_rmse']:,.0f}")
        print(f"Test RMSE: KSh {ensemble_metrics['test_rmse']:,.0f}")
        print(f"Train R²: {ensemble_metrics['train_r2']:.4f}")
        print(f"Test R²: {ensemble_metrics['test_r2']:.4f}")
        
        return ensemble_metrics
    
    def get_feature_importance(self, X, feature_names):
        """Get feature importance from the best tree-based model"""
        # Use XGBoost for feature importance
        model = self.models['xgboost']
        importance = model.feature_importances_
        
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print("\n" + "="*50)
        print("TOP 10 IMPORTANT FEATURES")
        print("="*50)
        print(feature_importance_df.head(10).to_string(index=False))
        
        return feature_importance_df
    
    def save_models(self, base_path='data/models'):
        """Save all models and metrics"""
        # Save ensemble model
        ensemble_path = f'{base_path}/ensemble_model.joblib'
        joblib.dump(self.ensemble_model, ensemble_path)
        print(f"\nEnsemble model saved to {ensemble_path}")
        
        # Save individual models
        for name, model in self.models.items():
            model_path = f'{base_path}/{name}_model.joblib'
            joblib.dump(model, model_path)
        
        # Save metrics
        metrics_path = f'{base_path}/model_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        print(f"Metrics saved to {metrics_path}")
    
    def load_model(self, model_path='data/models/ensemble_model.joblib'):
        """Load saved model"""
        self.ensemble_model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
    
    def predict(self, X):
        """Make predictions using the ensemble model"""
        if self.ensemble_model is None:
            raise ValueError("Model not trained or loaded")
        
        return self.ensemble_model.predict(X)
    
    def predict_with_confidence(self, X):
        """Predict with confidence intervals using individual model predictions"""
        predictions = []
        
        for name, model in self.models.items():
            if name in ['xgboost', 'lightgbm', 'catboost']:
                pred = model.predict(X)
                predictions.append(pred)
        
        predictions = np.array(predictions)
        
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # 95% confidence interval
        lower_bound = mean_pred - 1.96 * std_pred
        upper_bound = mean_pred + 1.96 * std_pred
        
        return {
            'prediction': mean_pred,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence': 1 - (std_pred / mean_pred)  
        }


def train_pipeline(csv_path='data/raw/nairobi_properties.csv'):
    """Complete training pipeline"""
    from data_preprocessing import PropertyDataPreprocessor
    
    # Step 1: Preprocess data
    print("="*50)
    print("STEP 1: DATA PREPROCESSING")
    print("="*50)
    preprocessor = PropertyDataPreprocessor()
    X, y, df = preprocessor.preprocess_data(csv_path, is_training=True)
    preprocessor.save_preprocessor()
    
    # Step 2: Split data
    print("\n" + "="*50)
    print("STEP 2: TRAIN-TEST SPLIT")
    print("="*50)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Step 3: Train models
    print("\n" + "="*50)
    print("STEP 3: TRAINING INDIVIDUAL MODELS")
    print("="*50)
    model = PropertyPriceModel()
    model.initialize_models()
    individual_results = model.train_individual_models(X_train, y_train, X_test, y_test)
    
    # Step 4: Create and evaluate ensemble
    print("\n" + "="*50)
    print("STEP 4: CREATING ENSEMBLE MODEL")
    print("="*50)
    model.create_ensemble(X_train, y_train)
    ensemble_metrics = model.evaluate_ensemble(X_train, y_train, X_test, y_test)
    
    # Step 5: Feature importance
    feature_importance = model.get_feature_importance(X, X.columns)
    
    # Step 6: Save everything
    print("\n" + "="*50)
    print("STEP 5: SAVING MODELS")
    print("="*50)
    model.metrics = {
        'individual_models': individual_results,
        'ensemble': ensemble_metrics,
        'feature_importance': feature_importance.to_dict('records')[:10],
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    model.save_models()
    
    print("\n✓ Training pipeline completed successfully!")
    
    return model, preprocessor


if __name__ == "__main__":
    model, preprocessor = train_pipeline()