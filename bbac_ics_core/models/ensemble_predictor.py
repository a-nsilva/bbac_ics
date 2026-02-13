#!/usr/bin/env python3
"""
BBAC ICS Framework - Ensemble Meta-Learner
Adaptively combines rule, behavioral, and ML layer scores using meta-classification.
"""
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class EnsemblePredictor:
    """
    Meta-learner that combines layer scores.
    
    Learns optimal fusion weights from training data instead of
    using fixed weighted voting.
    
    Supports:
    - Logistic Regression (default, fast, interpretable)
    - XGBoost (optional, more powerful)
    """
    
    def __init__(
        self,
        model_type: str = 'logistic_regression',
        random_state: int = 42
    ):
        """
        Initialize ensemble meta-learner.
        
        Args:
            model_type: 'logistic_regression' or 'xgboost'
            random_state: Random seed
        """
        self.model_type = model_type
        self.random_state = random_state
        
        # Initialize model
        if model_type == 'logistic_regression':
            self.model = LogisticRegression(
                random_state=random_state,
                max_iter=1000,
                class_weight='balanced'  # Handle imbalanced data
            )
        elif model_type == 'xgboost':
            try:
                from xgboost import XGBClassifier
                self.model = XGBClassifier(
                    random_state=random_state,
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    eval_metric='logloss'
                )
            except ImportError:
                raise ImportError(
                    "XGBoost not installed. Use 'pip install xgboost' or "
                    "set model_type='logistic_regression'"
                )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # Feature scaler
        self.scaler = StandardScaler()
        
        # Training status
        self.is_trained = False
        
        # Feature names
        self.feature_names = ['rule_score', 'behavioral_score', 'ml_score']
    
    def train(
        self,
        layer_scores: List[Tuple[float, float, float]],
        ground_truth: List[int]
    ):
        """
        Train meta-learner on layer outputs.
        
        Args:
            layer_scores: List of (rule_score, behavioral_score, ml_score) tuples
            ground_truth: List of binary labels (0=deny, 1=allow)
        """
        if len(layer_scores) != len(ground_truth):
            raise ValueError("Scores and labels must have same length")
        
        # Convert to numpy
        X = np.array(layer_scores)
        y = np.array(ground_truth)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        self.is_trained = True
    
    def predict_score(
        self,
        rule_score: float,
        behavioral_score: float,
        ml_score: float
    ) -> float:
        """
        Predict final score from layer scores.
        
        Args:
            rule_score: Policy layer score [0, 1]
            behavioral_score: Statistical layer score [0, 1]
            ml_score: Sequence layer score [0, 1]
            
        Returns:
            Final ensemble score [0, 1]
        """
        if not self.is_trained:
            # Fallback to simple average if not trained
            return (rule_score + behavioral_score + ml_score) / 3.0
        
        # Prepare features
        X = np.array([[rule_score, behavioral_score, ml_score]])
        X_scaled = self.scaler.transform(X)
        
        # Predict probability of positive class (allow)
        proba = self.model.predict_proba(X_scaled)[0]
        
        # Return probability of allow (class 1)
        return float(proba[1])
    
    def predict_batch(
        self,
        layer_scores: List[Tuple[float, float, float]]
    ) -> List[float]:
        """
        Predict scores for batch of requests.
        
        Args:
            layer_scores: List of (rule, behavioral, ml) score tuples
            
        Returns:
            List of ensemble scores
        """
        if not self.is_trained:
            # Fallback to simple average
            return [sum(scores) / 3.0 for scores in layer_scores]
        
        X = np.array(layer_scores)
        X_scaled = self.scaler.transform(X)
        
        probas = self.model.predict_proba(X_scaled)[:, 1]
        
        return probas.tolist()
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance (layer contributions).
        
        Returns:
            Dictionary mapping layer names to importance scores
        """
        if not self.is_trained:
            return {}
        
        if self.model_type == 'logistic_regression':
            # Use absolute coefficients as importance
            coefs = np.abs(self.model.coef_[0])
            # Normalize to sum to 1
            importance = coefs / coefs.sum()
            
        elif self.model_type == 'xgboost':
            # Use feature importance from XGBoost
            importance = self.model.feature_importances_
        
        return {
            name: float(imp)
            for name, imp in zip(self.feature_names, importance)
        }
    
    def get_learned_weights(self) -> Dict[str, float]:
        """
        Get learned weights (for logistic regression).
        
        Returns:
            Dictionary of layer weights
        """
        if not self.is_trained or self.model_type != 'logistic_regression':
            return {}
        
        coefs = self.model.coef_[0]
        
        return {
            'rule': float(coefs[0]),
            'behavioral': float(coefs[1]),
            'ml': float(coefs[2]),
            'intercept': float(self.model.intercept_[0])
        }
    
    def save(self, filepath: Path):
        """
        Save trained model to disk.
        
        Args:
            filepath: Path to save model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, filepath: Path):
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to saved model
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']
        self.feature_names = model_data['feature_names']
    
    def evaluate(
        self,
        layer_scores: List[Tuple[float, float, float]],
        ground_truth: List[int]
    ) -> Dict:
        """
        Evaluate meta-learner performance.
        
        Args:
            layer_scores: List of layer score tuples
            ground_truth: List of binary labels
            
        Returns:
            Evaluation metrics
        """
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score
        )
        
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Predict
        scores = self.predict_batch(layer_scores)
        predictions = [1 if s >= 0.5 else 0 for s in scores]
        
        # Compute metrics
        return {
            'accuracy': accuracy_score(ground_truth, predictions),
            'precision': precision_score(ground_truth, predictions, zero_division=0),
            'recall': recall_score(ground_truth, predictions, zero_division=0),
            'f1_score': f1_score(ground_truth, predictions, zero_division=0),
            'roc_auc': roc_auc_score(ground_truth, scores),
            'feature_importance': self.get_feature_importance(),
            'learned_weights': self.get_learned_weights()
        }
