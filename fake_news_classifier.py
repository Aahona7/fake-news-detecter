import pandas as pd
import numpy as np
from model_trainer import ModelTrainer

class FakeNewsClassifier:
    def __init__(self, test_size=0.2, random_state=42, max_features=10000):
        """
        Initialize the Fake News Classifier.
        
        Args:
            test_size (float): Size of test set
            random_state (int): Random state for reproducibility
            max_features (int): Maximum number of TF-IDF features
        """
        self.trainer = ModelTrainer(
            test_size=test_size,
            random_state=random_state,
            max_features=max_features
        )
        self.is_trained = False
        self.training_metrics = None
    
    def train(self, data, text_column='text', label_column='label'):
        """
        Train the classifier on the provided data.
        
        Args:
            data (pd.DataFrame): Training data
            text_column (str): Name of the text column
            label_column (str): Name of the label column
            
        Returns:
            dict: Training metrics
        """
        try:
            # Train the model with custom column names
            self.training_metrics = self.trainer.train_complete_pipeline(
                data, text_column=text_column, label_column=label_column
            )
            self.is_trained = True
            return self.training_metrics
        except Exception as e:
            raise Exception(f"Training failed: {str(e)}")
    
    def predict(self, text):
        """
        Predict whether a news article is fake or real.
        
        Args:
            text (str): News article text
            
        Returns:
            tuple: (prediction, confidence, probabilities)
                - prediction: 0 for real, 1 for fake
                - confidence: confidence score (0-1)
                - probabilities: [prob_real, prob_fake]
        """
        if not self.is_trained:
            raise Exception("Model must be trained before making predictions")
        
        if not text or not isinstance(text, str) or len(text.strip()) == 0:
            raise Exception("Invalid input text")
        
        return self.trainer.predict(text)
    
    def predict_batch(self, texts):
        """
        Predict multiple texts at once.
        
        Args:
            texts (list): List of text strings
            
        Returns:
            list: List of (prediction, confidence, probabilities) tuples
        """
        if not self.is_trained:
            raise Exception("Model must be trained before making predictions")
        
        results = []
        for text in texts:
            try:
                result = self.predict(text)
                results.append(result)
            except Exception as e:
                results.append((0, 0.5, [0.5, 0.5]))  # Default result for failed predictions
        
        return results
    
    def get_feature_importance(self, text, top_n=10):
        """
        Get the most important features for a prediction.
        
        Args:
            text (str): Input text
            top_n (int): Number of top features to return
            
        Returns:
            list: List of (feature, importance) tuples
        """
        if not self.is_trained:
            raise Exception("Model must be trained before getting feature importance")
        
        return self.trainer.get_feature_importance(text, top_n)
    
    def get_model_metrics(self):
        """
        Get the training metrics of the model.
        
        Returns:
            dict: Training metrics
        """
        if not self.is_trained:
            raise Exception("Model must be trained first")
        
        return self.training_metrics
    
    def save_model(self, filepath='fake_news_classifier.pkl'):
        """
        Save the trained classifier.
        
        Args:
            filepath (str): Path to save the model
        """
        if not self.is_trained:
            raise Exception("Model must be trained before saving")
        
        self.trainer.save_model(filepath)
    
    def load_model(self, filepath='fake_news_classifier.pkl'):
        """
        Load a pre-trained classifier.
        
        Args:
            filepath (str): Path to load the model from
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        success = self.trainer.load_model(filepath)
        if success:
            self.is_trained = True
        return success
    
    def get_prediction_explanation(self, text):
        """
        Get a detailed explanation of the prediction.
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Detailed prediction explanation
        """
        if not self.is_trained:
            raise Exception("Model must be trained before getting explanations")
        
        # Get prediction
        prediction, confidence, probabilities = self.predict(text)
        
        # Get feature importance
        feature_importance = self.get_feature_importance(text, top_n=10)
        
        # Create explanation
        explanation = {
            'prediction': 'Fake' if prediction == 1 else 'Real',
            'prediction_label': prediction,
            'confidence': confidence,
            'probabilities': {
                'real': probabilities[0],
                'fake': probabilities[1]
            },
            'top_features': feature_importance,
            'text_length': len(text),
            'word_count': len(text.split())
        }
        
        return explanation
