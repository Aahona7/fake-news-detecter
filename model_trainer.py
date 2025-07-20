import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import pickle
import os
from data_processor import DataProcessor

class ModelTrainer:
    def __init__(self, test_size=0.2, random_state=42, max_features=10000):
        """
        Initialize the model trainer.
        
        Args:
            test_size (float): Size of test set (0.1 to 0.4)
            random_state (int): Random state for reproducibility
            max_features (int): Maximum number of TF-IDF features
        """
        self.test_size = test_size
        self.random_state = random_state
        self.max_features = max_features
        
        # Initialize components
        self.data_processor = DataProcessor()
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),  # Unigrams and bigrams
            max_df=0.95,  # Ignore terms that appear in more than 95% of documents
            min_df=2      # Ignore terms that appear in less than 2 documents
        )
        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            C=1.0,
            solver='liblinear'
        )
        
        # Store training data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_vectorized = None
        self.X_test_vectorized = None
        
    def prepare_data(self, df, text_column='text', label_column='label'):
        """
        Prepare data for training.
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str): Name of text column
            label_column (str): Name of label column
            
        Returns:
            tuple: Processed data splits
        """
        # Preprocess data
        df_processed = self.data_processor.preprocess_dataframe(df, text_column, label_column)
        
        # Extract features and labels
        X = df_processed['processed_text']
        y = df_processed[label_column]
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def vectorize_data(self):
        """
        Convert text data to TF-IDF vectors.
        
        Returns:
            tuple: Vectorized training and test data
        """
        # Fit vectorizer on training data and transform
        self.X_train_vectorized = self.vectorizer.fit_transform(self.X_train)
        
        # Transform test data
        self.X_test_vectorized = self.vectorizer.transform(self.X_test)
        
        return self.X_train_vectorized, self.X_test_vectorized
    
    def train_model(self):
        """
        Train the logistic regression model.
        
        Returns:
            LogisticRegression: Trained model
        """
        # Train the model
        self.model.fit(self.X_train_vectorized, self.y_train)
        return self.model
    
    def evaluate_model(self):
        """
        Evaluate the trained model.
        
        Returns:
            dict: Evaluation metrics
        """
        # Make predictions
        y_pred = self.model.predict(self.X_test_vectorized)
        y_pred_proba = self.model.predict_proba(self.X_test_vectorized)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        # Classification report
        class_report = classification_report(self.y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba,
            'total_samples': len(self.X_train) + len(self.X_test),
            'train_samples': len(self.X_train),
            'test_samples': len(self.X_test),
            'n_features': self.X_train_vectorized.shape[1]
        }
        
        return metrics
    
    def train_complete_pipeline(self, df, text_column='text', label_column='label'):
        """
        Complete training pipeline.
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str): Name of text column
            label_column (str): Name of label column
            
        Returns:
            dict: Training metrics
        """
        # Prepare data
        self.prepare_data(df, text_column, label_column)
        
        # Vectorize data
        self.vectorize_data()
        
        # Train model
        self.train_model()
        
        # Evaluate model
        metrics = self.evaluate_model()
        
        return metrics
    
    def predict(self, text):
        """
        Make prediction on new text.
        
        Args:
            text (str): Input text
            
        Returns:
            tuple: (prediction, confidence, probabilities)
        """
        # Preprocess text
        processed_text = self.data_processor.preprocess_text(text)
        
        if not processed_text:
            return 0, 0.5, [0.5, 0.5]
        
        # Vectorize
        text_vectorized = self.vectorizer.transform([processed_text])
        
        # Predict
        prediction = self.model.predict(text_vectorized)[0]
        probabilities = self.model.predict_proba(text_vectorized)[0]
        confidence = max(probabilities)
        
        return prediction, confidence, probabilities
    
    def get_feature_importance(self, text, top_n=10):
        """
        Get feature importance for a given text.
        
        Args:
            text (str): Input text
            top_n (int): Number of top features to return
            
        Returns:
            list: List of (feature, importance) tuples
        """
        try:
            # Preprocess text
            processed_text = self.data_processor.preprocess_text(text)
            
            if not processed_text:
                return []
            
            # Vectorize
            text_vectorized = self.vectorizer.transform([processed_text])
            
            # Get feature names
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Get coefficients
            coefficients = self.model.coef_[0]
            
            # Get non-zero features for this text
            non_zero_indices = text_vectorized.nonzero()[1]
            
            # Get feature importance
            feature_importance = []
            for idx in non_zero_indices:
                feature_name = feature_names[idx]
                importance = abs(coefficients[idx] * text_vectorized[0, idx])
                feature_importance.append((feature_name, importance))
            
            # Sort by importance and return top_n
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            return feature_importance[:top_n]
            
        except Exception as e:
            print(f"Error getting feature importance: {e}")
            return []
    
    def save_model(self, filepath='fake_news_model.pkl'):
        """
        Save the trained model and vectorizer.
        
        Args:
            filepath (str): Path to save the model
        """
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'data_processor': self.data_processor
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath='fake_news_model.pkl'):
        """
        Load a trained model and vectorizer.
        
        Args:
            filepath (str): Path to load the model from
        """
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.vectorizer = model_data['vectorizer']
            self.data_processor = model_data['data_processor']
            
            return True
        return False
