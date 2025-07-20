import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self):
        """Initialize the data processor with NLTK components."""
        self.stemmer = PorterStemmer()
        self.stop_words = set()
        self._download_nltk_data()
    
    def _download_nltk_data(self):
        """Download required NLTK data."""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            print(f"Warning: Could not download NLTK data: {e}")
            # Fallback stopwords list
            self.stop_words = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
                'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
                'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
                'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
                'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
                'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
                'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after', 'above',
                'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                'further', 'then', 'once'
            }
    
    def clean_text(self, text):
        """
        Clean and preprocess text data.
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_text(self, text):
        """
        Tokenize text into words.
        
        Args:
            text (str): Text to tokenize
            
        Returns:
            list: List of tokens
        """
        try:
            tokens = word_tokenize(text)
        except:
            # Fallback tokenization
            tokens = text.split()
        
        return tokens
    
    def remove_stopwords(self, tokens):
        """
        Remove stopwords from tokens.
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: Filtered tokens
        """
        return [token for token in tokens if token not in self.stop_words and len(token) > 2]
    
    def stem_tokens(self, tokens):
        """
        Apply stemming to tokens.
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: Stemmed tokens
        """
        return [self.stemmer.stem(token) for token in tokens]
    
    def preprocess_text(self, text):
        """
        Complete preprocessing pipeline for text.
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Preprocessed text
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        
        if not cleaned_text:
            return ""
        
        # Tokenize
        tokens = self.tokenize_text(cleaned_text)
        
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Stem tokens
        tokens = self.stem_tokens(tokens)
        
        # Join back to string
        return ' '.join(tokens)
    
    def preprocess_dataframe(self, df, text_column='text', label_column='label'):
        """
        Preprocess a dataframe containing text data.
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str): Name of text column
            label_column (str): Name of label column
            
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        df_processed = df.copy()
        
        # Preprocess text
        df_processed['processed_text'] = df_processed[text_column].apply(self.preprocess_text)
        
        # Remove empty texts
        df_processed = df_processed[df_processed['processed_text'].str.len() > 0]
        
        # Reset index
        df_processed = df_processed.reset_index(drop=True)
        
        return df_processed
    
    def get_text_statistics(self, df, text_column='processed_text'):
        """
        Get statistics about the text data.
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str): Name of text column
            
        Returns:
            dict: Text statistics
        """
        if text_column not in df.columns:
            return {}
        
        texts = df[text_column].astype(str)
        
        # Calculate statistics
        word_counts = texts.apply(lambda x: len(x.split()))
        char_counts = texts.apply(len)
        
        stats = {
            'total_texts': len(texts),
            'avg_words': word_counts.mean(),
            'median_words': word_counts.median(),
            'max_words': word_counts.max(),
            'min_words': word_counts.min(),
            'avg_chars': char_counts.mean(),
            'median_chars': char_counts.median(),
            'max_chars': char_counts.max(),
            'min_chars': char_counts.min()
        }
        
        return stats
