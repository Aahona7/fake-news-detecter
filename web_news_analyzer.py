import requests
import trafilatura
import feedparser
from datetime import datetime
import pandas as pd
import streamlit as st
from fake_news_classifier import FakeNewsClassifier
import time
from urllib.parse import urljoin, urlparse
import re

class WebNewsAnalyzer:
    def __init__(self):
        self.classifier = None
        self.news_sources = {
            'RSS Feeds': {
                'BBC News': 'http://feeds.bbci.co.uk/news/rss.xml',
                'CNN': 'http://rss.cnn.com/rss/edition.rss',
                'Reuters': 'https://www.reuters.com/rssFeed/worldNews',
                'Associated Press': 'https://feeds.apnews.com/rss/apf-topnews',
                'NPR': 'https://www.npr.org/rss/rss.php?id=1001',
                'The Guardian': 'https://www.theguardian.com/world/rss',
                'ABC News': 'https://feeds.abcnews.go.com/abcnews/topstories',
                'NBC News': 'https://feeds.nbcnews.com/nbcnews/public/news'
            },
            'Direct URLs': {
                'BBC News': 'https://www.bbc.com/news',
                'CNN': 'https://www.cnn.com',
                'Reuters': 'https://www.reuters.com',
                'The Guardian': 'https://www.theguardian.com/us-news',
                'Associated Press': 'https://apnews.com',
                'NPR': 'https://www.npr.org/sections/news/',
                'ABC News': 'https://abcnews.go.com',
                'Washington Post': 'https://www.washingtonpost.com'
            }
        }
    
    def load_classifier(self):
        """Load the trained fake news classifier"""
        if self.classifier is None:
            self.classifier = FakeNewsClassifier()
            try:
                # Try to load existing model or train with generated data
                if hasattr(self.classifier, 'is_trained') and not self.classifier.is_trained:
                    st.info("Training classifier with generated dataset...")
                    # Load the generated dataset for training
                    import os
                    if os.path.exists('fake_news_dataset_5000.csv'):
                        training_data = pd.read_csv('fake_news_dataset_5000.csv')
                        self.classifier.train(training_data['text'], training_data['label'])
                    else:
                        st.error("No training dataset found. Please generate dataset first.")
                        return False
            except Exception as e:
                st.error(f"Error loading classifier: {str(e)}")
                return False
        return True
    
    def fetch_rss_articles(self, rss_url, max_articles=10):
        """Fetch articles from RSS feed"""
        articles = []
        try:
            # Add headers to avoid blocking
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            feed = feedparser.parse(rss_url)
            
            for entry in feed.entries[:max_articles]:
                try:
                    # Extract content from the article URL
                    article_url = entry.link
                    content = self.extract_article_content(article_url)
                    
                    if content and len(content.strip()) > 100:  # Ensure substantial content
                        articles.append({
                            'title': entry.title,
                            'content': content,
                            'url': article_url,
                            'published': entry.get('published', 'Unknown'),
                            'source': rss_url
                        })
                    
                    # Add delay to be respectful
                    time.sleep(1)
                    
                except Exception as e:
                    continue
                    
        except Exception as e:
            st.warning(f"Error fetching RSS feed {rss_url}: {str(e)}")
        
        return articles
    
    def extract_article_content(self, url):
        """Extract main content from article URL using trafilatura"""
        try:
            # Add headers to avoid blocking
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            downloaded = trafilatura.fetch_url(url, headers=headers)
            if downloaded:
                content = trafilatura.extract(downloaded)
                return content
            
        except Exception as e:
            pass
        
        return None
    
    def analyze_url(self, url):
        """Analyze a single URL for fake news"""
        try:
            content = self.extract_article_content(url)
            if not content or len(content.strip()) < 50:
                return None, "Could not extract sufficient content from URL"
            
            # Classify the content
            if not self.load_classifier():
                return None, "Classifier not available"
            
            prediction = self.classifier.predict(content)
            confidence = getattr(self.classifier.model, 'predict_proba', lambda x: [[0.5, 0.5]])([content])[0]
            
            result = {
                'url': url,
                'content': content[:500] + "..." if len(content) > 500 else content,
                'full_content': content,
                'prediction': 'Fake News' if prediction == 1 else 'Real News',
                'confidence': max(confidence) * 100,
                'fake_probability': confidence[1] * 100 if len(confidence) > 1 else prediction * 100,
                'real_probability': confidence[0] * 100 if len(confidence) > 1 else (1-prediction) * 100,
                'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return result, None
            
        except Exception as e:
            return None, f"Error analyzing URL: {str(e)}"
    
    def analyze_text(self, text):
        """Analyze raw text for fake news"""
        try:
            if not text or len(text.strip()) < 20:
                return None, "Text is too short for analysis"
            
            if not self.load_classifier():
                return None, "Classifier not available"
            
            prediction = self.classifier.predict(text)
            confidence = getattr(self.classifier.model, 'predict_proba', lambda x: [[0.5, 0.5]])([text])[0]
            
            result = {
                'text': text[:200] + "..." if len(text) > 200 else text,
                'full_text': text,
                'prediction': 'Fake News' if prediction == 1 else 'Real News',
                'confidence': max(confidence) * 100,
                'fake_probability': confidence[1] * 100 if len(confidence) > 1 else prediction * 100,
                'real_probability': confidence[0] * 100 if len(confidence) > 1 else (1-prediction) * 100,
                'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return result, None
            
        except Exception as e:
            return None, f"Error analyzing text: {str(e)}"
    
    def batch_analyze_rss(self, source_name, rss_url, max_articles=5):
        """Analyze multiple articles from RSS feed"""
        articles = self.fetch_rss_articles(rss_url, max_articles)
        results = []
        
        for article in articles:
            if not self.load_classifier():
                break
                
            try:
                prediction = self.classifier.predict(article['content'])
                confidence = getattr(self.classifier.model, 'predict_proba', lambda x: [[0.5, 0.5]])([article['content']])[0]
                
                result = {
                    'source': source_name,
                    'title': article['title'],
                    'url': article['url'],
                    'content': article['content'][:300] + "..." if len(article['content']) > 300 else article['content'],
                    'prediction': 'Fake News' if prediction == 1 else 'Real News',
                    'confidence': max(confidence) * 100,
                    'fake_probability': confidence[1] * 100 if len(confidence) > 1 else prediction * 100,
                    'real_probability': confidence[0] * 100 if len(confidence) > 1 else (1-prediction) * 100,
                    'published': article['published'],
                    'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                results.append(result)
                
            except Exception as e:
                continue
        
        return results
    
    def get_news_indicators(self, text):
        """Analyze text for fake news indicators"""
        indicators = {
            'sensational_words': 0,
            'caps_ratio': 0,
            'exclamation_count': 0,
            'question_count': 0,
            'suspicious_phrases': 0,
            'credible_sources': 0
        }
        
        # Sensational words commonly used in fake news
        sensational_words = [
            'SHOCKING', 'BREAKING', 'URGENT', 'EXPOSED', 'REVEALED', 'SECRET',
            'HIDDEN', 'CONSPIRACY', 'COVER-UP', 'LEAKED', 'EXCLUSIVE', 'BOMBSHELL'
        ]
        
        # Suspicious phrases
        suspicious_phrases = [
            'THEY DON\'T WANT YOU TO KNOW',
            'THE TRUTH THEY\'RE HIDING',
            'MAINSTREAM MEDIA WON\'T TELL YOU',
            'SHOCKING DISCOVERY',
            'GOVERNMENT COVER-UP',
            'BIG PHARMA',
            'SECRET AGENDA'
        ]
        
        # Credible source indicators
        credible_sources = [
            'according to official sources',
            'confirmed by',
            'peer-reviewed',
            'published in',
            'research shows',
            'study finds',
            'experts say',
            'officials stated'
        ]
        
        text_upper = text.upper()
        
        # Count indicators
        for word in sensational_words:
            indicators['sensational_words'] += text_upper.count(word)
        
        for phrase in suspicious_phrases:
            indicators['suspicious_phrases'] += text_upper.count(phrase)
        
        for phrase in credible_sources:
            indicators['credible_sources'] += text.lower().count(phrase.lower())
        
        # Calculate ratios
        total_chars = len(text)
        caps_chars = sum(1 for c in text if c.isupper())
        indicators['caps_ratio'] = (caps_chars / total_chars) * 100 if total_chars > 0 else 0
        
        indicators['exclamation_count'] = text.count('!')
        indicators['question_count'] = text.count('?')
        
        return indicators
