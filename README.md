# AI-Powered Fake News Detection System

A comprehensive web application that uses Machine Learning and Natural Language Processing to detect fake news articles with real-time internet data fetching capabilities.

## Features

- **Live News Analysis**: Fetch and analyze articles from BBC, Reuters, AP News, NPR, The Guardian
- **URL Analysis**: Extract and analyze content from any news website
- **Text Analysis**: Analyze any news article text for fake news patterns
- **Batch Processing**: Process multiple articles or URLs simultaneously
- **Real-time Classification**: Instant fake news detection with confidence scores
- **Advanced Web Scraping**: Clean content extraction from various news sources

## Installation

1. Install dependencies:
```bash
pip install -r deploy_requirements.txt
```

2. Download NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Usage

Run the application:
```bash
streamlit run web_app.py --server.port 5000
```

Then open your browser to `http://localhost:5000`

## Deployment

### Free Deployment Options:

1. **Streamlit Community Cloud** (Recommended)
   - Fork this repository to GitHub
   - Connect to Streamlit Community Cloud
   - Deploy with one click - completely free

2. **Heroku Free Tier**
   - Create `Procfile`: `web: streamlit run web_app.py --server.port=$PORT --server.address=0.0.0.0`
   - Deploy using Heroku CLI

3. **Railway**
   - Connect GitHub repository
   - Automatic deployment with Railway's free tier

## File Structure

- `web_app.py` - Main Streamlit application
- `web_news_analyzer.py` - Internet data fetching and analysis
- `fake_news_classifier.py` - ML classification engine  
- `model_trainer.py` - Model training logic
- `data_processor.py` - Text preprocessing
- `utils.py` - Utility functions
- `fake_news_dataset_5000.csv` - Training dataset (5,500 articles)

## How It Works

1. **Training**: Uses generated dataset of 5,500 real and fake news articles
2. **Web Scraping**: Fetches live content using RSS feeds and web scraping
3. **Analysis**: Applies NLP preprocessing and ML classification
4. **Results**: Shows prediction with confidence scores and detailed analysis

## Tech Stack

- **Frontend**: Streamlit
- **ML**: Scikit-learn, NLTK
- **Web Scraping**: Trafilatura, BeautifulSoup, Feedparser
- **Visualization**: Plotly
- **Data**: Pandas, NumPy

## API Usage

The system analyzes news from these sources:
- RSS feeds from major news outlets
- Direct URL content extraction
- Manual text input
- Batch file processing

No API keys required for basic functionality.