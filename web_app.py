import streamlit as st
import pandas as pd
from fake_news_classifier import FakeNewsClassifier
from web_news_analyzer import WebNewsAnalyzer
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

def main():
    st.set_page_config(
        page_title="AI-Powered Fake News Detection",
        page_icon="📰",
        layout="wide"
    )
    
    # Initialize components
    if 'classifier' not in st.session_state:
        st.session_state.classifier = FakeNewsClassifier()
    
    if 'web_analyzer' not in st.session_state:
        st.session_state.web_analyzer = WebNewsAnalyzer()
    
    # Header
    st.title("📰 AI-Powered Fake News Detection System")
    st.markdown("**Analyze live news from the internet with advanced AI detection**")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose analysis type:",
        ["🌐 Live News Analysis", "📝 Text Analysis", "🔗 URL Analysis", "📊 Batch Analysis", "⚙️ Model Training"]
    )
    
    if page == "🌐 Live News Analysis":
        show_live_news_analysis()
    elif page == "📝 Text Analysis":
        show_text_analysis()
    elif page == "🔗 URL Analysis":
        show_url_analysis()
    elif page == "📊 Batch Analysis":
        show_batch_analysis()
    elif page == "⚙️ Model Training":
        show_model_training()

def show_live_news_analysis():
    st.header("🌐 Live News Analysis")
    st.markdown("Fetch and analyze real news articles from major news sources")
    
    # News source selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        news_sources = {
            'BBC News': 'http://feeds.bbci.co.uk/news/rss.xml',
            'Reuters': 'https://www.reuters.com/rssFeed/worldNews',
            'Associated Press': 'https://feeds.apnews.com/rss/apf-topnews',
            'NPR': 'https://www.npr.org/rss/rss.php?id=1001',
            'The Guardian': 'https://www.theguardian.com/world/rss'
        }
        
        selected_source = st.selectbox("Select News Source:", list(news_sources.keys()))
        num_articles = st.slider("Number of articles to analyze:", 3, 15, 5)
    
    with col2:
        st.info("This will fetch live articles from the selected news source and analyze them for fake news patterns.")
    
    if st.button("🔍 Fetch & Analyze Live News", type="primary"):
        with st.spinner("Fetching articles from the internet..."):
            try:
                # Ensure model is trained
                ensure_model_trained()
                
                # Fetch and analyze articles
                rss_url = news_sources[selected_source]
                results = st.session_state.web_analyzer.batch_analyze_rss(
                    selected_source, rss_url, num_articles
                )
                
                if results:
                    st.success(f"Successfully analyzed {len(results)} articles from {selected_source}")
                    
                    # Display results
                    for i, result in enumerate(results, 1):
                        with st.expander(f"Article {i}: {result['title'][:80]}..."):
                            col1, col2, col3 = st.columns([2, 1, 1])
                            
                            with col1:
                                st.markdown(f"**Title:** {result['title']}")
                                st.markdown(f"**Content Preview:** {result['content']}")
                                st.markdown(f"**URL:** {result['url']}")
                                st.markdown(f"**Published:** {result['published']}")
                            
                            with col2:
                                prediction_color = "🔴" if result['prediction'] == 'Fake News' else "🟢"
                                st.markdown(f"**Prediction:**")
                                st.markdown(f"## {prediction_color} {result['prediction']}")
                                st.metric("Confidence", f"{result['confidence']:.1f}%")
                            
                            with col3:
                                # Create probability chart
                                fig = go.Figure(data=[
                                    go.Bar(
                                        x=['Real', 'Fake'],
                                        y=[result['real_probability'], result['fake_probability']],
                                        marker_color=['green', 'red']
                                    )
                                ])
                                fig.update_layout(height=300, showlegend=False)
                                st.plotly_chart(fig, use_container_width=True)
                    
                    # Summary statistics
                    st.markdown("### 📊 Analysis Summary")
                    
                    fake_count = sum(1 for r in results if r['prediction'] == 'Fake News')
                    real_count = len(results) - fake_count
                    avg_confidence = sum(r['confidence'] for r in results) / len(results)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Articles", len(results))
                    with col2:
                        st.metric("Real News", real_count)
                    with col3:
                        st.metric("Fake News", fake_count)
                    with col4:
                        st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
                    
                else:
                    st.warning("No articles could be fetched. The news source might be temporarily unavailable.")
                    
            except Exception as e:
                st.error(f"Error fetching news: {str(e)}")
                st.info("Try selecting a different news source or check your internet connection.")

def show_text_analysis():
    st.header("📝 Text Analysis")
    st.markdown("Analyze any news article text for fake news patterns")
    
    text_input = st.text_area(
        "Enter news article text:",
        height=300,
        help="Paste the complete news article you want to analyze"
    )
    
    if st.button("🔍 Analyze Text", type="primary") and text_input.strip():
        ensure_model_trained()
        
        with st.spinner("Analyzing text..."):
            result, error = st.session_state.web_analyzer.analyze_text(text_input)
            
            if result:
                display_analysis_result(result)
                
                # Additional analysis
                st.markdown("### 🔍 Detailed Analysis")
                indicators = st.session_state.web_analyzer.get_news_indicators(text_input)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Warning Signs:**")
                    st.metric("Sensational Words", indicators['sensational_words'])
                    st.metric("Suspicious Phrases", indicators['suspicious_phrases'])
                    st.metric("Caps Ratio", f"{indicators['caps_ratio']:.1f}%")
                
                with col2:
                    st.markdown("**Credibility Indicators:**")
                    st.metric("Credible Sources", indicators['credible_sources'])
                    st.metric("Exclamation Marks", indicators['exclamation_count'])
                    st.metric("Question Marks", indicators['question_count'])
                
            else:
                st.error(f"Analysis failed: {error}")

def show_url_analysis():
    st.header("🔗 URL Analysis")
    st.markdown("Extract and analyze content from any news website URL")
    
    url_input = st.text_input("Enter news article URL:", placeholder="https://example.com/news-article")
    
    if st.button("🔍 Analyze URL", type="primary") and url_input.strip():
        ensure_model_trained()
        
        with st.spinner("Extracting content from URL..."):
            result, error = st.session_state.web_analyzer.analyze_url(url_input)
            
            if result:
                st.success("Content successfully extracted and analyzed!")
                display_analysis_result(result)
                
                # Show extracted content
                with st.expander("📄 Extracted Content"):
                    st.text_area("Full Article Text:", value=result['full_content'], height=300)
                
            else:
                st.error(f"URL analysis failed: {error}")
                st.info("Make sure the URL is accessible and contains readable content.")

def show_batch_analysis():
    st.header("📊 Batch Analysis")
    st.markdown("Upload multiple articles or URLs for batch analysis")
    
    analysis_type = st.radio("Choose analysis type:", ["📁 Upload Text File", "🔗 Multiple URLs"])
    
    if analysis_type == "📁 Upload Text File":
        uploaded_file = st.file_uploader(
            "Upload CSV file with articles",
            type="csv",
            help="CSV should have a 'text' column with articles to analyze"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                if 'text' not in df.columns:
                    st.error("CSV must contain a 'text' column")
                    return
                
                st.success(f"File loaded with {len(df)} articles")
                
                if st.button("🔍 Analyze All Articles", type="primary"):
                    ensure_model_trained()
                    
                    with st.spinner("Analyzing articles..."):
                        results = []
                        progress_bar = st.progress(0)
                        
                        for i, text in enumerate(df['text']):
                            result, error = st.session_state.web_analyzer.analyze_text(text)
                            if result:
                                results.append({
                                    'Article': i+1,
                                    'Text Preview': text[:100] + "..." if len(text) > 100 else text,
                                    'Prediction': result['prediction'],
                                    'Confidence': f"{result['confidence']:.1f}%",
                                    'Fake Probability': f"{result['fake_probability']:.1f}%"
                                })
                            progress_bar.progress((i + 1) / len(df))
                        
                        if results:
                            results_df = pd.DataFrame(results)
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Download results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                "📥 Download Results",
                                data=csv,
                                file_name="batch_analysis_results.csv",
                                mime="text/csv"
                            )
                        
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    else:  # Multiple URLs
        st.text_area("Enter URLs (one per line):", key="url_batch", height=200)
        
        if st.button("🔍 Analyze All URLs", type="primary"):
            urls = st.session_state.url_batch.strip().split('\n')
            urls = [url.strip() for url in urls if url.strip()]
            
            if urls:
                ensure_model_trained()
                
                with st.spinner("Processing URLs..."):
                    results = []
                    progress_bar = st.progress(0)
                    
                    for i, url in enumerate(urls):
                        result, error = st.session_state.web_analyzer.analyze_url(url)
                        if result:
                            results.append({
                                'URL': url,
                                'Prediction': result['prediction'],
                                'Confidence': f"{result['confidence']:.1f}%",
                                'Content Preview': result['content'][:100] + "..."
                            })
                        else:
                            results.append({
                                'URL': url,
                                'Prediction': 'Error',
                                'Confidence': 'N/A',
                                'Content Preview': f'Failed: {error}'
                            })
                        progress_bar.progress((i + 1) / len(urls))
                    
                    if results:
                        results_df = pd.DataFrame(results)
                        st.dataframe(results_df, use_container_width=True)

def show_model_training():
    st.header("⚙️ Model Training")
    st.markdown("Train the fake news detection model")
    
    training_option = st.radio(
        "Choose training data:",
        ["🎲 Use Generated Dataset", "📁 Upload Custom Dataset"]
    )
    
    if training_option == "🎲 Use Generated Dataset":
        st.info("Train using the generated fake news dataset (5,500 articles)")
        
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training model... This may take a few minutes."):
                try:
                    # Load the generated dataset
                    if st.session_state.web_analyzer.load_classifier():
                        st.success("Model trained successfully!")
                        st.metric("Training Status", "✅ Ready")
                    else:
                        st.error("Training failed - dataset not found")
                        
                except Exception as e:
                    st.error(f"Training error: {str(e)}")
    
    else:
        uploaded_file = st.file_uploader(
            "Upload training CSV",
            type="csv",
            help="CSV should have 'text' and 'label' columns (0 for real, 1 for fake)"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                if 'text' not in df.columns or 'label' not in df.columns:
                    st.error("CSV must contain 'text' and 'label' columns")
                    return
                
                st.success(f"Dataset loaded: {len(df)} articles")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Articles", len(df))
                with col2:
                    st.metric("Real News", len(df[df['label'] == 0]))
                with col3:
                    st.metric("Fake News", len(df[df['label'] == 1]))
                
                if st.button("🚀 Train Custom Model", type="primary"):
                    with st.spinner("Training model..."):
                        try:
                            st.session_state.classifier.train(df['text'], df['label'])
                            st.success("Custom model trained successfully!")
                            
                        except Exception as e:
                            st.error(f"Training failed: {str(e)}")
                            
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")

def display_analysis_result(result):
    """Display analysis result in a formatted way"""
    st.markdown("### 📊 Analysis Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        prediction_color = "🔴" if result['prediction'] == 'Fake News' else "🟢"
        st.markdown(f"**Prediction:**")
        st.markdown(f"## {prediction_color} {result['prediction']}")
    
    with col2:
        st.metric("Confidence", f"{result['confidence']:.1f}%")
    
    with col3:
        st.metric("Analysis Time", result['analysis_time'])
    
    # Probability visualization
    fig = go.Figure(data=[
        go.Bar(
            x=['Real News', 'Fake News'],
            y=[result['real_probability'], result['fake_probability']],
            marker_color=['green', 'red']
        )
    ])
    fig.update_layout(
        title="Prediction Probabilities",
        yaxis_title="Probability (%)",
        showlegend=False,
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

def ensure_model_trained():
    """Ensure the model is trained before analysis"""
    if not hasattr(st.session_state.classifier, 'is_trained') or not st.session_state.classifier.is_trained:
        st.info("Training model with generated dataset...")
        try:
            # Try to load existing dataset
            import os
            if os.path.exists('fake_news_dataset_5000.csv'):
                training_data = pd.read_csv('fake_news_dataset_5000.csv')
                st.session_state.classifier.train(training_data['text'], training_data['label'])
                st.success("Model trained and ready!")
            else:
                st.error("Training dataset not found. Please generate dataset first or upload your own.")
                return False
        except Exception as e:
            st.error(f"Training failed: {str(e)}")
            return False
    return True

if __name__ == "__main__":
    main()