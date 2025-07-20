import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import pickle
from fake_news_classifier import FakeNewsClassifier
from utils import preprocess_text, load_sample_data

# Page configuration
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'classifier' not in st.session_state:
    st.session_state.classifier = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'training_metrics' not in st.session_state:
    st.session_state.training_metrics = None

def main():
    st.title("🔍 Fake News Detection System")
    st.markdown("### Using NLP and Machine Learning to Combat Misinformation")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a page:",
        ["🏠 Home", "🔧 Model Training", "🎯 Classify News", "📊 Performance Analysis"]
    )
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔧 Model Training":
        show_training_page()
    elif page == "🎯 Classify News":
        show_classification_page()
    elif page == "📊 Performance Analysis":
        show_analysis_page()

def show_home_page():
    st.header("Welcome to the Fake News Detection System")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        This system uses advanced Natural Language Processing and Machine Learning techniques to classify news articles as **real** or **fake**.
        
        ### 🚀 Key Features:
        - **Complete NLP Pipeline**: Text preprocessing, tokenization, and stopword removal
        - **TF-IDF Vectorization**: Converting text to numerical features
        - **Logistic Regression**: High-accuracy classification model
        - **Performance Metrics**: Comprehensive evaluation with confusion matrix
        - **Real-time Classification**: Instant results with confidence scores
        
        ### 🛠️ How it Works:
        1. **Data Preprocessing**: Clean and normalize text data
        2. **Feature Extraction**: Convert text to TF-IDF vectors
        3. **Model Training**: Train Logistic Regression classifier
        4. **Classification**: Predict fake/real with confidence scores
        5. **Visualization**: Display performance metrics and results
        """)
    
    with col2:
        st.info("""
        **Tech Stack:**
        - Python
        - Streamlit
        - Scikit-learn
        - NLTK
        - TF-IDF Vectorization
        - Logistic Regression
        """)
        
        if st.session_state.model_trained:
            st.success("✅ Model is trained and ready!")
        else:
            st.warning("⚠️ Model needs to be trained first")

def show_training_page():
    st.header("🔧 Model Training")
    
    # Data source selection
    st.markdown("### 📊 Select Data Source")
    data_source = st.radio(
        "Choose your data source:",
        ["Upload Your Dataset", "Use Sample Data"],
        help="Upload your own CSV file or use built-in sample data for testing"
    )
    
    data = None
    
    if data_source == "Upload Your Dataset":
        st.markdown("### 📁 Upload Your Dataset")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type=['csv'],
                help="Upload a CSV file with your news articles and labels"
            )
            
            if uploaded_file is not None:
                try:
                    # Read the uploaded file
                    data = pd.read_csv(uploaded_file)
                    
                    st.success(f"✅ File uploaded successfully! Shape: {data.shape}")
                    
                    # Show data preview
                    st.markdown("#### Data Preview:")
                    st.dataframe(data.head(), use_container_width=True)
                    
                    # Column mapping
                    st.markdown("#### Column Mapping:")
                    col_text, col_label = st.columns(2)
                    
                    with col_text:
                        text_column = st.selectbox(
                            "Text Column (news articles):",
                            data.columns.tolist(),
                            help="Select the column containing news article text"
                        )
                    
                    with col_label:
                        label_column = st.selectbox(
                            "Label Column (0=real, 1=fake):",
                            data.columns.tolist(),
                            index=1 if len(data.columns) > 1 else 0,
                            help="Select the column containing labels (0 for real, 1 for fake)"
                        )
                    
                    # Validate data
                    if text_column and label_column:
                        # Check label values
                        unique_labels = data[label_column].unique()
                        st.write(f"**Unique labels found:** {sorted(unique_labels)}")
                        
                        # Show label distribution
                        label_counts = data[label_column].value_counts()
                        st.write(f"**Label distribution:**")
                        for label, count in label_counts.items():
                            st.write(f"- Label {label}: {count} articles")
                        
                        # Validate labels are 0 and 1
                        if not all(label in [0, 1] for label in unique_labels):
                            st.warning("⚠️ Labels should be 0 (real) or 1 (fake). Please check your data.")
                        else:
                            # Store column names in session state
                            st.session_state.text_column = text_column
                            st.session_state.label_column = label_column
                
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
                    st.info("Please make sure your file is a valid CSV with proper encoding.")
        
        with col2:
            st.markdown("#### Expected Format:")
            st.info("""
            **CSV Requirements:**
            - Headers in first row
            - Text column: news articles
            - Label column: 0 (real) or 1 (fake)
            
            **Example:**
            ```
            text,label
            "This is a real news...",0
            "Fake story about...",1
            ```
            """)
            
            # Show sample CSV download
            if st.button("📥 Download Sample Format"):
                sample_data = load_sample_data()
                csv_data = sample_data.to_csv(index=False)
                st.download_button(
                    label="Download sample.csv",
                    data=csv_data,
                    file_name="sample_fake_news_data.csv",
                    mime="text/csv"
                )
    
    else:  # Use Sample Data
        st.markdown("### 🎯 Sample Dataset")
        st.info("""
        Using built-in sample dataset with 60+ news articles for demonstration.
        This includes both real and fake news examples for training.
        """)
        data = load_sample_data()
        
        # Show sample data info
        st.write(f"**Total articles:** {len(data)}")
        label_counts = data['label'].value_counts()
        st.write(f"**Real news:** {label_counts[0]} articles")
        st.write(f"**Fake news:** {label_counts[1]} articles")
        
        # Set default column names
        st.session_state.text_column = 'text'
        st.session_state.label_column = 'label'
    
    # Training Configuration
    if data is not None:
        st.markdown("### ⚙️ Training Configuration")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Model Parameters")
            test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
            random_state = st.number_input("Random State", 0, 1000, 42)
            max_features = st.number_input("Max TF-IDF Features", 1000, 20000, 10000, 1000)
        
        with col2:
            st.subheader("Dataset Statistics")
            st.metric("Total Samples", len(data))
            st.metric("Features", "TF-IDF Vectors")
            st.metric("Algorithm", "Logistic Regression")
        
        # Train button
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training model... This may take a few minutes."):
                try:
                    # Get column names from session state or defaults
                    text_col = st.session_state.get('text_column', 'text')
                    label_col = st.session_state.get('label_column', 'label')
                    
                    # Initialize classifier
                    classifier = FakeNewsClassifier(
                        test_size=test_size,
                        random_state=random_state,
                        max_features=max_features
                    )
                    
                    # Train model with custom column names
                    metrics = classifier.train(data, text_column=text_col, label_column=label_col)
                    
                    # Store in session state
                    st.session_state.classifier = classifier
                    st.session_state.model_trained = True
                    st.session_state.training_metrics = metrics
                    st.session_state.dataset_info = {
                        'source': data_source,
                        'shape': data.shape,
                        'text_column': text_col,
                        'label_column': label_col
                    }
                    
                    st.success("✅ Model trained successfully!")
                    
                    # Display training results
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                    with col2:
                        st.metric("Precision", f"{metrics['precision']:.3f}")
                    with col3:
                        st.metric("Recall", f"{metrics['recall']:.3f}")
                    with col4:
                        st.metric("F1-Score", f"{metrics['f1']:.3f}")
                    
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
                    st.info("Please check your data format and try again.")

def show_classification_page():
    st.header("🎯 Classify News Articles")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first in the 'Model Training' page.")
        return
    
    st.markdown("### Enter a news article to classify:")
    
    # Text input methods
    input_method = st.radio("Input method:", ["Text Area", "File Upload"])
    
    article_text = ""
    
    if input_method == "Text Area":
        article_text = st.text_area(
            "News Article Text:",
            height=200,
            placeholder="Paste your news article here..."
        )
    else:
        uploaded_file = st.file_uploader("Upload text file", type=['txt'])
        if uploaded_file is not None:
            article_text = str(uploaded_file.read(), "utf-8")
            st.text_area("Uploaded content:", article_text, height=200)
    
    if st.button("🔍 Classify Article", type="primary") and article_text.strip():
        with st.spinner("Analyzing article..."):
            try:
                # Get prediction
                prediction, confidence, probabilities = st.session_state.classifier.predict(article_text)
                
                # Display results
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    if prediction == 0:
                        st.success(f"✅ **REAL NEWS** (Confidence: {confidence:.1%})")
                    else:
                        st.error(f"❌ **FAKE NEWS** (Confidence: {confidence:.1%})")
                    
                    # Confidence meter
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = confidence * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Confidence Score"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "green"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Probability distribution
                    prob_df = pd.DataFrame({
                        'Class': ['Real', 'Fake'],
                        'Probability': [probabilities[0], probabilities[1]]
                    })
                    
                    fig = px.bar(
                        prob_df, 
                        x='Class', 
                        y='Probability',
                        title='Classification Probabilities',
                        color='Probability',
                        color_continuous_scale='RdYlGn_r'
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Feature analysis
                st.subheader("📊 Key Features Analysis")
                feature_importance = st.session_state.classifier.get_feature_importance(article_text, top_n=10)
                
                if feature_importance:
                    importance_df = pd.DataFrame(feature_importance, columns=['Feature', 'Importance'])
                    fig = px.bar(
                        importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Top Features Influencing Classification',
                        color='Importance',
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error during classification: {str(e)}")

def show_analysis_page():
    st.header("📊 Performance Analysis")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first to see performance metrics.")
        return
    
    metrics = st.session_state.training_metrics
    
    # Performance metrics
    st.subheader("📈 Model Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Accuracy",
            f"{metrics['accuracy']:.3f}",
            help="Overall correctness of the model"
        )
    with col2:
        st.metric(
            "Precision",
            f"{metrics['precision']:.3f}",
            help="True positives / (True positives + False positives)"
        )
    with col3:
        st.metric(
            "Recall",
            f"{metrics['recall']:.3f}",
            help="True positives / (True positives + False negatives)"
        )
    with col4:
        st.metric(
            "F1-Score",
            f"{metrics['f1']:.3f}",
            help="Harmonic mean of precision and recall"
        )
    
    # Confusion Matrix
    st.subheader("🎯 Confusion Matrix")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Confusion matrix heatmap
        cm = metrics['confusion_matrix']
        
        fig = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            title="Confusion Matrix",
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Real', 'Fake'],
            y=['Real', 'Fake'],
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Classification report
        st.subheader("📋 Classification Report")
        report_df = pd.DataFrame(metrics['classification_report']).transpose()
        st.dataframe(report_df.round(3), use_container_width=True)
    
    # Model insights
    st.subheader("🧠 Model Insights")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        dataset_info = st.session_state.get('dataset_info', {})
        st.info(f"""
        **Training Summary:**
        - Data source: {dataset_info.get('source', 'Sample Data')}
        - Total samples: {metrics.get('total_samples', 'N/A')}
        - Training samples: {metrics.get('train_samples', 'N/A')}
        - Test samples: {metrics.get('test_samples', 'N/A')}
        - Features: {metrics.get('n_features', 'N/A')}
        - Text column: {dataset_info.get('text_column', 'text')}
        - Label column: {dataset_info.get('label_column', 'label')}
        """)
    
    with col2:
        # Performance interpretation
        accuracy = metrics['accuracy']
        if accuracy >= 0.9:
            performance_level = "Excellent"
            color = "success"
        elif accuracy >= 0.8:
            performance_level = "Good"
            color = "info"
        elif accuracy >= 0.7:
            performance_level = "Fair"
            color = "warning"
        else:
            performance_level = "Needs Improvement"
            color = "error"
        
        st.markdown(f"""
        **Performance Assessment:**
        
        The model shows **{performance_level}** performance with an accuracy of {accuracy:.1%}.
        
        **Recommendations:**
        - {'✅' if accuracy >= 0.8 else '⚠️'} Model is {'ready for production' if accuracy >= 0.8 else 'suitable for testing but may need improvement'}
        - Consider collecting more training data if performance is below expectations
        - Fine-tune hyperparameters for better results
        """)

if __name__ == "__main__":
    main()
