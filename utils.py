import pandas as pd
import numpy as np
import re

def preprocess_text(text):
    """
    Basic text preprocessing function.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Preprocessed text
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
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def load_sample_data():
    """
    Load sample fake news data for training.
    This creates a representative dataset for demonstration purposes.
    
    Returns:
        pd.DataFrame: Sample dataset with text and label columns
    """
    # Sample real news articles (label = 0)
    real_news = [
        "The Federal Reserve announced today that interest rates will remain unchanged following their two-day policy meeting. The decision was unanimous among voting members and reflects the central bank's commitment to supporting economic recovery while monitoring inflation trends.",
        "Scientists at MIT have developed a new type of battery that charges 10 times faster than conventional lithium-ion batteries. The breakthrough could revolutionize electric vehicle technology and make renewable energy storage more efficient.",
        "The United Nations Climate Summit concluded with 195 countries agreeing to new emissions reduction targets. The agreement includes commitments to phase out coal power plants and increase investment in renewable energy infrastructure over the next decade.",
        "Local authorities reported that the new subway line connecting downtown to the airport will open six months ahead of schedule. The project, which cost $2.3 billion, is expected to reduce travel time by 40% and ease traffic congestion.",
        "Medical researchers published findings showing that a new treatment for Type 2 diabetes has proven effective in clinical trials. The medication helped 78% of participants achieve better blood sugar control with fewer side effects than existing treatments.",
        "The stock market closed higher today as technology stocks led gains following better-than-expected quarterly earnings reports. The S&P 500 rose 1.2% while the NASDAQ gained 1.8% in heavy trading volume.",
        "Education officials announced that standardized test scores have improved across all grade levels for the third consecutive year. The improvement is attributed to new teaching methods and increased funding for classroom technology.",
        "City planners unveiled designs for a new public park that will feature sustainable landscaping and renewable energy systems. Construction is scheduled to begin next spring and will create 200 temporary jobs.",
        "The World Health Organization reported that global vaccination rates have reached 85% for childhood immunizations, marking the highest level in the organization's history. This achievement represents progress toward eliminating preventable diseases.",
        "Archaeologists discovered ancient artifacts dating back 3,000 years at a dig site near the Mediterranean coast. The findings include pottery, tools, and coins that provide new insights into trade routes of the ancient world."
    ]
    
    # Sample fake news articles (label = 1)
    fake_news = [
        "BREAKING: Government officials secretly admit that all weather reports are fabricated to control agricultural markets. Internal documents reveal decades-long conspiracy to manipulate crop prices through false weather predictions.",
        "Shocking discovery: Common household item found to cure cancer in 24 hours. Big Pharma doesn't want you to know about this miracle cure that costs less than $5 and is available in every grocery store.",
        "Scientists confirm that the moon is actually hollow and contains alien technology. NASA has been hiding this information for 50 years while secretly working with extraterrestrial beings to mine lunar resources.",
        "URGENT WARNING: New smartphone update contains mind control software that can read your thoughts. Tech companies are using this technology to predict and influence your buying decisions without your knowledge.",
        "Exclusive report reveals that celebrities are using time travel technology to stay young. Hollywood insiders expose the secret underground facilities where stars go to reverse aging using advanced alien technology.",
        "Government whistleblower exposes plan to replace all birds with robotic surveillance drones. The massive operation began in 2001 and explains why you rarely see baby birds or bird graveyards anymore.",
        "Medical breakthrough suppressed by doctors: Drinking bleach actually boosts immune system and prevents all diseases. Pharmaceutical companies spend billions to hide this simple cure that threatens their profits.",
        "Ancient prophecy predicts that the world will end next Tuesday unless everyone shares this article exactly 47 times. Archaeological evidence from a recently discovered tablet confirms this dire warning from the past.",
        "EXPOSED: Your tap water contains microscopic robots programmed to alter your DNA. Government agencies have been secretly adding nanotechnology to public water supplies to create a more compliant population.",
        "Breaking investigation reveals that all historical events since 1800 have been staged by a secret society of time travelers. Napoleon, both World Wars, and the moon landing were all elaborate performances to hide the truth."
    ]
    
    # Create DataFrame
    data = []
    
    # Add real news
    for text in real_news:
        data.append({'text': text, 'label': 0})
    
    # Add fake news
    for text in fake_news:
        data.append({'text': text, 'label': 1})
    
    # Create additional synthetic examples to increase dataset size
    # This creates variations of the base examples
    additional_real = []
    additional_fake = []
    
    # Generate variations of real news
    for text in real_news:
        # Create slight variations
        words = text.split()
        if len(words) > 10:
            # Create a shorter version
            shorter = ' '.join(words[:len(words)//2]) + " according to official sources."
            additional_real.append(shorter)
            
            # Create a version with different ending
            different_ending = ' '.join(words[:-5]) + " officials confirmed in a statement."
            additional_real.append(different_ending)
    
    # Generate variations of fake news
    for text in fake_news:
        # Create variations with different sensational phrases
        sensational_phrases = [
            "SHOCKING TRUTH: ",
            "THEY DON'T WANT YOU TO KNOW: ",
            "BOMBSHELL REVELATION: ",
            "SECRET EXPOSED: "
        ]
        for phrase in sensational_phrases[:2]:  # Use first 2 phrases
            variation = phrase + text
            additional_fake.append(variation)
    
    # Add variations to data
    for text in additional_real:
        data.append({'text': text, 'label': 0})
    
    for text in additional_fake:
        data.append({'text': text, 'label': 1})
    
    df = pd.DataFrame(data)
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

def validate_text_input(text):
    """
    Validate text input for classification.
    
    Args:
        text (str): Input text
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not text or not isinstance(text, str):
        return False, "Text input is required"
    
    text = text.strip()
    if len(text) < 10:
        return False, "Text must be at least 10 characters long"
    
    if len(text) > 10000:
        return False, "Text must be less than 10,000 characters long"
    
    # Check if text contains meaningful content (not just special characters)
    clean_text = re.sub(r'[^a-zA-Z\s]', '', text)
    if len(clean_text.strip()) < 5:
        return False, "Text must contain meaningful content"
    
    return True, ""

def format_confidence_score(confidence):
    """
    Format confidence score for display.
    
    Args:
        confidence (float): Confidence score (0-1)
        
    Returns:
        str: Formatted confidence score
    """
    percentage = confidence * 100
    return f"{percentage:.1f}%"

def get_prediction_color(prediction, confidence):
    """
    Get color for prediction display based on prediction and confidence.
    
    Args:
        prediction (int): 0 for real, 1 for fake
        confidence (float): Confidence score (0-1)
        
    Returns:
        str: Color code for display
    """
    if confidence < 0.6:
        return "yellow"  # Low confidence
    elif prediction == 0:
        return "green"   # Real news with high confidence
    else:
        return "red"     # Fake news with high confidence

def analyze_text_complexity(text):
    """
    Analyze the complexity of input text.
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Text complexity metrics
    """
    if not text:
        return {}
    
    words = text.split()
    sentences = text.split('.')
    
    # Calculate metrics
    avg_word_length = np.mean([len(word) for word in words]) if words else 0
    avg_sentence_length = np.mean([len(sentence.split()) for sentence in sentences if sentence.strip()]) if sentences else 0
    
    # Count different types of words/characters
    caps_count = sum(1 for char in text if char.isupper())
    exclamation_count = text.count('!')
    question_count = text.count('?')
    
    complexity = {
        'word_count': len(words),
        'sentence_count': len([s for s in sentences if s.strip()]),
        'avg_word_length': avg_word_length,
        'avg_sentence_length': avg_sentence_length,
        'capitals_count': caps_count,
        'exclamations': exclamation_count,
        'questions': question_count,
        'caps_ratio': caps_count / len(text) if text else 0
    }
    
    return complexity
