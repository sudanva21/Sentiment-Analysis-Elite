from flask import Flask, render_template, request, session
from textblob import TextBlob
from transformers import pipeline
import uuid
import os
import nltk

app = Flask(__name__, template_folder='../templates', static_folder='../static')
app.secret_key = 'super-secret-sentiment-key'

# Serverless Configuration
# In Vercel/AWS Lambda, only /tmp is writable.
os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'
os.environ['HF_HOME'] = '/tmp/hf_home'
nltk.data.path.append("/tmp/nltk_data")

# Global variable for the model to allow lazy loading
sentiment_analyzer = None

def get_analyzer():
    global sentiment_analyzer
    if sentiment_analyzer is None:
        print("Loading model...")
        sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return sentiment_analyzer

def ensure_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/brown')
        nltk.data.find('corpora/noun_phrases')
    except LookupError:
        print("Downloading NLTK resources to /tmp/nltk_data...")
        nltk.download('punkt', download_dir='/tmp/nltk_data', quiet=True)
        nltk.download('brown', download_dir='/tmp/nltk_data', quiet=True)
        nltk.download('noun_phrases', download_dir='/tmp/nltk_data', quiet=True)
        # Verify textblob specific requirements
        nltk.download('punkt_tab', download_dir='/tmp/nltk_data', quiet=True)


@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment_result = None
    if request.method == 'POST':
        text = request.form.get('text_input')
        if text:
            # Ensure resources are available before processing
            ensure_nltk_resources()
            analyzer = get_analyzer()

            # 1. High-Accuracy Transformer Analysis
            model_outputs = analyzer(text[:512]) # Model has a limit of 512 tokens
            best_match = model_outputs[0]
            label_model = best_match['label'] # 'POSITIVE' or 'NEGATIVE'
            score_model = best_match['score'] # Confidence score 0-1
            
            # Map model results to our polarity system (-1 to 1)
            if label_model == 'POSITIVE':
                polarity = round(score_model, 2)
            else:
                polarity = round(-score_model, 2)

            # 2. Supplementary TextBlob Analysis (for Subjectivity and Noun Phrases)
            blob = TextBlob(text)
            subjectivity = round(blob.sentiment.subjectivity, 2)
            
            # Determine Emoji and Background Class based on high-accuracy polarity
            if polarity > 0.5:
                emoji = "🤩"
                bg_class = "bg-positive-intense"
            elif polarity > 0:
                emoji = "🙂"
                bg_class = "bg-positive"
            elif polarity < -0.5:
                emoji = "🤬"
                bg_class = "bg-negative-intense"
            elif polarity < 0:
                emoji = "🙁"
                bg_class = "bg-negative"
            else:
                emoji = "😐"
                bg_class = "bg-neutral"

            # Labels and Text Colors
            if polarity > 0.1: # Small threshold for Transformer noise
                label = "Positive"
                color_class = "text-green-600"
            elif polarity < -0.1:
                label = "Negative"
                color_class = "text-red-600"
            else:
                label = "Neutral"
                color_class = "text-gray-600"

            # Keyword Extraction & Sentence Analysis (Hybrid Approach)
            try:
                keywords = list(blob.noun_phrases)
            except Exception as e:
                print(f"Error extracting noun phrases: {e}")
                keywords = []

            sentences_data = []
            for sentence in blob.sentences:
                # Use model for sentences too for consistency
                s_text = str(sentence)
                try:
                    s_output = analyzer(s_text[:512])[0]
                    s_polarity = round(s_output['score'] if s_output['label'] == 'POSITIVE' else -s_output['score'], 2)
                except Exception as e:
                    s_polarity = 0
                
                s_label = "Positive" if s_polarity > 0.1 else "Negative" if s_polarity < -0.1 else "Neutral"
                sentences_data.append({
                    'text': s_text,
                    'polarity': s_polarity,
                    'label': s_label
                })

            polarity_percent = (polarity + 1) * 50
            subjectivity_percent = subjectivity * 100
            label_lower = label.lower()
            
            # Styles are now handled in the template
            # polarity_style = f"width: {polarity_percent}%; background-color: var(--{label_lower}-color);"
            # subjectivity_style = f"width: {subjectivity_percent}%; background-color: var(--primary-color);"

            sentiment_result = {
                'text': text,
                'polarity': polarity,
                'subjectivity': subjectivity,
                'label': label,
                'color_class': color_class,
                'polarity_percent': polarity_percent,
                'subjectivity_percent': subjectivity_percent,
                'label_lower': label_lower,
                'emoji': emoji,
                'bg_class': bg_class,
                'keywords': keywords,
                'sentences': sentences_data
            }

            # Update History in Session
            if 'history' not in session:
                session['history'] = []
            
            # Simple unique ID for the entry
            history_item = {
                'id': str(uuid.uuid4())[:8],
                'text_preview': text[:50] + "..." if len(text) > 50 else text,
                'label': label,
                'emoji': emoji,
                'polarity': polarity
            }
            
            # Keep only last 5 items
            new_history = [history_item] + session['history']
            session['history'] = new_history[:5]
            session.modified = True
            
    return render_template('index.html', result=sentiment_result, history=session.get('history', []))

if __name__ == '__main__':
    # Local development support
    app.run(debug=True)
