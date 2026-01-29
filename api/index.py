from flask import Flask, render_template, request, session
from textblob import TextBlob
import uuid
import os
import requests
import nltk

app = Flask(__name__, template_folder='../templates', static_folder='../static')
app.secret_key = 'super-secret-sentiment-key'

# Serverless Configuration for NLTK
# In Vercel, only /tmp is writable.
nltk.data.path.append("/tmp/nltk_data")

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
        nltk.download('punkt_tab', download_dir='/tmp/nltk_data', quiet=True)

# HF API Configuration
HF_API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
# Use environment variable if available, otherwise anonymous (might check rate limits)
HF_API_TOKEN = os.environ.get("HF_API_TOKEN") 
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"} if HF_API_TOKEN else {}

def query_hf_api(payload):
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        return response.json()
    except Exception as e:
        print(f"API Error: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment_result = None
    if request.method == 'POST':
        text = request.form.get('text_input')
        if text:
            # Ensure resources are available before processing
            ensure_nltk_resources()
            
            # 1. High-Accuracy Transformer Analysis via API
            # Note: API returns a list of lists [[{'label': 'POSITIVE', 'score': 0.9}]]
            api_response = query_hf_api({"inputs": text[:512]})
            
            # Default fallback if API fails
            label_model = 'NEUTRAL' 
            score_model = 0.0

            if api_response and isinstance(api_response, list) and len(api_response) > 0:
                 # Standard response format check
                 if isinstance(api_response[0], list):
                     best_match = api_response[0][0] # Usually sorted by score
                 elif isinstance(api_response[0], dict) and 'label' in api_response[0]:
                     best_match = api_response[0]
                 else:
                     best_match = None
                 
                 if best_match:
                    label_model = best_match.get('label', 'NEUTRAL')
                    score_model = best_match.get('score', 0.0)
            elif isinstance(api_response, dict) and 'error' in api_response:
                print(f"HF API Error: {api_response['error']}")
                # Fallback to TextBlob if API is loading or fails
                blob_fallback = TextBlob(text)
                pol_fallback = blob_fallback.sentiment.polarity
                if pol_fallback > 0:
                    label_model = 'POSITIVE'
                    score_model = pol_fallback
                else:
                    label_model = 'NEGATIVE'
                    score_model = abs(pol_fallback)

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
            if polarity > 0.1: # Small threshold
                label = "Positive"
                color_class = "text-green-600"
            elif polarity < -0.1:
                label = "Negative"
                color_class = "text-red-600"
            else:
                label = "Neutral"
                color_class = "text-gray-600"

            # Keyword Extraction
            try:
                keywords = list(blob.noun_phrases)
            except Exception as e:
                print(f"Error extracting noun phrases: {e}")
                keywords = []

            # Sentence Analysis - Simplified to TextBlob to save API calls per request
            # (Calling API for every sentence would be too slow/rate-limited)
            sentences_data = []
            for sentence in blob.sentences:
                s_text = str(sentence)
                s_pol = sentence.sentiment.polarity
                s_label = "Positive" if s_pol > 0.1 else "Negative" if s_pol < -0.1 else "Neutral"
                sentences_data.append({
                    'text': s_text,
                    'polarity': round(s_pol, 2),
                    'label': s_label
                })

            polarity_percent = (polarity + 1) * 50
            subjectivity_percent = subjectivity * 100
            label_lower = label.lower()

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
