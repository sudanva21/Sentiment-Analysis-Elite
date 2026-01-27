# Sentiment Analysis Elite

> A state-of-the-art NLP application delivering real-time emotional insight and granular textual analysis using Transformer-based Deep Learning.

## 📋 Project Overview
Sentiment Analysis Elite is a full-stack web application designed to bridge the gap between complex Deep Learning models and intuitive user experience. By integrating **DistilBERT**—a high-performance Transformer model—with a responsive, modern frontend, this tool provides instant, high-accuracy sentiment classification, subjectivity scoring, and key concept extraction.

This project demonstrates the practical application of NLP in a production-ready environment, emphasizing clean architecture, modern UI/UX principles, and robust backend engineering.

## 🚀 Key Features
- **Transformer-Powered Precision**: Utilizes `distilbert-base-uncased` fine-tuned on the SST-2 dataset for **91.1% accuracy**.
- **Hybrid Analysis Engine**: Combines Deep Learning (Transformers) for sentiment with Rule-Based NLP (TextBlob) for subjectivity and noun phrase extraction.
- **Dynamic Visualizations**: Real-time CSS animations and color-coded metrics that respond instantly to emotional context.
- **Granular Breakdown**: Deconstructs paragraphs into individual sentences to identify emotional shifts within the text.
- **History Tracking**: Session-based storage of recent analyses for quick comparison.
- **Premium UI**: Glassmorphism design, smooth transitions, and responsive layout.

## 🛠️ Technical Stack

### Backend
- **Framework**: Flask (Python 3.x)
- **AI/ML Engine**: Hugging Face Transformers, PyTorch/TensorFlow (backend)
- **NLP Utilities**: TextBlob (for linguistic features)
- **Architecture**: RESTful route handling with server-side session management.

### Frontend
- **Templating**: Jinja2
- **Styling**: Vanilla CSS3 with CSS Variables for dynamic theming.
- **Logic**: Vanilla JavaScript for DOM manipulation and runtime style injection.

## 📊 Model Specifications
The core of this application is the **DistilBERT** model. It offers BERT-level performance while being 40% smaller and 60% faster, making it ideal for web deployment.

| Metric | Value |
| :--- | :--- |
| **Model** | DistilBERT Base Uncased |
| **Training Data** | SST-2 (Stanford Sentiment Treebank) |
| **Validation Accuracy** | 91.1% (GLUE Benchmark) |
| **Parameters** | 66 Million |

*For a deep dive into the model architecture, see [report.md](report.md).*

## 📂 Project Structure
```
project-2/
├── app.py                # Application entry point & route logic
├── report.md             # Technical documentation of the AI model
├── requirements.txt      # Python dependencies
├── static/
│   └── css/
│       └── style.css     # Global styles and themes
└── templates/
    └── index.html        # Main interface template
```

## 🔧 Installation & Usage

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/sentiment-elite.git
    cd sentiment-elite
    ```

2.  **Install Dependencies**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application**
    ```bash
    python app.py
    ```
    *Note: The first run will download the ~260MB model file automatically.*

4.  **Access the Interface**
    Open your browser and navigate to `http://127.0.0.1:5000`.

## 💼 Business Use Cases
- **Customer Feedback**: Automated tagging of reviews and support tickets.
- **Brand Monitoring**: Real-time analysis of social media mentions.
- **Content Moderation**: Pre-filtering of user-generated content for negative tone.

---
*Built for the QSkill Internship Program.*
