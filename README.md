# 📊 Tweet Sentiment Analysis Dashboard

An interactive **Streamlit web application** that performs sentiment analysis on tweets using **Natural Language Processing (NLP)** and **Machine Learning**.

The system classifies tweets as **Positive or Negative**, provides **visual analytics**, supports **data filtering**, and includes a **basic sarcasm detection mechanism**.

---

## 🚀 Features

🔍 Predict sentiment of user-entered text (Positive / Negative)
-😏 Detect sarcasm and adjust prediction accordingly
-📊 Dashboard visualizations:
  - Sentiment distribution (donut chart)
  - Sentiment trend over time (line chart)
-🎯 Dynamic filters:
  - Keyword-based filtering
  - Sentiment filtering
  - Username filtering
  - Date range filtering
-📄 View filtered tweets in tabular format
-⚡ Fast and lightweight ML model (Logistic Regression)
## 🧠 Technologies Used

- Python
- Streamlit
- Scikit-learn
- NLTK
- Pandas
- Plotly

---

## 📂 Project Structure
sentiment-analysis-project/
│
├── app.py # Streamlit dashboard UI
├── model_train.py # Model training script
├── preprocess.py # Text preprocessing + sarcasm detection
├── dataset.csv / tweet_sentiment.csv # Dataset
├── sentiment_model.pkl # Trained ML model
├── tfidf_vectorizer.pkl # TF-IDF vectorizer
├── requirements.txt # Dependencies
└── README.md # Documentation


---

## ⚙️ How It Works

### 1. Data Preprocessing
- Lowercasing
- Removing URLs, mentions, hashtags
- Removing punctuation and stopwords
- Slang normalization
- Emoji removal

### 2. Feature Extraction
- TF-IDF (Term Frequency–Inverse Document Frequency)

### 3. Model
- Logistic Regression classifier

### 4. Sarcasm Detection
- Rule-based detection using contextual keywords
- Adjusts prediction when sarcasm is detected

## 📊 Dataset

This project uses a tweet sentiment dataset from Kaggle.

🔗 Dataset Link: https://www.kaggle.com/c/commonlitreadabilityprize/data

The dataset contains tweets labeled as:
- 0 → Negative
- 4 → Positive

Only the text and sentiment columns are used for training the model.

---
## ▶️ How to Run the Project

### Step 1: Clone the repository

```bash
git clone https://github.com/your-username/tweet-sentiment-analysis-dashboard.git
cd tweet-sentiment-analysis-dashboard

### Step 2: Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate

### Step 3: Install dependencies
```bash
pip install -r requirements.txt

###Step 4: Train the model
```bash
python model_train.py

###Step 5: Run the application
```bash
streamlit run app.py
