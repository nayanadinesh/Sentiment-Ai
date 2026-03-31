import os
import random
import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
from preprocess import clean_text, detect_sarcasm

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Tweet Sentiment Dashboard",
    page_icon="✨",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model_files():
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model_files()

# ---------------- LOAD DATASET IF AVAILABLE ----------------
@st.cache_data
def load_data():
    possible_files = ["tweet_sentiment.csv", "dataset.csv"]

    for file_name in possible_files:
        if os.path.exists(file_name):
            df = pd.read_csv(file_name, encoding="latin-1", header=None)
            df.columns = ["sentiment", "id", "date", "query", "user", "text"]

            df = df[["sentiment", "date", "user", "text"]].copy()
            df["sentiment"] = df["sentiment"].replace({
                0: "negative",
                4: "positive"
            })
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["text"] = df["text"].astype(str)
            df = df.dropna(subset=["date"]).copy()

            return df

    return pd.DataFrame(columns=["sentiment", "date", "user", "text"])

df = load_data()
dataset_available = not df.empty

# ---------------- SESSION STATE ----------------
if "sample_text" not in st.session_state:
    st.session_state["sample_text"] = ""

if "history" not in st.session_state:
    st.session_state["history"] = []

if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "Light"

# ---------------- SAMPLE TWEETS ----------------
sample_tweets = {
    "Positive Example": "I absolutely love this app, it works perfectly and feels so smooth.",
    "Negative Example": "This app is so frustrating, slow, and full of annoying bugs.",
    "Sarcasm Example": "Wow great job, the app crashed again. Amazing work.",
    "Multilingual Example": "This app is really good yaar, bahut acha hai.",
    "Mixed Emotion Example": "I like the design, but the app is still very slow and annoying."
}

# ---------------- THEME TOGGLE ----------------
st.sidebar.header("Appearance")
theme_choice = st.sidebar.radio("Choose theme", ["Light", "Dark"], index=0 if st.session_state["theme_mode"] == "Light" else 1)
st.session_state["theme_mode"] = theme_choice

if st.session_state["theme_mode"] == "Dark":
    bg_color = "#0f1720"
    app_bg = "linear-gradient(180deg, #0f1720 0%, #162330 100%)"
    card_bg = "rgba(21, 32, 43, 0.96)"
    text_main = "#f5f7fa"
    text_sub = "#a9b7c6"
    border_shadow = "0 8px 24px rgba(0,0,0,0.30)"
    input_text = "#ffffff"
else:
    bg_color = "#eef4fb"
    app_bg = "linear-gradient(180deg, #eef4fb 0%, #f8fbff 100%)"
    card_bg = "rgba(255,255,255,0.96)"
    text_main = "#1d3f5f"
    text_sub = "#66727f"
    border_shadow = "0 8px 24px rgba(0,0,0,0.08)"
    input_text = "#1d1d1d"

# ---------------- STYLING ----------------
st.markdown(f"""
<style>
    .stApp {{
        background: {app_bg};
    }}

    .topbar {{
        background: linear-gradient(90deg, #125ea8, #2490e8);
        padding: 24px 28px;
        border-radius: 22px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 10px 28px rgba(0,0,0,0.12);
    }}

    .topbar-title {{
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 8px;
        line-height: 1.1;
    }}

    .topbar-sub {{
        font-size: 17px;
        opacity: 0.97;
    }}

    .glass-card {{
        background: {card_bg};
        padding: 22px;
        border-radius: 22px;
        box-shadow: {border_shadow};
        margin-bottom: 18px;
        color: {text_main};
    }}

    .feature-card {{
        background: {card_bg};
        padding: 18px;
        border-radius: 18px;
        box-shadow: {border_shadow};
        text-align: center;
        height: 100%;
        color: {text_main};
    }}

    .feature-title {{
        font-size: 17px;
        font-weight: 700;
        color: {text_main};
        margin-top: 8px;
        margin-bottom: 6px;
    }}

    .feature-text {{
        font-size: 14px;
        color: {text_sub};
    }}

    .section-title {{
        font-size: 28px;
        font-weight: 800;
        color: {text_main};
        margin-bottom: 8px;
    }}

    .section-sub {{
        font-size: 15px;
        color: {text_sub};
        margin-bottom: 16px;
    }}

    .result-positive {{
        background: linear-gradient(90deg, #e8fff1, #f4fff8);
        border-left: 8px solid #2bb673;
        padding: 18px;
        border-radius: 16px;
        font-size: 22px;
        font-weight: 700;
        color: #176b43;
        margin-bottom: 12px;
    }}

    .result-negative {{
        background: linear-gradient(90deg, #fff0f0, #fff8f8);
        border-left: 8px solid #e14b4b;
        padding: 18px;
        border-radius: 16px;
        font-size: 22px;
        font-weight: 700;
        color: #8c2525;
        margin-bottom: 12px;
    }}

    .sarcasm-badge {{
        display: inline-block;
        background: #fff4db;
        color: #8a5a00;
        border: 1px solid #f1cf7b;
        padding: 8px 14px;
        border-radius: 999px;
        font-size: 14px;
        font-weight: 700;
        margin-bottom: 14px;
    }}

    .mini-note {{
        color: {text_sub};
        font-size: 14px;
    }}

    .footer-note {{
        text-align: center;
        color: {text_sub};
        font-size: 13px;
        margin-top: 22px;
        padding-bottom: 10px;
    }}

    .history-card {{
        background: {card_bg};
        padding: 14px;
        border-radius: 14px;
        box-shadow: {border_shadow};
        margin-bottom: 10px;
        color: {text_main};
    }}

    div.stButton > button {{
        border-radius: 12px;
        font-weight: 700;
        padding: 0.6rem 1rem;
        border: none;
    }}

    label, .stSelectbox label, .stTextArea label {{
        color: {text_main} !important;
    }}

    .stMarkdown, .stText, p, li, h1, h2, h3 {{
        color: {text_main};
    }}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="topbar">
    <div class="topbar-title">Tweet Sentiment Dashboard ✨</div>
    <div class="topbar-sub">
        Analyze tweet sentiment instantly, explore confidence scores, and detect sarcastic tone in a more interactive way.
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- FEATURE CARDS ----------------
f1, f2, f3, f4 = st.columns(4)

with f1:
    st.markdown("""
    <div class="feature-card">
        <div style="font-size:30px;">🧠</div>
        <div class="feature-title">AI Prediction</div>
        <div class="feature-text">Predicts positive or negative sentiment using a trained ML model.</div>
    </div>
    """, unsafe_allow_html=True)

with f2:
    st.markdown("""
    <div class="feature-card">
        <div style="font-size:30px;">😏</div>
        <div class="feature-title">Sarcasm Check</div>
        <div class="feature-text">Flags sarcastic wording and adjusts prediction when needed.</div>
    </div>
    """, unsafe_allow_html=True)

with f3:
    st.markdown("""
    <div class="feature-card">
        <div style="font-size:30px;">📊</div>
        <div class="feature-title">Confidence View</div>
        <div class="feature-text">Shows class confidence through a clear visual chart.</div>
    </div>
    """, unsafe_allow_html=True)

with f4:
    st.markdown("""
    <div class="feature-card">
        <div style="font-size:30px;">⚡</div>
        <div class="feature-title">Quick Testing</div>
        <div class="feature-text">Use examples, random tweets, and history to test faster.</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📈 Dashboard", "🕘 History"])

# ---------------- TAB 1: PREDICT ----------------
with tab1:
    left_col, right_col = st.columns([1.45, 1])

    with left_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Analyze a Tweet</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-sub">Type your own tweet below, select a sample, or generate a random example.</div>',
            unsafe_allow_html=True
        )

        selected_sample = st.selectbox(
            "Choose a sample tweet",
            ["None"] + list(sample_tweets.keys())
        )

        if selected_sample != "None":
            st.session_state["sample_text"] = sample_tweets[selected_sample]

        sample_col1, sample_col2 = st.columns(2)

        if sample_col1.button("🎲 Random Example", use_container_width=True):
            st.session_state["sample_text"] = random.choice(list(sample_tweets.values()))
            st.rerun()

        if sample_col2.button("🧹 Clear Text", use_container_width=True):
            st.session_state["sample_text"] = ""
            st.rerun()

        user_input = st.text_area(
            "Enter tweet text",
            value=st.session_state["sample_text"],
            height=170,
            placeholder="Type a tweet here..."
        )

        predict_clicked = st.button("Predict Sentiment 🚀", use_container_width=True)

        st.markdown(
            '<div class="mini-note">Try positive, negative, sarcastic, or mixed-language tweets to see how the app reacts.</div>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Quick Tips</div>', unsafe_allow_html=True)
        st.markdown("""
        - Use emotional words like **love**, **hate**, **great**, **annoying**
        - Sarcasm often mixes positive words with negative meaning
        - Short tweet-style text usually works best
        - Mixed language may still predict, but English gives more reliable results
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Try These</div>', unsafe_allow_html=True)
        st.write("**Positive:** I absolutely love this update, it works perfectly.")
        st.write("**Negative:** This app is so slow and frustrating.")
        st.write("**Sarcasm:** Wow amazing update, now it crashes every time.")
        st.markdown('</div>', unsafe_allow_html=True)

    if predict_clicked:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)

        if user_input.strip():
            cleaned_text = clean_text(user_input)
            vectorized_text = vectorizer.transform([cleaned_text])

            prediction = model.predict(vectorized_text)[0]
            probabilities = model.predict_proba(vectorized_text)[0]
            is_sarcasm = detect_sarcasm(user_input)

            if is_sarcasm and prediction == "positive":
                prediction = "negative"

            emoji = "😊" if prediction == "positive" else "😠"

            if prediction == "positive":
                st.markdown(
                    f'<div class="result-positive">{emoji} Predicted Sentiment: Positive</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="result-negative">{emoji} Predicted Sentiment: Negative</div>',
                    unsafe_allow_html=True
                )

            if is_sarcasm:
                st.markdown(
                    '<div class="sarcasm-badge">😏 Sarcasm detected</div>',
                    unsafe_allow_html=True
                )

            # save history
            st.session_state["history"].insert(0, {
                "text": user_input,
                "prediction": prediction.capitalize(),
                "sarcasm": "Yes" if is_sarcasm else "No"
            })
            st.session_state["history"] = st.session_state["history"][:10]

            result_col1, result_col2 = st.columns([1.1, 1])

            with result_col1:
                prob_df = pd.DataFrame({
                    "Sentiment": model.classes_,
                    "Confidence": probabilities
                })

                fig_bar = px.bar(
                    prob_df,
                    x="Sentiment",
                    y="Confidence",
                    text_auto=".2f"
                )
                fig_bar.update_layout(
                    height=320,
                    xaxis_title="Sentiment",
                    yaxis_title="Confidence",
                    margin=dict(t=20, b=20, l=20, r=20)
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with result_col2:
                st.markdown("### Model Interpretation")
                if prediction == "Positive":
                    st.write("The tweet contains stronger positive cues than negative ones.")
                else:
                    st.write("The tweet contains stronger negative cues or complaint-like wording.")

                if is_sarcasm:
                    st.write("Sarcastic wording was detected, so the result was adjusted to better reflect the intended tone.")

                st.write("**Cleaned text used by the model:**")
                st.code(cleaned_text)

        else:
            st.warning("Please enter some text before predicting.")

        st.markdown('</div>', unsafe_allow_html=True)

# ---------------- TAB 2: DASHBOARD ----------------
with tab2:
    if dataset_available:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Interactive analytics based on the available tweet dataset.</div>', unsafe_allow_html=True)

        total_tweets = len(df)
        positive_count = (df["sentiment"] == "positive").sum()
        negative_count = (df["sentiment"] == "negative").sum()

        m1, m2, m3 = st.columns(3)
        m1.metric("Tweets", f"{total_tweets:,}")
        m2.metric("Positive", f"{positive_count:,}")
        m3.metric("Negative", f"{negative_count:,}")

        st.markdown('</div>', unsafe_allow_html=True)

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Sentiment Distribution</div>', unsafe_allow_html=True)

            sentiment_counts = df["sentiment"].value_counts().reset_index()
            sentiment_counts.columns = ["Sentiment", "Count"]

            fig_donut = px.pie(
                sentiment_counts,
                names="Sentiment",
                values="Count",
                hole=0.65
            )
            fig_donut.update_layout(
                height=340,
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_donut, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with chart_col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Sentiment Trend</div>', unsafe_allow_html=True)

            trend_df = df.copy()
            trend_df["day"] = trend_df["date"].dt.date
            trend_counts = trend_df.groupby(["day", "sentiment"]).size().reset_index(name="count")

            fig_line = px.line(
                trend_counts,
                x="day",
                y="count",
                color="sentiment",
                markers=True
            )
            fig_line.update_layout(
                height=340,
                xaxis_title="Date",
                yaxis_title="Tweet Count",
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_line, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Sample Dataset Rows</div>', unsafe_allow_html=True)
        sample_df = df[["date", "user", "text", "sentiment"]].copy()
        sample_df["date"] = sample_df["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
        st.dataframe(sample_df.head(15), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Dashboard Preview</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-sub">Prediction is fully active. Add the dataset file later if you want charts and dataset analytics here.</div>',
            unsafe_allow_html=True
        )

        preview_df = pd.DataFrame({
            "Sentiment": ["Positive", "Negative"],
            "Count": [62, 38]
        })

        preview_col1, preview_col2 = st.columns(2)

        with preview_col1:
            fig_preview = px.pie(
                preview_df,
                names="Sentiment",
                values="Count",
                hole=0.65
            )
            fig_preview.update_layout(height=320, margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_preview, use_container_width=True)

        with preview_col2:
            preview_trend = pd.DataFrame({
                "Day": ["Mon", "Tue", "Wed", "Thu", "Fri"],
                "Positive": [10, 14, 9, 18, 15],
                "Negative": [7, 5, 8, 6, 9]
            })

            preview_long = preview_trend.melt(id_vars="Day", var_name="Sentiment", value_name="Count")
            fig_line_preview = px.line(
                preview_long,
                x="Day",
                y="Count",
                color="Sentiment",
                markers=True
            )
            fig_line_preview.update_layout(height=320, margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_line_preview, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

# ---------------- TAB 3: HISTORY ----------------
with tab3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Recent Prediction History</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Your latest predictions appear here during this session.</div>', unsafe_allow_html=True)

    if st.session_state["history"]:
        for item in st.session_state["history"]:
            st.markdown(f"""
            <div class="history-card">
                <b>Tweet:</b> {item['text']}<br>
                <b>Prediction:</b> {item['prediction']}<br>
                <b>Sarcasm:</b> {item['sarcasm']}
            </div>
            """, unsafe_allow_html=True)

        if st.button("Clear History"):
            st.session_state["history"] = []
            st.rerun()
    else:
        st.info("No predictions yet. Go to the Predict tab and try a tweet first.")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown(
    '<div class="footer-note">Built with Streamlit, NLP, and machine learning for tweet sentiment prediction.</div>',
    unsafe_allow_html=True
)