import os
import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
from preprocess import clean_text, detect_sarcasm

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Tweet Sentiment Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model_files():
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model_files()

# ---------------- LOAD DATA ----------------
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

# ---------------- STYLING ----------------
st.markdown("""
<style>
    .stApp {
        background-color: #eef3f8;
    }

    .topbar {
        background: linear-gradient(90deg, #0f5fa8, #1580d3);
        padding: 18px 24px;
        border-radius: 16px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.12);
    }

    .topbar-title {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .topbar-sub {
        font-size: 14px;
        opacity: 0.95;
    }

    .card {
        background-color: white;
        padding: 18px;
        border-radius: 18px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.08);
        margin-bottom: 18px;
    }

    .metric-card {
        background-color: white;
        padding: 18px;
        border-radius: 18px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.08);
        text-align: center;
        margin-bottom: 14px;
    }

    .metric-title {
        font-size: 14px;
        color: #666;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #0f5fa8;
    }

    .section-title {
        font-size: 22px;
        font-weight: 700;
        color: #183b56;
        margin-bottom: 10px;
    }

    .prediction-box {
        background: #eaf4ff;
        border-left: 6px solid #1580d3;
        padding: 16px;
        border-radius: 12px;
        font-size: 18px;
        font-weight: 600;
        color: #123b63;
        margin-bottom: 12px;
    }

    .info-box {
        background: #fff7e6;
        border-left: 6px solid #f0ad4e;
        padding: 14px;
        border-radius: 12px;
        color: #7a5710;
        margin-bottom: 18px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="topbar">
    <div class="topbar-title">Tweet Sentiment Dashboard</div>
    <div class="topbar-sub">Tweet sentiment prediction with optional analytics dashboard</div>
</div>
""", unsafe_allow_html=True)

# ---------------- DATASET NOTICE ----------------
if not dataset_available:
    st.markdown("""
    <div class="info-box">
        Dataset file not found. Dashboard analytics are disabled, but sentiment prediction still works using the pre-trained model files.
    </div>
    """, unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.header("Filters")

if dataset_available:
    keyword = st.sidebar.text_input("Keyword")

    selected_sentiments = st.sidebar.multiselect(
        "Sentiment",
        options=["positive", "negative"],
        default=["positive", "negative"]
    )

    user_filter = st.sidebar.text_input("Username contains")

    min_date = df["date"].min().date()
    max_date = df["date"].max().date()

    date_range = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
else:
    st.sidebar.info("Upload the dataset file to enable filters and analytics.")
    keyword = ""
    selected_sentiments = []
    user_filter = ""
    date_range = None

# ---------------- APPLY FILTERS ----------------
if dataset_available:
    filtered_df = df.copy()

    if keyword:
        filtered_df = filtered_df[
            filtered_df["text"].str.contains(keyword, case=False, na=False)
        ]

    if selected_sentiments:
        filtered_df = filtered_df[
            filtered_df["sentiment"].isin(selected_sentiments)
        ]

    if user_filter:
        filtered_df = filtered_df[
            filtered_df["user"].str.contains(user_filter, case=False, na=False)
        ]

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df["date"].dt.date >= start_date) &
            (filtered_df["date"].dt.date <= end_date)
        ]
else:
    filtered_df = pd.DataFrame(columns=["sentiment", "date", "user", "text"])

# ---------------- METRICS + CHARTS ----------------
if dataset_available:
    total_tweets = len(filtered_df)
    positive_count = (filtered_df["sentiment"] == "positive").sum()
    negative_count = (filtered_df["sentiment"] == "negative").sum()

    left_col, right_col = st.columns([3.2, 1.2])

    with right_col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Tweets</div>
            <div class="metric-value">{total_tweets}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Positive Tweets</div>
            <div class="metric-value">{positive_count}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Negative Tweets</div>
            <div class="metric-value">{negative_count}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Applied Filters")
        st.write(f"**Keyword:** {keyword if keyword else 'All'}")
        st.write(f"**Sentiment:** {', '.join(selected_sentiments) if selected_sentiments else 'All'}")
        st.write(f"**Username:** {user_filter if user_filter else 'All'}")
        st.markdown('</div>', unsafe_allow_html=True)

    with left_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Sentiment Distribution</div>', unsafe_allow_html=True)

        if not filtered_df.empty:
            sentiment_counts = filtered_df["sentiment"].value_counts().reset_index()
            sentiment_counts.columns = ["Sentiment", "Count"]

            fig_donut = px.pie(
                sentiment_counts,
                names="Sentiment",
                values="Count",
                hole=0.65
            )
            fig_donut.update_layout(
                height=360,
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.warning("No data found for the selected filters.")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Sentiment Trend Over Time</div>', unsafe_allow_html=True)

        if not filtered_df.empty:
            trend_df = filtered_df.copy()
            trend_df["day"] = trend_df["date"].dt.date

            trend_counts = (
                trend_df.groupby(["day", "sentiment"])
                .size()
                .reset_index(name="count")
            )

            fig_line = px.line(
                trend_counts,
                x="day",
                y="count",
                color="sentiment",
                markers=True
            )
            fig_line.update_layout(
                height=320,
                xaxis_title="Date",
                yaxis_title="Tweet Count",
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("No trend data available.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Filtered Tweets</div>', unsafe_allow_html=True)

    if not filtered_df.empty:
        display_df = filtered_df[["date", "user", "text", "sentiment"]].copy()
        display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
        st.dataframe(display_df.head(25), use_container_width=True)
    else:
        st.write("No matching tweets found.")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PREDICTION SECTION ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Analyze New Tweet</div>', unsafe_allow_html=True)

user_input = st.text_area("Enter tweet text", height=120)

if st.button("Predict Sentiment"):
    if user_input.strip():
        cleaned_text = clean_text(user_input)
        vectorized_text = vectorizer.transform([cleaned_text])

        prediction = model.predict(vectorized_text)[0]
        probabilities = model.predict_proba(vectorized_text)[0]

        is_sarcasm = detect_sarcasm(user_input)

        if is_sarcasm and prediction == "positive":
            prediction = "negative"

        if is_sarcasm:
            display_text = f"{prediction.capitalize()} (Sarcasm detected)"
        else:
            display_text = prediction.capitalize()

        st.markdown(
            f'<div class="prediction-box">Predicted Sentiment: {display_text}</div>',
            unsafe_allow_html=True
        )

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
            height=300,
            xaxis_title="Sentiment",
            yaxis_title="Confidence",
            margin=dict(t=20, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.warning("Please enter tweet text.")

st.markdown('</div>', unsafe_allow_html=True)