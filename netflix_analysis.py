import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from matplotlib.ticker import FuncFormatter
from collections import Counter
import sqlite3
import logging
import os
from io import BytesIO
import base64
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader

from ml_model import train_rf_model

# ------------------ Streamlit Page Setup ------------------
st.set_page_config(page_title="Netflix Dashboard", layout="wide")

# ------------------ Logging Setup ------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/netflix_dashboard.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ------------------ Constants ------------------
CSV_FILE = "netflix1.csv"
DB_FILE = "netflix_data.db"
TABLE_NAME = "netflix_data"

# ------------------ Load and Clean Data ------------------
@st.cache_data
def load_and_clean_data():
    try:
        df = pd.read_csv(CSV_FILE)
        logging.info("Dataset loaded successfully. Shape: %s", df.shape)
    except Exception as e:
        logging.error(f"Error loading CSV: {e}")
        st.error("Failed to load Netflix CSV file.")
        return pd.DataFrame()

    df.dropna(subset=['title', 'type', 'release_year'], inplace=True)

    for col in ['director', 'cast', 'country', 'date_added', 'rating', 'duration', 'listed_in']:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

    # Apply strip only to object (string) columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)

    if 'date_added' in df.columns:
        df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')

    df['duration_mins'] = df['duration'].str.extract(r'(\d+)').astype(float)
    df['genres_list'] = df['listed_in'].apply(lambda x: [genre.strip() for genre in x.split(',')] if pd.notnull(x) else [])
    df['genres'] = df['genres_list'].apply(lambda x: ', '.join(x))
    df['genre_encoded'] = pd.factorize(df['genres'])[0]

    return df

data = load_and_clean_data()

# ------------------ Save to SQLite DB ------------------
def save_to_database(df, db_path=DB_FILE):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS netflix_data (
                show_id TEXT, type TEXT, title TEXT, director TEXT,
                country TEXT, date_added TEXT, release_year INTEGER,
                rating TEXT, duration TEXT, listed_in TEXT,
                genres_list TEXT, genres TEXT
            )
        ''')

        cursor.execute(f'DELETE FROM {TABLE_NAME}')

        for _, row in df.iterrows():
            genres_list_str = ', '.join(row['genres_list']) if isinstance(row['genres_list'], list) else ''
            cursor.execute('''
                INSERT INTO netflix_data (
                    show_id, type, title, director, country,
                    date_added, release_year, rating, duration,
                    listed_in, genres_list, genres
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                row.get('show_id'), row.get('type'), row.get('title'), row.get('director'), row.get('country'),
                str(row.get('date_added')), row.get('release_year'), row.get('rating'), row.get('duration'),
                row.get('listed_in'), genres_list_str, row.get('genres')
            ))

        conn.commit()
        logging.info("Data inserted into SQLite successfully.")
    except Exception as e:
        logging.error(f"Error inserting into SQLite: {e}")
        st.error("Database write failed.")
    finally:
        conn.close()

if not data.empty:
    save_to_database(data)

# ------------------ Dashboard UI ------------------
st.title("🎬 Netflix Content Analysis Dashboard")
st.markdown("Visual insights into Netflix's catalog based on content type, country, genres, and trends over time.")
st.markdown("Use the filters to explore Netflix content by country, year, and type.")

# ----------------- Filters -----------------
st.sidebar.title("Filter Netflix Data")
year_filter = st.sidebar.slider("Select Release Year", 1940, 2025, 2015)
country_filter = st.sidebar.selectbox("Select Country", ['All'] + sorted(data['country'].dropna().unique()))

filtered_data = data.copy()
if year_filter:
    filtered_data = filtered_data[filtered_data['release_year'] == year_filter]
if country_filter != 'All':
    filtered_data = filtered_data[filtered_data['country'] == country_filter]

# ✅ Define filtered_df to avoid NameError
filtered_df = filtered_data.copy()

# ------------------ Genre Chart with Orientation + Labels ------------------
st.subheader("\U0001F4CA Genre Distribution Chart")
genre_counts = data['genres'].value_counts().head(10)
total = genre_counts.sum()

orientation = st.selectbox("Select Chart Orientation", ["Horizontal", "Vertical"])
fig, ax = plt.subplots(figsize=(10, 6))
palette = 'viridis'

if orientation == "Horizontal":
    sns.barplot(x=genre_counts.values, y=genre_counts.index, hue=genre_counts.index, palette=palette, ax=ax)
    for i, v in enumerate(genre_counts.values):
        ax.text(v + 1, i, f"{(v/total)*100:.1f}%", va='center')
    ax.set_xlabel("Count")
    ax.set_ylabel("Genre")
else:
    sns.barplot(y=genre_counts.values, x=genre_counts.index,hue=genre_counts.index, palette=palette, ax=ax)
    for i, v in enumerate(genre_counts.values):
        ax.text(i, v + 1, f"{(v/total)*100:.1f}%", ha='center')
    ax.set_ylabel("Count")
    ax.set_xlabel("Genre")
    plt.xticks(rotation=45)

st.pyplot(fig)

def export_chart(fig, format):
    buf = BytesIO()
    fig.savefig(buf, format=format, bbox_inches='tight')
    buf.seek(0)
    return buf

# ------------------ Charts ------------------
with st.expander("📺 Content by Type"):
    try:
        type_counts = data['type'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=type_counts.index, y=type_counts.values, hue=type_counts.index, palette='pastel', ax=ax, legend=False)
        ax.set_title("Content Count by Type")
        ax.set_xlabel("Type")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        # Download buttons
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("📥 Download Chart as PNG", export_chart(fig, "png"), file_name="content_type_chart.png")
        with col2:
            st.download_button("📄 Download Chart as PDF", export_chart(fig, "pdf"), file_name="content_type_chart.pdf")

    except Exception as e:
        st.error("Error loading content type chart.")
        logging.warning(f"KPI 1 error: {e}")

with st.expander("🌍 Top 10 Countries"):
    try:
        country_counts = data['country'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=country_counts.values, y=country_counts.index, hue=genre_counts.index, palette='viridis', ax=ax, legend=False)
        ax.set_title("Top 10 Countries with Most Content")
        ax.set_xlabel("Number of Titles")
        ax.set_ylabel("Country")
        st.pyplot(fig)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button("📥 Download Chart as PNG", export_chart(fig, "png"), file_name="top_countries_chart.png")
        with col2:
            st.download_button("📄 Download Chart as PDF", export_chart(fig, "pdf"), file_name="top_countries_chart.pdf")

    except Exception as e:
        st.error("Error loading country chart.")
        logging.warning(f"KPI 2 error: {e}")

with st.expander("📅 Content Added Over the Years"):
    try:
        if 'date_added' in data.columns:
            content_by_year = data['date_added'].dt.year.value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.lineplot(x=content_by_year.index, y=content_by_year.values, marker='o', ax=ax)
            ax.set_title("Content Added Over the Years")
            ax.set_xlabel("Year")
            ax.set_ylabel("Number of Titles")
            st.pyplot(fig)

            col1, col2 = st.columns(2)
            with col1:
                st.download_button("📥 Download Chart as PNG", export_chart(fig, "png"), file_name="content_by_year.png")
            with col2:
                st.download_button("📄 Download Chart as PDF", export_chart(fig, "pdf"), file_name="content_by_year.pdf")

    except Exception as e:
        st.error("Error loading timeline chart.")
        logging.warning(f"KPI 3 error: {e}")

with st.expander("🎭 Top 10 Genres"):
    try:
        all_genres = data['genres_list'].sum()
        genre_counts = pd.Series(Counter(all_genres)).sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=genre_counts.values, y=genre_counts.index, hue=genre_counts.index, palette='viridis', ax=ax, legend=False)
        ax.set_title("Top 10 Genres")
        ax.set_xlabel("Number of Titles")
        ax.set_ylabel("Genre")
        st.pyplot(fig)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button("📥 Download Chart as PNG", export_chart(fig, "png"), file_name="top_genres_chart.png")
        with col2:
            st.download_button("📄 Download Chart as PDF", export_chart(fig, "pdf"), file_name="top_genres_chart.pdf")

    except Exception as e:
        st.error("Error loading genre chart.")
        logging.warning(f"KPI 4 error: {e}")

# ------------------ Extra Visuals ------------------
st.subheader("📊 Rating Distribution (Filtered)")
fig1, ax1 = plt.subplots()
filtered_df['rating'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax1)
ax1.set_ylabel('')
st.pyplot(fig1)


# ---------- Animated Bar Chart by Release Year ----------
st.subheader("📽️ Animated Content Type Distribution Over Time")
df_anim = filtered_df.dropna(subset=['release_year', 'type'])
df_anim['release_year'] = df_anim['release_year'].astype(int)
df_grouped = df_anim.groupby(['release_year', 'type']).size().reset_index(name='count')
fig_anim = px.bar(
    df_grouped,
    x='type',
    y='count',
    color='type',
    animation_frame='release_year',
    title='Type Distribution Over Years (Animated)',
    labels={'type': 'Content Type', 'count': 'Number of Titles'},
    height=500
)
st.plotly_chart(fig_anim)
    
st.subheader("☁️ Genre Word Cloud")
wordcloud = WordCloud(width=800, height=400, background_color='black').generate(" ".join(filtered_df['listed_in'].dropna()))
st.image(wordcloud.to_array())


    
# ------------------ Export to Excel ------------------
def generate_excel_download(data):
    towrite = BytesIO()
    data.to_excel(towrite, index=False, sheet_name='Netflix Data')
    towrite.seek(0)
    return base64.b64encode(towrite.read()).decode()

st.subheader("📥 Download Filtered Data")
excel_data = generate_excel_download(filtered_df)
st.markdown(f'<a href="data:application/octet-stream;base64,{excel_data}" download="filtered_netflix.xlsx">📄 Download Excel File</a>', unsafe_allow_html=True)

# ------------------ Recommendation System ------------------
st.subheader("🤖 Recommend Similar Shows")
selected_title = st.selectbox("Select a Title for Recommendations", data['title'].dropna().unique())

def recommend_titles(selected_title, data, top_n=5):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['listed_in'].fillna(''))
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(data.index, index=data['title']).drop_duplicates()
    idx = indices[selected_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    recommended_indices = [i[0] for i in sim_scores]
    return data['title'].iloc[recommended_indices]

if selected_title:
    st.markdown("### 🔁 Similar Recommendations:")
    recommendations = recommend_titles(selected_title, data)
    st.write(recommendations)

# ------------------ Clustering ------------------
if st.button("Cluster Shows by Genre + Duration"):
    kmeans_data = data[['genre_encoded', 'duration_mins']].dropna()
    kmeans = KMeans(n_clusters=3, random_state=0)
    kmeans_data['cluster'] = kmeans.fit_predict(kmeans_data)

    fig, ax = plt.subplots()
    scatter = ax.scatter(kmeans_data['genre_encoded'], kmeans_data['duration_mins'], c=kmeans_data['cluster'], cmap='viridis')
    plt.xlabel('Genre Encoded')
    plt.ylabel('Duration (mins)')
    st.pyplot(fig)

# ------------------ Export PDF Report ------------------
if st.button("Export PDF Report"):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "Netflix Report")
    c.setFont("Helvetica", 12)
    c.drawString(100, 730, f"Total Shows: {len(data)}")
    c.drawString(100, 710, f"Filtered Shows: {len(filtered_df)}")
    c.drawString(100, 690, f"Unique Genres: {data['genres'].nunique()}")
    c.drawString(100, 670, f"Selected Year: {year_filter}")
    c.drawString(100, 650, f"Selected Country: {country_filter}")
    c.showPage()
    c.save()

    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    st.markdown(f'<a href="data:application/pdf;base64,{b64}" download="netflix_report.pdf">📄 Download PDF Report</a>', unsafe_allow_html=True)


        
# --------- Feature Engineering ----------
def preprocess_for_model(df):
    df = df.copy()
    df.dropna(subset=['duration'], inplace=True)
    df['duration'] = df['duration'].fillna('0 min')
    df['duration_int'] = df['duration'].str.extract(r'(\d+)').astype(float)
    df['duration_type'] = df['duration'].str.extract(r'([a-zA-Z]+)')
    return df

# --------- ML Classification Model ----------
def train_rf_model(df):
    df_model = preprocess_for_model(df)
    X = df_model[['duration_int', 'rating']]
    y = df_model['type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return classification_report(y_test, y_pred, output_dict=True)

# --------- ML Classification Report Section ----------
st.subheader("ML Classification: Random Forest on Type Prediction")

if st.checkbox("Run Classification Model"):
    try:
        report = train_rf_model(df)
        st.success("Classification model executed successfully.")
        
        # Display as a table
        st.write("### Classification Report")
        st.dataframe(pd.DataFrame(report).transpose())
    
    except Exception as e:
        st.error(f"Error running classification model: {e}")

# --------- KMeans Clustering ----------
def cluster_genres(df):
    genre_series = df['listed_in'].str.split(', ').explode().value_counts().head(10)
    clustered_data = pd.DataFrame({
        'genre': genre_series.index,
        'count': genre_series.values
    })

    kmeans = KMeans(n_clusters=3, random_state=0)
    clustered_data['cluster'] = kmeans.fit_predict(clustered_data[['count']])

    fig, ax = plt.subplots()
    sns.scatterplot(data=clustered_data, x='genre', y='count', hue='cluster', palette='Set2', s=100, ax=ax)
    plt.xticks(rotation=45)
    ax.set_title("KMeans Clustering of Top Genres")
    st.pyplot(fig) 

st.success("✅ Dashboard loaded successfully.")