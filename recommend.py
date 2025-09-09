# ===============================
# 🎬 Advanced Movie Recommendation Dashboard with Dark/Light Theme
# ===============================

import streamlit as st
import pandas as pd
import re
import difflib
import urllib.parse
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# 1️⃣ Load Dataset
# -------------------------------
df = pd.read_csv("imdb_storylines_cleaned.csv")  # Must have columns: Movie Name, Storyline
df.rename(columns={'Movie Name': 'title', 'Storyline': 'overview'}, inplace=True)

# -------------------------------
# 2️⃣ Text Cleaning
# -------------------------------
STOP_WORDS = set(ENGLISH_STOP_WORDS)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [t for t in text.split() if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)

df['clean_overview'] = df['overview'].apply(clean_text)

# -------------------------------
# 3️⃣ TF-IDF Vectorization
# -------------------------------
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(df['clean_overview'])

# -------------------------------
# 4️⃣ Recommendation Functions
# -------------------------------
def recommend_by_title(movie_title, top_n=10):

    """Recommend movies based on a given movie title"""
    titles = df['title'].dropna().astype(str).tolist()
    matches = difflib.get_close_matches(str(movie_title), titles, n=1, cutoff=0.6)

    if not matches:
        return pd.DataFrame()

    idx = df[df['title'] == matches[0]].index[0]
    similarity = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similarity[idx] = -1

    top_idx = similarity.argsort()[::-1]
    unique_titles, unique_rows = [], []

    for i in top_idx:
        movie = str(df.iloc[i]['title'])
        if movie not in unique_titles:
            unique_titles.append(movie)
            unique_rows.append(df.iloc[i])
        if len(unique_rows) >= top_n:
            break

    return pd.DataFrame(unique_rows)[['title', 'overview']]

def recommend_by_storyline(user_story, top_n=10):
    user_clean = clean_text(user_story)
    user_vector = tfidf.transform([user_clean])
    similarity = cosine_similarity(user_vector, tfidf_matrix).flatten()
    top_idx = similarity.argsort()[::-1]

    unique_titles, unique_rows = [], []
    for i in top_idx:
        movie = str(df.iloc[i]['title'])
        if movie not in unique_titles:
            unique_titles.append(movie)
            unique_rows.append(df.iloc[i])
        if len(unique_rows) >= top_n:
            break

    return pd.DataFrame(unique_rows)[['title', 'overview']]

# -------------------------------
# 5️⃣ Streamlit Interface
# -------------------------------
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")
st.markdown("<h1 style='text-align:center; color:#FF4500;'>🎬 Advanced Movie Recommendation Dashboard</h1>", unsafe_allow_html=True)

# -------------------------------
# 6️⃣ Theme Toggle
# -------------------------------
theme = st.radio("Select Theme:", ["Dark", "Light"], horizontal=True)
if theme == "Light":
    bg_color = "#C7D4D4"
    text_color = "#000"
    shadow_color = "#aaa"
    link_color = "#0077CC"
else:
    bg_color = "#1f1f1f"
    text_color = "#fff"
    shadow_color = "#FF4500"
    link_color = "#00CED1"

# -------------------------------
# 7️⃣ Search Mode
# -------------------------------
# 7️⃣ Search Mode
# -------------------------------
search_mode = st.radio("Search by:", ["Movie Title", "Storyline"], horizontal=True)

if search_mode == "Movie Title":
    movie_input = st.text_input("Enter Movie Title:", placeholder="Start typing...").strip()
    selected_movie = None

    if movie_input:
        titles = df['title'].dropna().astype(str).tolist()
        matches = difflib.get_close_matches(movie_input, titles, n=5, cutoff=0.3)
        selected_movie = st.selectbox("Did you mean?", options=matches if matches else [movie_input], index=0)

    if st.button("Get Recommendations") and selected_movie:
        results = recommend_by_title(selected_movie, top_n=10)

        if results.empty:
            st.warning("❌ No recommendations found!")
        else:
            for _, row in results.iterrows():
                title = str(row['title']) if pd.notna(row['title']) else "Unknown Title"
                overview = str(row['overview']) if pd.notna(row['overview']) else "No overview available"
                imdb_link = "https://www.imdb.com/find?q=" + urllib.parse.quote(title)

                st.markdown(f"""
                    <div style='background-color:{bg_color}; padding:15px; border-radius:10px; 
                                margin-bottom:15px; box-shadow:2px 2px 10px {shadow_color};'>
                        <h2 style='color:#FF4500; margin:5px 0;'>{title}</h2>
                        <p style='color:{text_color}; font-size:14px;'>{overview}</p>
                        <a href='{imdb_link}' target='_blank' style='color:{link_color}; font-weight:bold;'>🔗 View on IMDb</a>
                    </div>
                """, unsafe_allow_html=True)


elif search_mode == "Storyline":
    story_input = st.text_area("Enter Storyline/Plot:", placeholder="Describe a movie storyline...").strip()
    if st.button("Get Recommendations") and story_input:
        results = recommend_by_storyline(story_input, top_n=10)  # Corrected call

        if results.empty:
            st.warning("❌ No recommendations found!")
        else:
            for _, row in results.iterrows():
                title = str(row['title']) if pd.notna(row['title']) else "Unknown Title"
                overview = str(row['overview']) if pd.notna(row['overview']) else "No overview available"
                imdb_link = "https://www.imdb.com/find?q=" + urllib.parse.quote(title)

                st.markdown(f"""
                    <div style='background-color:{bg_color}; padding:15px; border-radius:10px; 
                                margin-bottom:15px; box-shadow:2px 2px 10px {shadow_color};'>
                        <h2 style='color:#FF4500; margin:5px 0;'>{title}</h2>
                        <p style='color:{text_color}; font-size:14px;'>{overview}</p>
                        <a href='{imdb_link}' target='_blank' style='color:{link_color}; font-weight:bold;'>🔗 View on IMDb</a>
                    </div>
                """, unsafe_allow_html=True)

