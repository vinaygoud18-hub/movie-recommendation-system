"""
================================================================================
  STREAMLIT WEB APP — Movie Recommendation System
================================================================================
Streamlit is a Python library that turns Python scripts into web apps.
No HTML/CSS/JavaScript needed!

HOW TO RUN:
  streamlit run app.py

Make sure you've run recommendation_engine.py first to generate model/ folder.
================================================================================
"""

import streamlit as st
import pickle
import os
import requests

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource   # cache so we don't reload on every interaction
def load_model(model_dir="model"):
    """Load the pre-computed movie DataFrame and similarity matrix."""
    movies_path     = os.path.join(model_dir, "movies.pkl")
    similarity_path = os.path.join(model_dir, "similarity.pkl")

    if not os.path.exists(movies_path) or not os.path.exists(similarity_path):
        st.error(
            "❌ Model files not found! "
            "Please run `python recommendation_engine.py` first to build the model."
        )
        st.stop()

    with open(movies_path, "rb") as f:
        movies = pickle.load(f)
    with open(similarity_path, "rb") as f:
        similarity = pickle.load(f)

    return movies, similarity


# ─────────────────────────────────────────────────────────────────────────────
# FETCH MOVIE POSTER FROM TMDB API (Optional Enhancement)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_poster(movie_title, api_key=None):
    """Fetch poster from OMDb using movie title."""
    if not api_key:
        return "https://via.placeholder.com/300x450/1a1a2e/ffffff?text=No+Poster"
    try:
        url = f"http://www.omdbapi.com/?t={movie_title}&apikey={api_key}"
        response = requests.get(url, timeout=5)
        data = response.json()
        poster = data.get("Poster")
        if poster and poster != "N/A":
            return poster
    except:
        pass
    return "https://via.placeholder.com/300x450/1a1a2e/ffffff?text=No+Poster"
# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDATION FUNCTION (duplicated here so app.py is self-contained)
# ─────────────────────────────────────────────────────────────────────────────

def recommend(movie_name, movies, similarity, top_n=10):
    """
    Return top_n movie recommendations with their movie_ids (for poster fetching).
    Returns list of (title, movie_id) tuples.
    """
    matches = movies[movies['title'].str.lower() == movie_name.lower()]
    if matches.empty:
        matches = movies[movies['title'].str.lower().str.contains(movie_name.lower())]
        if matches.empty:
            return []

    movie_index = matches.index[0]
    scores = list(enumerate(similarity[movie_index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_movies = scores[1: top_n + 1]

    results = []
    for idx, score in top_movies:
        row = movies.iloc[idx]
        results.append({
            "title":    row["title"],
            "movie_id": int(row["movie_id"]),
            "score":    round(score, 3)
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a dark, cinematic feel
st.markdown("""
<style>
    .stApp { background-color: #0e0e1a; color: #ffffff; }
    .movie-card {
        background: #1a1a2e;
        border-radius: 12px;
        padding: 12px;
        margin: 8px 0;
        border: 1px solid #2a2a4a;
        text-align: center;
    }
    .movie-title { font-size: 0.85em; font-weight: bold; margin-top: 8px; color: #e0e0ff; }
    .similarity-score { font-size: 0.75em; color: #888; }
    h1 { color: #e0b400 !important; }
    .stSelectbox label { color: #aaa; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
# 🎬 CineMatch
### Movie Recommendation System
""")
    st.markdown("---")

    st.subheader("⚙️ Settings")
    top_n = st.slider("Number of recommendations", min_value=5, max_value=20, value=10, step=1)

    st.markdown("---")
    st.subheader("🔑 OMDB/TMDB API Key (Optional)")
    api_key = st.text_input(
        "Enter your OMDB/TMDB API key to show posters",
        type="password",
        help="Get a free OMDB API key at http://www.omdbapi.com/apikey.aspx and TMDB key at https://www.themoviedb.org/settings/api"
        
    )

    st.markdown("---")
    st.markdown("""
    ### 📖 How it works
    1. Select a movie you like
    2. The system extracts its **genres, keywords, cast, director & overview**
    3. It finds movies with the most similar **content** using **Cosine Similarity**
    4. Returns the top matches!
    
    **Algorithm**: CountVectorizer + Cosine Similarity
    """)
    st.markdown("---")

    st.markdown("""
    👨‍💻 Developed by Vinay Goud

    🔗 [LinkedIn](https://www.linkedin.com/in/vinaygoud-vintapuram/)

    💻 [GitHub](https://github.com/vinaygoud18-hub)
    """)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────

st.title("🎬 CineMatch — Movie Recommendation System")
st.caption("Content-Based Filtering using TMDB 5000 Dataset")

# Load model
movies, similarity = load_model()

# Movie selector
movie_list = sorted(movies['title'].values)

col1, col2 = st.columns([3, 1])
with col1:
    selected_movie = st.selectbox(
    "🔍 Search for a movie you like:",
    movie_list,
    index=None,
    placeholder="Select a movie..."
)

with col2:
    st.write("")  # spacing
    st.write("")
    recommend_clicked = st.button("🎯 Recommend", use_container_width=True)

# ─── SHOW RECOMMENDATIONS ────────────────────────────────────────────────────

if selected_movie and recommend_clicked:
    with st.spinner("Finding similar movies..."):
        recommendations = recommend(selected_movie, movies, similarity, top_n)

    if not recommendations:
        st.warning(f"No recommendations found for '{selected_movie}'.")
    else:
        st.success(f"✅ Top {len(recommendations)} movies similar to **{selected_movie}**")
        st.markdown("---")

        # Display in a responsive grid (5 columns)
        cols = st.columns(5)
        for i, rec in enumerate(recommendations):
            col = cols[i % 5]
            with col:
                poster_url = fetch_poster(rec["title"], api_key if api_key else None)
                st.image(poster_url, use_column_width=True)
                st.markdown(
                    f"<div class='movie-title'>{rec['title']}</div>"
                    f"<div class='similarity-score'>Similarity: {rec['score']}</div>",
                    unsafe_allow_html=True
                )

        # Show raw table below
        with st.expander("📊 Show detailed similarity scores"):
            import pandas as pd
            rec_df = pd.DataFrame(recommendations)
            rec_df.index = range(1, len(rec_df) + 1)
            rec_df.columns = ["Title", "Movie ID", "Similarity Score"]
            st.dataframe(rec_df[["Title", "Similarity Score"]], use_container_width=True)

elif selected_movie and not recommend_clicked:
    # Show movie info
    row = movies[movies['title'] == selected_movie].iloc[0]
    st.info(f"📽️ You selected: **{selected_movie}** (Movie ID: {row['movie_id']}). Click **Recommend** to find similar movies!")

elif not selected_movie and recommend_clicked:
    st.warning("Please select a movie first.")

# Footer
st.markdown("---")
st.markdown(
    "<center style='color:#555'>Built by Vinay Goud using Python · Scikit-learn · Streamlit · TMDB Dataset</center>",
    unsafe_allow_html=True
)
