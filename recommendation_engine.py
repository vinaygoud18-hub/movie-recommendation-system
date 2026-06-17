"""
================================================================================
  CONTENT-BASED MOVIE RECOMMENDATION SYSTEM
  Built with: Python, Pandas, Scikit-learn
  Dataset: TMDB 5000 Movies Dataset

"""

import pandas as pd
import numpy as np
import ast
import pickle
import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Stemmer: use NLTK's PorterStemmer if available, else a lightweight fallback
try:
    from nltk.stem import PorterStemmer as _NltkStemmer
    _stemmer = _NltkStemmer()
    def _stem(word):
        return _stemmer.stem(word)
except ImportError:
    # Minimal suffix-stripping rules (covers most English inflections)
    def _stem(word):
        for suffix in ('ings','ing','tion','tions','ness','ful','ous','ly','es','ed','er','s'):
            if word.endswith(suffix) and len(word) - len(suffix) > 3:
                return word[:-len(suffix)]
        return word

# ─── STEP 1: LOAD DATA ───────────────────────────────────────────────────────
# We have two CSVs:
#   - movies.csv : title, genres, keywords, overview, etc.
#   - credits.csv: cast and crew info per movie

def load_data(movies_path, credits_path):
    """
    Load both CSVs and merge them on title.
    Returns a single DataFrame with all info we need.
    """
    print("[1/6] Loading data...")
    movies  = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)

    # Merge on 'title'. Both files share this column.
    # Think of it like a SQL JOIN: SELECT * FROM movies JOIN credits ON title
    df = movies.merge(credits, on='title')

    # Keep only the columns we'll use for recommendations
    df = df[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

    print(f"    Loaded {len(df)} movies.")
    return df


# ─── STEP 2: CLEAN DATA ──────────────────────────────────────────────────────
""" genres, keywords, cast, crew are stored as JSON strings (like a Python list
 of dicts). We need to *parse* them into actual Python lists.
 Example of genres value: "[{'id': 28, 'name': 'Action'}, {'id': 12, 'name': 'Adventure'}]"
 We want: ['Action', 'Adventure']"""

def parse_list_column(text):
    """Convert a stringified list-of-dicts into a list of 'name' values."""
    try:
        items = ast.literal_eval(text)   # safely parse string → Python list
        return [item['name'] for item in items]
    except:
        return []

def get_director(crew_text):
    """
    Extract the director's name from the crew column.
    The crew column is a list of dicts with 'job' and 'name' keys.
    """
    try:
        crew = ast.literal_eval(crew_text)
        for member in crew:
            if member.get('job') == 'Director':
                return [member['name']]   # return as list for easy joining later
        return []
    except:
        return []

def get_top_cast(cast_text, top_n=3):
    """
    Extract top N cast members (most billed = most famous for that movie).
    """
    try:
        cast = ast.literal_eval(cast_text)
        return [member['name'] for member in cast[:top_n]]
    except:
        return []

def clean_data(df):
    """
    Apply all parsing/cleaning steps to the DataFrame.
    """
    print("[2/6] Cleaning and parsing data...")

    # Drop rows where overview is missing (can't use for similarity)
    df = df.dropna(subset=['overview'])

    # Parse JSON string columns into Python lists
    df['genres']   = df['genres'].apply(parse_list_column)
    df['keywords'] = df['keywords'].apply(parse_list_column)
    df['cast']     = df['cast'].apply(get_top_cast)
    df['crew']     = df['crew'].apply(get_director)

    # Convert overview string to a list of words (split on spaces)
    df['overview'] = df['overview'].apply(lambda x: x.split())

    print(f"    Cleaned {len(df)} movies.")
    return df


# ─── STEP 3: COLLAPSE SPACES IN MULTI-WORD NAMES ─────────────────────────────
""" "Sam Raimi" as two words would confuse the vectorizer — "Sam" and "Raimi"
 are treated separately, and "Sam" might match other "Sam"s.
 Fix: collapse to "SamRaimi" so it's treated as ONE token."""

def collapse_spaces(df):
    """
    Remove spaces within each name/phrase so multi-word names
    are treated as a single token by the vectorizer.
    """
    print("[3/6] Collapsing spaces in multi-word tokens...")

    def join_tokens(lst):
        return [item.replace(" ", "") for item in lst]

    df['genres']   = df['genres'].apply(join_tokens)
    df['keywords'] = df['keywords'].apply(join_tokens)
    df['cast']     = df['cast'].apply(join_tokens)
    df['crew']     = df['crew'].apply(join_tokens)

    return df


# ─── STEP 4: BUILD TAGS ──────────────────────────────────────────────────────
# Combine all features into a single string called "tags".
# This becomes the document we'll vectorize.

def build_tags(df):
    """
    Concatenate overview words + genres + keywords + cast + director
    into a single string per movie.
    """
    print("[4/6] Building movie tags...")

    df['tags'] = (
        df['overview'] +
        df['genres']   +
        df['keywords'] +
        df['cast']     +
        df['crew']
    )

    # Join list back into a space-separated string (CountVectorizer needs strings)
    df['tags'] = df['tags'].apply(lambda x: " ".join(x).lower())

    return df[['movie_id', 'title', 'tags']]   # keep only what we need


# ─── STEP 5: VECTORIZE & COMPUTE SIMILARITY ──────────────────────────────────
""" TEXT → NUMBERS using CountVectorizer (Bag of Words)
 CountVectorizer builds a vocabulary from all tags, then represents
 each movie as a vector of word counts.

 Example:
   Vocabulary: [action, hero, villain, love, ...]
   "Avatar":   [3, 2, 1, 0, ...]
   "Titanic":  [0, 1, 0, 4, ...]

 COSINE SIMILARITY measures the angle between two vectors.
   cos(θ) = 1  → identical direction → very similar
   cos(θ) = 0  → perpendicular      → not similar at all

 Why cosine and not Euclidean distance?
 Because cosine ignores the *magnitude* (length) of vectors and focuses
 on *direction*, which is better for text (a long review vs short review
 about the same topic should still be "similar")."""

def build_similarity_matrix(df):
    """
    Vectorize tags using CountVectorizer and compute cosine similarity matrix.
    Returns (vectorizer, similarity_matrix).
    """
    print("[5/6] Vectorizing and computing cosine similarity...")

    # Apply stemming to reduce words to root form
    # "loving", "loved", "loves" → "love"
    # This reduces vocabulary size and improves matching
    def stem_text(text):
        return " ".join([_stem(word) for word in text.split()])

    df['tags'] = df['tags'].apply(stem_text)

    # max_features=5000 → keep only the 5000 most frequent words (reduces noise)
    # stop_words='english' → ignore "the", "is", "and", etc. (not informative)
    cv = CountVectorizer(max_features=5000, stop_words='english')

    # fit_transform: learn vocabulary AND convert tags to a word-count matrix
    # Shape: (num_movies, 5000)
    vectors = cv.fit_transform(df['tags']).toarray()

    # cosine_similarity computes similarity between EVERY pair of movies
    # Shape: (num_movies, num_movies)
    # similarity[i][j] = how similar movie i is to movie j
    similarity = cosine_similarity(vectors)

    print(f"    Similarity matrix shape: {similarity.shape}")
    return similarity


# ─── STEP 6: RECOMMEND FUNCTION ──────────────────────────────────────────────

def recommend(movie_name, df, similarity, top_n=10):
    """
    Given a movie title, return the top_n most similar movies.

    Parameters:
        movie_name (str): Exact or partial movie title
        df (DataFrame)  : Processed DataFrame with 'title' and 'tags'
        similarity      : Cosine similarity matrix
        top_n (int)     : Number of recommendations to return

    Returns:
        list of recommended movie titles (strings)
    """
    # Find the movie in our DataFrame (case-insensitive)
    matches = df[df['title'].str.lower() == movie_name.lower()]

    if matches.empty:
        # Try partial match
        matches = df[df['title'].str.lower().str.contains(movie_name.lower())]
        if matches.empty:
            return []

    # Get the integer index of the movie in the DataFrame
    movie_index = matches.index[0]

    # Get similarity scores for this movie vs all other movies
    # This is one row of the similarity matrix
    scores = list(enumerate(similarity[movie_index]))

    # Sort by similarity score (descending). scores[i] = (index, score)
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # Skip index 0 (the movie itself — always similarity = 1.0)
    top_movies = scores[1 : top_n + 1]

    # Retrieve titles using the indices
    recommended = [df.iloc[i[0]]['title'] for i in top_movies]
    return recommended


# ─── STEP 7: SAVE MODEL ARTIFACTS ────────────────────────────────────────────
""" We save the processed DataFrame and similarity matrix as .pkl files
 so the Streamlit app can load them instantly (no re-computation needed)."""

def save_artifacts(df, similarity, output_dir="."):
    """Save processed data and similarity matrix as pickle files."""
    print("[6/6] Saving model artifacts...")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "movies.pkl"), "wb") as f:
        pickle.dump(df, f)

    with open(os.path.join(output_dir, "similarity.pkl"), "wb") as f:
        pickle.dump(similarity, f)

    print(f"    Artifacts saved to '{output_dir}/'")


def load_artifacts(output_dir="."):
    """Load saved artifacts."""
    with open(os.path.join(output_dir, "movies.pkl"), "rb") as f:
        df = pickle.load(f)
    with open(os.path.join(output_dir, "similarity.pkl"), "rb") as f:
        similarity = pickle.load(f)
    return df, similarity


# ─── MAIN: RUN FULL PIPELINE ─────────────────────────────────────────────────

def build_model(movies_path, credits_path, output_dir="."):
    """
    End-to-end pipeline: load → clean → build tags → vectorize → save.
    Call this once to train and save your model.
    """
    df         = load_data(movies_path, credits_path)
    df         = clean_data(df)
    df         = collapse_spaces(df)
    df         = build_tags(df)
    similarity = build_similarity_matrix(df)
    save_artifacts(df, similarity, output_dir)
    print("\n✅ Model built successfully!")
    return df, similarity


if __name__ == "__main__":
    # ── Adjust these paths to where your CSVs are ──
    MOVIES_PATH  = "tmdb_5000_movies.csv"
    CREDITS_PATH = "tmdb_5000_credits.csv"
    OUTPUT_DIR   = "model"

    df, similarity = build_model(MOVIES_PATH, CREDITS_PATH, OUTPUT_DIR)

    # Quick test
    print("\n🎬 Test recommendations for 'Avatar':")
    recs = recommend("Avatar", df, similarity)
    for i, title in enumerate(recs, 1):
        print(f"  {i}. {title}")