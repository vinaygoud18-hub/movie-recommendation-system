# CineMatch — Movie Recommendation System

A content-based movie recommendation system that suggests movies similar to the one you pick. Built using Python, Scikit-learn and Streamlit.

**Dataset:** TMDB 5000 Movies from Kaggle (4,800+ movies)

---

## Demo

Select any movie → get 10 similar movies with posters instantly.

| Input | Recommendations |
|---|---|
| The Dark Knight | The Dark Knight Rises, Batman Begins, Batman Returns |
| Toy Story | Toy Story 2, Toy Story 3 |
| Avatar | Independence Day, Battle: Los Angeles |

---

## How it works

Each movie is described using 5 features — genres, keywords, top 3 cast members, director and plot overview. These are combined into a single text string, converted into word-count vectors using CountVectorizer, and compared using Cosine Similarity.

```
cos(θ) = (A · B) / (||A|| × ||B||)
```

Movies with the highest similarity score are returned as recommendations.

---

## Project structure

```
├── app.py                     # Streamlit web app
├── recommendation_engine.py   # ML pipeline — data cleaning, vectorization, similarity
├── setup.py                   # Run once to build the model
├── requirements.txt           # Dependencies
├── tmdb_5000_movies.csv       # Dataset
├── tmdb_5000_credits.csv      # Dataset
└── model/
    └── movies.pkl             # Saved after running setup.py
                               # similarity.pkl is not included (176MB — too large for GitHub)
                               # it gets generated when you run setup.py
```

---

## Run locally

```bash
# 1. Clone the repo
git clone https://github.com/vinaygoud18-hub/movie-recommendation-system.git
cd movie-recommendation-system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Build the model (generates similarity.pkl locally)
python setup.py

# 4. Start the app
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## Movie posters

Get a free OMDb API key at [omdbapi.com/apikey.aspx](http://www.omdbapi.com/apikey.aspx) and paste it in the sidebar. Free tier gives 1,000 requests/day which is more than enough.

---

## Tech stack

| Library | Used for |
|---|---|
| Pandas | loading and cleaning the dataset |
| Scikit-learn | CountVectorizer + cosine_similarity |
| Streamlit | web app UI |
| Pickle | saving the processed model |
| OMDb API | fetching movie posters |

---

## Note on similarity.pkl

The similarity matrix is a 4806 × 4806 array which comes out to ~176MB — over GitHub's 100MB file limit. It is excluded from this repo. Running `python setup.py` will generate it on your machine in under a minute.

---

Built by [Vinay Goud](https://github.com/vinaygoud18-hub) — 3rd year B.Tech CSE (Data Science)

