"""
setup.py — Run this ONCE before launching the Streamlit app.

It will:
1. Download required NLTK data (for stemming)
2. Build the model from your CSVs and save to model/ folder
"""

import nltk
import os
import sys

print("=" * 60)
print("  SETUP: Movie Recommendation System")
print("=" * 60)

# Download NLTK punkt (needed for some tokenization)
print("\n[Setup] Downloading NLTK data...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
print("    Done.")

# Run the model builder
print("\n[Setup] Building recommendation model from CSV files...")

# Default paths — change these if your CSVs are elsewhere
MOVIES_CSV  = "tmdb_5000_movies.csv"
CREDITS_CSV = "tmdb_5000_credits.csv"

# Check files exist
for path in [MOVIES_CSV, CREDITS_CSV]:
    if not os.path.exists(path):
        print(f"\n❌ ERROR: '{path}' not found!")
        print("   Please place both CSV files in the same folder as this script.")
        sys.exit(1)

# Build model
from recommendation_engine import build_model
df, similarity = build_model(MOVIES_CSV, CREDITS_CSV, output_dir="model")

print("\n" + "=" * 60)
print("  ✅ Setup complete!")
print("  Now run:  streamlit run app.py")
print("=" * 60)