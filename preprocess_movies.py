import pandas as pd
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from scipy.sparse import hstack
import pickle
import numpy as np

# --- 1. Load Data ---
print("Loading Movies.db...")
conn = sqlite3.connect("Movies.db")
df = pd.read_sql_query("SELECT * FROM movies", conn)
conn.close()

# --- 2. Feature Engineering & Preprocessing ---
print("Preprocessing features...")

# Handle NaNs (fill with empty string)
df['genres'] = df['genres'].fillna('')
df['cast_top5'] = df['cast_top5'].fillna('')
df['director'] = df['director'].fillna('')
df['decade'] = df['decade'].fillna('Unknown')

# Create processed text fields (replace '|' with ' ' for vectorizer)
df['genres_processed'] = df['genres'].str.replace('|', ' ', regex=False)
df['cast_processed'] = df['cast_top5'].str.replace('|', ' ', regex=False)
df['director_processed'] = df['director'].str.replace('|', ' ', regex=False)

# Handle tmdb_rating (fill NaNs with 5.0, scale 0-1)
df['tmdb_rating_norm'] = df['tmdb_rating'].fillna(5.0)

# --- 3. Vectorization ---
print("Applying vectorizers...")

# --- THIS IS THE FIX ---
# Define a named function for the tokenizer.
# pickle cannot serialize (save) lambda functions.
def split_tokenizer(text):
    return text.split(' ')
# --- END OF FIX ---

# Initialize vectorizers
# --- FIX: Use the named function 'split_tokenizer' ---
tf_genres = TfidfVectorizer(tokenizer=split_tokenizer, min_df=2)
tf_cast = TfidfVectorizer(tokenizer=split_tokenizer, max_features=500) # Limit cast to top 500
tf_director = TfidfVectorizer(tokenizer=split_tokenizer, max_features=300) # Limit directors
# --- END OF FIX ---

oh_decade = OneHotEncoder(handle_unknown='ignore')
scale_rating = MinMaxScaler()

# Apply vectorizers and track column slices
col_slices = {}
current_col = 0

genre_matrix = tf_genres.fit_transform(df['genres_processed'])
col_slices['genres'] = (current_col, current_col + genre_matrix.shape[1])
current_col += genre_matrix.shape[1]

cast_matrix = tf_cast.fit_transform(df['cast_processed'])
col_slices['cast'] = (current_col, current_col + cast_matrix.shape[1])
current_col += cast_matrix.shape[1]

director_matrix = tf_director.fit_transform(df['director_processed'])
col_slices['director'] = (current_col, current_col + director_matrix.shape[1])
current_col += director_matrix.shape[1]

decade_matrix = oh_decade.fit_transform(df[['decade']])
col_slices['decade'] = (current_col, current_col + decade_matrix.shape[1])
current_col += decade_matrix.shape[1]

rating_matrix = scale_rating.fit_transform(df[['tmdb_rating_norm']])
col_slices['rating'] = (current_col, current_col + rating_matrix.shape[1])
current_col += rating_matrix.shape[1]

print(f"Total features: {current_col}")

# --- 4. Combine Matrices ---
print("Combining matrices...")
final_matrix = hstack([
    genre_matrix,
    cast_matrix,
    director_matrix,
    decade_matrix,
    rating_matrix
]).tocsr() # Convert to CSR for efficient storage and computation

# --- 5. Create Mapping ---
# Create a map from tmdb_id -> matrix_row_index
id_to_index = {tmdb_id: i for i, tmdb_id in enumerate(df['tmdb_id'])}

# --- 6. Save Artifacts ---
print("Saving artifacts to movie_artifacts.pkl...")
artifacts = {
    'matrix': final_matrix,
    'id_to_index': id_to_index,
    'vectorizers': {
        'genres': tf_genres,
        'cast': tf_cast,
        'director': tf_director,
        'decade': oh_decade,
        'rating': scale_rating
    },
    'col_slices': col_slices
}

with open("movie_artifacts.pkl", "wb") as f:
    pickle.dump(artifacts, f)

print("✅ Preprocessing complete!")
print(f"Final matrix shape: {final_matrix.shape}")
print(f"Column slices saved: {col_slices}")

