import streamlit as st
import sqlite3
import pandas as pd
import math
from datetime import datetime, UTC, timedelta
import pickle
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# Import all our SQL helper functions
from sql_utils import (
    signup_user, login_user, get_user_id, 
    add_user_movie_history, create_tables,
    get_user_profile_vector, save_user_profile_vector,
    get_user_history
)

# Page config MUST be first Streamlit call
st.set_page_config(page_title="Movie Recommender", layout="wide")

# --- Config ---
MOVIES_DB_PATH = "Movies.db"      # your movie DB
USER_DB_PATH = "user.db"
ARTIFACTS_PATH = "movie_artifacts.pkl"
TOTAL_TO_SHOW = 96
PAGE_SIZE = 12   # 4 cols x 3 rows
POSTER_W = 200
POSTER_H = 300
PLACEHOLDER_POSTER = f"https://via.placeholder.com/{POSTER_W}x{POSTER_H}?text=No+Image"

# Ensure user DB tables exist (and migrate them if needed)
create_tables()

# --- THIS IS THE FIX ---
# This function MUST be defined here, at the top level,
# so that pickle can find it when loading the artifacts.
def split_tokenizer(text):
    return text.split(' ')
# --- END OF FIX ---

# --- NEW: Load ML Artifacts ---
@st.cache_data
def load_artifacts(path):
    """Loads the pickled artifacts file (matrix, maps, vectorizers)."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"FATAL ERROR: '{path}' not found. Please run preprocess_movies.py first.")
        return None
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None

# Load artifacts into a global-like variable
# ARTIFACTS will be None if loading fails
ARTIFACTS = load_artifacts(ARTIFACTS_PATH)
if ARTIFACTS:
    MOVIE_MATRIX = ARTIFACTS['matrix']
    ID_TO_INDEX = ARTIFACTS['id_to_index']
    # Invert map for easy lookup from index -> id
    INDEX_TO_ID = {i: tmdb_id for tmdb_id, i in ID_TO_INDEX.items()} 
    MATRIX_SHAPE = MOVIE_MATRIX.shape
    # --- NEW: Load column slices for debug chart ---
    COL_SLICES = ARTIFACTS.get('col_slices') 
else:
    # Set dummies so app doesn't crash, error is already shown
    MOVIE_MATRIX, ID_TO_INDEX, INDEX_TO_ID, MATRIX_SHAPE, COL_SLICES = [None] * 5

# --- SESSION INIT ---
def init_session():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = None
    if "page" not in st.session_state:
        st.session_state.page = 1
    if "search_query" not in st.session_state:
        st.session_state.search_query = ""
    if "selected_tmdb_id" not in st.session_state:
        st.session_state.selected_tmdb_id = None
    if "filtered_count" not in st.session_state:
        st.session_state.filtered_count = 0
    # --- NEW: View state for navigation ---
    if "view" not in st.session_state:
        st.session_state.view = "home" # 'home', 'popular', 'trending', 'history', 'genre'

init_session()

# --- DB helpers for movies ---
def load_movies(limit=TOTAL_TO_SHOW, search_query=None, popular=True):
    """
    Loads movies. By default, loads 'popular' (top-rated) movies.
    If search_query is provided, it filters by title.
    """
    conn = sqlite3.connect(MOVIES_DB_PATH)
    cur = conn.cursor()
    if search_query:
        q = f"%{search_query}%"
        cur.execute("SELECT * FROM movies WHERE title LIKE ? COLLATE NOCASE LIMIT ?", (q, limit))
    elif popular:
        # Default sort: By rating, to show "popular" movies
        cur.execute("SELECT * FROM movies ORDER BY tmdb_rating DESC LIMIT ?", (limit,))
    else:
        # Just get any movies
        cur.execute("SELECT * FROM movies LIMIT ?", (limit,))
        
    cols = [c[0] for c in cur.description]
    rows = cur.fetchall()
    conn.close()
    df = pd.DataFrame(rows, columns=cols)
    return df

@st.cache_data
def load_trending_movies(limit=TOTAL_TO_SHOW):
    """
    Loads movies released in the current/last year with a high rating.
    Sorts by most recent.
    """
    conn = sqlite3.connect(MOVIES_DB_PATH)
    cur = conn.cursor()
    
    current_year = datetime.now(UTC).year
    last_year = current_year - 1
    
    # This query is now safe, no Python comments inside
    cur.execute("""
        SELECT *
        FROM movies
        WHERE (year = ? OR year = ?) AND tmdb_rating >= 7
        ORDER BY year DESC, tmdb_rating DESC
        LIMIT ?
    """, (current_year, last_year, limit))
    
    rows = cur.fetchall()
    cols = [c[0] for c in cur.description]
    conn.close()
    
    if not rows:
        return pd.DataFrame()
        
    df = pd.DataFrame(rows, columns=cols)
    return df

@st.cache_data
def get_all_genres():
    """Fetches all unique genres from the movies DB."""
    conn = sqlite3.connect(MOVIES_DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT genres FROM movies WHERE genres IS NOT NULL")
    rows = cur.fetchall()
    conn.close()
    
    all_genres = set()
    for row in rows:
        genres = str(row[0]).split('|')
        all_genres.update(genres)
    
    return sorted(list(all_genres))

def load_movies_by_genre(genre, limit=TOTAL_TO_SHOW):
    """Loads movies filtered by a specific genre."""
    conn = sqlite3.connect(MOVIES_DB_PATH)
    cur = conn.cursor()
    # Use LIKE to find the genre in the "Action|Drama|..." string
    q = f"%{genre}%"
    cur.execute("SELECT * FROM movies WHERE genres LIKE ? ORDER BY tmdb_rating DESC LIMIT ?", (q, limit))
    
    cols = [c[0] for c in cur.description]
    rows = cur.fetchall()
    conn.close()
    df = pd.DataFrame(rows, columns=cols)
    return df


def get_movie_by_tmdb_id(tmdb_id):
    conn = sqlite3.connect(MOVIES_DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM movies WHERE tmdb_id = ?", (tmdb_id,))
    row = cur.fetchone()
    cols = [c[0] for c in cur.description] if cur.description else []
    conn.close()
    return dict(zip(cols, row)) if row else None

# --- Login / Signup UI ---
def login_signup_page():
    # ... (No changes to this function) ...
    st.title("🎬 Movie Recommender — Login / Sign Up")
    tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])

    with tab1:
        with st.form("login_form"):
            login_user_input = st.text_input("Username", key="login_user")
            login_pass = st.text_input("Password", type="password", key="login_pass")
            submitted = st.form_submit_button("Login")
            
            if submitted:
                if login_user_input and login_pass:
                    if login_user(login_user_input, login_pass):
                        st.session_state.logged_in = True
                        st.session_state.username = login_user_input
                        st.success("Logged in. Redirecting...")
                        st.rerun()
                    else:
                        st.error("Invalid credentials.")
                else:
                    st.warning("Fill username and password.")

    with tab2:
        with st.form("signup_form"):
            new_user = st.text_input("Choose a username", key="signup_user")
            new_pass = st.text_input("Choose a password", type="password", key="signup_pass")
            
            # --- FIX: DOB Calendar ---
            today = datetime.now(UTC)
            min_date = today - timedelta(days=365*100) # 100 years ago
            default_dob = today - timedelta(days=365*25) # Default to 25 years old
            
            dob = st.date_input(
                "Date of Birth", 
                value=default_dob, 
                min_value=min_date, 
                max_value=today, 
                key="signup_dob"
            )
            # --- END FIX ---
            
            submitted = st.form_submit_button("Sign Up")
            
            if submitted:
                if not new_user or not new_pass:
                    st.warning("Choose username & password.")
                else:
                    ok, msg = signup_user(new_user, new_pass, str(dob))
                    if ok:
                        st.session_state.logged_in = True
                        st.session_state.username = new_user
                        st.success(msg + " You are now logged in. Redirecting...")
                        st.rerun()
                    else:
                        st.error(msg)

# --- Save rating handler ---
def save_rating_for_selected(tmdb_id, rating_value):
    """
    This function now does two things:
    1. Saves the basic rating to the history (like before).
    2. Updates the user's ML profile vector.
    """
    if not st.session_state.logged_in:
        st.error("You must be logged in to rate a movie.")
        return
        
    # --- THIS IS THE FIX for the ValueError crash ---
    if ARTIFACTS is None or ID_TO_INDEX is None or MOVIE_MATRIX is None or MATRIX_SHAPE is None:
        st.error("ML model artifacts not loaded. Cannot update profile.")
        return
    # --- END OF FIX ---
        
    uid = get_user_id(st.session_state.username)
    if uid is None:
        st.error("User id not found.")
        return
    
    # --- 1. Save to History (Original logic) ---
    rating_float = float(rating_value)
    liked_status = 1 if rating_float >= 3 else 0
    add_user_movie_history(uid, tmdb_id, rating=rating_float, liked=liked_status, watched_on=datetime.now(UTC))
    st.success("Rating saved ✔️")

    # --- 2. Update ML Profile Vector (New logic) ---
    try:
        # Get the movie's vector
        movie_index = ID_TO_INDEX.get(tmdb_id)
        if movie_index is None:
            print(f"Warning: tmdb_id {tmdb_id} not in artifact map. Skipping profile update.")
            return
        movie_vector = MOVIE_MATRIX[movie_index]

        # Get the user's current profile
        current_vector = get_user_profile_vector(uid)
        
        # --- FIX: Check for stale profile on save ---
        if current_vector is not None and current_vector.shape[1] != MOVIE_MATRIX.shape[1]:
            print(f"Stale profile for user {uid} found during save. Resetting.")
            current_vector = None # Reset to create a new one
        
        if current_vector is None:
            # Create a new, empty vector if none exists
            current_vector = csr_matrix((1, MATRIX_SHAPE[1]), dtype=np.float64)

        # Define weights: 5-star is strong positive, 1-star is strong negative
        rating_weights = {
            5.0: 2.0,  # Love
            4.0: 1.0,  # Like
            3.0: 0.5,  # Meh
            2.0: -1.0, # Dislike
            1.0: -2.0  # Hate
        }
        weight = rating_weights.get(rating_float, 0.0)

        # Add the weighted movie vector to the user's profile
        new_vector = current_vector + (movie_vector * weight)
        
        # Save the new profile back to the DB
        save_user_profile_vector(uid, new_vector)
        print(f"User {uid} profile updated.")

    except Exception as e:
        st.error(f"Error updating profile: {e}")
        print(f"Error updating profile: {e}")

# --- UI: show details pane (reusable component) ---
def show_details_pane(tmdb_id):
    # ... (No changes to this function) ...
    movie = get_movie_by_tmdb_id(tmdb_id)
    if not movie:
        st.warning("Movie details not found in DB.")
        return
        
    left, right = st.columns([1, 2])
    poster = movie.get("poster_url") or PLACEHOLDER_POSTER
    with left:
        st.image(poster, width=POSTER_W) # Use st.image
    with right:
        st.markdown(f"### {movie.get('title')}")
        st.markdown(f"**Year:** {movie.get('year')}")
        st.markdown(f"**Genres:** {movie.get('genres')}")
        director = movie.get('director') or ""
        st.markdown(f"**Director:** {director}")
        cast = movie.get('cast_top5') or ""
        st.markdown(f"**Cast (top):** {cast}")
        st.markdown(f"**TMDb rating:** {movie.get('tmdb_rating')}")
        overview = movie.get('overview') if 'overview' in movie else movie.get('plot_short') if 'plot_short' in movie else None
        if overview:
            st.markdown("**Overview:**")
            st.write(overview)

        st.markdown("### Rate & Save")
        
        star_options = ["⭐" * i for i in range(1, 6)]
        star_map = {("⭐" * i): i for i in range(1, 6)}
        
        selected_star_display = st.radio(
            "Your rating (1-5)",
            options=star_options,
            index=4,  # Default to 5 stars ("⭐⭐⭐⭐⭐")
            key=f"rate_{tmdb_id}",
            horizontal=True
        )
        
        user_rating = star_map[selected_star_display]
        
        if st.button("Save rating", key=f"save_{tmdb_id}"):
            save_rating_for_selected(tmdb_id, user_rating)

# --- "Page" 2: Details View ---
def details_page(tmdb_id):
    # --- REFACTORED: Sidebar is now rendered in main() ---

    # "Back" button just clears the selected_id and reruns
    if st.button("⬅️ Back"):
        st.session_state.selected_tmdb_id = None
        st.rerun()
        
    st.markdown("---")
    with st.container(border=True):
        show_details_pane(tmdb_id)

# --- NEW: Recommendation Logic ---
def get_recommendations(user_id):
    """
    Generates a DataFrame of recommended movies for a user.
    Returns None if no profile exists or no recs can be made.
    """
    # --- FIX: Ambiguity check ---
    if ARTIFACTS is None or ID_TO_INDEX is None or MOVIE_MATRIX is None or MATRIX_SHAPE is None:
        print("Artifacts not loaded, cannot get recommendations.")
        return None

    user_vector = get_user_profile_vector(user_id)
    if user_vector is None:
        print(f"No profile vector for user {user_id}. Returning default movies.")
        return None

    # --- THIS IS THE FIX ---
    # Check if the user's profile shape matches the movie matrix shape
    if user_vector.shape[1] != MOVIE_MATRIX.shape[1]:
        print(f"WARNING: Stale profile for user {user_id}. Vector shape ({user_vector.shape[1]}) does not match matrix shape ({MOVIE_MATRIX.shape[1]}).")
        print("Resetting user profile to a new empty vector.")
        
        # Create a new, empty vector with the *correct* shape
        new_empty_vector = csr_matrix((1, MATRIX_SHAPE[1]), dtype=np.float64)
        # Save it, replacing the old, bad one
        save_user_profile_vector(user_id, new_empty_vector)
        
        # Return None for this run, so the user sees "Popular Movies"
        # The next rating will build on the new, correct vector.
        return None
    # --- END OF FIX ---

    # Calculate similarity scores
    scores = cosine_similarity(user_vector, MOVIE_MATRIX)[0]

    # Get list of movies user has already seen/rated
    seen_history = get_user_history(user_id)
    seen_tmdb_ids = {row[0] for row in seen_history} # Use a set for fast O(1) lookups

    # Create a list of (tmdb_id, score) tuples, filtering out seen movies
    score_tuples = []
    for i, score in enumerate(scores):
        tmdb_id = INDEX_TO_ID.get(i)
        if tmdb_id and tmdb_id not in seen_tmdb_ids:
            score_tuples.append((tmdb_id, score))

    # Sort by score in descending order
    sorted_scores = sorted(score_tuples, key=lambda x: x[1], reverse=True)

    # Get the top N tmdb_ids
    top_tmdb_ids = [tmdb_id for tmdb_id, score in sorted_scores[:TOTAL_TO_SHOW]]

    if not top_tmdb_ids:
        print("No recommendations found after filtering.")
        return None

    # Fetch movie details for these IDs from the DB
    conn = sqlite3.connect(MOVIES_DB_PATH)
    id_placeholders = ','.join('?' for _ in top_tmdb_ids)
    query = f"SELECT * FROM movies WHERE tmdb_id IN ({id_placeholders})"
    
    try:
        df = pd.read_sql_query(query, conn, params=top_tmdb_ids)
    except Exception as e:
        print(f"Error fetching recommended movies: {e}")
        conn.close()
        return None
    finally:
        conn.close()

    # Re-order the DataFrame to match the recommendation score order
    df['tmdb_id'] = pd.Categorical(df['tmdb_id'], categories=top_tmdb_ids, ordered=True)
    df = df.sort_values('tmdb_id')
    
    return df

# --- NEW: Reusable function to render the movie grid ---
def render_movie_grid(df):
    """Takes a dataframe and renders the poster grid and pagination."""
    if df is None or df.empty:
        st.info("No movies found.")
        return

    # cap to first TOTAL_TO_SHOW
    df = df.head(TOTAL_TO_SHOW).reset_index(drop=True)
    total = len(df)
    total_pages = math.ceil(total / PAGE_SIZE)
    if total_pages == 0: total_pages = 1
    if st.session_state.page > total_pages: st.session_state.page = total_pages

    # Grid of posters
    st.markdown("---")
    start = (st.session_state.page - 1) * PAGE_SIZE
    end = start + PAGE_SIZE
    page_df = df.iloc[start:end].reset_index(drop=True)

    cols = st.columns(4)
    
    def set_selected_movie(id):
        st.session_state.selected_tmdb_id = id
        
    for i, row in page_df.iterrows():
        col = cols[i % 4]
        poster = row.get("poster_url") or PLACEHOLDER_POSTER
        tmdb_id = row.get("tmdb_id")
        if tmdb_id is None:
            col.warning("Movie with missing ID found, skipping.")
            continue

        col.image(poster, width=POSTER_W)
        
        title_html = f"<div style='font-weight:700; text-align:center; color: inherit; height: 3em; overflow: hidden;'>{row.get('title')}</div>"
        col.markdown(title_html, unsafe_allow_html=True)
        
        col.button(
            "View Details", 
            key=f"view_{tmdb_id}_{i}", 
            on_click=set_selected_movie,
            args=(tmdb_id,),
            use_container_width=True
        )
        col.markdown("<br>", unsafe_allow_html=True)

    # Pagination controls
    st.markdown("---")
    pag_col1, pag_col2, pag_col3 = st.columns([1,2,1])
    with pag_col1:
        if st.button("⬅️ Previous"):
            if st.session_state.page > 1:
                st.session_state.page -= 1
                st.session_state.selected_tmdb_id = None
                st.rerun()
    with pag_col2:
        st.write(f"Page {st.session_state.page} of {total_pages} (showing {total} results)")
    with pag_col3:
        if st.button("Next ➡️"):
            if st.session_state.page < total_pages:
                st.session_state.page += 1
                st.session_state.selected_tmdb_id = None
                st.rerun()

# --- NEW: Page Function ---
def home_page():
    # --- UPDATED: Recommendation/Load Logic ---
    df = None
    if st.session_state.search_query:
        st.header(f"Search results for '{st.session_state.search_query}'")
        df = load_movies(limit=TOTAL_TO_SHOW, search_query=st.session_state.search_query)
        # Clear search query in state so we return to recs next time
        st.session_state.search_query = "" 
    else:
        # If no search, try to get recommendations
        uid = get_user_id(st.session_state.username)
        df = get_recommendations(uid)
        
        if df is not None and not df.empty:
            st.header("Recommended For You")
        else:
            # Fallback: No profile or no recs, show popular movies
            st.header("Popular Movies")
            df = load_movies(limit=TOTAL_TO_SHOW, popular=True)

    # Render the grid
    render_movie_grid(df)

# --- NEW: Page Function ---
def popular_page():
    st.header("Popular Movies")
    df = load_movies(limit=TOTAL_TO_SHOW, popular=True)
    render_movie_grid(df)

# --- NEW: Page Function ---
def trending_page():
    st.header("Trending Movies")
    df = load_trending_movies(limit=TOTAL_TO_SHOW)
    render_movie_grid(df)

# --- NEW: Page Function ---
def genre_page():
    st.header("Browse by Genre")
    all_genres = get_all_genres()
    if not all_genres:
        st.error("No genres found in database.")
        return

    selected_genre = st.selectbox("Select a genre:", all_genres)
    
    if selected_genre:
        df = load_movies_by_genre(selected_genre)
        render_movie_grid(df)

# --- NEW: Page Function ---
def history_page():
    st.header("Your Rating History")
    uid = get_user_id(st.session_state.username)
    history = get_user_history(uid)
    
    if not history:
        st.info("You have not rated any movies yet.")
        return

    # Fetch details for the movies in history
    tmdb_ids = [row[0] for row in history if row[0] is not None]
    if not tmdb_ids:
        st.info("No movie IDs found in your history.")
        return

    conn = sqlite3.connect(MOVIES_DB_PATH)
    id_placeholders = ','.join('?' for _ in tmdb_ids)
    query = f"SELECT * FROM movies WHERE tmdb_id IN ({id_placeholders})"
    
    try:
        df = pd.read_sql_query(query, conn, params=tmdb_ids)
    except Exception as e:
        print(f"Error fetching history details: {e}")
        st.error("Could not load movie details for your history.")
        return
    finally:
        conn.close()

    # Create a lookup map for movie details
    movie_details_map = {row['tmdb_id']: row for i, row in df.iterrows()}
    
    # Create a map for user ratings from history
    rating_map = {row[0]: row[1] for row in history} # tmdb_id -> rating

    st.markdown("---")
    # --- FIX: Duplicate Key Error ---
    # We iterate over tmdb_ids with an index 'i' to create a unique key
    for i, tmdb_id in enumerate(tmdb_ids):
        movie = movie_details_map.get(tmdb_id)
        
        # --- FIX: ValueError (Series ambiguous) ---
        if movie is None:
            continue
        # --- END FIX ---
            
        rating = rating_map.get(tmdb_id)
        
        with st.container(border=True):
            c1, c2 = st.columns([1, 4])
            poster = movie.get('poster_url') or PLACEHOLDER_POSTER
            c1.image(poster)
            c2.markdown(f"### {movie.get('title')}")
            # Handle cases where rating might be None
            if rating:
                c2.markdown(f"**Your Rating:** {"⭐" * int(rating)} ({rating}/5)")
            else:
                c2.markdown(f"**Your Rating:** Not Rated")
            c2.markdown(f"**Genres:** {movie.get('genres')}")
            c2.markdown(f"**Year:** {movie.get('year')}")
            # --- FIX: Use 'i' in the key to make it unique ---
            if c2.button("View Again", key=f"hist_view_{tmdb_id}_{i}", use_container_width=True):
                st.session_state.selected_tmdb_id = tmdb_id
                st.rerun()


# --- Main Router ---
def main():
    
    # 1. Check login status FIRST
    if not st.session_state.logged_in:
        login_signup_page()
        return

    # --- NEW: Render Sidebar Navigation on EVERY logged-in page ---
    with st.sidebar:
        st.title(f"👋 {st.session_state.username}")
        st.markdown("---")
        
        if st.button("Recommended For You", use_container_width=True):
            st.session_state.view = "home"
            st.session_state.page = 1
            st.session_state.selected_tmdb_id = None
            st.rerun()
        if st.button("Popular Movies", use_container_width=True):
            st.session_state.view = "popular"
            st.session_state.page = 1
            st.session_state.selected_tmdb_id = None
            st.rerun()
        if st.button("Trending Movies", use_container_width=True):
            st.session_state.view = "trending"
            st.session_state.page = 1
            st.session_state.selected_tmdb_id = None
            st.rerun()
        if st.button("Browse by Genre", use_container_width=True):
            st.session_state.view = "genre"
            st.session_state.page = 1
            st.session_state.selected_tmdb_id = None
            st.rerun()
        if st.button("Your History", use_container_width=True):
            st.session_state.view = "history"
            st.session_state.page = 1
            st.session_state.selected_tmdb_id = None
            st.rerun()
            
        st.markdown("---")
        if st.button("Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.selected_tmdb_id = None
            st.session_state.page = 1
            st.session_state.view = "home"
            st.rerun()

        # --- NEW: Developer Window ---
        if "debug" in st.query_params:
            uid = get_user_id(st.session_state.username)
            if uid:
                with st.expander("DEV: User Profile Vector"):
                    profile = get_user_profile_vector(uid)
                    if profile is not None:
                        st.write(f"Shape: {profile.shape}")
                        st.write(f"Non-zero elements: {profile.nnz}")
                        
                        # --- FIX: Debug Bar Chart (Genre-Only) ---
                        if COL_SLICES and ARTIFACTS and 'vectorizers' in ARTIFACTS and 'genres' in ARTIFACTS['vectorizers']:
                            try:
                                # 1. Get genre names from the saved vectorizer
                                genre_vectorizer = ARTIFACTS['vectorizers']['genres']
                                genre_names = genre_vectorizer.get_feature_names_out()
                                
                                # 2. Get the genre slice indices
                                start, end = COL_SLICES['genres']
                                
                                # 3. Extract the genre weights from the user's profile
                                profile_dense = profile.todense()
                                genre_weights = profile_dense[0, start:end]
                                
                                # 4. Create DataFrame
                                # Squeeze to 1D array
                                genre_weights_squeezed = np.asarray(genre_weights).squeeze()
                                
                                df = pd.DataFrame({
                                    'Genre': genre_names,
                                    'Weight': genre_weights_squeezed
                                })
                                
                                # Filter to only show genres with non-zero weights for clarity
                                df_to_plot = df[df['Weight'] != 0].set_index('Genre')
                                
                                if not df_to_plot.empty:
                                    st.write("Genre Weights (Real-time):")
                                    st.bar_chart(df_to_plot)
                                else:
                                    st.write("No genre weights yet. Rate some movies!")
                                
                            except Exception as e:
                                st.error(f"Error plotting genre chart: {e}")
                                
                        else:
                            st.warning("`col_slices` or 'genres' vectorizer not in artifacts. Cannot show chart.")
                        
                        # --- THIS IS THE FIX ---
                        # Removed the nested st.expander, which is not allowed.
                        st.write("Full Vector Dataframe:")
                        st.dataframe(profile.todense())
                        # --- END FIX ---
                    else:
                        st.write("No profile vector found.")
                        
    # 2. If logged in, proceed to routing
    
    # First, check if we are viewing details
    if st.session_state.selected_tmdb_id:
        details_page(st.session_state.selected_tmdb_id)
    
    # If not, show the main page based on the view state
    else:
        # --- NEW: Search bar moved here, outside pages ---
        with st.form("search_form"):
            home_btn_col, q_col, search_btn_col = st.columns([2, 8, 2])
            
            with home_btn_col:
                if st.form_submit_button("🏠 Home"):
                    st.session_state.view = "home"
                    st.session_state.search_query = ""
                    st.session_state.page = 1
                    st.rerun()
            with q_col:
                query = st.text_input("🔍 Search movies", value=st.session_state.search_query, label_visibility="collapsed", placeholder="Search movies by title...")
            with search_btn_col:
                search_btn = st.form_submit_button("Search")
                
            if search_btn:
                st.session_state.search_query = query
                st.session_state.page = 1
                st.session_state.view = "home" # Search always goes to home
                st.rerun()

        # --- NEW: Main router ---
        if st.session_state.view == "home":
            home_page()
        elif st.session_state.view == "popular":
            popular_page()
        elif st.session_state.view == "trending":
            trending_page()
        elif st.session_state.view == "genre":
            genre_page()
        elif st.session_state.view == "history":
            history_page()
        else:
            home_page() # Default


if __name__ == "__main__":
    if not ARTIFACTS:
        st.error("Application cannot start: movie_artifacts.pkl failed to load.")
    else:
        main()

