# 🎬 Movie Recommendation System

A content-based movie recommender system built with **Streamlit**, **SQLite**, and **scikit-learn**. 

The app lets users sign up, rate movies, and receive personalized recommendations powered by TF-IDF features and cosine similarity over a movie feature matrix.

---

## 🔍 Overview

This project provides an end-to-end pipeline:

1. **Data collection** from TMDb via their public API (Bollywood & Hollywood movies from 2010–2025).
2. **Storage** in a SQLite database (`Movies.db`).
3. **Feature engineering & model artifacts** via `preprocess_movies.py`:
   - TF‑IDF features for genres, cast, and director.
   - One-hot encoding for decade.
   - Normalized TMDb rating.
   - Combined into a sparse feature matrix stored in `movie_artifacts.pkl`.
4. **Interactive web app** via Streamlit (`app.py`):
   - User authentication and history tracking in `user.db`.
   - Per-user preference vector learned from ratings.
   - Cosine-similarity based recommendations.

---

## ✨ Features

- **User accounts**
  - Sign up with username, password (SHA‑256 hash), and date of birth.
  - Login/logout handled via `user.db` (see `sql_utils.py`).

- **Personalized recommendations**
  - Each rating updates a **profile vector** stored in `user_preferences`.
  - Recommendations are produced via cosine similarity between the user vector and the movie feature matrix.
  - Negative ratings push the profile away from disliked movies; positive ratings pull it closer.

- **Movie browsing views**
  - **Recommended For You** (default home for logged-in users).
  - **Popular Movies** – highest TMDb rating.
  - **Trending Movies** – recent releases (current and previous year) with good ratings.
  - **Browse by Genre** – filter movies by genre.
  - **Your History** – movies you have rated, with rating details and quick “View Again” shortcut.

- **Rich movie metadata**
  - Title, year, genres, top cast, director, TMDb rating, overview/plot, and poster.

- **Developer debug view** (optional)
  - When run with a `?debug=1` query parameter, shows:
    - User profile vector shape and density.
    - Bar chart of **genre weights** for the logged-in user.

---

## 🧱 Project Structure

- `app.py` – Main Streamlit application and UI/router.
- `preprocess_movies.py` – Builds feature matrix and saves `movie_artifacts.pkl`.
- `imdbpy_bollywood_scraper.py` – TMDb scraper (Bollywood & Hollywood, 2010–2025) that writes `movies_2010_2025_tmdb.csv`.
- `sql_utils.py` – All user and history related SQLite helpers for `user.db`.
- `Movies.db` – SQLite database containing the `movies` table (movie catalog).
- `movie_artifacts.pkl` – Pickled artifacts used by the recommender (feature matrix, ID mappings, vectorizers, column slices).
- `movies_2010_2025_tmdb.csv` – Raw movie dump produced by the scraper.
- `user.db` – SQLite database for authentication, user profiles, and rating history.
- `EDA.ipynb` – Notebook for exploratory data analysis (optional, not needed to run the app).

---

## 🛠 Tech Stack

- **Language:** Python 3
- **Web UI:** [Streamlit](https://streamlit.io/)
- **Databases:** SQLite (`Movies.db`, `user.db`)
- **ML / Data:**
  - `pandas`, `numpy`
  - `scikit-learn` (`TfidfVectorizer`, `OneHotEncoder`, `MinMaxScaler`)
  - `scipy.sparse`
- **Data source:** TMDb via `tmdbsimple`

---

## 📦 Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/Movie_Reccmendation.git
   cd Movie_Reccmendation
   ```

2. **Create a virtual environment (optional but recommended)**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # on Windows
   ```

3. **Install dependencies**

   Install packages via `pip` (adapt as needed):

   ```bash
   pip install streamlit pandas numpy scipy scikit-learn tmdbsimple tqdm
   ```

   The standard library covers `sqlite3`, `hashlib`, and `datetime`.

---

## 📡 Data Collection (TMDb)

If you want to **regenerate the dataset** instead of using the provided `Movies.db` and `movies_2010_2025_tmdb.csv`:

1. **Get a TMDb API key**
   - Create a TMDb account and request an API key.

2. **Configure the scraper**
   - Open `imdbpy_bollywood_scraper.py`.
   - Replace the placeholder value of `TMDB_API_KEY` with your own key, or modify the script to read from an environment variable.

3. **Run the scraper**

   ```bash
   python imdbpy_bollywood_scraper.py
   ```

   This will progressively write/update `movies_2010_2025_tmdb.csv` in the project root.

4. **Create / update `Movies.db`**
   - Load `movies_2010_2025_tmdb.csv` into a SQLite database named `Movies.db`.
   - Create a `movies` table with at least the columns used in `app.py` and `preprocess_movies.py` (e.g. `tmdb_id`, `title`, `year`, `genres`, `cast_top5`, `director`, `tmdb_rating`, `decade`, `poster_url`, `overview` / `plot_short`, `language`, `imdb_id`, `source`).
   - You can do this via a notebook (`EDA.ipynb`), a small script, or a GUI tool like DB Browser for SQLite.

If you already have a working `Movies.db`, you can skip this section.

---

## 🧮 Building the Recommendation Artifacts

Once `Movies.db` is in place, build the feature matrix and artifacts used by the app.

1. **Run the preprocessing script**

   ```bash
   python preprocess_movies.py
   ```

   This will:

   - Read the `movies` table from `Movies.db`.
   - Preprocess and vectorize genres, cast, director, decade, and ratings.
   - Build a sparse feature matrix and an `id_to_index` mapping.
   - Save everything into `movie_artifacts.pkl` in the project root.

2. **Verify output**
   - The script prints the final matrix shape and column slice information.
   - Make sure `movie_artifacts.pkl` exists before starting the Streamlit app.

---

## 🗄 User Database Setup

The user database (`user.db`) is automatically migrated/created using `sql_utils.create_tables()`.

You can initialize it manually if desired:

```bash
python sql_utils.py
```

This will create (if missing):

- `users` – basic user info and password hashes.
- `user_preferences` – one row per user with a serialized profile vector.
- `user_movie_history` – rating history (movie IDs, ratings, liked flag, timestamps).

Running `app.py` also calls `create_tables()` on startup, so in most cases you do not need to run anything manually.

---

## 🚀 Running the App

1. Make sure you have:
   - `Movies.db` with a `movies` table.
   - `movie_artifacts.pkl` created by `preprocess_movies.py`.

2. Start Streamlit:

   ```bash
   streamlit run app.py
   ```

3. Open the URL shown in your terminal (usually `http://localhost:8501`).

4. **Sign up / log in**
   - Create a new account on the **Sign Up** tab.
   - Log in from the **Login** tab.

5. **Use the app**
   - Browse **Recommended For You**, **Popular**, **Trending**, **Browse by Genre**, and **Your History** from the sidebar.
   - Click **View Details** on any movie to see its metadata and poster, and to rate it using 1–5 stars.
   - Ratings immediately update your profile vector; keep rating to improve recommendations.

---

## 🧪 Developer Notes

- **Debug / inspection mode**
  - Add `?debug=1` to the Streamlit URL, e.g. `http://localhost:8501/?debug=1`.
  - The sidebar will show a developer expander with:
    - Shape and non-zero count of the user profile vector.
    - A bar chart of genre weights extracted from the vector, if available.

- **Retraining**
  - If you change the movie catalog (`Movies.db`) or add new columns/features:
    1. Update `preprocess_movies.py` accordingly.
    2. Re-run `python preprocess_movies.py` to regenerate `movie_artifacts.pkl`.

- **User profile compatibility**
  - The app checks for stale profile vectors (shape mismatch vs. current matrix) and automatically resets them if needed.

---

## ⚖️ License

Others
---

## 🙌 Acknowledgements

- [TMDb](https://www.themoviedb.org/) for movie metadata and images.
- The Python open‑source ecosystem: Streamlit, pandas, numpy, SciPy, scikit‑learn, and others.
