# save as robust_tmdb_scraper.py
import tmdbsimple as tmdb
import pandas as pd
import time, random, os, socket
from tqdm import tqdm
from collections import deque
from datetime import datetime

# ---------------- CONFIG ----------------
TMDB_API_KEY = "85da8892926938ea6a5f5c92340ca110"
START_YEAR = 2010
END_YEAR = 2025
BOLLY_LIMIT = 25
HOLLY_LIMIT = 25
CSV_PATH = "movies_2010_2025_tmdb.csv"

# Rate limiter: keep < max_requests in given window
max_requests_per_window = 28  # safe under TMDb ~30/10s limit
time_window_seconds = 10

# retry/backoff config
MAX_RETRIES = 5
BASE_DELAY = 1.0  # seconds base for backoff
# ----------------------------------------

tmdb.API_KEY = TMDB_API_KEY

# sliding window timestamps of recent requests
request_timestamps = deque()

def rate_limit_wait():
    """Ensure we don't exceed max requests in the time window."""
    now = time.time()
    # drop old timestamps
    while request_timestamps and now - request_timestamps[0] > time_window_seconds:
        request_timestamps.popleft()
    if len(request_timestamps) >= max_requests_per_window:
        # need to wait until earliest timestamp is older than window
        earliest = request_timestamps[0]
        wait = time_window_seconds - (now - earliest) + 0.05
        print(f"⏳ Rate limit reached — sleeping {wait:.2f}s")
        time.sleep(wait)
    # when proceeding, we'll append timestamp in caller after request

def safe_tmdb_info(movie_id, retries=MAX_RETRIES):
    """Fetch movie details with credits and external_ids in one call using append_to_response."""
    movies = tmdb.Movies(movie_id)
    attempt = 0
    while attempt < retries:
        try:
            rate_limit_wait()
            # append credits and external ids (imdb_id)
            resp = movies.info(append_to_response="credits,external_ids")
            # record this request timestamp
            request_timestamps.append(time.time())
            # small jitter after successful request
            time.sleep(random.uniform(0.2, 0.5))
            return resp
        except (ConnectionResetError, socket.error) as e:
            attempt += 1
            backoff = BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            print(f"⚠️ Connection reset / socket error (attempt {attempt}/{retries}) — backoff {backoff:.1f}s — {e}")
            time.sleep(backoff)
        except Exception as e:
            attempt += 1
            # handle 429 or other HTTP errors more politely
            backoff = BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            print(f"[WARN] TMDb request failed (attempt {attempt}/{retries}) — backoff {backoff:.1f}s — {e}")
            time.sleep(backoff)
    print(f"[ERR] Failed to fetch movie {movie_id} after {retries} attempts.")
    return None

def fetch_by_discover(year, language_code, region, limit):
    """Use Discover endpoint to collect movie ids and then fetch info for each id."""
    discover = tmdb.Discover()
    collected = []
    page = 1
    while len(collected) < limit:
        # safe discover calls should also be rate limited (wrap same logic)
        try:
            rate_limit_wait()
            resp = discover.movie(with_original_language=language_code, year=year, region=region, page=page)
            request_timestamps.append(time.time())
        except Exception as e:
            print(f"[WARN] Discover failed for {year} lang={language_code} page={page}: {e}")
            # small wait then retry page (no aggressive retry here)
            time.sleep(2 + random.uniform(0,1))
            page += 1
            if page > 100: break
            continue

        results = resp.get("results") or []
        if not results:
            break
        for m in results:
            if len(collected) >= limit:
                break
            collected.append(m)  # keep discover result (includes id & title)
        page += 1
        # gentle delay between discover pages
        time.sleep(random.uniform(0.5, 1.2))
    return collected[:limit]

def extract_movie_record(movie_disc_item, year, lang):
    """From a discover item, fetch full info (one request) and extract record fields."""
    movie_id = movie_disc_item["id"]
    tm = safe_tmdb_info(movie_id)
    if not tm:
        return None
    # credits under tm['credits']
    credits = tm.get("credits", {})
    cast = [c.get("name") for c in credits.get("cast", [])[:5]]
    crew = credits.get("crew", [])
    directors = [c.get("name") for c in crew if c.get("job") == "Director"]
    genres = [g.get("name") for g in tm.get("genres", [])]

    poster_path = tm.get("poster_path")
    poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None

    imdb_id = None
    ext = tm.get("external_ids") or {}
    imdb_id = ext.get("imdb_id")

    record = {
        "title": tm.get("title") or movie_disc_item.get("title"),
        "year": year,
        "language": lang,
        "genres": "|".join(genres) if genres else None,
        "director": directors[0] if directors else None,
        "cast_top5": "|".join([c for c in cast if c]),
        "tmdb_rating": tm.get("vote_average"),
        "imdb_id": imdb_id,
        "poster_url": poster_url,
        "tmdb_id": movie_id,
        "source": "tmdb_discover"
    }
    return record

def save_progress(df):
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"💾 Saved {len(df)} records to {CSV_PATH}")

# ---------- main loop ----------
if __name__ == "__main__":
    # resume if exists
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        done_pairs = set((int(r["year"]), r["language"]) for _, r in df.iterrows())
        print(f"Resuming; loaded {len(df)} rows; done language-year combos: {sorted(done_pairs)[:10]} ...")
    else:
        df = pd.DataFrame()
        done_pairs = set()

    for year in range(START_YEAR, END_YEAR + 1):
        for lang, region, limit, label in [("hi", "IN", BOLLY_LIMIT, "Bollywood"), ("en", "US", HOLLY_LIMIT, "Hollywood")]:
            if (year, lang) in done_pairs:
                print(f"⏭️ Skipping {label} {year} (already done).")
                continue

            print(f"\n📅 Fetching {label} ({lang}) for {year} (target {limit})...")
            discovered = fetch_by_discover(year, lang, region, limit)
            print(f"  → Discover returned {len(discovered)} candidate items")

            year_records = []
            for item in tqdm(discovered, desc=f"{label} {year}"):
                try:
                    rec = extract_movie_record(item, year, lang)
                    if rec:
                        year_records.append(rec)
                        # save each record to minimize loss
                        df = pd.concat([df, pd.DataFrame([rec])], ignore_index=True)
                        save_progress(df)
                except Exception as e:
                    print(f"[ERR] processing item {item.get('id')}: {e}")
                # small random pause between movies
                time.sleep(random.uniform(0.4, 1.0))

            print(f"✅ {label} {year}: added {len(year_records)} records this run.")
            # cooldown after finishing a language-year
            time.sleep(random.uniform(2.5, 5.0))

    print("\n🎉 DONE. Total records:", len(df))
