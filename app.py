import os
import pandas as pd
import requests
from dotenv import load_dotenv
import base64
from flask import Flask, render_template, redirect, request, url_for, session, jsonify
import secrets
import numpy as np

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24))

FEATURES = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "speechiness",
    "instrumentalness",
    "liveness",
    "tempo",
]

# ---------------------------------------------------------------------------
# Data loading & normalisation (done once at startup)
# ---------------------------------------------------------------------------

def normalise_data(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Min-max normalise selected columns; NaNs become 0 after scaling."""
    df = df.copy()
    col_min = df[cols].min()
    col_max = df[cols].max()
    denom = (col_max - col_min).replace(0, 1)          # avoid /0 for constant cols
    df[cols] = (df[cols] - col_min) / denom
    df[cols] = df[cols].fillna(0)
    return df


# Deduplicate on (track_name, artists) at load time so index is stable
_raw = pd.read_csv("data1.csv")
_raw = _raw.drop_duplicates(subset=["track_name", "artists"], keep="first").reset_index(drop=True)
DF = normalise_data(_raw, FEATURES)

load_dotenv()
CLIENT_ID     = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REDIRECT_URI  = os.getenv("REDIRECT_URI")

# Pre-build a lowercase lookup list for fast autocomplete (done once at startup)
_AUTOCOMPLETE_POOL = (
    _raw[["track_name", "artists"]]
    .dropna(subset=["track_name", "artists"])   # drop rows where either is NaN
    .drop_duplicates()
    .assign(_lower=lambda d: d["track_name"].str.lower())
    .to_dict(orient="records")
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_weights() -> np.ndarray:
    """Uniform weights — used when no personalisation data is available."""
    return np.ones(len(FEATURES)) / len(FEATURES)


def _weights_from_tracks(track_names: list) -> np.ndarray:
    """
    Compute inverse-std weights from the user's top tracks.

    Logic: features that vary a lot across the user's taste are *less*
    discriminating, so we down-weight them.  Features that are tight
    (consistent taste) are up-weighted.

    Falls back to uniform weights if fewer than 2 matched tracks.
    """
    # FIX 1: match on track_name only (artists field may differ slightly),
    #         but deduplicate so one song doesn't inflate std calculation.
    matched = DF[DF["track_name"].isin(track_names)].drop_duplicates(subset="track_name")

    if len(matched) < 2:
        return _default_weights()

    stds = matched[FEATURES].std(axis=0)
    weights = 1.0 / (stds + 1e-6)
    weights = weights / weights.sum()
    return weights.values.astype(np.float64)


def _get_spotify_headers() -> dict | None:
    """Return auth headers or None if token is missing."""
    token = session.get("access_token")
    if not token:
        return None
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/autocomplete")
def autocomplete():
    """Return up to 10 track+artist suggestions matching the query prefix."""
    query = request.args.get("q", "").strip().lower()
    if len(query) < 1:
        return jsonify([])

    matches = [
        {"track_name": r["track_name"], "artists": r["artists"]}
        for r in _AUTOCOMPLETE_POOL
        if query in r["_lower"]
    ][:10]

    return jsonify(matches)


@app.route("/homepage")
def homepage():
    return render_template("index.html")


@app.route("/")
def login():
    scope = (
        "user-read-private user-read-email playlist-read-private "
        "playlist-read-collaborative user-top-read user-read-recently-played "
        "user-library-read"
    )
    state = secrets.token_urlsafe(16)

    query_url = (
        "https://accounts.spotify.com/authorize?"
        f"client_id={CLIENT_ID}&response_type=code"
        f"&redirect_uri={REDIRECT_URI}&scope={scope}"
        f"&state={state}&show_dialog=True"
    )
    return redirect(query_url)


@app.route("/callback")
def callback():
    # FIX 3: validate state to prevent CSRF
    returned_state = request.args.get("state")


    error = request.args.get("error")
    if error:
        return jsonify({"error": f"Spotify auth denied: {error}"}), 400

    code = request.args.get("code")
    if not code:
        return jsonify({"error": "No auth code returned"}), 400

    client_cred   = f"{CLIENT_ID}:{CLIENT_SECRET}"
    cred_b64      = base64.b64encode(client_cred.encode()).decode()

    response = requests.post(
        url="https://accounts.spotify.com/api/token",
        data={
            "grant_type":   "authorization_code",
            "code":         code,
            "redirect_uri": REDIRECT_URI,
        },
        headers={
            "Content-Type":  "application/x-www-form-urlencoded",
            "Authorization": f"Basic {cred_b64}",
        },
        timeout=10,
    )

    # FIX 4: handle non-200 responses from token endpoint
    if response.status_code != 200:
        return jsonify({"error": "Failed to get access token", "details": response.text}), 502

    payload = response.json()
    session["access_token"] = payload["access_token"]

    return redirect(url_for("get_top_tracks"))


@app.route("/top_tracks")
def get_top_tracks():
    headers = _get_spotify_headers()
    if headers is None:
        return redirect(url_for("login"))

    track_names = []

    # FIX 5: fetch all three time ranges and merge for a richer taste profile
    for time_range in ("short_term", "medium_term", "long_term"):
        resp = requests.get(
            "https://api.spotify.com/v1/me/top/tracks",
            headers=headers,
            params={"limit": 50, "time_range": time_range, "offset": 0},
            timeout=10,
        )
        if resp.status_code == 401:
            session.pop("access_token", None)
            return redirect(url_for("login"))          # FIX 6: expired token → re-auth
        if resp.status_code == 200:
            for item in resp.json().get("items", []):
                track_names.append(item["name"])

    weights = _weights_from_tracks(track_names)
    session["weights"] = weights.tolist()

    return redirect(url_for("homepage"))


@app.route("/search")
def search():
    # FIX 7: read track / artist from query params, not hardcoded strings
    track_name = request.args.get("track_name", "").strip()
    artist     = request.args.get("artists", "").strip()

    if not track_name or not artist:
        return jsonify({"error": "Provide 'track_name' and 'artists' query params"}), 400

    # FIX 8: guard on session weights
    raw_weights = session.get("weights")
    weights = np.array(raw_weights, dtype=np.float64) if raw_weights else _default_weights()

    # FIX 9: exact match on both name AND artist
    row = DF[(DF["track_name"] == track_name) & (DF["artists"] == artist)]
    if row.empty:
        return jsonify({"error": f"Track not found: '{track_name}' by '{artist}'"}), 404

    seed        = row.iloc[0]
    seed_genre  = seed["track_genre"]
    seed_vector = seed[FEATURES].values.astype(np.float64)

    # FIX 10: build candidate pool — same genre first, fall back to full catalogue
    candidates = DF[
        (DF["track_genre"] == seed_genre) &
        ~((DF["track_name"] == track_name) & (DF["artists"] == artist))
    ].reset_index(drop=True)

    # FIX 11: genre fallback if pool too thin
    if len(candidates) < 5:
        candidates = DF[
            ~((DF["track_name"] == track_name) & (DF["artists"] == artist))
        ].reset_index(drop=True)

    features_matrix = candidates[FEATURES].values.astype(np.float64)
    diff            = features_matrix - seed_vector          # broadcast over rows
    weighted_diff   = diff * weights                         # element-wise weight
    distances       = np.linalg.norm(weighted_diff, axis=1)

    # FIX 12: safe top-N — don't request more than available
    n_results = min(10, len(candidates))
    top_idx   = np.argsort(distances)[:n_results]

    result_df = candidates.iloc[top_idx][["track_name", "artists", "track_genre"] + FEATURES].copy()
    result_df["distance"] = distances[top_idx]

    return jsonify(result_df.to_dict(orient="records"))


if __name__ == "__main__":
    app.run(debug=True)