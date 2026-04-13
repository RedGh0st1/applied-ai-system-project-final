"""ToneMatch 1.0 — content-based music recommender."""
import csv
from typing import List, Dict, Tuple
from dataclasses import dataclass


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    subgenre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float
    mode: int           # 0 = minor, 1 = major
    instrumentalness: float


@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    target_valence: float
    target_bpm: float
    target_acoustic: float
    target_inst: float
    preferred_mode: int  # 0 = minor, 1 = major
    likes_acoustic: bool


# ── Scoring constants ─────────────────────────────────────────────────────────
#
# Genre is the strongest categorical signal, while mood is a
# useful secondary signal for session-level intent.
#
# Point budget breakdown:
#   Categorical  (genre + mood + subgenre + mode) : max ~6.5 pts
#   Continuous   (energy + valence + inst)        : max  3.5 pts
#   Total maximum                                 : ~10.0 pts

POINTS_MOOD_EXACT     = 1.0   # mood == user.favorite_mood
POINTS_MOOD_ADJACENT  = 0.5   # mood is semantically close to favorite
POINTS_GENRE_EXACT    = 2.0   # genre == user.favorite_genre
POINTS_SUBGENRE_EXACT = 1.5   # subgenre match (stacks on genre match only)
POINTS_MODE_MATCH     = 1.0   # song.mode == user.preferred_mode
POINTS_MODE_MISMATCH  = -1.0  # wrong key feel penalty

MAX_PTS_ENERGY  = 2.0   # full points when energy == target_energy
MAX_PTS_VALENCE = 1.0   # full points when valence == target_valence
MAX_PTS_INST    = 0.5   # full points when instrumentalness == target_inst

# Moods that are close enough to earn partial credit
_ADJACENT_MOODS: Dict[str, set] = {
    "chill":       {"relaxed", "focused", "serene"},
    "relaxed":     {"chill", "focused", "romantic"},
    "focused":     {"chill", "relaxed", "serene"},
    "happy":       {"uplifting", "euphoric", "energetic"},
    "energetic":   {"happy", "euphoric", "intense", "groovy"},
    "intense":     {"aggressive", "energetic", "frantic", "passionate"},
    "aggressive":  {"intense", "frantic"},
    "moody":       {"dark", "melancholic", "wistful"},
    "dark":        {"moody", "melancholic", "sad"},
    "melancholic": {"sad", "dark", "moody", "wistful"},
    "sad":         {"melancholic", "wistful", "dark"},
    "wistful":     {"sad", "melancholic", "nostalgic"},
    "nostalgic":   {"wistful", "relaxed"},
    "euphoric":    {"happy", "energetic", "uplifting"},
    "uplifting":   {"happy", "euphoric"},
    "groovy":      {"energetic", "happy", "confident"},
    "romantic":    {"relaxed", "happy"},
    "confident":   {"energetic", "groovy"},
    "serene":      {"chill", "focused", "relaxed"},
    "passionate":  {"intense", "energetic"},
    "frantic":     {"intense", "aggressive"},
}


# ── Shared scoring helpers ────────────────────────────────────────────────────

def _proximity(value: float, target: float, max_pts: float,
               tolerance: float = 0.5) -> float:
    """
    Linear-decay proximity score.
    Returns max_pts at a perfect match, decays to 0.0 at tolerance distance.

    Example (energy, max_pts=1.5, tolerance=0.5):
      diff = 0.00 → 1.50 pts  (perfect match)
      diff = 0.25 → 0.75 pts  (half tolerance)
      diff = 0.50 → 0.00 pts  (at tolerance boundary)
      diff > 0.50 → 0.00 pts  (clamped)
    """
    return max(0.0, max_pts * (1.0 - abs(value - target) / tolerance))


def _score_dict(song: Dict, user: Dict) -> Tuple[float, List[str]]:
    """
    Score a single song dict against a user prefs dict.
    Returns (score, reasons) where reasons lists every rule that fired.
    """
    score = 0.0
    hits: List[str] = []

    # ── Mood scoring ──────────────────────────────────────────────────────────
    s_mood = song.get("mood", "")
    u_mood = user.get("favorite_mood", "")
    if s_mood == u_mood:
        score += POINTS_MOOD_EXACT
        hits.append(f"mood '{s_mood}' match (+{POINTS_MOOD_EXACT})")
    elif s_mood in _ADJACENT_MOODS.get(u_mood, set()):
        score += POINTS_MOOD_ADJACENT
        hits.append(f"adjacent mood '{s_mood}' (+{POINTS_MOOD_ADJACENT})")

    # ── Genre + subgenre scoring ──────────────────────────────────────────────
    s_genre = song.get("genre", "")
    u_genre = user.get("favorite_genre", "")
    if s_genre == u_genre:
        score += POINTS_GENRE_EXACT
        hits.append(f"genre '{s_genre}' match (+{POINTS_GENRE_EXACT})")
        if song.get("subgenre", "") == user.get("favorite_subgenre", ""):
            score += POINTS_SUBGENRE_EXACT
            hits.append(
                f"subgenre '{song.get('subgenre')}' match (+{POINTS_SUBGENRE_EXACT})"
            )

    # ── Mode scoring (major / minor key feel) ─────────────────────────────────
    s_mode = int(song.get("mode", -1))
    u_mode = int(user.get("preferred_mode", -1))
    key_label = "major" if s_mode == 1 else "minor"
    if s_mode == u_mode:
        score += POINTS_MODE_MATCH
        hits.append(f"{key_label} key match (+{POINTS_MODE_MATCH})")
    else:
        score += POINTS_MODE_MISMATCH
        hits.append(f"{key_label} key mismatch ({POINTS_MODE_MISMATCH})")

    # ── Continuous proximity scores ───────────────────────────────────────────
    e_pts = _proximity(
        float(song.get("energy", 0.5)),
        user.get("target_energy", 0.5),
        MAX_PTS_ENERGY,
    )
    score += e_pts
    hits.append(
        f"energy {float(song.get('energy', 0)):.2f} "
        f"(target {user.get('target_energy', 0.5):.2f}) +{e_pts:.2f}"
    )

    v_pts = _proximity(
        float(song.get("valence", 0.5)),
        user.get("target_valence", 0.5),
        MAX_PTS_VALENCE,
    )
    score += v_pts
    hits.append(
        f"valence {float(song.get('valence', 0)):.2f} "
        f"(target {user.get('target_valence', 0.5):.2f}) +{v_pts:.2f}"
    )

    i_pts = _proximity(
        float(song.get("instrumentalness", 0.5)),
        user.get("target_inst", 0.5),
        MAX_PTS_INST,
        tolerance=0.4,
    )
    score += i_pts
    hits.append(
        f"inst {float(song.get('instrumentalness', 0)):.2f} "
        f"(target {user.get('target_inst', 0.5):.2f}) +{i_pts:.2f}"
    )

    return max(0.0, score), hits


def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    Score a single song dictionary against user preferences.

    Recipe:
    - +2.0 points for a genre match
    - +1.0 point for a mood match
    - +0.0 to +2.0 points for energy similarity
    """
    return _score_dict(song, user_prefs)


def _score_song_obj(song: Song, user: UserProfile) -> Tuple[float, List[str]]:
    """Bridge: converts Song + UserProfile dataclasses to dicts for _score_dict."""
    song_dict = {
        "mood":             song.mood,
        "genre":            song.genre,
        "subgenre":         song.subgenre,
        "mode":             song.mode,
        "energy":           song.energy,
        "valence":          song.valence,
        "instrumentalness": song.instrumentalness,
    }
    user_dict = {
        "favorite_mood":    user.favorite_mood,
        "favorite_genre":   user.favorite_genre,
        "preferred_mode":   user.preferred_mode,
        "target_energy":    user.target_energy,
        "target_valence":   user.target_valence,
        "target_inst":      user.target_inst,
    }
    return _score_dict(song_dict, user_dict)


# ── Functional API (used by src/main.py) ──────────────────────────────────────

def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
    print(f"Loading songs from {csv_path}...")
    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            songs.append({
                "id":               int(row["id"]),
                "title":            row["title"],
                "artist":           row["artist"],
                "genre":            row["genre"],
                "subgenre":         row["subgenre"],
                "mood":             row["mood"],
                "energy":           float(row["energy"]),
                "tempo_bpm":        float(row["tempo_bpm"]),
                "valence":          float(row["valence"]),
                "danceability":     float(row["danceability"]),
                "acousticness":     float(row["acousticness"]),
                "mode":             int(row["mode"]),
                "instrumentalness": float(row["instrumentalness"]),
            })
    print(f"  Loaded {len(songs)} songs.")
    return songs


def recommend_songs(
    user_prefs: Dict, songs: List[Dict], k: int = 5
) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py
    """
    scored = []
    for song in songs:
        score, reasons = score_song(user_prefs, song)
        scored.append((song, score, " | ".join(reasons)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


# ── OOP API (used by tests/test_recommender.py) ───────────────────────────────

class Recommender:
    """
    ToneMatch 1.0 — content-based music recommender.
    Matches songs to a user profile using energy, valence, acousticness,
    mood, and genre as the primary vibe signals.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        scored = [
            (song, _score_song_obj(song, user)[0])
            for song in self.songs
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [song for song, _ in scored[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        _, reasons = _score_song_obj(song, user)
        return " | ".join(reasons)
