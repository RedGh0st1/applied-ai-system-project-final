"""ToneMatch 1.0 — content-based music recommender."""
import csv
import math
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
    popularity: int
    release_decade: str
    explicit: int          # 0 or 1
    language: str
    duration_sec: int
    loudness_db: float
    speechiness: float
    liveness: float
    detailed_mood_tags: str   # pipe-separated, e.g. "euphoric|bright|summery"
    cultural_region: str
    vocal_gender: str
    era_feel: str


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
    preferred_era: str        # "retro", "contemporary", or "" (no preference)
    preferred_language: str   # "English", "Korean", "Spanish", "Instrumental", or "" (any)
    preferred_region: str     # "Western", "Caribbean", "Latin American", "East Asian", or ""
    preferred_vocal_gender: str  # "male", "female", "mixed", "none", or ""
    allow_explicit: bool      # if False, explicit songs are hard-penalised
    target_popularity: float  # 0.0–1.0  (normalised from 0–100)
    target_liveness: float    # 0.0–1.0
    target_speechiness: float # 0.0–1.0
    preferred_mood_tags: list # list of strings, e.g. ["euphoric", "athletic", "bright"]


# ── Scoring constants ─────────────────────────────────────────────────────────
#
# Fix 1 (Independence Fix): MAX_PTS for continuous variables raised so they
#   can meaningfully compete with a categorical genre match. Energy + Valence +
#   Inst now max at 4.5 pts vs. genre's 2.0.
# Fix 4 (Balance Fix): POINTS_MODE_MISMATCH changed from -1.0 → 0.0.
#   Reward-only mode scoring prevents a single key-feel miss from wiping out
#   an otherwise strong genre + mood + energy alignment.
#
# Sensitivity experiment — Weight Shift:
#   Energy doubled (2.5 → 5.0) so numerical feel competes with categorical labels.
#   Genre halved  (2.0 → 1.0) so a string match no longer dominates the budget.
#   Genre partial halved (1.0 → 0.5) to keep the ratio consistent.
#
# Point budget breakdown:
#   Categorical  (genre + mood + subgenre + mode) : max  4.5 pts  (was 6.0)
#   Continuous   (energy + valence + inst)        : max  7.0 pts  (was 4.5)
#   Total maximum                                 : ~11.5 pts

POINTS_MOOD_EXACT     = 1.0   # mood == user.favorite_mood
POINTS_MOOD_ADJACENT  = 0.5   # mood is semantically close to favorite
POINTS_GENRE_EXACT    = 1.0   # weight shift: was 2.0
POINTS_GENRE_PARTIAL  = 0.5   # weight shift: was 1.0 (keeps partial/exact ratio at 0.5)
POINTS_SUBGENRE_EXACT = 1.5   # subgenre match (stacks on genre match only)
POINTS_MODE_MATCH     = 1.0   # song.mode == user.preferred_mode
POINTS_MODE_MISMATCH  = 0.0   # Fix 4: was -1.0; reward-only prevents a 2-pt swing from one signal

MAX_PTS_ENERGY  = 5.0   # weight shift: was 2.5
MAX_PTS_VALENCE = 1.5   # Fix 1: raised from 1.0
MAX_PTS_INST    = 0.5   # unchanged; multiplicative penalty handles extreme mismatches

# Fix 3 (Gaussian Scoring): sigma controls bell-curve width.
#   Smaller sigma → sharper peak (rewards precision more).
#   At sigma=0.25, diff=0.25 earns ~61% of max_pts; diff=0.5 earns ~14%.
#   Unlike linear decay, the score never hard-cuts to 0 — every song earns
#   some proximity credit, letting the system differentiate "kind of neutral"
#   from "perfectly neutral."
PROXIMITY_SIGMA      = 0.25   # used for energy and valence
PROXIMITY_SIGMA_INST = 0.20   # tighter for instrumentalness (more decisive signal)

# Fix 1 (Independence Fix): Multiplicative penalty for extreme inst mismatch.
#   If |song_inst - target_inst| > threshold, total score is multiplied by
#   INST_PENALTY_FACTOR. This lets a "Lyric Lover" actually be penalised for
#   getting a fully instrumental track, even if the genre/mood is perfect.
INST_PENALTY_THRESHOLD = 0.70
INST_PENALTY_FACTOR    = 0.60

# Fix 5 (Soft Floor): A tiny energy-scaled epsilon replaces the hard 0.0 clamp.
#   Prevents true score ties in the dead zone; ensures Python's stable sort
#   always has a meaningful secondary signal to break on.
SCORE_SOFT_FLOOR_BASE = 0.0001

# ── New-attribute scoring constants ───────────────────────────────────────────
POINTS_ERA_MATCH        = 1.0   # song.era_feel == user.preferred_era
POINTS_LANGUAGE_MATCH   = 0.75  # song.language == user.preferred_language
POINTS_REGION_MATCH     = 0.75  # song.cultural_region == user.preferred_region
POINTS_VOCAL_MATCH      = 0.5   # song.vocal_gender == user.preferred_vocal_gender
POINTS_PER_MOOD_TAG     = 0.4   # per overlapping tag in detailed_mood_tags
MAX_PTS_MOOD_TAGS       = 1.0   # cap: at most 1.0 pts from mood-tag overlap
MAX_PTS_POPULARITY      = 0.5   # Gaussian proximity to target_popularity (normalised)
MAX_PTS_LIVENESS        = 0.75  # Gaussian proximity to target_liveness
MAX_PTS_SPEECHINESS     = 0.75  # Gaussian proximity to target_speechiness
EXPLICIT_PENALTY_FACTOR = 0.0   # multiplier when allow_explicit=False and song.explicit=1
PROXIMITY_SIGMA_POP     = 0.25  # sigma for popularity Gaussian
PROXIMITY_SIGMA_LIVE    = 0.25  # sigma for liveness Gaussian
PROXIMITY_SIGMA_SPEECH  = 0.20  # sigma for speechiness Gaussian

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

# Fix 2 (Semantic Fallback): Genre map from unrecognized / niche genres to the
#   nearest catalog parent genre. When a user's favorite_genre is not in the
#   catalog, we fall back to this map and award POINTS_GENRE_PARTIAL instead of
#   POINTS_GENRE_EXACT. This prevents a hard-miss from silently zeroing the
#   heaviest single signal.
GENRE_MAP: Dict[str, str] = {
    "bossa nova":       "jazz",
    "brazilian jazz":   "jazz",
    "samba":            "jazz",
    "bebop":            "jazz",
    "swing":            "jazz",
    "blues":            "r&b",
    "soul blues":       "soul",
    "trap":             "hip-hop",
    "drill":            "hip-hop",
    "phonk":            "hip-hop",
    "industrial":       "metal",
    "black metal":      "metal",
    "death metal":      "metal",
    "hardcore":         "metal",
    "post-rock":        "rock",
    "emo":              "rock",
    "shoegaze":         "rock",
    "progressive rock": "rock",
    "new wave":         "pop",
    "synth pop":        "pop",
    "k-pop":            "pop",
    "chillwave":        "lofi",
    "lo-fi":            "lofi",
    "vaporwave":        "ambient",
    "new age":          "ambient",
    "drone":            "ambient",
    "techno":           "edm",
    "trance":           "edm",
    "house":            "edm",
    "dubstep":          "edm",
    "garage":           "edm",
    "flamenco":         "classical",
    "baroque":          "classical",
    "opera":            "classical",
    "orchestra":        "classical",
    "neoclassical":     "classical",
}


# ── Ranking strategies ───────────────────────────────────────────────────────
#
# Each strategy is a dict of per-signal multipliers applied on top of the
# baseline constants.  A multiplier of 1.0 = unchanged.  0.0 = disabled.
# _score_dict_strategy() reads these to scale each signal bucket at score time.
#
# Signal budget at baseline (mult=1.0):
#   genre(1.0) + subgenre(1.5) + mood(1.0) + mode(1.0)  → max  4.5 pts categorical
#   energy(5.0) + valence(1.5) + inst(0.5)               → max  7.0 pts continuous
#   era(1.0) + lang(0.75) + region(0.75) + vocal(0.5)
#     + tags(1.0) + pop(0.5) + live(0.75) + speech(0.75) → max  6.0 pts new-attr
#   Total baseline maximum                               → ~17.5 pts

STRATEGIES: Dict[str, Dict] = {

    # ── 1. Genre-First ────────────────────────────────────────────────────────
    # Catalog identity is king.  Genre + subgenre multiplied 4×; continuous
    # signals shrunk to 25% so a string match dominates purely numeric fit.
    # Best for: users who care deeply about staying inside one genre world.
    "genre-first": {
        "description": "Catalog identity first — genre and subgenre dominate, feel is secondary.",
        "genre_mult":    4.0,   # genre exact:    1.0 × 4  =  4.0 pts
        "subgenre_mult": 3.0,   # subgenre:       1.5 × 3  =  4.5 pts
        "mood_mult":     1.0,
        "mode_mult":     1.0,
        "energy_mult":   0.25,  # energy max:     5.0 × 0.25 = 1.25 pts
        "valence_mult":  0.30,  # valence max:    1.5 × 0.30 = 0.45 pts
        "inst_mult":     0.50,
        "era_mult":      0.50,
        "language_mult": 0.50,
        "region_mult":   0.50,
        "vocal_mult":    0.50,
        "tag_per_mult":  0.50,
        "tag_max_mult":  0.50,
        "pop_mult":      0.25,
        "live_mult":     0.25,
        "speech_mult":   0.25,
    },

    # ── 2. Mood-First ─────────────────────────────────────────────────────────
    # Emotional state is king.  Mood exact = 4 pts, mood tags up to 2.5 pts,
    # valence (emotional positivity/negativity) tripled.  Genre is a soft nudge.
    # Best for: session-aware recommendations ("I'm feeling sad right now").
    "mood-first": {
        "description": "Emotional state first — mood, valence, and mood tags dominate.",
        "genre_mult":    0.40,
        "subgenre_mult": 0.40,
        "mood_mult":     4.00,  # mood exact:  1.0 × 4  = 4.0 pts
        "mode_mult":     1.00,
        "energy_mult":   0.50,  # energy max:  5.0 × 0.5 = 2.5 pts
        "valence_mult":  3.00,  # valence max: 1.5 × 3  = 4.5 pts (emotional tone)
        "inst_mult":     0.50,
        "era_mult":      1.00,
        "language_mult": 1.00,
        "region_mult":   1.00,
        "vocal_mult":    1.00,
        "tag_per_mult":  2.50,  # per-tag:     0.4 × 2.5 = 1.0 pts; max = 2.5 pts
        "tag_max_mult":  2.50,
        "pop_mult":      0.50,
        "live_mult":     0.50,
        "speech_mult":   0.50,
    },

    # ── 3. Energy-Focused ─────────────────────────────────────────────────────
    # Physical intensity only.  Energy multiplied 3× to 15 pts max — it
    # overwhelms everything else.  Every other signal shrunk to 20–50%.
    # Best for: workout playlists, DJ sets, or any context where BPM / drive
    # matters more than genre or emotional nuance.
    "energy-focused": {
        "description": "Physical intensity only — energy dominant, everything else secondary.",
        "genre_mult":    0.20,
        "subgenre_mult": 0.20,
        "mood_mult":     0.20,
        "mode_mult":     0.50,
        "energy_mult":   3.00,  # energy max:  5.0 × 3 = 15.0 pts — overwhelming
        "valence_mult":  0.20,
        "inst_mult":     0.20,
        "era_mult":      0.25,
        "language_mult": 0.25,
        "region_mult":   0.25,
        "vocal_mult":    0.25,
        "tag_per_mult":  0.25,
        "tag_max_mult":  0.25,
        "pop_mult":      0.25,
        "live_mult":     0.25,
        "speech_mult":   0.25,
    },

    # ── 4. Vibe-Match ─────────────────────────────────────────────────────────
    # Cultural and contextual identity.  Era, region, language, vocal gender,
    # and detailed mood tags all multiplied 2.5–3×.  Genre and energy shrunk.
    # Best for: "I want something that sounds like it's from a specific place
    # and time" — surfaces cross-genre songs with matching cultural texture.
    "vibe-match": {
        "description": "Cultural & contextual identity — era, region, language, vocal gender, tags dominate.",
        "genre_mult":    0.30,
        "subgenre_mult": 0.30,
        "mood_mult":     0.75,
        "mode_mult":     0.50,
        "energy_mult":   0.40,
        "valence_mult":  0.50,
        "inst_mult":     0.50,
        "era_mult":      3.00,  # era:         1.0 × 3   = 3.0 pts
        "language_mult": 2.50,  # language:    0.75 × 2.5 = 1.875 pts
        "region_mult":   2.50,  # region:      0.75 × 2.5 = 1.875 pts
        "vocal_mult":    2.50,  # vocal gender: 0.5 × 2.5 = 1.25 pts
        "tag_per_mult":  3.00,  # per-tag:     0.4 × 3   = 1.2 pts; max = 3.0 pts
        "tag_max_mult":  3.00,
        "pop_mult":      1.00,
        "live_mult":     1.50,
        "speech_mult":   1.50,
    },

    # ── 5. Discovery ──────────────────────────────────────────────────────────
    # Genre completely disabled (0×).  Valence quadrupled — emotional fit is
    # the only categorical anchor.  Mood tags doubled.  Forces the system to
    # surface surprising cross-genre songs the user would never search for.
    # Best for: "Show me something I wouldn't normally pick myself."
    "discovery": {
        "description": "Genre-blind discovery — genre disabled; valence + tags surface unexpected fits.",
        "genre_mult":    0.00,  # genre and subgenre DISABLED
        "subgenre_mult": 0.00,
        "mood_mult":     1.50,
        "mode_mult":     1.00,
        "energy_mult":   1.20,
        "valence_mult":  4.00,  # valence max: 1.5 × 4 = 6.0 pts — primary signal
        "inst_mult":     1.00,
        "era_mult":      1.00,
        "language_mult": 1.00,
        "region_mult":   1.00,
        "vocal_mult":    1.00,
        "tag_per_mult":  2.00,  # per-tag: 0.4 × 2 = 0.8 pts; max = 2.0 pts
        "tag_max_mult":  2.00,
        "pop_mult":      1.00,
        "live_mult":     1.00,
        "speech_mult":   1.00,
    },
}


# ── Strategy pattern ──────────────────────────────────────────────────────────
#
# RankingStrategy wraps one config from STRATEGIES and exposes two methods:
#
#   .score(song, user)       → (float, [str])   — score one song
#   .rank(user, songs, k=5)  → [(song, score, reasons_str)]  — full top-k
#
# Callers never touch the raw cfg dict or call _score_dict_strategy directly.
# To switch modes, pass a different strategy object — the scoring logic is
# identical; only the multipliers change.
#
# Registry:
#   STRATEGY_REGISTRY  — dict[name → RankingStrategy]
#   get_strategy(name) — look up by name, raises ValueError if unknown

class RankingStrategy:
    """
    One ranking mode encapsulated as an object.

    Attributes
    ----------
    name        Short identifier, e.g. "genre-first".
    description One-line summary shown in comparisons.

    Methods
    -------
    score(song, user)         Score one song dict against one user dict.
    rank(user_prefs, songs)   Return top-k (song, score, reasons) tuples.
    """

    def __init__(self, name: str, description: str, cfg: Dict) -> None:
        self.name        = name
        self.description = description
        self._cfg        = cfg

    # ── Public interface ──────────────────────────────────────────────────────

    def score(self, song: Dict, user: Dict) -> Tuple[float, List[str]]:
        """Score a single song. Returns (numeric_score, list_of_reason_strings)."""
        return _score_dict_strategy(song, user, self._cfg)

    def rank(
        self,
        user_prefs: Dict,
        songs: List[Dict],
        k: int = 5,
    ) -> List[Tuple[Dict, float, str]]:
        """
        Return the top-k songs for user_prefs under this strategy.
        Output shape: [(song_dict, score, reasons_string), ...]
        Same shape as recommend_songs() so callers can swap strategies freely.
        """
        scored = [
            (song, *self.score(song, user_prefs))
            for song in songs
        ]
        # score() returns (float, list[str]); unpack into (song, float, list[str])
        packed = [
            (song, sc, " | ".join(reasons))
            for song, sc, reasons in scored
        ]
        packed.sort(key=lambda x: (x[1], -x[0]["id"]), reverse=True)
        return packed[:k]

    def __repr__(self) -> str:
        return f"RankingStrategy(name={self.name!r})"


# Build one RankingStrategy object per entry in STRATEGIES.
# Consumed by get_strategy() and by main.py for the comparison table.
STRATEGY_REGISTRY: Dict[str, "RankingStrategy"] = {
    name: RankingStrategy(name, cfg["description"], cfg)
    for name, cfg in STRATEGIES.items()
}


def get_strategy(name: str) -> "RankingStrategy":
    """
    Return the RankingStrategy for *name*.

    Usage
    -----
        strategy = get_strategy("mood-first")
        top5     = strategy.rank(user_prefs, songs)

    Raises
    ------
    ValueError if *name* is not in STRATEGY_REGISTRY.
    """
    if name not in STRATEGY_REGISTRY:
        available = ", ".join(sorted(STRATEGY_REGISTRY))
        raise ValueError(
            f"Unknown strategy {name!r}. Available: {available}"
        )
    return STRATEGY_REGISTRY[name]


# ── Shared scoring helpers ────────────────────────────────────────────────────

def _proximity(value: float, target: float, max_pts: float,
               sigma: float = PROXIMITY_SIGMA) -> float:
    """
    Fix 3 — Gaussian (RBF) proximity score.
    Returns max_pts at a perfect match and decays as a bell curve away from
    the target. Unlike the previous linear decay, this never hard-cuts to
    zero — every song earns some credit, creating meaningful differentiation
    even for neutral targets like 0.5.

    Formula: max_pts * exp( -(value - target)^2 / (2 * sigma^2) )

    Example (energy, max_pts=2.5, sigma=0.25):
      diff = 0.00 → 2.50 pts  (perfect match)
      diff = 0.25 → 1.52 pts  (~61% — smooth decay)
      diff = 0.50 → 0.34 pts  (meaningful but non-zero)
      diff = 1.00 → 0.01 pts  (effectively negligible)
    """
    return max_pts * math.exp(-((value - target) ** 2) / (2 * sigma ** 2))


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
    # Fix 2: Exact match first; fall back to GENRE_MAP for partial credit.
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
    else:
        mapped = GENRE_MAP.get(u_genre.lower(), "")
        if mapped and s_genre == mapped:
            score += POINTS_GENRE_PARTIAL
            hits.append(
                f"mapped genre '{u_genre}' → '{mapped}' (+{POINTS_GENRE_PARTIAL})"
            )

    # ── Mode scoring (major / minor key feel) ─────────────────────────────────
    # Fix 4: Reward-only. Mismatch earns 0 instead of -1.0, so a single
    #   key-feel miss cannot override a strong genre + mood + energy alignment.
    s_mode = int(song.get("mode", -1))
    u_mode = int(user.get("preferred_mode", -1))
    key_label = "major" if s_mode == 1 else "minor"
    if s_mode == u_mode:
        score += POINTS_MODE_MATCH
        hits.append(f"{key_label} key match (+{POINTS_MODE_MATCH})")
    else:
        score += POINTS_MODE_MISMATCH   # 0.0 — no penalty, no reward
        hits.append(f"{key_label} key mismatch (+{POINTS_MODE_MISMATCH})")

    # ── Continuous proximity scores (Fix 3: Gaussian RBF) ────────────────────
    e_pts = _proximity(
        float(song.get("energy", 0.5)),
        user.get("target_energy", 0.5),
        MAX_PTS_ENERGY,
        sigma=PROXIMITY_SIGMA,
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
        sigma=PROXIMITY_SIGMA,
    )
    score += v_pts
    hits.append(
        f"valence {float(song.get('valence', 0)):.2f} "
        f"(target {user.get('target_valence', 0.5):.2f}) +{v_pts:.2f}"
    )

    song_inst = float(song.get("instrumentalness", 0.5))
    target_inst = user.get("target_inst", 0.5)
    i_pts = _proximity(
        song_inst,
        target_inst,
        MAX_PTS_INST,
        sigma=PROXIMITY_SIGMA_INST,
    )
    score += i_pts
    hits.append(
        f"inst {song_inst:.2f} "
        f"(target {target_inst:.2f}) +{i_pts:.2f}"
    )

    # Fix 1: Multiplicative penalty for extreme instrumentalness mismatch.
    #   After all additive signals are summed, if inst is far from target,
    #   scale the whole score down. This lets "Lyric Lover" users actually
    #   feel the difference instead of just missing 0.5 pts.
    inst_diff = abs(song_inst - target_inst)
    if inst_diff > INST_PENALTY_THRESHOLD:
        score *= INST_PENALTY_FACTOR
        hits.append(
            f"inst penalty: diff {inst_diff:.2f} > {INST_PENALTY_THRESHOLD} "
            f"→ score ×{INST_PENALTY_FACTOR}"
        )

    # ── New-attribute scoring ─────────────────────────────────────────────────────

    # Explicit filter (hard penalty — multiplies entire score to near-zero)
    if not user.get("allow_explicit", True) and int(song.get("explicit", 0)) == 1:
        score *= EXPLICIT_PENALTY_FACTOR
        hits.append("explicit filter: score zeroed (allow_explicit=False)")

    # Era feel match
    u_era = user.get("preferred_era", "")
    s_era = song.get("era_feel", "")
    if u_era and s_era == u_era:
        score += POINTS_ERA_MATCH
        hits.append(f"era '{s_era}' match (+{POINTS_ERA_MATCH})")

    # Language match ("Instrumental" always passes — it works in any language)
    u_lang = user.get("preferred_language", "")
    s_lang = song.get("language", "")
    if u_lang and s_lang != "Instrumental":
        if s_lang == u_lang:
            score += POINTS_LANGUAGE_MATCH
            hits.append(f"language '{s_lang}' match (+{POINTS_LANGUAGE_MATCH})")

    # Cultural region match
    u_region = user.get("preferred_region", "")
    s_region = song.get("cultural_region", "")
    if u_region and s_region == u_region:
        score += POINTS_REGION_MATCH
        hits.append(f"region '{s_region}' match (+{POINTS_REGION_MATCH})")

    # Vocal gender match ("none" = instrumental, matches any user who prefers instrumentals)
    u_vg = user.get("preferred_vocal_gender", "")
    s_vg = song.get("vocal_gender", "")
    if u_vg and s_vg == u_vg:
        score += POINTS_VOCAL_MATCH
        hits.append(f"vocal gender '{s_vg}' match (+{POINTS_VOCAL_MATCH})")

    # Detailed mood-tag overlap
    u_tags = set(user.get("preferred_mood_tags", []))
    s_tags = set(song.get("detailed_mood_tags", "").split("|"))
    if u_tags:
        matching_tags = u_tags & s_tags
        tag_pts = min(len(matching_tags) * POINTS_PER_MOOD_TAG, MAX_PTS_MOOD_TAGS)
        score += tag_pts
        if matching_tags:
            hits.append(f"mood tags {sorted(matching_tags)} overlap +{tag_pts:.2f}")

    # Popularity proximity (Gaussian, target_popularity is 0.0–1.0 normalised)
    u_pop = user.get("target_popularity", None)
    if u_pop is not None:
        s_pop = float(song.get("popularity", 50)) / 100.0
        pop_pts = _proximity(s_pop, u_pop, MAX_PTS_POPULARITY, sigma=PROXIMITY_SIGMA_POP)
        score += pop_pts
        hits.append(f"popularity {s_pop:.2f} (target {u_pop:.2f}) +{pop_pts:.2f}")

    # Liveness proximity (Gaussian)
    u_live = user.get("target_liveness", None)
    if u_live is not None:
        s_live = float(song.get("liveness", 0.15))
        live_pts = _proximity(s_live, u_live, MAX_PTS_LIVENESS, sigma=PROXIMITY_SIGMA_LIVE)
        score += live_pts
        hits.append(f"liveness {s_live:.2f} (target {u_live:.2f}) +{live_pts:.2f}")

    # Speechiness proximity (Gaussian)
    u_speech = user.get("target_speechiness", None)
    if u_speech is not None:
        s_speech = float(song.get("speechiness", 0.05))
        speech_pts = _proximity(
            s_speech, u_speech, MAX_PTS_SPEECHINESS, sigma=PROXIMITY_SIGMA_SPEECH
        )
        score += speech_pts
        hits.append(f"speechiness {s_speech:.2f} (target {u_speech:.2f}) +{speech_pts:.2f}")

    # Fix 5: Soft floor — energy-scaled epsilon prevents true 0.0 ties.
    song_energy = float(song.get("energy", 0.5))
    soft_floor = SCORE_SOFT_FLOOR_BASE * (1.0 + song_energy)
    return max(soft_floor, score), hits


def _score_dict_strategy(song: Dict, user: Dict, cfg: Dict) -> Tuple[float, List[str]]:
    """
    Score a song using per-signal multipliers from a STRATEGIES config dict.
    Each signal's contribution is scaled by its multiplier before accumulation.
    A multiplier of 0.0 disables a signal entirely; 1.0 = baseline behaviour.
    The explicit hard-filter and soft floor always apply regardless of strategy.
    """
    score = 0.0
    hits: List[str] = []

    m_mood    = cfg.get("mood_mult",    1.0)
    m_genre   = cfg.get("genre_mult",   1.0)
    m_sub     = cfg.get("subgenre_mult",1.0)
    m_mode    = cfg.get("mode_mult",    1.0)
    m_energy  = cfg.get("energy_mult",  1.0)
    m_valence = cfg.get("valence_mult", 1.0)
    m_inst    = cfg.get("inst_mult",    1.0)
    m_era     = cfg.get("era_mult",     1.0)
    m_lang    = cfg.get("language_mult",1.0)
    m_region  = cfg.get("region_mult",  1.0)
    m_vocal   = cfg.get("vocal_mult",   1.0)
    m_tag_per = cfg.get("tag_per_mult", 1.0)
    m_tag_max = cfg.get("tag_max_mult", 1.0)
    m_pop     = cfg.get("pop_mult",     1.0)
    m_live    = cfg.get("live_mult",    1.0)
    m_speech  = cfg.get("speech_mult",  1.0)

    # ── Mood ─────────────────────────────────────────────────────────────────
    s_mood = song.get("mood", "")
    u_mood = user.get("favorite_mood", "")
    if s_mood == u_mood:
        pts = POINTS_MOOD_EXACT * m_mood
        score += pts
        hits.append(f"mood exact '{s_mood}' +{pts:.2f}")
    elif s_mood in _ADJACENT_MOODS.get(u_mood, set()):
        pts = POINTS_MOOD_ADJACENT * m_mood
        score += pts
        hits.append(f"adjacent mood '{s_mood}' +{pts:.2f}")

    # ── Genre + subgenre ─────────────────────────────────────────────────────
    s_genre = song.get("genre", "")
    u_genre = user.get("favorite_genre", "")
    if s_genre == u_genre:
        pts = POINTS_GENRE_EXACT * m_genre
        score += pts
        hits.append(f"genre '{s_genre}' +{pts:.2f}")
        if song.get("subgenre", "") == user.get("favorite_subgenre", ""):
            pts2 = POINTS_SUBGENRE_EXACT * m_sub
            score += pts2
            hits.append(f"subgenre '{song.get('subgenre')}' +{pts2:.2f}")
    else:
        mapped = GENRE_MAP.get(u_genre.lower(), "")
        if mapped and s_genre == mapped:
            pts = POINTS_GENRE_PARTIAL * m_genre
            score += pts
            hits.append(f"mapped genre '{u_genre}'→'{mapped}' +{pts:.2f}")

    # ── Mode ─────────────────────────────────────────────────────────────────
    s_mode = int(song.get("mode", -1))
    u_mode = int(user.get("preferred_mode", -1))
    key_label = "major" if s_mode == 1 else "minor"
    if s_mode == u_mode:
        pts = POINTS_MODE_MATCH * m_mode
        score += pts
        hits.append(f"{key_label} key +{pts:.2f}")

    # ── Continuous (Gaussian) ─────────────────────────────────────────────────
    e_pts = _proximity(float(song.get("energy", 0.5)),
                       user.get("target_energy", 0.5),
                       MAX_PTS_ENERGY * m_energy)
    score += e_pts
    hits.append(f"energy {float(song.get('energy',0)):.2f} +{e_pts:.2f}")

    v_pts = _proximity(float(song.get("valence", 0.5)),
                       user.get("target_valence", 0.5),
                       MAX_PTS_VALENCE * m_valence)
    score += v_pts
    hits.append(f"valence {float(song.get('valence',0)):.2f} +{v_pts:.2f}")

    song_inst   = float(song.get("instrumentalness", 0.5))
    target_inst = user.get("target_inst", 0.5)
    i_pts = _proximity(song_inst, target_inst,
                       MAX_PTS_INST * m_inst, sigma=PROXIMITY_SIGMA_INST)
    score += i_pts
    hits.append(f"inst {song_inst:.2f} +{i_pts:.2f}")

    if abs(song_inst - target_inst) > INST_PENALTY_THRESHOLD:
        score *= INST_PENALTY_FACTOR
        hits.append(f"inst penalty ×{INST_PENALTY_FACTOR}")

    # ── Explicit — always hard filter regardless of strategy ──────────────────
    if not user.get("allow_explicit", True) and int(song.get("explicit", 0)) == 1:
        score *= EXPLICIT_PENALTY_FACTOR
        hits.append("explicit: zeroed")

    # ── New attributes ────────────────────────────────────────────────────────
    u_era = user.get("preferred_era", "")
    if u_era and song.get("era_feel", "") == u_era:
        pts = POINTS_ERA_MATCH * m_era
        score += pts
        hits.append(f"era '{u_era}' +{pts:.2f}")

    u_lang = user.get("preferred_language", "")
    s_lang = song.get("language", "")
    if u_lang and s_lang != "Instrumental" and s_lang == u_lang:
        pts = POINTS_LANGUAGE_MATCH * m_lang
        score += pts
        hits.append(f"lang '{s_lang}' +{pts:.2f}")

    u_region = user.get("preferred_region", "")
    if u_region and song.get("cultural_region", "") == u_region:
        pts = POINTS_REGION_MATCH * m_region
        score += pts
        hits.append(f"region '{u_region}' +{pts:.2f}")

    u_vg = user.get("preferred_vocal_gender", "")
    if u_vg and song.get("vocal_gender", "") == u_vg:
        pts = POINTS_VOCAL_MATCH * m_vocal
        score += pts
        hits.append(f"vocal '{u_vg}' +{pts:.2f}")

    u_tags = set(user.get("preferred_mood_tags", []))
    s_tags = set(song.get("detailed_mood_tags", "").split("|"))
    if u_tags:
        matching = u_tags & s_tags
        tag_pts  = min(len(matching) * POINTS_PER_MOOD_TAG * m_tag_per,
                       MAX_PTS_MOOD_TAGS * m_tag_max)
        score += tag_pts
        if matching:
            hits.append(f"tags {sorted(matching)} +{tag_pts:.2f}")

    u_pop = user.get("target_popularity", None)
    if u_pop is not None:
        s_pop   = float(song.get("popularity", 50)) / 100.0
        pop_pts = _proximity(s_pop, u_pop, MAX_PTS_POPULARITY * m_pop,
                             sigma=PROXIMITY_SIGMA_POP)
        score += pop_pts
        hits.append(f"pop {s_pop:.2f} +{pop_pts:.2f}")

    u_live = user.get("target_liveness", None)
    if u_live is not None:
        s_live   = float(song.get("liveness", 0.15))
        live_pts = _proximity(s_live, u_live, MAX_PTS_LIVENESS * m_live,
                              sigma=PROXIMITY_SIGMA_LIVE)
        score += live_pts
        hits.append(f"live {s_live:.2f} +{live_pts:.2f}")

    u_speech = user.get("target_speechiness", None)
    if u_speech is not None:
        s_speech   = float(song.get("speechiness", 0.05))
        speech_pts = _proximity(s_speech, u_speech, MAX_PTS_SPEECHINESS * m_speech,
                                sigma=PROXIMITY_SIGMA_SPEECH)
        score += speech_pts
        hits.append(f"speech {s_speech:.2f} +{speech_pts:.2f}")

    soft_floor = SCORE_SOFT_FLOOR_BASE * (1.0 + float(song.get("energy", 0.5)))
    return max(soft_floor, score), hits


def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    Score a single song dictionary against user preferences.

    Recipe:
    - +2.0 points for a genre match (or +1.0 for a mapped/partial genre match)
    - +1.0 point for a mood match
    - +0.0 to +2.5 points for energy similarity (Gaussian)
    - +0.0 to +1.5 points for valence similarity (Gaussian)
    - +0.0 to +0.5 points for instrumentalness similarity (Gaussian)
    - ×0.6 multiplier if instrumentalness diff > 0.7 (extreme mismatch penalty)
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
                "popularity":          int(row["popularity"]),
                "release_decade":      row["release_decade"],
                "explicit":            int(row["explicit"]),
                "language":            row["language"],
                "duration_sec":        int(row["duration_sec"]),
                "loudness_db":         float(row["loudness_db"]),
                "speechiness":         float(row["speechiness"]),
                "liveness":            float(row["liveness"]),
                "detailed_mood_tags":  row["detailed_mood_tags"],
                "cultural_region":     row["cultural_region"],
                "vocal_gender":        row["vocal_gender"],
                "era_feel":            row["era_feel"],
            })
    print(f"  Loaded {len(songs)} songs.")
    return songs


def recommend_songs(
    user_prefs: Dict, songs: List[Dict], k: int = 5
) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py

    Fix 5: Sorted by (score DESC, id ASC) so that when scores are equal or
    near-equal after the soft floor, catalog order (lower id wins) is the
    tiebreaker — deterministic and transparent.
    """
    scored = []
    for song in songs:
        score, reasons = score_song(user_prefs, song)
        scored.append((song, score, " | ".join(reasons)))

    # Primary: score descending. Secondary: catalog id ascending (lower id wins ties).
    scored.sort(key=lambda x: (x[1], -x[0]["id"]), reverse=True)
    return scored[:k]


def recommend_with_strategy(
    user_prefs: Dict, songs: List[Dict], strategy_name: str, k: int = 5
) -> List[Tuple[Dict, float, str]]:
    """
    Rank songs using a named strategy from STRATEGY_REGISTRY.
    Returns top-k as (song_dict, score, reasons_string), same shape as
    recommend_songs so callers can swap between them transparently.

    Available strategy names: "genre-first", "mood-first", "energy-focused",
    "vibe-match", "discovery".
    """
    return get_strategy(strategy_name).rank(user_prefs, songs, k)


# ── OOP API (used by tests/test_recommender.py) ───────────────────────────────

class Recommender:
    """
    ToneMatch 1.0 — content-based music recommender.
    Matches songs to a user profile using energy, valence, acousticness,
    mood, and genre as the primary vibe signals.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        """Store the song catalog for use across recommendation calls."""
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return the top-k songs ranked by composite score against the user profile."""
        scored = [
            (song, _score_song_obj(song, user)[0])
            for song in self.songs
        ]
        scored.sort(key=lambda x: (x[1], -x[0].id), reverse=True)
        return [song for song, _ in scored[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Return a human-readable string listing every scoring rule that fired for this song."""
        _, reasons = _score_song_obj(song, user)
        return " | ".join(reasons)
