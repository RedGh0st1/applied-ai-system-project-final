"""
ToneMatch 1.0 — Music Recommender Simulation runner.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from .recommender import load_songs, recommend_songs

# ── User profiles ─────────────────────────────────────────────────────────────

PROFILES = {
    # ── Original profiles ─────────────────────────────────────────────────────
    "High-Energy Pop": {
        "favorite_genre":    "pop",
        "favorite_subgenre": "dance pop",
        "favorite_mood":     "happy",
        "target_energy":     0.88,   # loud, bright, driven
        "target_valence":    0.82,   # upbeat and positive
        "target_bpm":        128.0,  # dance-floor pace
        "target_acoustic":   0.10,   # fully produced, no acoustic texture
        "target_inst":       0.05,   # wants vocals front and center
        "preferred_mode":    1,      # major key — bright and resolved
        "likes_acoustic":    False,
    },

    "Chill Lo-Fi": {
        "favorite_genre":    "lofi",
        "favorite_subgenre": "lofi hip-hop",
        "favorite_mood":     "focused",
        "target_energy":     0.40,   # low intensity, background listening
        "target_valence":    0.58,   # calm-positive, not sad
        "target_bpm":        78.0,   # slow, unhurried groove
        "target_acoustic":   0.80,   # warm, organic texture
        "target_inst":       0.87,   # near-fully instrumental for focus
        "preferred_mode":    1,      # major key — relaxed, not tense
        "likes_acoustic":    True,
    },

    "Deep Intense Rock": {
        "favorite_genre":    "rock",
        "favorite_subgenre": "hard rock",
        "favorite_mood":     "aggressive",
        "target_energy":     0.93,   # maximum physical intensity
        "target_valence":    0.35,   # dark and driven, not cheerful
        "target_bpm":        155.0,  # fast, relentless tempo
        "target_acoustic":   0.08,   # distortion and amplification, no acoustic
        "target_inst":       0.10,   # raw vocals, shouting encouraged
        "preferred_mode":    0,      # minor key — tension and aggression
        "likes_acoustic":    False,
    },

    # ── Adversarial profiles — designed to stress-test scoring logic ──────────

    # Fix 1 (Independence Fix): mood and energy pull in opposite directions.
    # Before the fix, genre + mood dominated and the continuous signals were
    # decorative. After: Gaussian energy scoring contributes meaningfully and
    # the mode reward (not penalty) keeps the ranking honest.
    "The Contradiction (Fix 1 — Independence)": {
        "favorite_genre":    "pop",
        "favorite_subgenre": "dance pop",
        "favorite_mood":     "sad",        # pulls toward quiet, minor, dark songs
        "target_energy":     0.92,         # pulls toward high-intensity songs
        "target_valence":    0.10,         # very dark / negative
        "target_bpm":        130.0,
        "target_acoustic":   0.10,
        "target_inst":       0.05,
        "preferred_mode":    0,            # minor key
        "likes_acoustic":    False,
    },

    # Fix 2 (Semantic Fallback): "bossa nova" is not in the catalog.
    # Before the fix, POINTS_GENRE_EXACT (2.0) was never awarded and the user
    # silently got random results. After: GENRE_MAP maps "bossa nova" → "jazz",
    # awarding POINTS_GENRE_PARTIAL (1.0) so Jazz songs surface correctly.
    "The Genre Ghost (Fix 2 — Semantic Fallback)": {
        "favorite_genre":    "bossa nova",  # not in catalog; maps to "jazz"
        "favorite_subgenre": "brazilian jazz",
        "favorite_mood":     "relaxed",
        "target_energy":     0.38,
        "target_valence":    0.68,
        "target_bpm":        90.0,
        "target_acoustic":   0.80,
        "target_inst":       0.45,
        "preferred_mode":    1,
        "likes_acoustic":    True,
    },

    # Fix 3 (Gaussian Scoring): all continuous targets at 0.5 (dead center).
    # Before the fix, linear proximity with target=0.5 awarded partial points
    # to almost everything, making categorical signals overwhelmingly dominant.
    # After: Gaussian RBF creates a true peak at 0.5 — songs that are
    # "perfectly neutral" score measurably higher than ones that are just close.
    "The Agnostic (Fix 3 — Gaussian Resolution)": {
        "favorite_genre":    "pop",
        "favorite_subgenre": "",
        "favorite_mood":     "happy",
        "target_energy":     0.50,   # dead center on every continuous axis
        "target_valence":    0.50,
        "target_bpm":        100.0,
        "target_acoustic":   0.50,
        "target_inst":       0.50,
        "preferred_mode":    1,
        "likes_acoustic":    True,
    },

    # Fix 4 (Balance Fix): user loves happy pop but prefers minor key.
    # Before the fix, POINTS_MODE_MISMATCH = -1.0 caused a 2-pt swing that
    # could flip a perfect genre + mood match into a loss. After: mismatch
    # earns 0 instead of -1.0 so strong mood/genre wins still win.
    "The Minor Happy (Fix 4 — Symmetric Mode)": {
        "favorite_genre":    "pop",
        "favorite_subgenre": "indie pop",
        "favorite_mood":     "happy",
        "target_energy":     0.75,
        "target_valence":    0.80,   # bright and positive ...
        "target_bpm":        120.0,
        "target_acoustic":   0.30,
        "target_inst":       0.10,
        "preferred_mode":    0,      # ... but prefers minor key — the contradiction
        "likes_acoustic":    False,
    },

    # Fix 1 (Independence Fix — Lyric Lover variant): target_inst=0.95 signals
    # a strong desire for fully instrumental music. Before the fix, the 0.5-pt
    # max for inst was too small to beat a genre match — vocal songs still won.
    # After: the ×0.6 multiplicative penalty fires when inst diff > 0.7,
    # scaling down vocal-heavy songs even when genre/mood is a perfect match.
    "The Lyric Lover (Fix 1 — Inst Penalty)": {
        "favorite_genre":    "lofi",
        "favorite_subgenre": "lofi hip-hop",
        "favorite_mood":     "focused",
        "target_energy":     0.38,
        "target_valence":    0.55,
        "target_bpm":        78.0,
        "target_acoustic":   0.80,
        "target_inst":       0.95,   # wants fully instrumental
        "preferred_mode":    1,
        "likes_acoustic":    True,
    },

    # Fix 5 (Sorting Fix): niche genre + extreme targets push most songs to
    # near-zero. Before the fix, max(0.0, score) created a "dead zone" where
    # multiple songs tied at exactly 0.0 and catalog order decided the winner
    # silently. After: soft floor = 0.0001 × (1 + energy) ensures no two songs
    # share an identical score, and (score DESC, id ASC) sorting makes
    # tie-breaking transparent and deterministic.
    "The Mismatch Maximizer (Fix 5 — Soft Floor + Tie-break)": {
        "favorite_genre":    "classical",
        "favorite_subgenre": "baroque",   # not in catalog
        "favorite_mood":     "serene",
        "target_energy":     0.05,        # very low — most songs miss badly
        "target_valence":    0.10,        # very dark
        "target_bpm":        50.0,
        "target_acoustic":   0.99,
        "target_inst":       0.99,
        "preferred_mode":    0,           # minor
        "likes_acoustic":    True,
    },

    # Uses: era_feel, detailed_mood_tags, popularity, liveness
    "The Retro Soul Digger": {
        "favorite_genre":       "soul",
        "favorite_subgenre":    "southern soul",
        "favorite_mood":        "melancholic",
        "target_energy":        0.38,
        "target_valence":       0.35,
        "target_bpm":           70.0,
        "target_acoustic":      0.65,
        "target_inst":          0.10,
        "preferred_mode":       0,
        "likes_acoustic":       True,
        # new fields
        "preferred_era":        "retro",
        "preferred_language":   "English",
        "preferred_region":     "Western",
        "preferred_vocal_gender": "female",
        "allow_explicit":       True,
        "target_popularity":    0.45,   # prefers under-the-radar tracks
        "target_liveness":      0.45,   # likes that live, raw feeling
        "target_speechiness":   0.10,
        "preferred_mood_tags":  ["heartbroken", "raw", "soulful", "lonesome", "weary"],
    },

    # Uses: language, cultural_region, detailed_mood_tags, speechiness
    "The Global Dance Floor": {
        "favorite_genre":       "latin",
        "favorite_subgenre":    "salsa",
        "favorite_mood":        "passionate",
        "target_energy":        0.88,
        "target_valence":       0.88,
        "target_bpm":           170.0,
        "target_acoustic":      0.20,
        "target_inst":          0.10,
        "preferred_mode":       1,
        "likes_acoustic":       False,
        # new fields
        "preferred_era":        "retro",
        "preferred_language":   "Spanish",
        "preferred_region":     "Latin American",
        "preferred_vocal_gender": "mixed",
        "allow_explicit":       False,
        "target_popularity":    0.75,
        "target_liveness":      0.45,
        "target_speechiness":   0.09,
        "preferred_mood_tags":  ["fiery", "celebratory", "sensual", "danceable"],
    },

    # Uses: explicit filter, speechiness, vocal_gender, mood_tags
    "The Clean Rap Fan": {
        "favorite_genre":       "hip-hop",
        "favorite_subgenre":    "boom bap",
        "favorite_mood":        "confident",
        "target_energy":        0.72,
        "target_valence":       0.68,
        "target_bpm":           95.0,
        "target_acoustic":      0.15,
        "target_inst":          0.05,
        "preferred_mode":       0,
        "likes_acoustic":       False,
        # new fields
        "preferred_era":        "retro",
        "preferred_language":   "English",
        "preferred_region":     "Western",
        "preferred_vocal_gender": "male",
        "allow_explicit":       False,   # hard filter — no explicit tracks
        "target_popularity":    0.65,
        "target_liveness":      0.12,
        "target_speechiness":   0.38,   # wants high-speechiness rap delivery
        "preferred_mood_tags":  ["street", "rhythmic", "braggadocious"],
    },
}


def main() -> None:
    songs = load_songs("data/songs.csv")

    for profile_name, user_prefs in PROFILES.items():
        print(f"\n{'=' * 62}")
        print(f"  PROFILE: {profile_name}")
        print(f"{'=' * 62}")

        recommendations = recommend_songs(user_prefs, songs, k=5)

        for idx, rec in enumerate(recommendations, start=1):
            song, score, explanation = rec
            print(f"\n  {idx}. {song['title']} by {song['artist']}")
            print(f"     Genre: {song['genre']} / {song['mood']}  |  Score: {score:.4f}")
            print("     Reasons:")
            for reason in explanation.split(" | "):
                print(f"       - {reason}")
        print()


if __name__ == "__main__":
    main()