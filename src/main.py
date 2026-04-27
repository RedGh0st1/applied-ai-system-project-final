"""
ToneMatch — Music Recommender with integrated AI features.

Application flow
----------------
Section 1  Profile recommendations
           Scoring engine ranks songs (retrieval).
           For AI-narrated profiles Claude formulates the response using the
           ranked songs as context (generation) — this IS the RAG loop when
           the scoring engine acts as the retriever.
           Non-narrated profiles fall back to the raw score output.

Section 2  Strategy comparison
           Side-by-side numerical view of how the 5 ranking strategies
           diverge on three representative profiles.

Section 3  RAG — natural language queries   [requires ANTHROPIC_API_KEY]
           classify_query_intent → retrieve_songs_for_query → rag_recommend
           → validate_recommendation_relevance.
           Claude's grounded answer is the primary output; guardrail scores
           are printed to show automated validation is active.

Section 4  Step-by-step playlist planning   [requires ANTHROPIC_API_KEY]
           Claude reasons through an occasion's energy arc, maps moods to
           phases, and selects specific catalog songs for each phase.
"""

import logging
import os
import textwrap

logger = logging.getLogger(__name__)

from .recommender import (
    load_songs,
    recommend_with_strategy,
    get_strategy,
    STRATEGIES,
)
from .ai_features import (
    generate_recommendation_response,
    rag_recommend,
    plan_playlist_for_occasion,
    classify_query_intent,
    agentic_recommend,
)

# ── Active strategy ───────────────────────────────────────────────────────────
# Options: "genre-first" | "mood-first" | "energy-focused" | "vibe-match" | "discovery"
ACTIVE_STRATEGY = "genre-first"

# ── User profiles ─────────────────────────────────────────────────────────────

PROFILES = {
    # ── Original profiles ─────────────────────────────────────────────────────
    "High-Energy Pop": {
        "favorite_genre":    "pop",
        "favorite_subgenre": "dance pop",
        "favorite_mood":     "happy",
        "target_energy":     0.88,
        "target_valence":    0.82,
        "target_bpm":        128.0,
        "target_acoustic":   0.10,
        "target_inst":       0.05,
        "preferred_mode":    1,
        "likes_acoustic":    False,
    },

    "Chill Lo-Fi": {
        "favorite_genre":    "lofi",
        "favorite_subgenre": "lofi hip-hop",
        "favorite_mood":     "focused",
        "target_energy":     0.40,
        "target_valence":    0.58,
        "target_bpm":        78.0,
        "target_acoustic":   0.80,
        "target_inst":       0.87,
        "preferred_mode":    1,
        "likes_acoustic":    True,
    },

    "Deep Intense Rock": {
        "favorite_genre":    "rock",
        "favorite_subgenre": "hard rock",
        "favorite_mood":     "aggressive",
        "target_energy":     0.93,
        "target_valence":    0.35,
        "target_bpm":        155.0,
        "target_acoustic":   0.08,
        "target_inst":       0.10,
        "preferred_mode":    0,
        "likes_acoustic":    False,
    },

    # ── Adversarial profiles ──────────────────────────────────────────────────

    "The Contradiction (Fix 1 — Independence)": {
        "favorite_genre":    "pop",
        "favorite_subgenre": "dance pop",
        "favorite_mood":     "sad",
        "target_energy":     0.92,
        "target_valence":    0.10,
        "target_bpm":        130.0,
        "target_acoustic":   0.10,
        "target_inst":       0.05,
        "preferred_mode":    0,
        "likes_acoustic":    False,
    },

    "The Genre Ghost (Fix 2 — Semantic Fallback)": {
        "favorite_genre":    "bossa nova",
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

    "The Agnostic (Fix 3 — Gaussian Resolution)": {
        "favorite_genre":    "pop",
        "favorite_subgenre": "",
        "favorite_mood":     "happy",
        "target_energy":     0.50,
        "target_valence":    0.50,
        "target_bpm":        100.0,
        "target_acoustic":   0.50,
        "target_inst":       0.50,
        "preferred_mode":    1,
        "likes_acoustic":    True,
    },

    "The Minor Happy (Fix 4 — Symmetric Mode)": {
        "favorite_genre":    "pop",
        "favorite_subgenre": "indie pop",
        "favorite_mood":     "happy",
        "target_energy":     0.75,
        "target_valence":    0.80,
        "target_bpm":        120.0,
        "target_acoustic":   0.30,
        "target_inst":       0.10,
        "preferred_mode":    0,
        "likes_acoustic":    False,
    },

    "The Lyric Lover (Fix 1 — Inst Penalty)": {
        "favorite_genre":    "lofi",
        "favorite_subgenre": "lofi hip-hop",
        "favorite_mood":     "focused",
        "target_energy":     0.38,
        "target_valence":    0.55,
        "target_bpm":        78.0,
        "target_acoustic":   0.80,
        "target_inst":       0.95,
        "preferred_mode":    1,
        "likes_acoustic":    True,
    },

    "The Mismatch Maximizer (Fix 5 — Soft Floor + Tie-break)": {
        "favorite_genre":    "classical",
        "favorite_subgenre": "baroque",
        "favorite_mood":     "serene",
        "target_energy":     0.05,
        "target_valence":    0.10,
        "target_bpm":        50.0,
        "target_acoustic":   0.99,
        "target_inst":       0.99,
        "preferred_mode":    0,
        "likes_acoustic":    True,
    },

    "The Retro Soul Digger": {
        "favorite_genre":         "soul",
        "favorite_subgenre":      "southern soul",
        "favorite_mood":          "melancholic",
        "target_energy":          0.38,
        "target_valence":         0.35,
        "target_bpm":             70.0,
        "target_acoustic":        0.65,
        "target_inst":            0.10,
        "preferred_mode":         0,
        "likes_acoustic":         True,
        "preferred_era":          "retro",
        "preferred_language":     "English",
        "preferred_region":       "Western",
        "preferred_vocal_gender": "female",
        "allow_explicit":         True,
        "target_popularity":      0.45,
        "target_liveness":        0.45,
        "target_speechiness":     0.10,
        "preferred_mood_tags":    ["heartbroken", "raw", "soulful", "lonesome", "weary"],
    },

    "The Global Dance Floor": {
        "favorite_genre":         "latin",
        "favorite_subgenre":      "salsa",
        "favorite_mood":          "passionate",
        "target_energy":          0.88,
        "target_valence":         0.88,
        "target_bpm":             170.0,
        "target_acoustic":        0.20,
        "target_inst":            0.10,
        "preferred_mode":         1,
        "likes_acoustic":         False,
        "preferred_era":          "retro",
        "preferred_language":     "Spanish",
        "preferred_region":       "Latin American",
        "preferred_vocal_gender": "mixed",
        "allow_explicit":         False,
        "target_popularity":      0.75,
        "target_liveness":        0.45,
        "target_speechiness":     0.09,
        "preferred_mood_tags":    ["fiery", "celebratory", "sensual", "danceable"],
    },

    "The Clean Rap Fan": {
        "favorite_genre":         "hip-hop",
        "favorite_subgenre":      "boom bap",
        "favorite_mood":          "confident",
        "target_energy":          0.72,
        "target_valence":         0.68,
        "target_bpm":             95.0,
        "target_acoustic":        0.15,
        "target_inst":            0.05,
        "preferred_mode":         0,
        "likes_acoustic":         False,
        "preferred_era":          "retro",
        "preferred_language":     "English",
        "preferred_region":       "Western",
        "preferred_vocal_gender": "male",
        "allow_explicit":         False,
        "target_popularity":      0.65,
        "target_liveness":        0.12,
        "target_speechiness":     0.38,
        "preferred_mood_tags":    ["street", "rhythmic", "braggadocious"],
    },
}

# ── AI-narrated profiles ──────────────────────────────────────────────────────
# For these profiles the scoring engine acts as the retriever and Claude acts
# as the generator — the AI narrative replaces the raw score dump.
# Other profiles still run but show the raw numerical output.
AI_NARRATED_PROFILES = {
    "High-Energy Pop",
    "Chill Lo-Fi",
    "The Retro Soul Digger",
}

# ── RAG queries for Section 3 ─────────────────────────────────────────────────
RAG_QUERIES = [
    "I need something dark and moody for a late-night drive",
    "upbeat songs to hype me up for a morning workout",
    "chill background music for studying late at night",
]

# ── Occasion for Section 4 ────────────────────────────────────────────────────
PLANNING_OCCASION = "a dinner party that starts relaxed and gradually builds to dancing"

# ── Queries for Section 5 (agentic workflow) ──────────────────────────────────
# Chosen to require multi-tool reasoning: each needs at least one catalog
# search plus a ranking step, and the right answer is non-obvious from the
# query words alone.
AGENT_QUERIES = [
    "I want something calm but not too quiet for a yoga class",
    "Find me dark and intense music that isn't typical metal",
]

# ── Strategy comparison config ────────────────────────────────────────────────
STRATEGY_DEMO_PROFILES = [
    "High-Energy Pop",
    "The Contradiction (Fix 1 — Independence)",
    "The Retro Soul Digger",
]

STRATEGY_ORDER = [
    "genre-first",
    "mood-first",
    "energy-focused",
    "vibe-match",
    "discovery",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pad(text: str, width: int) -> str:
    return text[:width].ljust(width)


def _wrap(text: str, width: int = 70, indent: str = "  ") -> str:
    """Wrap long text to width, preserving paragraph breaks."""
    paragraphs = text.split("\n")
    wrapped = []
    for para in paragraphs:
        if para.strip():
            wrapped.append(textwrap.fill(para, width=width, initial_indent=indent,
                                         subsequent_indent=indent))
        else:
            wrapped.append("")
    return "\n".join(wrapped)


# ── Section 2: strategy comparison ───────────────────────────────────────────

def print_strategy_comparison(songs: list) -> None:
    col = 28
    strat_labels = {
        "genre-first":    "Genre-First",
        "mood-first":     "Mood-First",
        "energy-focused": "Energy-Focused",
        "vibe-match":     "Vibe-Match",
        "discovery":      "Discovery",
    }

    for profile_name in STRATEGY_DEMO_PROFILES:
        user_prefs = PROFILES[profile_name]
        width = col * len(STRATEGY_ORDER) + len(STRATEGY_ORDER) - 1

        print(f"\n{'╔' + '═' * width + '╗'}")
        print(f"  STRATEGY COMPARISON — {profile_name}")
        print(f"{'╚' + '═' * width + '╝'}")

        header  = " | ".join(_pad(strat_labels[s], col) for s in STRATEGY_ORDER)
        descs   = " | ".join(_pad(STRATEGIES[s]["description"], col) for s in STRATEGY_ORDER)
        print(f"\n  {header}")
        print(f"  {descs}")
        print(f"  {'─' * (col * len(STRATEGY_ORDER) + (len(STRATEGY_ORDER) - 1) * 3)}")

        results = {s: recommend_with_strategy(user_prefs, songs, s, k=5) for s in STRATEGY_ORDER}
        for rank in range(5):
            row = []
            for s in STRATEGY_ORDER:
                song, score, _ = results[s][rank]
                row.append(_pad(f"{rank+1}. {song['title']} [{score:.1f}]", col))
            print(f"  {' | '.join(row)}")

        top_ones   = {s: results[s][0][0]["title"] for s in STRATEGY_ORDER}
        unique_top = set(top_ones.values())
        print(f"\n  #1 winners: {len(unique_top)} distinct song(s)")
        for s in STRATEGY_ORDER:
            print(f"    {strat_labels[s]:16s}  →  {top_ones[s]}")
        print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    ToneMatch main loop.

    Sections
    --------
    1  Profile recommendations — AI narrative (selected) or raw scores (others)
    2  Strategy comparison     — numerical side-by-side table
    3  RAG queries             — classify → retrieve → generate → validate
    4  Occasion planning       — step-by-step playlist curation
    """
    # Configure logging: INFO to console, DEBUG available via LOG_LEVEL=DEBUG
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    songs      = load_songs("data/songs.csv")
    strategy   = get_strategy(ACTIVE_STRATEGY)
    ai_enabled = bool(os.environ.get("ANTHROPIC_API_KEY"))

    logger.info("ToneMatch started — strategy=%r, ai_enabled=%s, catalog=%d songs",
                ACTIVE_STRATEGY, ai_enabled, len(songs))
    print(f"\n  Strategy : {strategy.name!r} — {strategy.description}")
    print(f"  AI mode  : {'enabled' if ai_enabled else 'disabled (set ANTHROPIC_API_KEY)'}")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — Profile-based recommendations
    # ═══════════════════════════════════════════════════════════════════════════
    #
    # The scoring engine ranks songs against each profile (retrieval).
    # For AI-narrated profiles, Claude receives those ranked songs as context
    # and formulates the recommendation narrative (generation).
    # For other profiles the raw score output is shown as-is.

    print(f"\n\n{'#' * 62}")
    print("  SECTION 1 — PROFILE RECOMMENDATIONS")
    print(f"{'#' * 62}")

    for profile_name, user_prefs in PROFILES.items():
        print(f"\n{'=' * 62}")
        print(f"  PROFILE: {profile_name}")
        print(f"{'=' * 62}")

        # Retrieval: score every song and return top-5
        recommendations = strategy.rank(user_prefs, songs, k=5)

        narrate = ai_enabled and profile_name in AI_NARRATED_PROFILES

        if narrate:
            # Generation: Claude synthesizes the scored results into a narrative.
            # The AI actively uses the match signals as evidence — it does not
            # invent reasons or fall back to generic descriptions.
            print("\n  [AI-narrated — scoring engine retrieved, Claude generated]\n")
            try:
                narrative = generate_recommendation_response(user_prefs, recommendations)
                print(_wrap(narrative, width=62))
            except Exception:
                logger.exception("AI narration failed for profile %r — falling back to scores",
                                 profile_name)
                narrate = False  # fall through to raw output below

        if not narrate:
            # Fallback: raw score output
            for idx, (song, score, explanation) in enumerate(recommendations, start=1):
                print(f"\n  {idx}. {song['title']} by {song['artist']}")
                print(f"     Genre: {song['genre']} / {song['mood']}  |  Score: {score:.4f}")
                print("     Reasons:")
                for reason in explanation.split(" | "):
                    print(f"       - {reason}")
        print()

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — Strategy comparison (numerical)
    # ═══════════════════════════════════════════════════════════════════════════

    print(f"\n\n{'#' * 62}")
    print("  SECTION 2 — STRATEGY COMPARISON")
    print(f"{'#' * 62}")
    print("  Same profile, five different weighting strategies.\n")
    print_strategy_comparison(songs)

    if not ai_enabled:
        print("\n  [Sections 3 & 4 require ANTHROPIC_API_KEY]\n")
        return

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — RAG: natural language queries
    # ═══════════════════════════════════════════════════════════════════════════
    #
    # Pipeline: classify intent → keyword retrieval → Claude generation
    #           → relevance validation (automated guardrail).
    # Claude's answer is the primary output; guardrail scores show that the
    # system validates its own outputs automatically.

    print(f"\n\n{'#' * 62}")
    print("  SECTION 3 — RAG: NATURAL LANGUAGE RECOMMENDATIONS")
    print(f"{'#' * 62}")
    print("  Pipeline: classify → retrieve → generate → validate\n")

    for query in RAG_QUERIES:
        logger.info("RAG query: %r", query)
        print(f"  Query    : {query!r}")

        # Step 1 — classify free-text into structured intent
        intent = classify_query_intent(query)
        print(f"  Intent   : genre={intent.get('genre','?')!r}, "
              f"mood={intent.get('mood','?')!r}, "
              f"energy={intent.get('energy','?')!r}, "
              f"occasion={intent.get('occasion','?')!r}")

        # Steps 2–4 — full RAG pipeline (retrieval + generation + guardrails)
        result  = rag_recommend(query, songs)
        safe    = "✓ safe" if result["safety_check"]["safe"] else "✗ flagged"
        rel_chk = result["relevance_check"] or {}
        rel     = f"{rel_chk.get('score', '?')}/5"
        issues  = rel_chk.get("issues", [])

        print(f"  Safety   : {safe}")
        print(f"  Relevance: {rel}" + (f"  issues: {issues}" if issues else ""))
        print(f"\n  Recommendation:\n")
        print(_wrap(result["answer"], width=62))
        if rel_chk.get("score", 5) < 3:
            print("\n  [GUARDRAIL] Relevance score below threshold — "
                  "results may not fully match query intent.")
        print(f"\n  {'─' * 58}\n")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — Step-by-step playlist planning
    # ═══════════════════════════════════════════════════════════════════════════
    #
    # Claude reasons through the occasion's energy arc, maps moods to phases,
    # and selects specific songs from the catalog — shown step by step.

    print(f"\n{'#' * 62}")
    print("  SECTION 4 — STEP-BY-STEP PLAYLIST PLANNING")
    print(f"{'#' * 62}")
    print(f"  Occasion: {PLANNING_OCCASION!r}\n")

    logger.info("Planning playlist for occasion: %r", PLANNING_OCCASION)
    plan = plan_playlist_for_occasion(PLANNING_OCCASION, songs)
    print(_wrap(plan, width=62))

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 5 — Agentic workflow: multi-step reasoning with observable steps
    # ═══════════════════════════════════════════════════════════════════════════
    #
    # Claude is given two catalog tools (search_catalog, rank_songs) and decides
    # autonomously which to call, in what order, and with what arguments.
    # Every tool call and its result are printed as observable intermediate steps
    # so the reasoning chain is visible — not just the final answer.

    print(f"\n\n{'#' * 62}")
    print("  SECTION 5 — AGENTIC WORKFLOW")
    print(f"{'#' * 62}")
    print("  Claude decides which catalog tools to call and in what order.")
    print("  Intermediate steps are shown so the reasoning chain is visible.\n")

    for query in AGENT_QUERIES:
        logger.info("Agentic query: %r", query)
        print(f"  Query    : {query!r}\n")

        result = agentic_recommend(query, songs)

        # Print each intermediate step
        for step in result["steps"]:
            tool    = step["tool"]
            inputs  = step["inputs"]
            summary = step["result_summary"]

            if tool == "search_catalog":
                field = inputs.get("field", "?")
                value = inputs.get("value", "?")
                print(f"  Step {step['step']} [{tool}]  field={field!r}  value={value!r}")
            else:
                ids   = inputs.get("song_ids", [])
                desc  = inputs.get("description", "?")
                print(f"  Step {step['step']} [{tool}]  ids={ids}  desc={desc!r}")

            print(f"         → {summary}")

            # Show the song titles returned by each step
            songs_out = step["result"].get("songs") or step["result"].get("ranked") or []
            if songs_out:
                titles = ", ".join(s["title"] for s in songs_out[:5])
                print(f"         → songs: {titles}")
            print()

        print("  Final answer:\n")
        print(_wrap(result["answer"], width=62))

        if result["final_songs"]:
            print("\n  Songs selected by agent:")
            for s in result["final_songs"]:
                print(f"    • {s['title']} by {s['artist']}"
                      f"  [{s['genre']} / {s['mood']}  energy {s['energy']:.2f}]")

        print(f"\n  {'─' * 58}\n")

    logger.info("ToneMatch run complete")
    print()


if __name__ == "__main__":
    main()
