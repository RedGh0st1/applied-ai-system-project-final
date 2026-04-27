"""
ToneMatch AI Features — RAG, summarization, planning, explain, and guardrails.

Feature map
-----------
1. Summarize        summarize_profile(), summarize_recommendations()
2. Retrieve / RAG   retrieve_songs_for_query(), rag_recommend()
3. Plan             plan_playlist_for_occasion()
4. Explain/Classify explain_song_score(), classify_query_intent()
5. Guardrails       validate_query_safety(), validate_recommendation_relevance()

All Claude calls use prompt caching on the static system prompt to reduce
latency and cost on repeated requests.

Requires: ANTHROPIC_API_KEY environment variable.
"""

import json
import re
from typing import Dict, List, Tuple

import anthropic

# ── Client & model selection ──────────────────────────────────────────────────

_client = anthropic.Anthropic()

MODEL_MAIN = "claude-sonnet-4-6"        # summarization, RAG generation, planning
MODEL_FAST = "claude-haiku-4-5-20251001"  # guardrails, classify, explain (cheap + fast)


# ── Keyword maps for retrieval ────────────────────────────────────────────────

_MOOD_KEYWORDS: Dict[str, List[str]] = {
    "happy":      ["happy", "upbeat", "cheerful", "joyful", "positive", "fun", "bright"],
    "sad":        ["sad", "melancholy", "melancholic", "heartbroken", "gloomy", "weepy"],
    "energetic":  ["energetic", "hype", "pumped", "workout", "gym", "running", "sprint"],
    "chill":      ["chill", "relaxed", "calm", "mellow", "background", "easy"],
    "focused":    ["focused", "concentrate", "study", "work", "productive", "deep work"],
    "romantic":   ["romantic", "love", "date", "evening", "intimate", "sensual"],
    "aggressive": ["aggressive", "intense", "angry", "hard", "heavy", "brutal"],
    "dark":       ["dark", "brooding", "moody", "noir", "gothic"],
    "euphoric":   ["euphoric", "ecstatic", "rave", "dance floor", "party", "festival"],
    "nostalgic":  ["nostalgic", "throwback", "vintage", "retro", "classic", "old school"],
    "melancholic":["melancholic", "bittersweet", "lonesome", "wistful", "pensive"],
    "confident":  ["confident", "swagger", "bold", "power", "boss"],
    "passionate": ["passionate", "fiery", "intense feel", "soulful"],
}

_GENRE_KEYWORDS: Dict[str, List[str]] = {
    "pop":       ["pop", "mainstream", "chart", "radio"],
    "rock":      ["rock", "guitar", "band", "indie rock"],
    "lofi":      ["lofi", "lo-fi", "lofi hip hop", "study beats", "chill beats"],
    "hip-hop":   ["hip-hop", "rap", "hip hop", "urban", "trap", "boom bap"],
    "jazz":      ["jazz", "saxophone", "swing", "bebop", "jazz club"],
    "classical": ["classical", "orchestra", "piano", "symphony", "baroque"],
    "edm":       ["edm", "electronic", "dance", "club", "techno", "house", "trance"],
    "r&b":       ["r&b", "soul", "rhythm and blues", "groove", "neo soul"],
    "metal":     ["metal", "heavy", "thrash", "screaming", "hardcore"],
    "ambient":   ["ambient", "atmospheric", "meditation", "spa", "drone"],
    "latin":     ["latin", "salsa", "reggaeton", "cumbia", "bossa nova"],
    "soul":      ["soul", "motown", "funk", "gospel", "soulful"],
    "country":   ["country", "folk", "bluegrass", "americana"],
}

_ENERGY_KEYWORDS: Dict[str, List[str]] = {
    "high": ["hype", "workout", "gym", "run", "dance", "party", "intense",
             "energetic", "pump", "fast", "upbeat", "rave", "sprint", "power"],
    "low":  ["sleep", "relax", "calm", "chill", "study", "focus", "ambient",
             "background", "quiet", "gentle", "slow", "peaceful", "meditate"],
}


# ── Retrieval helpers ─────────────────────────────────────────────────────────

def _word_in(keyword: str, text: str) -> bool:
    """True if *keyword* appears as a whole word in *text* (word-boundary match)."""
    return bool(re.search(r"\b" + re.escape(keyword) + r"\b", text))


def _kw_hits(q: str, keyword_map: Dict[str, List[str]], song_val: str) -> float:
    """Return 2.0 if any keyword for *song_val* appears as a whole word in *q*."""
    return 2.0 if any(_word_in(kw, q) for kw in keyword_map.get(song_val, [])) else 0.0


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_songs_for_query(query: str, songs: List[Dict], k: int = 8) -> List[Dict]:
    """
    Keyword-based retrieval: score every song against a natural language query
    and return the top-k most relevant ones.

    This is the R in RAG — no embeddings required, runs fully offline.
    Query flags are pre-computed once outside the song loop for efficiency.
    """
    q = query.lower()

    wants_high     = any(_word_in(kw, q) for kw in _ENERGY_KEYWORDS["high"])
    wants_low      = any(_word_in(kw, q) for kw in _ENERGY_KEYWORDS["low"])
    wants_acoustic = any(_word_in(kw, q) for kw in ["acoustic", "unplugged", "raw"])
    wants_inst     = any(_word_in(kw, q) for kw in
                         ["instrumental", "no lyrics", "no vocals", "wordless"])

    scored: List[Tuple[Dict, float]] = []
    for song in songs:
        score = (_kw_hits(q, _MOOD_KEYWORDS,  song.get("mood",  ""))
                 + _kw_hits(q, _GENRE_KEYWORDS, song.get("genre", "")))

        if wants_high:
            score += float(song.get("energy", 0.5)) * 1.5
        if wants_low:
            score += (1.0 - float(song.get("energy", 0.5))) * 1.5
        if wants_acoustic:
            score += float(song.get("acousticness", 0.0))
        if wants_inst:
            score += float(song.get("instrumentalness", 0.0)) * 1.5

        if song.get("title", "").lower() in q:
            score += 5.0
        if song.get("artist", "").lower() in q:
            score += 3.0

        for tag in song.get("detailed_mood_tags", "").split("|"):
            if tag and _word_in(tag, q):
                score += 0.5

        scored.append((song, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in scored[:k]]


def _songs_to_context(songs: List[Dict]) -> str:
    """Render songs as a numbered text block for use in prompts."""
    lines = []
    for i, s in enumerate(songs, 1):
        lines.append(
            f'{i}. "{s["title"]}" by {s["artist"]} '
            f'[genre={s.get("genre","?")}, mood={s.get("mood","?")}, '
            f'energy={float(s.get("energy",0.5)):.2f}, '
            f'valence={float(s.get("valence",0.5)):.2f}, '
            f'tags={s.get("detailed_mood_tags","none")}]'
        )
    return "\n".join(lines)


# ── Guardrails ────────────────────────────────────────────────────────────────

def validate_query_safety(query: str) -> Dict:
    """
    Input guardrail: confirm the query is appropriate for a music recommender.
    Returns {"safe": bool, "reason": str}.
    Uses the fast model — cheap and low-latency.
    """
    resp = _client.messages.create(
        model=MODEL_FAST,
        max_tokens=120,
        messages=[{
            "role": "user",
            "content": (
                "Is this query appropriate for a music recommendation service? "
                'Reply with JSON only — {"safe": true/false, "reason": "one sentence"}.\n'
                f"Query: {query!r}"
            ),
        }],
    )
    return _parse_json(
        resp.content[0].text,
        default={"safe": True, "reason": "parse error; defaulting safe"},
    )


def validate_recommendation_relevance(query: str, recommendations: List[Dict]) -> Dict:
    """
    Output guardrail: verify the retrieved songs are relevant to the query.
    Returns {"valid": bool, "score": int (1–5), "issues": list[str]}.
    Uses the fast model to keep cost low even when called after every RAG request.
    """
    songs_text = _songs_to_context(recommendations)
    resp = _client.messages.create(
        model=MODEL_FAST,
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": (
                f"User query: {query!r}\n\n"
                f"Recommended songs:\n{songs_text}\n\n"
                "Rate relevance 1 (irrelevant) to 5 (perfect). "
                'Reply with JSON only — {"valid": true/false, "score": 1-5, "issues": ["..."]}.'
            ),
        }],
    )
    result = _parse_json(resp.content[0].text, default={"score": 3, "issues": []})
    result["valid"] = result.get("score", 3) >= 3
    return result


def _parse_json(text: str, default: Dict) -> Dict:
    """Extract and parse the first JSON object from a Claude response string."""
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])
    except (json.JSONDecodeError, ValueError):
        return default


# ── RAG: full pipeline ────────────────────────────────────────────────────────

_RAG_SYSTEM = (
    "You are ToneMatch AI, a music recommendation assistant powered by a curated catalog. "
    "When given a list of retrieved songs and a user request, recommend the best matches. "
    "Reference exact song titles and artists. Explain in 2-3 sentences why each song fits. "
    "If none of the retrieved songs are a good fit, say so honestly."
)


def rag_recommend(query: str, songs: List[Dict]) -> Dict:
    """
    Full RAG pipeline — retrieve, generate, and validate.

    Steps
    -----
    1. Input guardrail  — reject unsafe / off-topic queries before any LLM call.
    2. Retrieval        — keyword-score every song, return top 8 as context.
    3. Generation       — Claude generates a natural-language recommendation
                          grounded in the retrieved songs.
    4. Output guardrail — automated relevance check; flags low-quality outputs.

    Returns
    -------
    {
        "answer":          str,
        "retrieved_songs": list[dict],
        "safety_check":    {"safe": bool, "reason": str},
        "relevance_check": {"valid": bool, "score": int, "issues": list[str]},
    }
    """
    # 1. Input guardrail
    safety = validate_query_safety(query)
    if not safety.get("safe", True):
        return {
            "answer": f"Cannot process request: {safety.get('reason', 'query flagged as unsafe')}",
            "retrieved_songs": [],
            "safety_check": safety,
            "relevance_check": None,
        }

    # 2. Retrieval
    retrieved = retrieve_songs_for_query(query, songs, k=8)
    catalog_ctx = _songs_to_context(retrieved)

    # 3. Generation with prompt-cached system prompt
    resp = _client.messages.create(
        model=MODEL_MAIN,
        max_tokens=600,
        system=[{
            "type": "text",
            "text": _RAG_SYSTEM,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{
            "role": "user",
            "content": (
                f"Retrieved catalog entries:\n{catalog_ctx}\n\n"
                f"User request: {query}"
            ),
        }],
    )
    answer = resp.content[0].text

    # 4. Output guardrail
    relevance = validate_recommendation_relevance(query, retrieved[:5])

    return {
        "answer": answer,
        "retrieved_songs": retrieved,
        "safety_check": safety,
        "relevance_check": relevance,
    }


# ── Summarization ─────────────────────────────────────────────────────────────

_PROFILE_SUMMARIZER_SYSTEM = (
    "You are a music taste analyst. Given a user preference profile with numerical "
    "and categorical attributes, write 2-3 vivid, specific sentences describing what "
    "kind of music listener this person is. Avoid repeating raw numbers — translate "
    "them into human language (e.g. 'energy: 0.9' → 'craves high-octane tracks')."
)

_RECS_SUMMARIZER_SYSTEM = (
    "You are ToneMatch AI. Given a user's profile summary and their top song picks, "
    "write 2-3 sentences explaining the common thread between the recommendations "
    "and why they match this listener's taste. Be specific about the sonic qualities."
)


def summarize_profile(user_prefs: Dict) -> str:
    """
    Summarize a user preference dict in natural language using Claude.
    The static system prompt is cached to reduce cost on repeated calls.
    """
    resp = _client.messages.create(
        model=MODEL_MAIN,
        max_tokens=250,
        system=[{
            "type": "text",
            "text": _PROFILE_SUMMARIZER_SYSTEM,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{
            "role": "user",
            "content": f"User profile:\n{json.dumps(user_prefs, indent=2)}",
        }],
    )
    return resp.content[0].text


def summarize_recommendations(
    user_prefs: Dict,
    recommendations: List[Tuple],
) -> str:
    """
    Summarize a recommendation result list in natural language.
    recommendations: list of (song_dict, score, reasons_str) tuples.
    """
    profile_line = (
        f"Genre: {user_prefs.get('favorite_genre', '?')}, "
        f"Mood: {user_prefs.get('favorite_mood', '?')}, "
        f"Energy: {user_prefs.get('target_energy', '?')}"
    )
    picks = "\n".join(
        f'{i}. "{r[0]["title"]}" by {r[0]["artist"]} (score {r[1]:.2f})'
        for i, r in enumerate(recommendations, 1)
    )
    resp = _client.messages.create(
        model=MODEL_MAIN,
        max_tokens=300,
        system=[{
            "type": "text",
            "text": _RECS_SUMMARIZER_SYSTEM,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{
            "role": "user",
            "content": f"Profile: {profile_line}\n\nTop picks:\n{picks}",
        }],
    )
    return resp.content[0].text


# ── Step-by-step planning ─────────────────────────────────────────────────────

_PLANNER_SYSTEM = (
    "You are a professional music curator. When asked to plan a playlist for an occasion, "
    "think step by step:\n"
    "  Step 1 — Identify the energy arc the occasion needs (build-up, plateau, wind-down, etc.)\n"
    "  Step 2 — Map moods to each phase of the occasion.\n"
    "  Step 3 — Select specific songs from the provided catalog for each phase.\n"
    "Show your reasoning at each step. End with a numbered final playlist of 5 songs."
)


def plan_playlist_for_occasion(occasion: str, songs: List[Dict]) -> str:
    """
    Use Claude to plan a 5-song playlist for a given occasion, step by step.
    Songs are sampled from the catalog to keep the prompt size manageable.
    """
    sample = songs[:40]  # first 40 songs as the planning context
    catalog_ctx = _songs_to_context(sample)

    resp = _client.messages.create(
        model=MODEL_MAIN,
        max_tokens=900,
        system=[{
            "type": "text",
            "text": _PLANNER_SYSTEM,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{
            "role": "user",
            "content": (
                f"Occasion: {occasion}\n\n"
                f"Available songs:\n{catalog_ctx}\n\n"
                "Plan a 5-song playlist step by step."
            ),
        }],
    )
    return resp.content[0].text


# ── Explain / Debug / Classify ────────────────────────────────────────────────

def explain_song_score(
    song: Dict,
    user_prefs: Dict,
    score: float,
    reasons: str,
) -> str:
    """
    Explain in plain English why a song received its numerical score.
    Uses the fast model — this is called per-song so speed matters.
    """
    resp = _client.messages.create(
        model=MODEL_FAST,
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": (
                f'Explain in 2 plain English sentences why "{song["title"]}" '
                f"by {song['artist']} scored {score:.2f} for this listener.\n\n"
                f"Song: genre={song.get('genre','?')}, mood={song.get('mood','?')}, "
                f"energy={float(song.get('energy',0.5)):.2f}\n"
                f"Listener: genre={user_prefs.get('favorite_genre','?')}, "
                f"mood={user_prefs.get('favorite_mood','?')}, "
                f"energy={user_prefs.get('target_energy','?')}\n"
                f"Signals fired: {reasons[:300]}"
            ),
        }],
    )
    return resp.content[0].text


def classify_query_intent(query: str) -> Dict:
    """
    Classify a free-text music query into structured intent.
    Returns {"genre": str, "mood": str, "energy": "high"|"medium"|"low", "occasion": str}.
    Useful for converting natural language input into profile-compatible fields.
    """
    resp = _client.messages.create(
        model=MODEL_FAST,
        max_tokens=150,
        messages=[{
            "role": "user",
            "content": (
                "Classify this music query into structured intent. "
                'Reply with JSON only: {"genre": "...", "mood": "...", '
                '"energy": "high|medium|low", "occasion": "..."}.\n'
                f"Query: {query!r}"
            ),
        }],
    )
    return _parse_json(
        resp.content[0].text,
        default={"genre": "", "mood": "", "energy": "medium", "occasion": ""},
    )
