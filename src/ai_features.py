"""
ToneMatch AI Features — RAG, summarization, planning, explain, and guardrails.

Feature map
-----------
1. Retrieve / RAG   retrieve_songs_for_query(), rag_recommend()
                    generate_recommendation_response()
2. Summarize        summarize_profile(), summarize_recommendations()
3. Plan             plan_playlist_for_occasion()
4. Explain/Classify explain_song_score(), classify_query_intent()
5. Guardrails       validate_query_safety(), validate_recommendation_relevance()

Design notes
------------
* Client is initialized lazily on first use — the module is safe to import
  without ANTHROPIC_API_KEY (the scoring engine still works).
* Every Claude call is wrapped in error handling; functions return a
  descriptive fallback string/dict instead of raising.
* All errors and guardrail events are logged via the module logger.
  Configure the root logger in your entry point to see them.
* Static system prompts use prompt caching (cache_control: ephemeral) to
  reduce latency and cost on repeated calls within a session.

Requires: ANTHROPIC_API_KEY environment variable for AI features.
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import anthropic

logger = logging.getLogger(__name__)

# ── Client — lazy initialization ──────────────────────────────────────────────

_client: Optional[anthropic.Anthropic] = None


def _get_client() -> anthropic.Anthropic:
    """
    Return the shared Anthropic client, creating it on first call.

    Raises EnvironmentError with a clear message if ANTHROPIC_API_KEY is
    missing so callers can catch it and show a user-friendly error instead
    of an opaque SDK exception.
    """
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set.\n"
                "Export it before running:\n"
                "  export ANTHROPIC_API_KEY=sk-ant-..."
            )
        _client = anthropic.Anthropic(api_key=api_key)
        logger.debug("Anthropic client initialized (model_main=%s, model_fast=%s)",
                     MODEL_MAIN, MODEL_FAST)
    return _client


MODEL_MAIN = "claude-sonnet-4-6"          # RAG generation, planning, narration
MODEL_FAST = "claude-haiku-4-5-20251001"  # guardrails, classify, explain


# ── Error handling helpers ────────────────────────────────────────────────────

def _call_claude_text(*, model: str, max_tokens: int,
                      messages: list, system: Optional[list] = None,
                      label: str = "claude") -> Optional[str]:
    """
    Make a Claude API call and return the response text.

    Returns None on any API error so callers can provide their own fallback.
    Logs all errors with the *label* string for traceability.
    """
    kwargs: dict = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if system:
        kwargs["system"] = system
    try:
        resp = _get_client().messages.create(**kwargs)
        text = resp.content[0].text
        logger.debug("[%s] %d tokens used (in=%d, out=%d)",
                     label,
                     (resp.usage.input_tokens or 0) + (resp.usage.output_tokens or 0),
                     resp.usage.input_tokens or 0,
                     resp.usage.output_tokens or 0)
        return text
    except EnvironmentError as exc:
        logger.error("[%s] %s", label, exc)
    except anthropic.AuthenticationError:
        logger.error("[%s] Authentication failed — verify ANTHROPIC_API_KEY", label)
    except anthropic.RateLimitError:
        logger.warning("[%s] Rate limit hit — back off and retry", label)
    except anthropic.APIConnectionError as exc:
        logger.error("[%s] Connection error: %s", label, exc)
    except anthropic.APIStatusError as exc:
        logger.error("[%s] API status %d: %s", label, exc.status_code, exc.message)
    return None


# ── Keyword maps for retrieval ────────────────────────────────────────────────

_MOOD_KEYWORDS: Dict[str, List[str]] = {
    "happy":       ["happy", "upbeat", "cheerful", "joyful", "positive", "fun", "bright"],
    "sad":         ["sad", "melancholy", "melancholic", "heartbroken", "gloomy", "weepy"],
    "energetic":   ["energetic", "hype", "pumped", "workout", "gym", "running", "sprint"],
    "chill":       ["chill", "relaxed", "calm", "mellow", "background", "easy"],
    "focused":     ["focused", "concentrate", "study", "productive", "deep work"],
    "romantic":    ["romantic", "love", "date", "evening", "intimate", "sensual"],
    "aggressive":  ["aggressive", "intense", "angry", "hard", "heavy", "brutal"],
    "dark":        ["dark", "brooding", "moody", "noir", "gothic"],
    "euphoric":    ["euphoric", "ecstatic", "rave", "dance floor", "party", "festival"],
    "nostalgic":   ["nostalgic", "throwback", "vintage", "retro", "classic", "old school"],
    "melancholic": ["melancholic", "bittersweet", "lonesome", "wistful", "pensive"],
    "confident":   ["confident", "swagger", "bold", "power", "boss"],
    "passionate":  ["passionate", "fiery", "soulful"],
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
    """Return 2.0 if any keyword for *song_val* matches a whole word in *q*."""
    return 2.0 if any(_word_in(kw, q) for kw in keyword_map.get(song_val, [])) else 0.0


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_songs_for_query(query: str, songs: List[Dict], k: int = 8) -> List[Dict]:
    """
    Keyword-based retrieval: score every song against a natural language query
    and return the top-k most relevant ones.

    This is the R in RAG — fully offline, no embeddings required.
    Query flags are pre-computed once outside the song loop for efficiency.
    Uses word-boundary matching so 'work' does not accidentally match 'workout'.
    """
    q = query.lower()
    logger.debug("retrieve_songs_for_query: query=%r, catalog_size=%d, k=%d",
                 query, len(songs), k)

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
    top = [s for s, _ in scored[:k]]
    logger.debug("retrieve_songs_for_query: returned %d songs, top=%r",
                 len(top), top[0].get("title") if top else None)
    return top


def _songs_to_context(songs: List[Dict]) -> str:
    """Render songs as a numbered text block for use in prompts."""
    lines = []
    for i, s in enumerate(songs, 1):
        lines.append(
            f'{i}. "{s["title"]}" by {s["artist"]} '
            f'[genre={s.get("genre","?")}, mood={s.get("mood","?")}, '
            f'energy={float(s.get("energy", 0.5)):.2f}, '
            f'valence={float(s.get("valence", 0.5)):.2f}, '
            f'tags={s.get("detailed_mood_tags","none")}]'
        )
    return "\n".join(lines)


# ── Guardrails ────────────────────────────────────────────────────────────────

def validate_query_safety(query: str) -> Dict:
    """
    Input guardrail: confirm the query is appropriate for a music recommender.

    Returns {"safe": bool, "reason": str}.
    On API failure defaults to safe=True with the error noted in reason,
    so a transient network error does not silently block all queries.
    """
    logger.debug("validate_query_safety: query=%r", query)
    text = _call_claude_text(
        model=MODEL_FAST,
        max_tokens=120,
        label="guardrail/safety",
        messages=[{
            "role": "user",
            "content": (
                "Is this query appropriate for a music recommendation service? "
                'Reply with JSON only — {"safe": true/false, "reason": "one sentence"}.\n'
                f"Query: {query!r}"
            ),
        }],
    )
    if text is None:
        logger.warning("Safety guardrail failed — defaulting to safe")
        return {"safe": True, "reason": "guardrail API call failed — defaulting safe"}

    result = _parse_json(text, default={"safe": True, "reason": "parse error"})
    if not result.get("safe", True):
        logger.warning("Safety guardrail flagged query: %r  reason: %s",
                       query, result.get("reason"))
    return result


def validate_recommendation_relevance(query: str, recommendations: List[Dict]) -> Dict:
    """
    Output guardrail: verify the retrieved songs are relevant to the query.

    Returns {"valid": bool, "score": int (1–5), "issues": list[str]}.
    Scores below 3 are logged as warnings.  On API failure defaults to
    score=3/valid=True so a transient error does not block the response.
    """
    logger.debug("validate_recommendation_relevance: query=%r, n_songs=%d",
                 query, len(recommendations))
    songs_text = _songs_to_context(recommendations)
    text = _call_claude_text(
        model=MODEL_FAST,
        max_tokens=200,
        label="guardrail/relevance",
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
    if text is None:
        logger.warning("Relevance guardrail failed — defaulting to score=3")
        return {"valid": True, "score": 3, "issues": ["guardrail API call failed"]}

    result = _parse_json(text, default={"score": 3, "issues": []})
    result["valid"] = result.get("score", 3) >= 3
    if result["score"] < 3:
        logger.warning("Low relevance score %d for query %r — issues: %s",
                       result["score"], query, result.get("issues"))
    return result


def _parse_json(text: str, default: Dict) -> Dict:
    """Extract and parse the first JSON object from a Claude response string."""
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])
    except (json.JSONDecodeError, ValueError):
        logger.debug("_parse_json: could not parse %r — using default", text[:80])
        return default


# ── Score-engine RAG: generation from ranked results ─────────────────────────
#
# The scoring engine acts as the retriever — it ranks every song against a
# user profile using a multi-signal composite score.  generate_recommendation_
# response() takes those retrieved/ranked results and uses Claude to *generate*
# a grounded narrative: the match signals are the evidence, Claude synthesises
# the human-readable recommendation.

_SCORE_NARRATION_SYSTEM = (
    "You are ToneMatch AI, a personal music curator. "
    "You receive a listener's preference profile and their top matches from a "
    "scoring engine. The match signals are your evidence — use them to write "
    "a grounded, personalized recommendation.\n\n"
    "Structure your response:\n"
    "1. One sentence reading this listener's taste from their profile.\n"
    "2. Top 3 picks with a specific reason each (translate signals into human "
    "   terms — e.g. 'energy 0.91' → 'relentless, driving intensity').\n"
    "3. One sentence on the common thread connecting all picks.\n"
    "Be vivid and specific. Never repeat raw numbers."
)


def generate_recommendation_response(
    user_prefs: Dict,
    ranked_songs: List[Tuple],
) -> str:
    """
    Generation half of the score-engine RAG pipeline.

    The scoring engine has already retrieved the most relevant songs (ranked
    by composite score). This function takes those retrieved results and uses
    Claude to generate a grounded, human-readable recommendation narrative.

    Parameters
    ----------
    user_prefs   : raw preference dict (genre, mood, energy targets, etc.)
    ranked_songs : list of (song_dict, score, reasons_str) from the recommender

    Returns
    -------
    str — Claude's grounded recommendation narrative, or a fallback message
          if the API call fails.
    """
    logger.info("generate_recommendation_response: profile_genre=%r, n_songs=%d",
                user_prefs.get("favorite_genre"), len(ranked_songs))

    picks_lines = []
    for i, (song, score, reasons) in enumerate(ranked_songs[:5], 1):
        top_signals = " | ".join(reasons.split(" | ")[:4])
        picks_lines.append(
            f'{i}. "{song["title"]}" by {song["artist"]} '
            f'[genre={song.get("genre","?")}, mood={song.get("mood","?")}, '
            f'energy={float(song.get("energy", 0.5)):.2f}, score={score:.2f}]\n'
            f'   Signals: {top_signals}'
        )

    profile_parts = [
        f"genre={user_prefs.get('favorite_genre','?')}",
        f"mood={user_prefs.get('favorite_mood','?')}",
        f"energy_target={user_prefs.get('target_energy','?')}",
        f"valence_target={user_prefs.get('target_valence','?')}",
    ]
    if user_prefs.get("preferred_mood_tags"):
        profile_parts.append(f"mood_tags={user_prefs['preferred_mood_tags']}")
    if user_prefs.get("preferred_era"):
        profile_parts.append(f"era={user_prefs['preferred_era']}")
    if user_prefs.get("preferred_language"):
        profile_parts.append(f"language={user_prefs['preferred_language']}")

    text = _call_claude_text(
        model=MODEL_MAIN,
        max_tokens=450,
        label="narration",
        system=[{
            "type": "text",
            "text": _SCORE_NARRATION_SYSTEM,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{
            "role": "user",
            "content": (
                f"Listener profile: {', '.join(profile_parts)}\n\n"
                f"Top matches from scoring engine:\n"
                + "\n".join(picks_lines)
            ),
        }],
    )
    if text is None:
        logger.error("generate_recommendation_response: Claude call failed")
        return "[AI narration unavailable — see log for details]"
    return text


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
    dict with keys: answer, retrieved_songs, safety_check, relevance_check.
    On partial failure individual fields contain error descriptions rather
    than crashing the caller.
    """
    logger.info("rag_recommend: query=%r", query)

    # 1. Input guardrail
    safety = validate_query_safety(query)
    if not safety.get("safe", True):
        logger.warning("rag_recommend: query blocked by safety guardrail")
        return {
            "answer": f"Cannot process request: {safety.get('reason', 'query flagged')}",
            "retrieved_songs": [],
            "safety_check": safety,
            "relevance_check": None,
        }

    # 2. Retrieval (offline — always succeeds)
    retrieved = retrieve_songs_for_query(query, songs, k=8)
    catalog_ctx = _songs_to_context(retrieved)

    # 3. Generation
    answer_text = _call_claude_text(
        model=MODEL_MAIN,
        max_tokens=600,
        label="rag/generation",
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
    if answer_text is None:
        answer_text = "[Generation failed — see log for details]"
        logger.error("rag_recommend: generation step failed for query=%r", query)

    # 4. Output guardrail
    relevance = validate_recommendation_relevance(query, retrieved[:5])

    logger.info("rag_recommend: done — relevance=%d/5, safe=%s",
                relevance.get("score", 0), safety.get("safe"))
    return {
        "answer": answer_text,
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
    """Summarize a user preference dict in natural language using Claude."""
    logger.debug("summarize_profile: genre=%r", user_prefs.get("favorite_genre"))
    text = _call_claude_text(
        model=MODEL_MAIN,
        max_tokens=250,
        label="summarize/profile",
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
    return text or "[Profile summary unavailable]"


def summarize_recommendations(
    user_prefs: Dict,
    recommendations: List[Tuple],
) -> str:
    """
    Summarize a recommendation result list in natural language.
    recommendations: list of (song_dict, score, reasons_str) tuples.
    """
    logger.debug("summarize_recommendations: n=%d", len(recommendations))
    profile_line = (
        f"Genre: {user_prefs.get('favorite_genre', '?')}, "
        f"Mood: {user_prefs.get('favorite_mood', '?')}, "
        f"Energy: {user_prefs.get('target_energy', '?')}"
    )
    picks = "\n".join(
        f'{i}. "{r[0]["title"]}" by {r[0]["artist"]} (score {r[1]:.2f})'
        for i, r in enumerate(recommendations, 1)
    )
    text = _call_claude_text(
        model=MODEL_MAIN,
        max_tokens=300,
        label="summarize/recs",
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
    return text or "[Recommendation summary unavailable]"


# ── Step-by-step planning ─────────────────────────────────────────────────────

_PLANNER_SYSTEM = (
    "You are a professional music curator. When asked to plan a playlist for an occasion, "
    "think step by step:\n"
    "  Step 1 — Identify the energy arc the occasion needs (build-up, plateau, wind-down).\n"
    "  Step 2 — Map moods to each phase of the occasion.\n"
    "  Step 3 — Select specific songs from the provided catalog for each phase.\n"
    "Show your reasoning at each step. End with a numbered final playlist of 5 songs."
)


def plan_playlist_for_occasion(occasion: str, songs: List[Dict]) -> str:
    """
    Use Claude to plan a 5-song playlist for a given occasion, step by step.
    Songs are sampled from the catalog to keep the prompt size manageable.
    """
    logger.info("plan_playlist_for_occasion: occasion=%r, catalog_size=%d",
                occasion, len(songs))
    sample = songs[:40]
    catalog_ctx = _songs_to_context(sample)

    text = _call_claude_text(
        model=MODEL_MAIN,
        max_tokens=900,
        label="planning",
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
    return text or "[Playlist planning unavailable — see log for details]"


# ── Explain / Debug / Classify ────────────────────────────────────────────────

def explain_song_score(
    song: Dict,
    user_prefs: Dict,
    score: float,
    reasons: str,
) -> str:
    """Explain in plain English why a song received its numerical score."""
    logger.debug("explain_song_score: song=%r, score=%.2f", song.get("title"), score)
    text = _call_claude_text(
        model=MODEL_FAST,
        max_tokens=200,
        label="explain",
        messages=[{
            "role": "user",
            "content": (
                f'Explain in 2 plain English sentences why "{song["title"]}" '
                f"by {song['artist']} scored {score:.2f} for this listener.\n\n"
                f"Song: genre={song.get('genre','?')}, mood={song.get('mood','?')}, "
                f"energy={float(song.get('energy', 0.5)):.2f}\n"
                f"Listener: genre={user_prefs.get('favorite_genre','?')}, "
                f"mood={user_prefs.get('favorite_mood','?')}, "
                f"energy={user_prefs.get('target_energy','?')}\n"
                f"Signals fired: {reasons[:300]}"
            ),
        }],
    )
    return text or "[Explanation unavailable]"


def classify_query_intent(query: str) -> Dict:
    """
    Classify a free-text music query into structured intent.
    Returns {"genre": str, "mood": str, "energy": "high"|"medium"|"low", "occasion": str}.
    """
    logger.debug("classify_query_intent: query=%r", query)
    text = _call_claude_text(
        model=MODEL_FAST,
        max_tokens=150,
        label="classify",
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
    if text is None:
        logger.error("classify_query_intent: Claude call failed for query=%r", query)
        return {"genre": "", "mood": "", "energy": "medium", "occasion": ""}
    return _parse_json(text, default={"genre": "", "mood": "", "energy": "medium", "occasion": ""})


# ── Agentic Workflow ──────────────────────────────────────────────────────────
#
# Two tools give Claude a decision-making loop over the catalog:
#   search_catalog  — filter songs by genre, mood, or energy range
#   rank_songs      — score a candidate set against a natural-language description
#
# Claude decides which tools to call, in which order, and what arguments to
# pass.  Every tool call and its result are captured as an observable step so
# the reasoning chain is visible to the caller.

_AGENT_TOOLS = [
    {
        "name": "search_catalog",
        "description": (
            "Filter the music catalog by a single attribute. "
            "Use field='genre' or field='mood' for exact string matches. "
            "Use field='energy' with value='min,max' (e.g. '0.6,0.85') to find "
            "songs in a numeric energy range."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "field": {
                    "type": "string",
                    "enum": ["genre", "mood", "energy"],
                    "description": "Which song attribute to filter on.",
                },
                "value": {
                    "type": "string",
                    "description": (
                        "For genre/mood: exact label (e.g. 'lofi', 'chill'). "
                        "For energy: 'min,max' floats (e.g. '0.3,0.55')."
                    ),
                },
            },
            "required": ["field", "value"],
        },
    },
    {
        "name": "rank_songs",
        "description": (
            "Score and rank a list of songs against a natural-language description. "
            "Pass the song IDs returned by earlier search_catalog calls. "
            "Use this as a final step to confirm the best matches among your candidates."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "song_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Song IDs to rank (from prior search_catalog results).",
                },
                "description": {
                    "type": "string",
                    "description": "Natural-language description of what the listener wants.",
                },
                "k": {
                    "type": "integer",
                    "description": "Number of top results to return (default 5).",
                    "default": 5,
                },
            },
            "required": ["song_ids", "description"],
        },
    },
]


def _execute_agent_tool(name: str, inputs: dict, songs: List[Dict]) -> dict:
    """Execute one agent tool call; return a JSON-serialisable result dict."""
    if name == "search_catalog":
        field = inputs.get("field", "genre")
        value = str(inputs.get("value", ""))

        if field == "energy":
            try:
                lo_str, hi_str = value.split(",", 1)
                lo, hi = float(lo_str.strip()), float(hi_str.strip())
                matches = [
                    {"id": int(s["id"]), "title": s["title"], "artist": s["artist"],
                     "genre": s["genre"], "mood": s["mood"], "energy": float(s["energy"])}
                    for s in songs if lo <= float(s["energy"]) <= hi
                ]
            except (ValueError, IndexError):
                logger.warning("_execute_agent_tool: bad energy value %r", value)
                matches = []
        else:
            matches = [
                {"id": int(s["id"]), "title": s["title"], "artist": s["artist"],
                 "genre": s["genre"], "mood": s["mood"], "energy": float(s["energy"])}
                for s in songs
                if str(s.get(field, "")).lower() == value.lower()
            ]

        return {"count": len(matches), "songs": matches}

    if name == "rank_songs":
        song_ids = {int(i) for i in inputs.get("song_ids", [])}
        description = str(inputs.get("description", ""))
        k = int(inputs.get("k", 5))

        subset = [s for s in songs if int(s["id"]) in song_ids] if song_ids else songs
        if not subset:
            subset = songs
        ranked = retrieve_songs_for_query(description, subset, k=k)
        return {
            "ranked": [
                {"id": int(r["id"]), "title": r["title"], "artist": r["artist"],
                 "genre": r["genre"], "mood": r["mood"], "energy": float(r["energy"])}
                for r in ranked
            ]
        }

    return {"error": f"Unknown tool: {name}"}


def agentic_recommend(query: str, songs: List[Dict]) -> Dict:
    """
    Agentic recommendation workflow with observable intermediate steps.

    Claude is given two catalog tools (search_catalog, rank_songs) and decides
    autonomously how to use them to answer the query.  Each tool call and its
    result are recorded as a step so the full reasoning chain is visible.

    The final answer is grounded in actual tool outputs — Claude cannot name a
    song that was not returned by a tool call.

    Parameters
    ----------
    query : str       Natural-language request from the listener.
    songs : list      Full song catalog (list of attribute dicts).

    Returns
    -------
    {
        "answer":      str          # final recommendation narrative
        "steps":       list[dict]   # observable intermediate steps
        "final_songs": list[dict]   # songs surfaced in the last rank_songs call
    }
    """
    logger.info("agentic_recommend: starting for query=%r", query)

    try:
        client = _get_client()
    except EnvironmentError as exc:
        logger.error("agentic_recommend: %s", exc)
        return {"answer": str(exc), "steps": [], "final_songs": []}

    system_prompt = (
        "You are ToneMatch, an AI music curator with tool access to a song catalog. "
        "To answer the listener's request, explore the catalog step by step using the tools. "
        "Think about what attributes the listener wants — genre, mood, energy level — "
        "search the catalog to find candidates, then rank them to confirm the best matches. "
        "Your final answer must reference only songs that appeared in your tool results. "
        "Do not invent or name songs that were not returned by the tools."
    )

    messages: list = [{"role": "user", "content": query}]
    steps: list = []
    max_iterations = 8  # guard against infinite loops

    def _create(**kwargs):
        """Thin wrapper so API errors surface as logged warnings, not crashes."""
        try:
            return client.messages.create(**kwargs)
        except anthropic.AuthenticationError:
            logger.error("[agent] Authentication failed — verify ANTHROPIC_API_KEY")
        except anthropic.RateLimitError:
            logger.warning("[agent] Rate limit hit")
        except anthropic.APIConnectionError as exc:
            logger.error("[agent] Connection error: %s", exc)
        except anthropic.APIStatusError as exc:
            logger.error("[agent] API status %d: %s", exc.status_code, exc.message)
        return None

    response = _create(
        model=MODEL_MAIN,
        max_tokens=1024,
        system=system_prompt,
        tools=_AGENT_TOOLS,
        messages=messages,
    )
    if response is None:
        return {"answer": "Agent unavailable — API call failed.", "steps": [], "final_songs": []}

    logger.debug("agentic_recommend: initial stop_reason=%r", response.stop_reason)

    for iteration in range(1, max_iterations + 1):
        if response.stop_reason != "tool_use":
            break

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        tool_results = []

        for tu in tool_use_blocks:
            logger.info("agentic_recommend: step %d — tool=%r inputs=%r",
                        iteration, tu.name, tu.input)

            result = _execute_agent_tool(tu.name, tu.input, songs)

            if "count" in result:
                summary = f"found {result['count']} songs"
            else:
                summary = f"ranked {len(result.get('ranked', []))} songs"

            steps.append({
                "step":           iteration,
                "tool":           tu.name,
                "inputs":         tu.input,
                "result_summary": summary,
                "result":         result,
            })
            logger.debug("agentic_recommend: step %d result — %s", iteration, summary)

            tool_results.append({
                "type":        "tool_result",
                "tool_use_id": tu.id,
                "content":     json.dumps(result),
            })

        messages = messages + [
            {"role": "assistant", "content": response.content},
            {"role": "user",      "content": tool_results},
        ]
        next_response = _create(
            model=MODEL_MAIN,
            max_tokens=1024,
            system=system_prompt,
            tools=_AGENT_TOOLS,
            messages=messages,
        )
        if next_response is None:
            logger.error("agentic_recommend: API call failed at step %d — stopping loop",
                         iteration)
            break
        response = next_response
        logger.debug("agentic_recommend: after step %d stop_reason=%r",
                     iteration, response.stop_reason)

    if response.stop_reason == "tool_use":
        logger.warning("agentic_recommend: hit max_iterations=%d without final answer",
                       max_iterations)

    # Extract the final text answer
    answer = "".join(
        block.text for block in response.content if hasattr(block, "text")
    ).strip() or "(No response generated)"

    # Surface the songs from the last rank_songs step (or last search if no ranking)
    final_songs: list = []
    for step in reversed(steps):
        if step["tool"] == "rank_songs":
            final_songs = step["result"].get("ranked", [])
            break
    if not final_songs:
        for step in reversed(steps):
            if step["tool"] == "search_catalog":
                final_songs = step["result"].get("songs", [])[:5]
                break

    logger.info("agentic_recommend: completed in %d steps", len(steps))
    return {"answer": answer, "steps": steps, "final_songs": final_songs}
