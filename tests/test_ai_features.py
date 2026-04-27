"""
Tests for ToneMatch AI features.

Test categories
---------------
Unit tests   — retrieve_songs_for_query, classify_query_intent, _parse_json
               These run fully offline and require no API key.

Integration  — RAG pipeline, guardrails, summarization, planning, explain.
               These call the Claude API and require ANTHROPIC_API_KEY.
               Marked with @pytest.mark.integration.

Guardrail    — validate_query_safety, validate_recommendation_relevance.
               Subset of integration tests focused on safety / quality gates.
               Marked with @pytest.mark.guardrail (also @pytest.mark.integration).

Run unit tests only:
    pytest tests/test_ai_features.py -m "not integration"

Run all tests (requires ANTHROPIC_API_KEY):
    pytest tests/test_ai_features.py
"""

import os
import pytest
from src.ai_features import (
    retrieve_songs_for_query,
    validate_query_safety,
    validate_recommendation_relevance,
    rag_recommend,
    summarize_profile,
    summarize_recommendations,
    plan_playlist_for_occasion,
    explain_song_score,
    classify_query_intent,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_SONGS = [
    {
        "id": 1, "title": "Sunrise City", "artist": "Neon Echo",
        "genre": "pop", "mood": "happy", "energy": 0.82, "valence": 0.84,
        "instrumentalness": 0.05, "acousticness": 0.18,
        "detailed_mood_tags": "euphoric|bright|summery",
    },
    {
        "id": 2, "title": "Midnight Coding", "artist": "LoRoom",
        "genre": "lofi", "mood": "focused", "energy": 0.42, "valence": 0.56,
        "instrumentalness": 0.85, "acousticness": 0.71,
        "detailed_mood_tags": "calm|studious|sleepy",
    },
    {
        "id": 3, "title": "Storm Runner", "artist": "Voltline",
        "genre": "rock", "mood": "intense", "energy": 0.91, "valence": 0.48,
        "instrumentalness": 0.10, "acousticness": 0.10,
        "detailed_mood_tags": "powerful|driving|cinematic",
    },
    {
        "id": 4, "title": "Library Rain", "artist": "Paper Lanterns",
        "genre": "lofi", "mood": "chill", "energy": 0.35, "valence": 0.60,
        "instrumentalness": 0.90, "acousticness": 0.86,
        "detailed_mood_tags": "cozy|introspective|gentle",
    },
    {
        "id": 5, "title": "Saturday Groove", "artist": "The Weekenders",
        "genre": "pop", "mood": "happy", "energy": 0.78, "valence": 0.88,
        "instrumentalness": 0.03, "acousticness": 0.22,
        "detailed_mood_tags": "bright|danceable|fun",
    },
]

SAMPLE_USER = {
    "favorite_genre": "pop",
    "favorite_mood": "happy",
    "target_energy": 0.80,
    "target_valence": 0.82,
    "target_inst": 0.05,
}

requires_api = pytest.mark.integration


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — no API key required
# ═══════════════════════════════════════════════════════════════════════════════

class TestRetrieval:
    """Keyword-based retrieval — fully offline."""

    def test_returns_at_most_k_songs(self):
        results = retrieve_songs_for_query("happy pop music", SAMPLE_SONGS, k=2)
        assert len(results) <= 2

    def test_pop_query_ranks_pop_first(self):
        results = retrieve_songs_for_query("upbeat pop radio song", SAMPLE_SONGS, k=5)
        assert results[0]["genre"] == "pop"

    def test_study_query_ranks_lofi_first(self):
        results = retrieve_songs_for_query("study music lofi calm", SAMPLE_SONGS, k=5)
        assert results[0]["genre"] == "lofi"

    def test_workout_query_ranks_high_energy_first(self):
        results = retrieve_songs_for_query("gym workout pump run intense", SAMPLE_SONGS, k=3)
        assert float(results[0]["energy"]) >= 0.75

    def test_relaxed_query_ranks_low_energy_first(self):
        results = retrieve_songs_for_query("relax sleep ambient gentle calm", SAMPLE_SONGS, k=5)
        assert float(results[0]["energy"]) <= 0.55

    def test_returns_all_songs_when_k_exceeds_catalog(self):
        results = retrieve_songs_for_query("music", SAMPLE_SONGS, k=100)
        assert len(results) == len(SAMPLE_SONGS)

    def test_instrumental_query_boosts_high_instrumentalness(self):
        results = retrieve_songs_for_query("instrumental no vocals no lyrics", SAMPLE_SONGS, k=3)
        assert float(results[0]["instrumentalness"]) > 0.5

    def test_title_mention_scores_highest(self):
        results = retrieve_songs_for_query("play Midnight Coding for me", SAMPLE_SONGS, k=5)
        assert results[0]["title"] == "Midnight Coding"


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS — require ANTHROPIC_API_KEY
# ═══════════════════════════════════════════════════════════════════════════════

class TestGuardrails:
    """Input and output guardrails via Claude."""

    @requires_api
    def test_safety_check_music_query_is_safe(self):
        result = validate_query_safety("recommend chill lofi songs for studying")
        assert "safe" in result
        assert result["safe"] is True

    @requires_api
    def test_safety_check_returns_reason(self):
        result = validate_query_safety("upbeat pop playlist for a party")
        assert "reason" in result
        assert isinstance(result["reason"], str)

    @requires_api
    def test_relevance_check_returns_expected_shape(self):
        result = validate_recommendation_relevance("happy pop songs", SAMPLE_SONGS[:3])
        assert "valid" in result
        assert "score" in result
        assert "issues" in result

    @requires_api
    def test_relevance_score_in_valid_range(self):
        result = validate_recommendation_relevance("upbeat pop party music", SAMPLE_SONGS[:3])
        assert 1 <= result["score"] <= 5

    @requires_api
    def test_relevant_songs_score_at_least_3(self):
        # Query clearly matches the catalog: "pop happy music" → pop/happy songs
        result = validate_recommendation_relevance(
            "happy upbeat pop music",
            [s for s in SAMPLE_SONGS if s["genre"] == "pop"],
        )
        assert result["score"] >= 3
        assert result["valid"] is True


class TestRAGPipeline:
    """Full RAG: retrieve → generate → validate."""

    @requires_api
    def test_returns_all_expected_keys(self):
        result = rag_recommend("I want chill study music", SAMPLE_SONGS)
        assert "answer" in result
        assert "retrieved_songs" in result
        assert "safety_check" in result
        assert "relevance_check" in result

    @requires_api
    def test_answer_is_non_empty_string(self):
        result = rag_recommend("energetic workout rock songs", SAMPLE_SONGS)
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 30

    @requires_api
    def test_retrieves_songs_before_generation(self):
        result = rag_recommend("something happy and upbeat", SAMPLE_SONGS)
        assert len(result["retrieved_songs"]) > 0

    @requires_api
    def test_safety_check_marked_safe_for_valid_query(self):
        result = rag_recommend("relaxing lofi beats for focus", SAMPLE_SONGS)
        assert result["safety_check"]["safe"] is True

    @requires_api
    def test_relevance_check_runs_after_generation(self):
        result = rag_recommend("calm background music for work", SAMPLE_SONGS)
        assert result["relevance_check"] is not None
        assert "score" in result["relevance_check"]


class TestSummarization:
    """Profile and recommendation summarization."""

    @requires_api
    def test_summarize_profile_returns_string(self):
        result = summarize_profile(SAMPLE_USER)
        assert isinstance(result, str)
        assert len(result) > 20

    @requires_api
    def test_summarize_recommendations_returns_string(self):
        recs = [(s, 8.5, "genre match | mood match") for s in SAMPLE_SONGS[:3]]
        result = summarize_recommendations(SAMPLE_USER, recs)
        assert isinstance(result, str)
        assert len(result) > 20


class TestPlanning:
    """Step-by-step playlist planning."""

    @requires_api
    def test_plan_returns_non_empty_string(self):
        result = plan_playlist_for_occasion("late night study session", SAMPLE_SONGS)
        assert isinstance(result, str)
        assert len(result) > 50

    @requires_api
    def test_plan_mentions_steps(self):
        result = plan_playlist_for_occasion("morning jog", SAMPLE_SONGS)
        # Claude should mention at least one numbered step or "Step"
        assert "step" in result.lower() or "1." in result or "2." in result


class TestExplainAndClassify:
    """Explain scoring and classify query intent."""

    @requires_api
    def test_explain_song_score_returns_string(self):
        song = SAMPLE_SONGS[0]
        result = explain_song_score(song, SAMPLE_USER, score=9.2, reasons="mood match | genre match")
        assert isinstance(result, str)
        assert len(result) > 20

    @requires_api
    def test_classify_query_returns_expected_keys(self):
        result = classify_query_intent("I need energetic pop music for my workout")
        assert "genre" in result
        assert "mood" in result
        assert "energy" in result
        assert "occasion" in result

    @requires_api
    def test_classify_query_energy_field_is_valid(self):
        result = classify_query_intent("calm relaxing music for sleep")
        assert result.get("energy") in ("low", "medium", "high")

    @requires_api
    def test_classify_high_energy_query(self):
        result = classify_query_intent("intense gym workout music pump me up")
        assert result.get("energy") == "high"

    @requires_api
    def test_classify_low_energy_query(self):
        result = classify_query_intent("gentle ambient music to fall asleep")
        assert result.get("energy") == "low"
