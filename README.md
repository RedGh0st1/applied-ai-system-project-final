# ToneMatch AI

> **Built on top of the Music Recommender Simulation** — a Codepath AI 110 project — enhanced with Retrieval-Augmented Generation, natural language queries, AI-powered playlist planning, and automated safety guardrails.

---

## Original Project: Music Recommender Simulation

The foundation of this work is the [**Music Recommender Simulation**](../ai110-module3show-musicrecommendersimulation-starter), a Codepath AI 110 assignment (`codepath_AI110/ai110-module3show-musicrecommendersimulation-starter`). Its goals were to represent songs and user taste profiles as structured data, design a hand-crafted scoring algorithm that could turn that data into ranked recommendations, and evaluate where the system worked well and where it failed. The original system scored every song in a 25-track catalog against a user profile using a combination of categorical signals (genre, mood, musical key) and Gaussian proximity scores on continuous features like energy, valence, and instrumentalness — returning the top-k results with per-signal explanations.

**ToneMatch AI** keeps that scoring engine intact and layers four AI capabilities on top: natural language query understanding, RAG-based recommendation generation, occasion-aware playlist planning, and dual-stage safety guardrails.

---

## Title and Summary

**ToneMatch AI** is a music recommendation system that understands what you are in the mood for — even when you describe it in plain English.

You can hand it a structured taste profile ("I like high-energy pop, major key, target energy 0.88") and it will score and rank songs, then have Claude write a narrative recommendation grounded in those exact scores. Or you can type a free-text query like *"dark and moody for a late-night drive"* and it will classify your intent, retrieve matching songs offline, generate a grounded answer with Claude, then automatically validate that the answer actually fits what you asked for. For more complex needs — a dinner party playlist that evolves from chill to dancing — a dedicated planning mode has Claude reason through an energy arc and select specific songs for each phase.

The project matters because it demonstrates how a small, transparent rule-based system and a large language model can be composed without replacing one with the other. The scoring engine stays auditable and offline; Claude handles language understanding and narrative generation. The seam between them is explicit, testable, and documented.

---

## Architecture Overview

The system has three independent execution paths, visualized in [`assets/system_diagram.svg`](assets/system_diagram.svg).

```text
① Profile-Based Path
   User Profile → Scoring Engine (5 strategies) → Response Generator (Claude Sonnet) → Ranked Recommendations

② NL Query RAG Path
   User Query → Intent Classifier (Claude Haiku)
              → Input Guardrail: validate_query_safety() (Claude Haiku)  ← BEFORE retrieval
              → Keyword Retriever: retrieve_songs_for_query() (offline, no API)
              → RAG Generator: rag_recommend() (Claude Sonnet)
              → Output Guardrail: validate_recommendation_relevance() (Claude Haiku)
              → RAG Answer

③ Planner Path
   Occasion → Planner: plan_playlist_for_occasion() (Claude Sonnet, samples catalog[:40]) → Playlist Plan
```

**Shared data store:** `data/songs.csv` — 25 songs with 25+ attributes each. The Scoring Engine and Keyword Retriever both read from it; the Planner samples the first 40 entries directly.

**Two Claude models are used deliberately:**

- `claude-sonnet-4-6` — for generation tasks that require reasoning and narrative quality (response generation, RAG answers, planning)
- `claude-haiku-4-5-20251001` — for fast, cheap classification and guardrail validation (intent classification, safety checks, relevance scoring)

**Human and automated review** sit at the bottom of the pipeline. Pytest covers unit tests (fully offline) and integration tests (live API). Human review validates that guardrail outputs and AI narrations actually make sense, since automated tests can only check format and range.

---

## Setup Instructions

### Prerequisites

- Python 3.10 or higher
- An [Anthropic API key](https://console.anthropic.com/) for AI features (Sections 3 and 4; the scoring engine runs without one)

### 1. Clone and enter the project

```bash
git clone <your-repo-url>
cd applied-ai-system-project-final
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
.venv\Scripts\activate          # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Export your API key

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # macOS / Linux
set ANTHROPIC_API_KEY=sk-ant-...      # Windows CMD
```

The application detects this at startup. If it is missing, Sections 1 and 2 (scoring engine) still run fully; Sections 3 and 4 are skipped with a clear message.

### 5. Run the application

```bash
python -m src.main
```

This runs all four sections in sequence:

- **Section 1** — Profile-based recommendations with optional AI narration
- **Section 2** — Side-by-side comparison of all 5 ranking strategies
- **Section 3** — RAG pipeline for three natural language queries
- **Section 4** — Step-by-step playlist plan for a specific occasion

### 6. Run the test suite

```bash
# Unit tests only (no API key required)
pytest tests/ -m "not integration"

# All tests including live API calls
pytest tests/
```

---

## Sample Interactions

### Interaction 1 — Profile-Based Recommendation (AI-narrated)

**Input:** `"High-Energy Pop"` profile — genre: pop, mood: happy, target energy: 0.88, target valence: 0.82

**Scoring engine output (top 3 retrieved):**

```text
1. Gym Hero         pop / intense     energy 0.93   score 10.37
2. Sunrise City     pop / happy       energy 0.82   score  9.85
3. Neon Seoul       k-pop / energetic energy 0.85   score  8.45
```

**Claude-generated narrative (Section 1, AI-narrated mode):**

```text
Based on your taste profile, here are your top picks:

Gym Hero leads the list — its energy of 0.93 sits right at the top of
your target range and it earns full marks for both genre and mood. If
you want the most intense option, this is it.

Sunrise City is the most tonally aligned pick: valence 0.84 matches
your 0.82 target almost exactly, giving it a bright, euphoric feel
without tipping into aggression. It's the most "feel-good" of the three.

Neon Seoul earns its spot through energy (0.85) and a high valence
(0.86), even though k-pop isn't your stated genre. Its rhythmic drive
makes it a natural fit when Gym Hero and Sunrise City are already in
rotation.
```

---

### Interaction 2 — Natural Language RAG Query

**Input query:** `"I need something dark and moody for a late-night drive"`

**Pipeline trace:**

```text
Intent   : genre='rock', mood='moody', energy='medium', occasion='night drive'
Safety   : ✓ safe
Relevance: 4/5
```

**Claude-generated answer (Section 3, RAG output):**

```text
For a late-night drive with a dark, moody atmosphere, here are three
tracks from the catalog worth queuing up:

Neon Rainfall (synthwave / moody, energy 0.62, valence 0.38) — slow
pulse, minor key, built for empty highways after midnight.

Storm Runner (rock / intense, energy 0.91, valence 0.48) — for when
the drive turns urgent. High energy but emotionally heavy; valence
under 0.50 keeps it dark rather than aggressive.

Midnight Mirage (ambient / melancholic, energy 0.30, valence 0.32) —
the quieter option. Best for long stretches where the city disappears
and you want something that sits in the background without demanding
attention.
```

---

### Interaction 3 — Occasion Playlist Planning

**Input occasion:** `"a dinner party that starts relaxed and gradually builds to dancing"`

**Claude planning output (Section 4, abridged):**

```text
Step 1 — Arrival & Settling (energy 0.25–0.40)
  Goal: create a calm, welcoming atmosphere while guests arrive.
  Songs: Coffee Shop Stories, Library Rain, Island Morning

Step 2 — Dinner Conversation (energy 0.40–0.55)
  Goal: maintain energy just above background without dominating
  the room. Rhythmic but not distracting.
  Songs: Saturday Groove, Bossa Vida, Rooftop Lights

Step 3 — Post-dinner Transition (energy 0.55–0.72)
  Goal: signal the shift from eating to socializing. Tempo
  increases, mood lifts.
  Songs: Crown Walk, Salsa Fuego, Funky Parliament

Step 4 — Dancing (energy 0.78–0.95)
  Goal: full dance floor energy. High valence, strong beat.
  Songs: Gym Hero, Neon Seoul, Sunrise City
```

---

## Design Decisions

### Why keep the rule-based scoring engine instead of letting Claude do everything?

The scoring engine is auditable. Every recommendation comes with a per-signal breakdown ("genre match | mood match | energy proximity: 0.93 vs 0.88") that can be inspected, debugged, and unit-tested. If I handed all ranking to Claude, I would get plausible-sounding recommendations with no way to verify or improve them systematically. The engine also runs completely offline — no API cost, no latency, no rate limits.

The trade-off: the engine is rigid. It cannot understand nuance in user intent that isn't pre-encoded as a number. That is exactly where Claude fills the gap.

### Why put the input guardrail BEFORE retrieval, not after?

`validate_query_safety()` fires as the first step inside `rag_recommend()`, before any retrieval or generation happens. The reason is cost and correctness: if a query is unsafe, there is no point spending API calls on retrieval and generation. Blocking early also means an unsafe query never touches the catalog or Claude's generation prompt, which limits exposure regardless of what subsequent filters might catch.

### Why two Claude models instead of one?

Guardrail and classification calls (`validate_query_safety`, `validate_recommendation_relevance`, `classify_query_intent`) are called on every query and need to be fast and cheap. Haiku is sufficient for structured JSON extraction and yes/no safety judgments. Sonnet is reserved for tasks where output quality is the primary goal — narrative generation and multi-step planning — where the higher cost is justified.

### Why five ranking strategies instead of one?

A single scoring formula cannot serve every listener. "Genre-first" weights genre heavily and is the right default for someone who knows what genre they want. "Discovery" deliberately penalizes genre repetition to surface songs the user would not expect. "Vibe-match" leans on valence and mood over genre labels. Rather than trying to find one formula that works for everyone, the system makes the weighting explicit and swappable. The side-by-side comparison in Section 2 is designed specifically to show how the same profile produces different top-5 lists under each strategy.

### Trade-offs accepted

| Decision | What was gained | What was lost |
| --- | --- | --- |
| 25-song catalog | Fast iteration, full control over every edge case | No ability to stress-test catalog-density effects |
| Keyword retrieval (no embeddings) | Fully offline, no vector store setup | Misses semantic synonyms ("joyful" vs "happy") |
| Prompt caching on system prompts | Reduced latency and cost on repeated runs | Slightly more complex prompt construction |
| No streaming | Simpler code and testing | Slower perceived response for long Sonnet outputs |

---

## Testing Summary

The project has **31 tests** across two files.

### `tests/test_ai_features.py` — 27 tests

**8 unit tests** (no API key, fully offline) cover `retrieve_songs_for_query()`:

- Returns at most `k` results
- Pop queries rank pop songs first
- Study queries rank lofi first
- Gym queries rank high-energy songs first (energy ≥ 0.75)
- Instrumental queries boost high-instrumentalness songs
- Title mentions score highest (direct name matching)

These passed consistently because the retrieval logic is deterministic keyword scoring with no randomness.

**19 integration tests** (require `ANTHROPIC_API_KEY`) cover:

- Guardrails: safe queries return `{"safe": True}`, relevance scores fall in the 1–5 range, clearly matched queries score ≥ 3
- RAG pipeline: all four expected keys present, answer is a non-empty string, safety check fires before generation
- Summarization: `summarize_profile()` and `summarize_recommendations()` both return non-trivial strings
- Planning: `plan_playlist_for_occasion()` returns a multi-step plan with numbered steps
- Classification: `classify_query_intent()` returns genre/mood/energy/occasion keys, energy is always one of "low"/"medium"/"high"

**What worked:** Unit tests were reliable from day one. Integration tests exposed that guardrail relevance scores vary by phrasing — a query that is objectively a match can score 3 instead of 4 depending on how Claude interprets the match criteria that session.

**What was harder than expected:** Testing that Claude's structured JSON output is always parseable. The `_parse_json()` helper (which strips markdown code fences and retries) was added after integration tests revealed that Claude occasionally wraps JSON in \`\`\`json blocks even when instructed not to.

### `tests/test_recommender.py` — 4 tests

Covers the core scoring engine: score computation, top-k ordering, genre match bonus, and edge cases with profiles that have no matching genre in the catalog.

---

## Reflection

Building ToneMatch AI clarified something that is easy to miss when reading about AI systems: **composition is harder than any individual component**. Getting Claude to return a grounded recommendation is straightforward. Getting the safety check to fire before — not after — retrieval, routing the result back through a relevance validator, and surfacing a useful output when any step fails requires deliberate sequencing that the model itself cannot enforce.

The guardrail implementation was the most instructive part of the project. I initially placed `validate_query_safety()` after retrieval, which meant every request was already making two API calls before the safety decision happened. Moving it to the first step required rethinking the entire `rag_recommend()` function — and that redesign revealed that the order of operations in a pipeline is a design choice with real cost and correctness implications, not just an implementation detail.

I also learned that LLMs are very good at producing plausible-sounding outputs that do not actually match the request. The output guardrail (`validate_recommendation_relevance`) was supposed to catch this, but it relies on Claude evaluating its own outputs — which means it tends to be generous. A score of 3/5 from the guardrail does not necessarily mean the recommendation is bad; it means Claude thinks there is some mismatch. The number is useful as a signal but not a verdict.

More broadly, this project changed how I think about what "AI" means in an application. The scoring engine — written entirely in Python with no model calls — does the heaviest lifting for structured profiles. Claude handles the parts where language and context matter. The system is better for having both. Real AI applications are almost never just a model; they are pipelines where the non-AI parts are just as important to get right.
