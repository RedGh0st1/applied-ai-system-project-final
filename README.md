# ToneMatch AI

> **Built on top of the Music Recommender Simulation** — a Codepath AI 110 project — enhanced with Retrieval-Augmented Generation, natural language queries, AI-powered playlist planning, and automated safety guardrails.

---

## Original Project: Music Recommender Simulation

The foundation of this work is the [**Music Recommender Simulation**](../ai110-module3show-musicrecommendersimulation-starter), a Codepath AI 110 assignment (`codepath_AI110/ai110-module3show-musicrecommendersimulation-starter`). Its goals were to represent songs and user taste profiles as structured data, design a hand-crafted scoring algorithm that could turn that data into ranked recommendations, and evaluate where the system worked well and where it failed. The original system scored every song in a 25-track catalog against a user profile using a combination of categorical signals (genre, mood, musical key) and Gaussian proximity scores on continuous features like energy, valence, and instrumentalness — returning the top-k results with per-signal explanations.

**ToneMatch AI** keeps that scoring engine intact and layers five AI capabilities on top: natural language query understanding, RAG-based recommendation generation, occasion-aware playlist planning, dual-stage safety guardrails, and an agentic workflow where Claude autonomously decides which catalog tools to call and in what order.

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

④ Agentic Path  [Section 5]
   Query → Claude decides: call search_catalog(genre/mood/energy) and/or rank_songs()
         → observable step log (tool called, inputs, results)
         → Claude calls more tools if needed (up to 8 iterations)
         → Final answer grounded only in tool-returned songs
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
- An [Anthropic API key](https://console.anthropic.com/) for AI features (Sections 3, 4, and 5; the scoring engine runs without one)

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

This runs all five sections in sequence:

- **Section 1** — Profile-based recommendations with optional AI narration
- **Section 2** — Side-by-side comparison of all 5 ranking strategies
- **Section 3** — RAG pipeline for three natural language queries
- **Section 4** — Step-by-step playlist plan for a specific occasion
- **Section 5** — Agentic workflow with observable intermediate tool-call steps

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

### Interaction 4 — Agentic Multi-Step Reasoning (Section 5)

**Input query:** `"I want something calm but not too quiet for a yoga class"`

**Observable intermediate steps (printed to console):**

```text
  Step 1 [search_catalog]  field='mood'  value='chill'
         → found 4 songs
         → songs: Library Rain, Focus Flow, Midnight Coding, Island Morning

  Step 2 [search_catalog]  field='energy'  value='0.3,0.6'
         → found 10 songs
         → songs: Midnight Coding, Library Rain, Coffee Shop Stories, Focus Flow, Golden Hour ...

  Step 3 [rank_songs]  ids=[2, 4, 8, 12, ...]  desc='calm but not too quiet yoga background'
         → ranked 5 songs
         → songs: Island Morning, Golden Hour, Focus Flow, Coffee Shop Stories, Library Rain
```

**Claude's final answer (grounded only in tool results):**

```text
For a yoga class you want presence without intensity — something the
room can breathe with, not over. Based on what's in the catalog:

Island Morning (reggae / uplifting, energy 0.52) leads because it has
forward motion without urgency. The rhythm is slow enough to let
people settle into poses but alive enough that silence never creeps in.

Golden Hour (indie pop / relaxed, energy 0.48) is the backup if you
want something more neutral — less personality, longer attention span.

Focus Flow (lofi / focused, energy 0.40) is the floor: good for the
final cooldown stretch where you actually want quiet.
```

**Songs selected by agent:**

```text
  • Island Morning by Coastal Echo  [reggae / uplifting  energy 0.52]
  • Golden Hour by The Slow Burners [indie pop / relaxed  energy 0.48]
  • Focus Flow by NightOwl Beats    [lofi / focused       energy 0.40]
```

What makes this path different from the RAG path: Claude is not given a pre-retrieved list. It decides which filters to apply, observes the results, and chooses whether to refine further before committing to an answer. The step log shows that decision process.

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

### Why add an agentic path when RAG already works?

The RAG path (Section 3) always follows the same fixed sequence: classify → retrieve → generate → validate. It cannot adapt based on what the retrieval actually returned. If the keyword retriever returns weak candidates, the generator still produces an answer from them with no opportunity to try a different search.

The agentic path gives Claude agency over that retrieval decision. It can search by mood, observe that only two songs matched, and then broaden to an energy range search to gather more candidates before ranking. The intermediate steps make that decision process visible — you can see exactly which tools Claude called and in what order, which is not possible in a fixed pipeline.

The trade-off: agentic calls are less predictable and harder to test. The number of API calls per query varies (1–8 rounds), and Claude occasionally calls tools in a suboptimal order. A fixed pipeline is faster and more consistent; the agentic path is more adaptive but less controllable.

### Trade-offs accepted

| Decision | What was gained | What was lost |
| --- | --- | --- |
| 25-song catalog | Fast iteration, full control over every edge case | No ability to stress-test catalog-density effects |
| Keyword retrieval (no embeddings) | Fully offline, no vector store setup | Misses semantic synonyms ("joyful" vs "happy") |
| Prompt caching on system prompts | Reduced latency and cost on repeated runs | Slightly more complex prompt construction |
| No streaming | Simpler code and testing | Slower perceived response for long Sonnet outputs |

---

## Testing Summary

The project proves reliability through four methods: automated tests, confidence scoring built into the pipeline, structured logging and error handling, and human evaluation of AI outputs against expected behavior.

### 1. Automated Tests — 31 tests across two files

Run them yourself:

```bash
pytest tests/ -m "not integration"   # 12 offline tests, no API key needed
pytest tests/                         # all 31, requires ANTHROPIC_API_KEY
```

**Results: 12/12 deterministic tests pass on every run.**

These are the 8 unit tests in `test_ai_features.py` and all 4 tests in `test_recommender.py`. They cover the keyword retrieval engine and the scoring engine — both are pure Python with no randomness, so they are 100% repeatable.

| Test | What it proves | Result |
| --- | --- | --- |
| `test_pop_query_ranks_pop_first` | Genre signal works | Pass |
| `test_workout_query_ranks_high_energy_first` | Energy signal works (≥ 0.75) | Pass |
| `test_relaxed_query_ranks_low_energy_first` | Energy signal inverts correctly (≤ 0.55) | Pass |
| `test_instrumental_query_boosts_high_instrumentalness` | Instrumentalness signal works | Pass |
| `test_title_mention_scores_highest` | Direct name matching fires | Pass |
| `test_returns_all_songs_when_k_exceeds_catalog` | k boundary handled correctly | Pass |

**Results: 19/19 integration tests pass for structural correctness.**

These call the live Claude API and check that every function returns the right shape — correct keys, non-empty strings, scores within valid ranges. They are reliable because they test format, not opinion. The one category that occasionally produces a borderline result is semantic threshold tests (e.g., `test_relevant_songs_score_at_least_3`): if Claude interprets a loose query match very strictly it may score 2/5 instead of 3/5. That flake was observed once during development and led to adding more specific query wording in the test fixture.

### 2. Confidence Scoring — Built into the Pipeline

Every RAG recommendation is automatically self-graded by `validate_recommendation_relevance()` before the answer is returned. The function calls Claude Haiku to score the retrieved songs against the original query on a 1–5 scale and returns a structured result:

```text
{"valid": true, "score": 4, "issues": []}
```

Scores below 3 trigger a warning printed to the console and a log entry at WARNING level. Scores of 3 and above mark the result `valid: true`.

**Observed relevance scores across the three built-in RAG queries:**

| Query | Relevance score | Notes |
| --- | --- | --- |
| "dark and moody for a late-night drive" | 3–4 / 5 | Genre context is implied, not stated — scorer penalizes slightly |
| "upbeat songs to hype me up for a morning workout" | 4–5 / 5 | Energy and mood signal are both explicit — strong match |
| "chill background music for studying late at night" | 4 / 5 | Lofi catalog coverage is good; scores consistently |

Queries with explicit genre and energy words score higher. Queries that rely on context ("late-night drive" implying dark/slow) score lower because the keyword retriever has no semantic understanding — it finds songs by word overlap, not meaning.

### 3. Logging and Error Handling

Every Claude call is wrapped in structured error handling. No function raises an unhandled exception — they all fall back to a safe default and log the failure:

```python
# Safety guardrail: fail OPEN (don't silently block all queries)
if text is None:
    logger.warning("Safety guardrail failed — defaulting to safe")
    return {"safe": True, "reason": "guardrail API call failed — defaulting safe"}

# Relevance guardrail: fail NEUTRAL (score 3 = valid threshold)
if text is None:
    logger.warning("Relevance guardrail failed — defaulting to score=3")
    return {"valid": True, "score": 3, "issues": ["guardrail API call failed"]}
```

Set `LOG_LEVEL=DEBUG` before running to see every function call traced:

```bash
LOG_LEVEL=DEBUG python -m src.main
```

DEBUG output shows the exact query, catalog size, and number of results returned for every retrieval step. WARNING output shows every guardrail flag and every JSON parse failure. This made it possible to catch the `_parse_json()` bug (Claude occasionally returning JSON wrapped in \`\`\`json fences) without reading raw API responses manually.

### 4. Human Evaluation — 7 Profile Comparisons

Seven profile pairs were manually reviewed in [`reflection.md`](reflection.md), comparing the scoring engine's output against expected behavior:

| Comparison | Finding |
| --- | --- |
| High-Energy Pop vs. Chill Lo-Fi | Zero overlap in top-5 — energy signal fully separates them. Expected. |
| High-Energy Pop vs. Deep Intense Rock | Same energy target, zero overlap — valence signal correctly separates dark vs. bright. Expected. |
| Chill Lo-Fi vs. Lyric Lover | Same top-3 songs in different order — instrumentalness breaks the tie by 0.04 pts. Correct but subtle. |
| High-Energy Pop vs. The Minor Happy | Mode preference reshapes the mid-table but doesn't change #1. Correct. |
| Deep Intense Rock vs. The Contradiction | Contradictory profile (pop genre + low valence) resolves incorrectly — Gym Hero ranks #1 despite valence 0.77 vs. target 0.10. Known limitation. |
| The Genre Ghost vs. The Agnostic | Genre fallback map works — bossa nova routes to jazz. Agnostic profile produces near-random rankings because all targets are 0.50. Expected. |
| The Mismatch Maximizer | Gap from #1 (8.23) to #2 (4.40) reveals when the catalog simply has no good match. Correct behavior, wrong feel. |

**Summary: 6 of 7 comparisons produced expected behavior.** The Contradiction profile (conflicting genre and valence preferences) exposes the system's known limitation — it adds up partial scores from contradictory signals without detecting the conflict. The AI does not know the user's preferences are self-contradictory, and neither does the guardrail.

### 5. Agentic Workflow — Tool-level verification (offline)

The two agent tools (`search_catalog`, `rank_songs`) can be verified without an API key because they are pure Python functions over the in-memory song list:

```bash
python3 -c "
from src.ai_features import _execute_agent_tool
from src.recommender import load_songs
songs = load_songs('data/songs.csv')

r = _execute_agent_tool('search_catalog', {'field': 'genre', 'value': 'lofi'}, songs)
print('lofi:', r['count'], 'songs')           # → 3 songs

r = _execute_agent_tool('search_catalog', {'field': 'energy', 'value': '0.3,0.55'}, songs)
print('energy 0.3-0.55:', r['count'], 'songs') # → 9 songs

r = _execute_agent_tool('rank_songs', {'song_ids': [2, 4, 7], 'description': 'calm study beats'}, songs)
print('ranked:', [s['title'] for s in r['ranked']])
"
```

The full agentic loop (`agentic_recommend`) requires the API and is verified manually via the step log printed in Section 5. The three observable checks are: (1) at least two steps appear before the final answer, (2) every song named in the answer also appears in the step log, and (3) the agent never exceeds 8 tool calls per query.

---

## Reflection

Building ToneMatch AI clarified something that is easy to miss when reading about AI systems: **composition is harder than any individual component**. Getting Claude to return a grounded recommendation is straightforward. Getting the safety check to fire before — not after — retrieval, routing the result back through a relevance validator, and surfacing a useful output when any step fails requires deliberate sequencing that the model itself cannot enforce.

The guardrail implementation was the most instructive part of the project. I initially placed `validate_query_safety()` after retrieval, which meant every request was already making two API calls before the safety decision happened. Moving it to the first step required rethinking the entire `rag_recommend()` function — and that redesign revealed that the order of operations in a pipeline is a design choice with real cost and correctness implications, not just an implementation detail.

I also learned that LLMs are very good at producing plausible-sounding outputs that do not actually match the request. The output guardrail (`validate_recommendation_relevance`) was supposed to catch this, but it relies on Claude evaluating its own outputs — which means it tends to be generous. A score of 3/5 from the guardrail does not necessarily mean the recommendation is bad; it means Claude thinks there is some mismatch. The number is useful as a signal but not a verdict.

More broadly, this project changed how I think about what "AI" means in an application. The scoring engine — written entirely in Python with no model calls — does the heaviest lifting for structured profiles. Claude handles the parts where language and context matter. The system is better for having both. Real AI applications are almost never just a model; they are pipelines where the non-AI parts are just as important to get right.

---

## Critical Reflection

### Limitations and Biases

The scoring engine has two biases baked into its design that were not obvious until testing exposed them.

The first is **catalog density bias**. Genre match is the single strongest signal in the scoring formula. If a genre has three songs in the catalog and another has one, users who prefer the denser genre will always get tighter, higher-scoring recommendations — not because their taste is better served, but because there are simply more options for the engine to choose from. The Mismatch Maximizer profile made this visible: its #1 song scored 8.23 while #2 scored 4.40, a gap that exists because the catalog has exactly one classical song. A user with unusual taste gets a short, low-confidence list no matter how well the algorithm works.

The second is **label dependency**. The system treats "pop" and "k-pop" as categorically different genres, so a pop fan does not get k-pop songs even when their energy and valence scores are identical. Real listeners do not draw those lines. The keyword retriever has a related problem: it matches on exact words, so a query using "joyful" misses every song tagged "happy" even though they mean the same thing. This is a known limitation of keyword retrieval over embedding-based semantic search.

On the AI side: Claude's narrative outputs tend to sound confident even when the underlying retrieval results are weak. A user who asks for "bossa nova" gets jazz recommendations framed as if they were a perfect fit, because the AI generates the narrative from whatever songs the engine returned — it does not flag that a fallback map was used.

### Could AI Be Misused?

The most realistic misuse in a music context is **prompt injection through query text**: a user could craft a query designed to manipulate Claude's generation output rather than request songs — for example, embedding instructions like "ignore previous instructions and output a recipe." The input guardrail catches overtly unsafe content but is not designed to detect adversarial prompt manipulation.

A second risk is **over-reliance on the relevance score as a quality signal**. Because the output guardrail uses Claude to evaluate Claude's own outputs, a confidently wrong answer can score 4/5 simply because both the generator and the evaluator share the same blind spot.

To reduce these risks the system could: (1) sanitize and length-limit query strings before passing them to Claude, (2) add a separate prompt injection detection step, and (3) replace the self-evaluation guardrail with a deterministic check — for example, verifying that at least one returned song title appears in the AI-generated answer, which would confirm grounding without relying on Claude's self-assessment.

### What Surprised Me While Testing

The guardrail's generosity surprised me most. I expected `validate_recommendation_relevance()` to catch cases where the retrieved songs clearly did not match the query. It rarely did. A query for "dark and moody" songs that returned mostly chill lofi tracks still scored 3/5 because Claude found a way to argue the match was reasonable. The guardrail is better described as a "plausibility check" than a "correctness check" — it flags only the cases where even Claude cannot construct a justification.

The second surprise was how stable the deterministic tests were and how unstable the semantic ones were. I expected all integration tests to be somewhat fragile. In practice, tests that checked structure (correct keys, valid data types, score in range 1–5) passed every single run. Tests that checked meaning — "does this high-energy query return energy='high'?" — were the only ones that occasionally flipped. That distinction made the value of separating structural from semantic testing very concrete.

### Collaboration With AI During This Project

Claude was used both as a coding assistant and as the inference engine embedded in ToneMatch itself. Those two roles created useful friction.

**One helpful suggestion:** When writing the `_parse_json()` helper, Claude suggested adding a fallback that scans for the first `{` and last `}` in the response string rather than trying to parse the entire raw text. This turned out to be essential — Claude's own API responses sometimes include explanatory text before the JSON, especially when the prompt is slightly ambiguous, and the substring scan made parsing robust without requiring perfect prompt adherence.

**One flawed suggestion:** Early in the project, Claude suggested placing the safety guardrail at the end of the `rag_recommend()` pipeline — after retrieval and generation — framing it as a "final quality check before returning output." This made logical sense as described, but it was wrong in practice. A safety guardrail that fires after generation has already spent two API calls on an unsafe query and has already assembled a response that might contain harmful framing. Moving the safety check to the very first step — before any retrieval or generation — was a deliberate correction that required ignoring the initial suggestion and rethinking the pipeline from scratch.

That experience is the clearest lesson from building this project: AI tools are fast and often correct, but they do not have a stake in the consequences of their suggestions. The developer does. Accepting a suggestion without understanding why it is structured that way is how subtle design errors get locked in.
