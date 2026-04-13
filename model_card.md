# Model Card: ToneMatch 1.0

---

## 1. Model Name

**ToneMatch 1.0**

A content-based music recommender that matches songs to a listener based on how a song *feels*, not just what genre label it carries.

---

## 2. Intended Use

ToneMatch is designed to suggest songs from a small catalog that match a user's current listening mood and energy level.

It is built for classroom exploration. It is not a production system.

It assumes the user can describe their preferences in advance — things like "I like pop, I want high energy, I prefer major key songs." It does not learn from listening history or feedback.

**Not intended for:** real-world deployment, personalization at scale, or any context where a bad recommendation has real consequences.

---

## 3. How the Model Works

Every song in the catalog gets a score. The highest-scoring songs are recommended.

The score is built from two types of signals:

**Categorical signals** check whether a song matches the user's stated preferences exactly.
- Genre match: the song's genre matches the user's favorite genre. This is the strongest single signal.
- Subgenre match: only counts if the genre already matched. Rewards specificity.
- Mood match: exact match scores full points. Songs with a similar mood score partial points.
- Key feel (major vs. minor): a match adds points. A mismatch adds nothing.

**Continuous signals** measure how close a song's numbers are to the user's targets using a bell curve. The closer the match, the more points. Unlike the categorical signals, these never hit zero — every song earns at least a little credit.
- Energy: how loud and intense the song feels. Worth the most of any single signal.
- Valence: how positive or negative the song feels emotionally.
- Instrumentalness: how much of the song is instruments vs. vocals.

One extra rule: if a song's instrumentalness is very far from what the user wants (more than 0.70 apart), the entire score is multiplied by 0.60 as a penalty. This stops vocal songs from sneaking into the top results for a user who explicitly wants instrumentals.

If a user's favorite genre is not in the catalog, the system checks a lookup table of related genres (for example, "bossa nova" maps to "jazz") and awards partial genre points instead of nothing.

Scores are added up and the top five songs are returned.

---

## 4. Data

The catalog has **25 songs**.

Each song has 13 attributes: id, title, artist, genre, subgenre, mood, energy, tempo, valence, danceability, acousticness, mode, and instrumentalness.

**Genres covered:** pop, lofi, rock, ambient, jazz, synthwave, indie pop, r&b, hip-hop, classical, country, edm, reggae, funk, soul, metal, blues, folk, trap, latin, drum and bass, k-pop.

**Moods covered:** happy, chill, intense, focused, relaxed, romantic, moody, serene, euphoric, uplifting, groovy, confident, melancholic, sad, wistful, nostalgic, aggressive, frantic, dark, passionate, energetic.

**Known gaps in the data:**
- Lo-fi has 3 songs. Every other genre has exactly 1. This creates unequal recommendation quality across genres.
- 64% of songs are in a major key. Only 36% are minor. Minor-key listeners have fewer songs competing for them.
- Very quiet songs (energy below 0.25) are rare — only 2 songs fit that description. Users who want calm, ambient music get weaker matches than users who want high-energy music.
- The catalog does not include non-Western genres, spoken word, or anything experimental. It reflects a narrow slice of recorded music.

---

## 5. Strengths

The system works best when a user's preferences are internally consistent and the catalog has good coverage of their genre.

**Chill Lo-Fi users** get strong results. Three lofi songs compete for the top spots and they all score in the 10–11 point range. The recommendations feel right.

**High-energy users** are well-served too. There are enough high-energy songs (energy above 0.85) that the system can distinguish between them using valence and mode.

**The bell-curve scoring** is a genuine improvement over simple cutoffs. A user who wants energy 0.88 still gets meaningful credit for a song at 0.82 — the score does not suddenly drop to zero the way a threshold would. This produces smoother, more sensible rankings across the catalog.

**The genre mapping** handles niche genre requests gracefully. A user who says their favorite genre is "bossa nova" still gets a jazz song at the top instead of a random result.

---

## 6. Limitations and Bias

**Catalog density creates a lo-fi filter bubble.**
The 25-song catalog has three lo-fi tracks but only one song for 20 other genres. A lo-fi listener is almost guaranteed three genre-matched songs in their top results — not because the system is working better for them, but because there are simply more songs to match against. A rock listener has only one song that can ever earn a genre bonus. If that one song is a poor energy match, the rest of their top five fills with genre-miss songs the system settled for. Recommendation quality is tied to catalog size per genre, which is an accident of data collection.

**The system cannot detect contradictory preferences.**
If a user says they want pop genre but also low valence and a sad mood, the system just adds up the points for each signal separately. It will recommend a pop song at the top even if that pop song is bright and upbeat — because the genre points outweigh the valence penalty. There is no logic that notices the conflict or warns the user.

**Minor-key listeners are at a structural disadvantage.**
64% of songs are major-key. The mode bonus is worth 1 point. Minor-key users earn that bonus on only 9 out of 25 songs, while major-key users earn it on 16. In a tight ranking where everything else is close, major-key users consistently win the tiebreaker more often simply because the catalog was built that way.

**Energy and genre can override everything else.**
Energy is now the highest-weighted signal. In testing, a salsa song appeared in the top five for a pop listener purely because its energy was a perfect match. The system had no way to flag that the recommendation crossed a genre boundary the user probably cared about. A number being right does not mean the song is right.

---

## 7. Evaluation

Nine user profiles were tested against the full 25-song catalog.

**Standard profiles:** High-Energy Pop, Chill Lo-Fi, Deep Intense Rock. These tested whether the system returned sensible results for straightforward, internally consistent preferences.

**Adversarial profiles** were designed to break the system or expose edge cases:
- *The Contradiction* — high energy with a sad mood and low valence. Tests whether conflicting preferences produce coherent output.
- *The Genre Ghost* — favorite genre not in the catalog. Tests the genre mapping fallback.
- *The Agnostic* — all continuous targets set to 0.5. Tests what happens when numerical signals are neutral.
- *The Minor Happy* — happy mood with minor key preference. Tests whether a single conflicting preference disrupts otherwise good results.
- *The Lyric Lover* — instrumental target of 0.95. Tests whether the instrumentalness penalty actually changes recommendations.
- *The Mismatch Maximizer* — extreme targets designed to score most songs near zero. Tests tie-breaking and score floor behavior.

Two structural experiments were also run. First, energy weight was doubled and genre weight was halved to see whether continuous signals could override categorical ones. Second, the mood signal was disabled entirely to see what the remaining signals would recommend on their own.

**What the evaluation found:**

The clearest surprise was that Gym Hero ranked first for the High-Energy Pop profile even though its mood is "intense," not "happy." It won because a subgenre match awards more points than a mood match. The system never considered whether "intense" and "happy" are compatible — it just counted the points.

Removing mood had almost no effect on the Chill Lo-Fi profile. The top three songs stayed the same. Mood was acting as a small bonus rather than a real differentiator, because genre and energy had already sorted the results before mood could matter.

The Mismatch Maximizer showed what the output looks like when the catalog simply does not have what a user wants. Moonlight Study scored 8.23 and ranked first clearly. The next song scored 4.40. The remaining three hovered around 3.0–3.6, practically tied. The system produced a ranked list that looked normal but was actually ranking degrees of failure.

---

## 8. Future Work

**1. Balance the catalog.**
The single most impactful improvement would be adding more songs per genre so that every genre has at least 3–5 representatives. Right now lo-fi users get a structurally better experience than everyone else for no reason related to the algorithm.

**2. Make the mood adjacency graph symmetric.**
Seven mood connections currently work in one direction only. A user who likes "groovy" can partially match a "happy" song, but a user who likes "happy" cannot partially match a "groovy" song. Making every connection bidirectional would make the system treat equivalent tastes equally.

**3. Add a conflict warning.**
When a user's genre preference and valence/mood preference point in opposite directions, the system should surface a note rather than silently recommending something that satisfies one signal at the expense of the other. Even a simple message like "No songs matched both your genre and mood preferences — showing closest energy match instead" would make the output more honest.

---

## 9. Personal Reflection

Building this system made it clear that a recommender does not need to be complex to produce interesting results — and that interesting results are not always correct results.

The most surprising discovery was that removing a signal (mood) had almost no visible effect on one profile. That means the signal was in the formula but not doing real work. It is easy to assume that more features means better recommendations. This project showed that a feature only matters if the catalog has enough variety to let it differentiate songs from each other.

The experiment that changed my thinking the most was the salsa song appearing in a pop listener's top five after the weight shift. It was a perfect energy match. It was also a completely wrong recommendation. That gap — between a number being accurate and a recommendation being right — is what real recommender systems spend enormous effort trying to close, and this project helped make that gap concrete and visible rather than abstract.