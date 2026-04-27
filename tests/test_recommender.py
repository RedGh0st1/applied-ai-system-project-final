from src.recommender import Song, UserProfile, Recommender


def _make_song(id: int, title: str, genre: str, mood: str,
               energy: float, valence: float, mode: int = 1) -> Song:
    """Create a Song with all required fields, using sensible defaults for unused ones."""
    return Song(
        id=id,
        title=title,
        artist="Test Artist",
        genre=genre,
        subgenre="",
        mood=mood,
        energy=energy,
        tempo_bpm=120.0,
        valence=valence,
        danceability=0.7,
        acousticness=0.2,
        mode=mode,
        instrumentalness=0.05,
        popularity=70,
        release_decade="2020s",
        explicit=0,
        language="English",
        duration_sec=210,
        loudness_db=-6.0,
        speechiness=0.05,
        liveness=0.12,
        detailed_mood_tags="",
        cultural_region="Western",
        vocal_gender="female",
        era_feel="contemporary",
    )


def make_small_recommender() -> Recommender:
    songs = [
        _make_song(1, "Test Pop Track", genre="pop", mood="happy",
                   energy=0.8, valence=0.9),
        _make_song(2, "Chill Lofi Loop", genre="lofi", mood="chill",
                   energy=0.4, valence=0.6),
    ]
    return Recommender(songs)


def _make_user(**kwargs) -> UserProfile:
    """Create a UserProfile with all required fields, using sensible defaults."""
    defaults = dict(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        target_valence=0.8,
        target_bpm=120.0,
        target_acoustic=0.2,
        target_inst=0.05,
        preferred_mode=1,
        likes_acoustic=False,
        preferred_era="",
        preferred_language="",
        preferred_region="",
        preferred_vocal_gender="",
        allow_explicit=True,
        target_popularity=0.7,
        target_liveness=0.15,
        target_speechiness=0.05,
        preferred_mood_tags=[],
    )
    defaults.update(kwargs)
    return UserProfile(**defaults)


def test_recommend_returns_songs_sorted_by_score():
    user = _make_user(favorite_genre="pop", favorite_mood="happy", target_energy=0.8)
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)

    assert len(results) == 2
    # The pop/happy/high-energy song should score higher than lofi/chill/low-energy
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_explain_recommendation_returns_non_empty_string():
    user = _make_user(favorite_genre="pop", favorite_mood="happy", target_energy=0.8)
    rec = make_small_recommender()
    song = rec.songs[0]

    explanation = rec.explain_recommendation(user, song)
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


def test_recommend_respects_k_limit():
    user = _make_user()
    rec = make_small_recommender()
    assert len(rec.recommend(user, k=1)) == 1


def test_lofi_profile_prefers_lofi_song():
    user = _make_user(
        favorite_genre="lofi",
        favorite_mood="chill",
        target_energy=0.4,
        target_valence=0.6,
    )
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)
    assert results[0].genre == "lofi"
