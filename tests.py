"""
tests.py — Unit tests for parser.py helpers and parse() pipeline.

Run with:  python -m pytest tests.py -v
           python tests.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Stub playwright-dependent scraper so parser can be imported without it.
import types

_stub = types.ModuleType("scraper")
_stub.BasicData = dict
_stub.ResultItem = dict
_stub.ScrapeResult = dict
sys.modules.setdefault("scraper", _stub)

from parser import (  # noqa: E402
    SearchResult,
    dedup,
    exact_key_set,
    normalize_for_match,
    normalize_text,
    parse,
    parse_song_type,
    parse_synonyms,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _basic(
    anime_title: str = "Naruto",
    song_type_raw: str = "OP1",
    title_raw: str = "R★O★C★K★S",
    artist_values: list | None = None,
    spotify_link: str | None = "https://open.spotify.com/track/abc123",
    source_index: int = 0,
    unreleased: bool = False,
) -> dict:
    return {
        "anime_title": anime_title,
        "song_type_raw": song_type_raw,
        "title_raw": title_raw,
        "artist_values": artist_values or ["Hound Dog"],
        "spotify_link": spotify_link,
        "source_index": source_index,
        "unreleased": unreleased,
    }


def _item(
    basic: dict, portal: dict | None = None, *, include_portal_key: bool = True
) -> dict:
    """
    include_portal_key=True  → portal_data key present (Phase 1+2 or explicit fetch)
    include_portal_key=False → portal_data key absent  (Phase 1-only scrape)
    """
    item: dict = {"basic_data": basic}
    if include_portal_key:
        item["portal_data"] = portal
    return item


def _scrape(query: str, items: list[dict]) -> dict:
    return {"query": query, "results": items, "raw_html_snapshot": ""}


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------


class TestNormalizeText(unittest.TestCase):

    def test_casefold(self):
        self.assertEqual(normalize_text("NARUTO"), "naruto")

    def test_strips_whitespace(self):
        self.assertEqual(normalize_text("  sword art  "), "sword art")

    def test_collapses_internal_whitespace(self):
        self.assertEqual(normalize_text("dragon\t\tball"), "dragon ball")

    def test_nfkc_ligature(self):
        self.assertEqual(normalize_text("\ufb01lm"), "film")

    def test_nbsp(self):
        self.assertEqual(normalize_text("one\u00a0piece"), "one piece")


# ---------------------------------------------------------------------------
# normalize_for_match
# ---------------------------------------------------------------------------


class TestNormalizeForMatch(unittest.TestCase):

    def test_casefold(self):
        self.assertEqual(normalize_for_match("Naruto"), "naruto")

    def test_curly_apostrophe(self):
        self.assertEqual(normalize_for_match("Re\u2019Zero"), "re'zero")

    def test_modifier_apostrophe(self):
        self.assertEqual(normalize_for_match("Re\u02bcZero"), "re'zero")

    def test_em_dash(self):
        self.assertEqual(
            normalize_for_match("Attack on Titan\u2014Final"), "attack on titan-final"
        )

    def test_en_dash(self):
        self.assertEqual(
            normalize_for_match("My Hero\u2013Academia"), "my hero-academia"
        )

    def test_fullwidth_exclamation(self):
        self.assertEqual(
            normalize_for_match("Sword Art Online\uff01"), "sword art online!"
        )

    def test_fullwidth_question(self):
        self.assertEqual(normalize_for_match("Is It Wrong\uff1f"), "is it wrong?")

    def test_curly_quotes(self):
        self.assertEqual(normalize_for_match("\u201cHello\u201d"), '"hello"')

    def test_whitespace_collapse(self):
        self.assertEqual(normalize_for_match("  Demon   Slayer  "), "demon slayer")

    def test_plain_ascii_unchanged(self):
        self.assertEqual(
            normalize_for_match("Fullmetal Alchemist"), "fullmetal alchemist"
        )


# ---------------------------------------------------------------------------
# parse_song_type
# ---------------------------------------------------------------------------


class TestParseSongType(unittest.TestCase):

    def test_op_with_number(self):
        self.assertEqual(parse_song_type("OP1"), ("op", 1))

    def test_ed_with_number(self):
        self.assertEqual(parse_song_type("ED2"), ("ed", 2))

    def test_op_no_number(self):
        self.assertEqual(parse_song_type("OP"), ("op", None))

    def test_insert_song(self):
        self.assertEqual(parse_song_type("Insert Song"), ("insert song", None))

    def test_ost_large_number(self):
        self.assertEqual(parse_song_type("OST12"), ("ost", 12))

    def test_already_lowercase(self):
        self.assertEqual(parse_song_type("op"), ("op", None))

    def test_empty(self):
        kind, seq = parse_song_type("")
        self.assertEqual(kind, "")
        self.assertIsNone(seq)


# ---------------------------------------------------------------------------
# parse_synonyms
# ---------------------------------------------------------------------------


class TestParseSynonyms(unittest.TestCase):

    def test_single(self):
        self.assertEqual(parse_synonyms("Naruto"), ["Naruto"])

    def test_multiple(self):
        self.assertEqual(
            parse_synonyms("Naruto, Naruto Shippuden"), ["Naruto", "Naruto Shippuden"]
        )

    def test_deduplication(self):
        self.assertEqual(parse_synonyms("A, A, B"), ["A", "B"])

    def test_strips_spaces(self):
        self.assertEqual(
            parse_synonyms("  Bleach ,  Bleach 2 "), ["Bleach", "Bleach 2"]
        )

    def test_empty_string(self):
        self.assertEqual(parse_synonyms(""), [])


# ---------------------------------------------------------------------------
# dedup
# ---------------------------------------------------------------------------


class TestDedup(unittest.TestCase):

    def test_removes_duplicates(self):
        self.assertEqual(dedup(["a", "b", "a"]), ["a", "b"])

    def test_strips_items(self):
        self.assertEqual(dedup(["  x  ", "y"]), ["x", "y"])

    def test_drops_empty(self):
        self.assertEqual(dedup(["a", "", "  ", "b"]), ["a", "b"])

    def test_preserves_order(self):
        self.assertEqual(dedup(["c", "a", "b"]), ["c", "a", "b"])

    def test_empty_input(self):
        self.assertEqual(dedup([]), [])


# ---------------------------------------------------------------------------
# exact_key_set
# ---------------------------------------------------------------------------


class TestExactKeySet(unittest.TestCase):

    def test_normalises_entries(self):
        keys = exact_key_set(["Naruto", "NARUTO"])
        self.assertEqual(keys, {"naruto"})

    def test_punct_canonicalised(self):
        self.assertIn("re'zero", exact_key_set(["Re\u02bcZero"]))

    def test_skips_empty(self):
        keys = exact_key_set(["", "  ", "Bleach"])
        self.assertNotIn("", keys)
        self.assertIn("bleach", keys)


# ---------------------------------------------------------------------------
# parse() — portal_data key semantics
# ---------------------------------------------------------------------------


class TestParsePortalKeySemantics(unittest.TestCase):
    """
    Tests specifically for the three states of portal_data:
      absent  → Phase 1-only; no advanced logic
      None    → requested, all retries failed
      {}      → requested, no info button found
      {dict}  → synonyms extracted
    """

    def test_portal_key_absent_no_advanced_attempt(self):
        # Phase 1-only scrape: portal_data key not in item
        item = _item(_basic(anime_title="Bleach"), include_portal_key=False)
        r = parse(_scrape("q", [item]), exact_titles=["Naruto"])[0]
        self.assertFalse(r.advanced_attempted)
        self.assertIsNone(r.advanced_synonyms)
        self.assertIsNone(r.advanced_error)

    def test_portal_key_absent_match_still_works(self):
        # Title match works even without portal
        item = _item(_basic(anime_title="Naruto"), include_portal_key=False)
        r = parse(_scrape("q", [item]), exact_titles=["Naruto"])[0]
        self.assertTrue(r.matched_query)
        self.assertFalse(r.advanced_attempted)

    def test_portal_none_sets_error(self):
        # portal_data key present but None → retries exhausted
        item = _item(_basic(anime_title="Bleach"), portal=None)
        r = parse(_scrape("q", [item]), exact_titles=["Naruto"])[0]
        self.assertTrue(r.advanced_attempted)
        self.assertEqual(r.advanced_error, "portal extraction failed")

    def test_portal_empty_dict_no_attempt(self):
        # portal_data={} → info button not found; no attempt recorded
        item = _item(_basic(anime_title="Bleach"), portal={})
        r = parse(_scrape("q", [item]), exact_titles=["Naruto"])[0]
        self.assertFalse(r.advanced_attempted)
        self.assertIsNone(r.advanced_synonyms)

    def test_portal_with_synonyms_attempt_recorded(self):
        item = _item(
            _basic(anime_title="Shingeki no Kyojin"),
            portal={"synonyms": ["Attack on Titan"]},
        )
        r = parse(_scrape("q", [item]), exact_titles=["Attack on Titan"])[0]
        self.assertTrue(r.advanced_attempted)
        self.assertEqual(r.advanced_synonyms, ["Attack on Titan"])

    def test_portal_error_key_propagated(self):
        item = _item(
            _basic(anime_title="Bleach"),
            portal={"synonyms": [], "error": "dialog timed out"},
        )
        r = parse(_scrape("q", [item]), exact_titles=["Naruto"])[0]
        self.assertEqual(r.advanced_error, "dialog timed out")

    def test_portal_none_on_already_matched_no_advanced(self):
        # Even if portal=None, a card that already matched by title should
        # not enter the advanced logic branch.
        item = _item(_basic(anime_title="Naruto"), portal=None)
        r = parse(_scrape("q", [item]), exact_titles=["Naruto"])[0]
        self.assertTrue(r.matched_query)
        self.assertFalse(r.advanced_attempted)
        self.assertIsNone(r.advanced_error)


# ---------------------------------------------------------------------------
# parse() — title matching
# ---------------------------------------------------------------------------


class TestParseTitleMatching(unittest.TestCase):

    def test_exact_match(self):
        item = _item(_basic(anime_title="Naruto"), portal={})
        self.assertTrue(
            parse(_scrape("Naruto", [item]), exact_titles=["Naruto"])[0].matched_query
        )

    def test_no_match(self):
        item = _item(_basic(anime_title="Bleach"), portal={})
        self.assertFalse(
            parse(_scrape("Naruto", [item]), exact_titles=["Naruto"])[0].matched_query
        )

    def test_case_insensitive(self):
        item = _item(_basic(anime_title="NARUTO"), portal={})
        self.assertTrue(
            parse(_scrape("naruto", [item]), exact_titles=["Naruto"])[0].matched_query
        )

    def test_punct_normalisation(self):
        item = _item(_basic(anime_title="Attack on Titan\u2014Final Season"), portal={})
        self.assertTrue(
            parse(_scrape("q", [item]), exact_titles=["Attack on Titan-Final Season"])[
                0
            ].matched_query
        )

    def test_synonym_match(self):
        item = _item(
            _basic(anime_title="Shingeki no Kyojin"),
            portal={"synonyms": ["Attack on Titan", "AoT"]},
        )
        r = parse(_scrape("q", [item]), exact_titles=["Attack on Titan"])[0]
        self.assertTrue(r.matched_query)
        self.assertEqual(r.advanced_matched_synonym, "Attack on Titan")

    def test_synonym_no_match(self):
        item = _item(
            _basic(anime_title="Shingeki no Kyojin"),
            portal={"synonyms": ["AoT"]},
        )
        r = parse(_scrape("q", [item]), exact_titles=["Attack on Titan"])[0]
        self.assertFalse(r.matched_query)
        self.assertIsNone(r.advanced_matched_synonym)

    def test_defaults_to_query(self):
        item = _item(_basic(anime_title="Naruto"), include_portal_key=False)
        self.assertTrue(parse(_scrape("Naruto", [item]))[0].matched_query)


# ---------------------------------------------------------------------------
# parse() — field mapping and filtering
# ---------------------------------------------------------------------------


class TestParseFields(unittest.TestCase):

    def test_unreleased_dropped(self):
        item = _item(_basic(unreleased=True), portal={})
        self.assertEqual(parse(_scrape("q", [item])), [])

    def test_empty_title_dropped(self):
        item = _item(_basic(title_raw=""), portal={})
        self.assertEqual(parse(_scrape("q", [item])), [])

    def test_song_type_and_sequence(self):
        item = _item(_basic(song_type_raw="ED3"), portal={})
        r = parse(_scrape("q", [item]))[0]
        self.assertEqual(r.song_type, "ed")
        self.assertEqual(r.sequence, 3)

    def test_artists_deduped(self):
        item = _item(_basic(artist_values=["Band", "Band", "Vocalist"]), portal={})
        self.assertEqual(parse(_scrape("q", [item]))[0].artists, ["Band", "Vocalist"])

    def test_spotify_link_preserved(self):
        link = "https://open.spotify.com/track/xyz"
        r = parse(_scrape("q", [_item(_basic(spotify_link=link), portal={})]))[0]
        self.assertEqual(r.spotify_link, link)

    def test_spotify_link_none(self):
        r = parse(_scrape("q", [_item(_basic(spotify_link=None), portal={})]))[0]
        self.assertIsNone(r.spotify_link)

    def test_source_index_preserved(self):
        items = [
            _item(_basic(source_index=0, anime_title="A"), portal={}),
            _item(_basic(source_index=1, anime_title="B"), portal={}),
        ]
        results = parse(_scrape("q", items))
        self.assertEqual(results[0].source_index, 0)
        self.assertEqual(results[1].source_index, 1)

    def test_title_normalised(self):
        r = parse(_scrape("q", [_item(_basic(title_raw="  MY SONG  "), portal={})]))[0]
        self.assertEqual(r.title, "my song")

    def test_returns_search_result_instances(self):
        item = _item(_basic(), portal={})
        self.assertIsInstance(parse(_scrape("q", [item]))[0], SearchResult)

    def test_multiple_mixed_items(self):
        items = [
            _item(_basic(anime_title="Naruto", source_index=0), portal={}),
            _item(
                _basic(anime_title="Bleach", source_index=1, unreleased=True), portal={}
            ),
            _item(_basic(anime_title="One Piece", source_index=2), portal={}),
        ]
        results = parse(_scrape("q", items), exact_titles=["Naruto"])
        self.assertEqual(len(results), 2)  # Bleach dropped
        self.assertTrue(results[0].matched_query)
        self.assertFalse(results[1].matched_query)

    def test_mixed_portal_presence(self):
        # Some items have portal key, some don't — both coexist in one ScrapeResult
        items = [
            _item(
                _basic(anime_title="Naruto", source_index=0), include_portal_key=False
            ),
            _item(
                _basic(anime_title="Bleach", source_index=1),
                portal={"synonyms": ["Bleach 2"]},
            ),
            _item(_basic(anime_title="One Piece", source_index=2), portal=None),
        ]
        results = parse(_scrape("q", items), exact_titles=["Naruto"])
        self.assertTrue(results[0].matched_query)  # matched by title, no portal
        self.assertFalse(results[0].advanced_attempted)
        self.assertTrue(results[1].advanced_attempted)  # had synonyms
        self.assertTrue(results[2].advanced_attempted)  # portal=None → error
        self.assertEqual(results[2].advanced_error, "portal extraction failed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
