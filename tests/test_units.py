"""Unit tests: toy-corpus checks with known right answers."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goldatfive.corpus import Corpus, Chunk
from goldatfive.metrics import rank_of, stage_attribution
from goldatfive.rerankers import (lexical_rerank, merge_llm_order,
                                  parse_llm_order)
from goldatfive.retrievers import Index, bm25_rank, rrf_fuse, tfidf_rank
from goldatfive.taxonomy import classify
from goldatfive.text import content_terms, tokenize


def toy_corpus() -> Corpus:
    docs = [
        ("alpha#1", "alpha", 0, "Cats", "cats sleep sixteen hours a day"),
        ("alpha#2", "alpha", 1, "Cats", "cats groom their fur daily"),
        ("beta#1", "beta", 0, "Dogs", "dogs need walks and training"),
        ("beta#2", "beta", 1, "Dogs", "dogs bark at the mail carrier"),
    ]
    return Corpus([Chunk(id=i, doc_id=d, index=n, title=t, text=x)
                   for i, d, n, t, x in docs])


class TestRetrievers(unittest.TestCase):
    def setUp(self):
        self.index = Index(toy_corpus())

    def test_bm25_finds_cat_chunk(self):
        self.assertEqual(bm25_rank(self.index, "how long do cats sleep")[0],
                         "alpha#1")

    def test_tfidf_finds_dog_chunk(self):
        self.assertEqual(tfidf_rank(self.index, "dogs bark mail")[0],
                         "beta#2")

    def test_rrf_prefers_consensus(self):
        fused = rrf_fuse([["a", "b", "c"], ["b", "a", "c"]])
        self.assertEqual(set(fused[:2]), {"a", "b"})
        self.assertEqual(fused[2], "c")

    def test_full_ranking_returned(self):
        self.assertEqual(len(bm25_rank(self.index, "cats")), 4)


class TestMetrics(unittest.TestCase):
    def test_attribution(self):
        s1 = ["a", "b", "gold", "c", "d", "e"]
        self.assertEqual(
            stage_attribution(s1, ["gold", "a", "b", "c", "d"], "gold"),
            "held")
        self.assertEqual(
            stage_attribution(s1, ["a", "b", "c", "d", "e", "gold"],
                              "gold"),
            "harm")
        self.assertEqual(
            stage_attribution(["a", "b", "c", "d", "e", "gold"],
                              ["gold", "a", "b", "c", "d"], "gold"),
            "lift")
        self.assertEqual(rank_of(s1, "gold"), 3)
        self.assertIsNone(rank_of(s1, "zzz"))


class TestRerankers(unittest.TestCase):
    def test_lexical_promotes_coverage(self):
        corpus = toy_corpus()
        by_id = corpus.by_id()
        pool = ["beta#1", "alpha#1", "beta#2", "alpha#2"]
        out = lexical_rerank("how long do cats sleep", pool, by_id)
        self.assertEqual(out[0], "alpha#1")

    def test_llm_merge_keeps_pool(self):
        out = merge_llm_order(["a#1", "b#1", "c#1"], ["c#1", "a#1"])
        self.assertEqual(out, ["c#1", "a#1", "b#1"])

    def test_parse_llm_order_from_json(self):
        out = parse_llm_order('["b#1", "a#1"]', ["a#1", "b#1"])
        self.assertEqual(out, ["b#1", "a#1"])


class TestTaxonomy(unittest.TestCase):
    def setUp(self):
        self.corpus = toy_corpus()
        self.index = Index(self.corpus)

    def test_harm_detected(self):
        s1 = ["alpha#1", "beta#1", "beta#2", "alpha#2"]
        final = ["beta#1", "beta#2", "alpha#2", "beta#1", "beta#2",
                 "alpha#1"]
        cat = classify("cats sleep", "alpha#1", {"alpha#1"}, s1,
                       final, True, self.index, self.corpus)
        self.assertIsNotNone(cat)
        self.assertEqual(cat[0], "reranker_harm")

    def test_hit_returns_none(self):
        s1 = ["alpha#1", "beta#1", "beta#2", "alpha#2"]
        cat = classify("cats sleep", "alpha#1", {"alpha#1"}, s1, s1,
                       False, self.index, self.corpus)
        self.assertIsNone(cat)

    def _split_corpus(self) -> Corpus:
        docs = [
            ("d#1", "d", 0, "Departure",
             "the extended departure programme keeps your room until "
             "one in the afternoon and the programme cost is set by "
             "membership"),
            ("d#2", "d", 1, "Departure",
             "members pay nothing for it; everyone else pays thirty "
             "five dollars"),
            ("e#1", "e", 0, "Other", "boats sail across the wide ocean"),
            ("e#2", "e", 1, "Other", "trains run on steel rails northward"),
            ("e#3", "e", 2, "Other", "chefs chop onions in the kitchen"),
            ("e#4", "e", 3, "Other", "bankers count coins at the vault"),
        ]
        return Corpus([Chunk(id=i, doc_id=d, index=n, title=t, text=x)
                       for i, d, n, t, x in docs])

    def test_straddle_detected(self):
        corpus = self._split_corpus()
        index = Index(corpus)
        final = ["e#1", "e#2", "e#3", "e#4", "d#1", "d#2"]
        cat = classify("what does the extended departure programme "
                       "cost", "d#2", {"d#2"}, final, final, False,
                       index, corpus)
        self.assertIsNotNone(cat)
        self.assertEqual(cat[0], "chunk_boundary_straddle")

    def test_vocabulary_detected(self):
        corpus = self._split_corpus()
        index = Index(corpus)
        final = ["e#1", "e#2", "e#3", "e#4", "d#1", "d#2"]
        cat = classify("is there a sauna and a hammam", "d#2",
                       {"d#2"}, final, final, False, index, corpus)
        self.assertIsNotNone(cat)
        self.assertEqual(cat[0], "vocabulary_mismatch")

    def test_flooding_detected(self):
        near = "the pet fee is fifty dollars per stay for dogs"
        docs = [
            ("p#1", "p", 0, "Pets", near),
            ("p#2", "p", 1, "Pets", near + " at the hotel"),
            ("p#3", "p", 2, "Pets", near + " for guests"),
            ("g#1", "g", 0, "Gold",
             "the shuttle fare is fifteen dollars per person"),
            ("f#1", "f", 0, "Filler", "breakfast is served at seven"),
            ("f#2", "f", 1, "Filler", "the pool opens at six"),
        ]
        corpus = Corpus([Chunk(id=i, doc_id=d, index=n, title=t,
                               text=x)
                         for i, d, n, t, x in docs])
        index = Index(corpus)
        final = ["p#1", "p#2", "p#3", "f#1", "f#2", "g#1"]
        cat = classify("shuttle fare person", "g#1", {"g#1"}, final,
                       final, False, index, corpus)
        self.assertIsNotNone(cat)
        self.assertEqual(cat[0], "near_duplicate_flooding")


class TestText(unittest.TestCase):
    def test_content_terms_drop_stopwords(self):
        self.assertEqual(content_terms("What is the fare?"), ["fare"])
        self.assertEqual(tokenize("Wi-Fi, 24/7!"),
                         ["wi", "fi", "24", "7"])


if __name__ == "__main__":
    unittest.main()
