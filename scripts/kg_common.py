"""Shared helpers for the SWO69 KG-quality analysis scripts (T1/T2/T3/T5).

Normalization + entity matching rules follow SWO69/SWO_PLAN.md §3.1:
lowercasing, article/punctuation cleanup, word-boundary substring matching,
and (optionally) embedding cosine similarity above a 0.85 threshold.
"""

from __future__ import annotations

import re
import unicodedata

_WORD_BOUNDARY = r"(?<![\w])"

_ARTICLES = ("the", "a", "an")


def normalize_surface(text: str) -> str:
    """Canonical form used for exact matching: unicode-fold, lowercase,
    strip punctuation at word edges, collapse whitespace, drop a leading
    English article."""
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", text)
    s = s.lower()
    s = re.sub(r"[^0-9a-z぀-ヿ㐀-鿿' \-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for art in _ARTICLES:
        if s.startswith(art + " "):
            s = s[len(art) + 1 :].strip()
            break
    return s


def word_boundary_substring(haystack: str, needle: str) -> bool:
    """True if `needle` occurs in `haystack` at a word boundary (both sides)."""
    if not needle or not haystack:
        return False
    if needle == haystack:
        return True
    pattern = _WORD_BOUNDARY + re.escape(needle) + r"(?![\w])"
    return re.search(pattern, haystack) is not None


def entity_variants(surface: str, aliases: list[str] | None = None) -> list[str]:
    """Normalized candidate surfaces for one entity (itself + aliases)."""
    out = [normalize_surface(surface)]
    for a in aliases or []:
        n = normalize_surface(a)
        if n and n not in out:
            out.append(n)
    return [v for v in out if v]


def surface_in_text(surface: str, text: str) -> bool:
    """Whether an entity surface appears in free text (retrieval context).

    Uses the same rules as graph-node matching: normalized exact or
    word-boundary substring, on the primary surface or any alias.
    """
    hay = normalize_surface(text)
    for needle in entity_variants(surface):
        if word_boundary_substring(hay, needle):
            return True
    return False


def match_entity(
    gold_surface: str,
    gold_aliases: list[str] | None,
    candidate_surface: str,
    candidate_aliases: list[str] | None = None,
) -> bool:
    """Entity equivalence per §3.1 (exact, alias, word-boundary substring)."""
    gold_vs = entity_variants(gold_surface, gold_aliases)
    cand_vs = entity_variants(candidate_surface, candidate_aliases)
    for g in gold_vs:
        for c in cand_vs:
            if g == c:
                return True
            if word_boundary_substring(c, g) or word_boundary_substring(g, c):
                return True
    return False
